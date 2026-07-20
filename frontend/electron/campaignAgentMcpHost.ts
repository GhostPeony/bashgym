import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { randomBytes, timingSafeEqual } from 'node:crypto'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js'
import { isInitializeRequest } from '@modelcontextprotocol/sdk/types.js'
import { z } from 'zod'

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const ARTIFACT_CURSOR = /^a1\.[A-Za-z0-9_-]{11}$/
const SHA256 = /^[0-9a-f]{64}$/
const CANONICAL_UTC_SECOND = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/
const LAUNCH_HEADER = 'X-BashGym-MCP-Launch'
const MAX_REQUEST_BYTES = 64 * 1024

const identifier = z.string().regex(IDENTIFIER)
const nullableIdentifier = identifier.nullable()
const scopeSchema = z
  .object({
    workspaceId: identifier,
    campaignId: identifier
  })
  .strict()
const observationSchema = z
  .object({
    schemaVersion: z.literal('campaign_agent_observation.v1'),
    scope: scopeSchema,
    campaign: z
      .object({
        status: z.enum([
          'draft',
          'validating',
          'ready',
          'active',
          'paused',
          'awaiting_authority',
          'cancelling',
          'completed',
          'exhausted',
          'failed',
          'cancelled'
        ]),
        version: z.number().int().safe().min(1),
        manifestRevision: z.number().int().safe().min(1),
        activeStudyId: nullableIdentifier,
        activeActionId: nullableIdentifier,
        latestEventCursor: z.number().int().safe().min(0)
      })
      .strict(),
    agent: z
      .object({
        attachmentId: identifier,
        attachmentVersion: z.number().int().safe().min(1),
        agentFamily: z.enum(['codex', 'hermes']),
        agentPrincipalId: identifier,
        authorizedCapability: z.literal('campaign_observe')
      })
      .strict()
  })
  .strict()
const artifactItemSchema = z
  .object({
    artifactId: identifier,
    producerActionId: nullableIdentifier,
    sha256: z.string().regex(SHA256),
    sizeBytes: z.number().int().safe().min(0),
    schemaName: identifier,
    sealed: z.boolean(),
    valid: z.boolean(),
    createdAt: z.string().regex(CANONICAL_UTC_SECOND).datetime({ offset: false })
  })
  .strict()
const artifactPageSchema = z
  .object({
    schemaVersion: z.literal('campaign_agent_artifact_page.v1'),
    scope: scopeSchema,
    items: z.array(artifactItemSchema).max(50),
    nextCursor: z.string().regex(ARTIFACT_CURSOR).nullable(),
    hasMore: z.boolean()
  })
  .strict()
const artifactArgsSchema = z
  .object({
    afterCursor: z.string().regex(ARTIFACT_CURSOR).optional(),
    limit: z.number().int().min(1).max(50).optional()
  })
  .strict()

export interface CampaignAgentArtifactArgs {
  afterCursor?: string
  limit?: number
}

export interface CampaignAgentMcpHostOptions {
  terminalId: string
  generation: string
  scope: { workspaceId: string; campaignId: string }
  observe(): Promise<unknown>
  artifacts(args: CampaignAgentArtifactArgs): Promise<unknown>
}

export interface CampaignAgentMcpLaunch {
  url: string
  headers: Record<typeof LAUNCH_HEADER, string>
  terminalId: string
  generation: string
}

function publicIdentifier(value: string, label: string): string {
  if (!IDENTIFIER.test(value)) throw new Error(`Campaign agent MCP ${label} is invalid`)
  return value
}

function writeJson(res: ServerResponse, status: number, message: string): void {
  if (res.headersSent) return
  const body = JSON.stringify({ error: message })
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
    'Cache-Control': 'no-store'
  })
  res.end(body)
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
    size += bytes.length
    if (size > MAX_REQUEST_BYTES) throw new Error('request_too_large')
    chunks.push(bytes)
  }
  if (size === 0) throw new Error('invalid_json')
  try {
    return JSON.parse(Buffer.concat(chunks, size).toString('utf8')) as unknown
  } catch {
    throw new Error('invalid_json')
  } finally {
    chunks.forEach((chunk) => chunk.fill(0))
  }
}

function authorityUnavailable() {
  return {
    isError: true as const,
    content: [{ type: 'text' as const, text: 'Campaign authority is unavailable.' }]
  }
}

export class CampaignAgentMcpHost {
  private static readonly activeGenerations = new Set<string>()
  private readonly terminalId: string
  private readonly generation: string
  private readonly generationKey: string
  private readonly scope: { workspaceId: string; campaignId: string }
  private readonly launchSecret = randomBytes(32)
  private readonly route = randomBytes(32).toString('base64url')
  private httpServer: Server | null = null
  private mcpServer: McpServer | null = null
  private transport: StreamableHTTPServerTransport | null = null
  private activeSessionId: string | null = null
  private initializationClaimed = false
  private authorityAvailable = true
  private closed = false
  private generationClaimed = false

  constructor(private readonly options: CampaignAgentMcpHostOptions) {
    this.terminalId = publicIdentifier(options.terminalId, 'terminal id')
    this.generation = publicIdentifier(options.generation, 'PTY generation')
    this.generationKey = `${this.terminalId}\0${this.generation}`
    this.scope = scopeSchema.parse(options.scope)
  }

  private launchAuthorized(req: IncomingMessage): boolean {
    const supplied = req.headers['x-bashgym-mcp-launch']
    if (typeof supplied !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(supplied)) return false
    const decoded = Buffer.from(supplied, 'base64url')
    try {
      return (
        decoded.length === this.launchSecret.length && timingSafeEqual(decoded, this.launchSecret)
      )
    } finally {
      decoded.fill(0)
    }
  }

  private sameScope(value: { workspaceId: string; campaignId: string }): boolean {
    return (
      value.workspaceId === this.scope.workspaceId && value.campaignId === this.scope.campaignId
    )
  }

  private makeMcpServer(): McpServer {
    const server = new McpServer({ name: 'bashgym-campaign-agent', version: '1.0.0' })
    server.registerTool(
      'campaign_observe',
      {
        description: 'Read the bounded public state of the authorized AutoResearch campaign.',
        inputSchema: z.object({}).strict(),
        outputSchema: observationSchema,
        annotations: {
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false
        }
      },
      async () => {
        if (!this.authorityAvailable || this.closed) return authorityUnavailable()
        try {
          const value = observationSchema.parse(await this.options.observe())
          if (!this.sameScope(value.scope)) return authorityUnavailable()
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(value) }],
            structuredContent: value
          }
        } catch {
          return authorityUnavailable()
        }
      }
    )
    server.registerTool(
      'campaign_artifacts',
      {
        description:
          'Read one bounded, URI-free artifact page for the authorized AutoResearch campaign.',
        inputSchema: artifactArgsSchema,
        outputSchema: artifactPageSchema,
        annotations: {
          readOnlyHint: true,
          destructiveHint: false,
          idempotentHint: true,
          openWorldHint: false
        }
      },
      async (args) => {
        if (!this.authorityAvailable || this.closed) return authorityUnavailable()
        try {
          const validatedArgs = artifactArgsSchema.parse(args)
          const value = artifactPageSchema.parse(await this.options.artifacts(validatedArgs))
          const limit = validatedArgs.limit ?? 20
          if (
            !this.sameScope(value.scope) ||
            value.items.length > limit ||
            value.hasMore !== (value.nextCursor !== null)
          )
            return authorityUnavailable()
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(value) }],
            structuredContent: value
          }
        } catch {
          return authorityUnavailable()
        }
      }
    )
    return server
  }

  private async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (this.closed || !this.launchAuthorized(req)) {
      writeJson(res, this.closed ? 410 : 401, this.closed ? 'host_closed' : 'unauthorized')
      return
    }
    if (req.url !== `/${this.route}`) {
      writeJson(res, 404, 'not_found')
      return
    }
    if (!['GET', 'POST', 'DELETE'].includes(req.method ?? '')) {
      writeJson(res, 405, 'method_not_allowed')
      return
    }
    const suppliedSession = req.headers['mcp-session-id']
    if (
      suppliedSession !== undefined &&
      (typeof suppliedSession !== 'string' || suppliedSession !== this.activeSessionId)
    ) {
      writeJson(res, 404, 'session_unavailable')
      return
    }

    try {
      let body: unknown = undefined
      if (req.method === 'POST') body = await readJson(req)
      if (!this.transport) {
        if (req.method !== 'POST' || suppliedSession !== undefined || !isInitializeRequest(body)) {
          writeJson(res, 400, 'initialization_required')
          return
        }
        if (this.initializationClaimed) {
          writeJson(res, 409, 'session_already_claimed')
          return
        }
        this.initializationClaimed = true
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => randomBytes(32).toString('base64url'),
          onsessioninitialized: (sessionId) => {
            this.activeSessionId = sessionId
          }
        })
        const server = this.makeMcpServer()
        this.transport = transport
        this.mcpServer = server
        await server.connect(transport)
      } else if (suppliedSession === undefined || suppliedSession !== this.activeSessionId) {
        writeJson(res, 409, 'session_already_claimed')
        return
      }
      await this.transport.handleRequest(req, res, body)
    } catch {
      writeJson(res, 400, 'invalid_request')
    }
  }

  async start(): Promise<CampaignAgentMcpLaunch> {
    if (this.closed) throw new Error('Campaign agent MCP host is closed')
    if (!this.authorityAvailable) throw new Error('Campaign authority is unavailable')
    if (this.httpServer) throw new Error('Campaign agent MCP host is already started')
    if (CampaignAgentMcpHost.activeGenerations.has(this.generationKey)) {
      throw new Error('PTY generation already has an active MCP host')
    }
    CampaignAgentMcpHost.activeGenerations.add(this.generationKey)
    this.generationClaimed = true
    const server = createServer((req, res) => {
      void this.handle(req, res)
    })
    this.httpServer = server
    try {
      await new Promise<void>((resolve, reject) => {
        const onError = (error: Error) => {
          server.off('listening', onListening)
          reject(error)
        }
        const onListening = () => {
          server.off('error', onError)
          resolve()
        }
        server.once('error', onError)
        server.once('listening', onListening)
        server.listen({ host: '127.0.0.1', port: 0, exclusive: true })
      })
    } catch (error) {
      this.httpServer = null
      CampaignAgentMcpHost.activeGenerations.delete(this.generationKey)
      this.generationClaimed = false
      throw error
    }
    const address = server.address()
    if (!address || typeof address === 'string') {
      await this.close()
      throw new Error('Campaign agent MCP loopback listener is unavailable')
    }
    return {
      url: `http://127.0.0.1:${address.port}/${this.route}`,
      headers: { [LAUNCH_HEADER]: this.launchSecret.toString('base64url') },
      terminalId: this.terminalId,
      generation: this.generation
    }
  }

  lock(): void {
    this.authorityAvailable = false
  }

  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    this.authorityAvailable = false
    const mcpServer = this.mcpServer
    const httpServer = this.httpServer
    this.mcpServer = null
    this.transport = null
    this.httpServer = null
    this.activeSessionId = null
    this.launchSecret.fill(0)
    if (this.generationClaimed) {
      CampaignAgentMcpHost.activeGenerations.delete(this.generationKey)
      this.generationClaimed = false
    }
    await mcpServer?.close().catch(() => undefined)
    if (httpServer) {
      await new Promise<void>((resolve) => httpServer.close(() => resolve()))
    }
  }
}
