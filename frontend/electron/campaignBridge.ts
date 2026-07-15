import { randomBytes, randomUUID } from 'node:crypto'

export type CampaignMethod = 'GET' | 'POST'
export type CampaignQueryValue = string | number | boolean
export type CampaignBody = Record<string, unknown>
export type CampaignQuery = Record<string, CampaignQueryValue>

export interface CampaignResponse {
  ok: boolean
  status: number
  data?: unknown
  error?: string
}

type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>

const IDENTIFIER = '[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}'
const GET_ROUTES = [
  /^\/api\/campaign-auth\/capabilities$/,
  /^\/api\/campaigns$/,
  /^\/api\/campaigns\/templates$/,
  new RegExp(`^/api/campaigns/${IDENTIFIER}$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/autoresearch$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/(?:events|artifacts|attempts|comparisons|proposals|studies|evidence|ledger)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/manifest/[1-9][0-9]*$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/studies/${IDENTIFIER}$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/attempts/${IDENTIFIER}/metrics$`),
]
const POST_ROUTES = [
  /^\/api\/campaigns$/,
  /^\/api\/campaigns\/from-template$/,
  new RegExp(`^/api/campaigns/${IDENTIFIER}/(?:start|pause|resume|cancel|conclude|advance|protected-lease|protected-result|promotion|export)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/manifest/revise$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/proposals$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/autoresearch/(?:baseline|candidates|results)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/proposals/${IDENTIFIER}/withdraw$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/proposals/${IDENTIFIER}/code-lineage/(?:prepare|capture)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/actions/${IDENTIFIER}/(?:retry|force-stop)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/studies/${IDENTIFIER}/abandon$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/budget/amend$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/sources/${IDENTIFIER}/approve$`),
]
const MAX_ROUTE_LENGTH = 512
const MAX_BODY_BYTES = 64 * 1024
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const MAX_QUERY_ENTRIES = 8
const MAX_QUERY_VALUE_LENGTH = 512

// A cold desktop backend rebuilds the local trace index before FastAPI reports
// readiness. Real workspaces can take more than a minute, so the Electron main
// process must not terminate an otherwise healthy child after a short timeout.
export const MANAGED_BACKEND_STARTUP_TIMEOUT_MS = 120_000

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function queryKeysForRoute(route: string): ReadonlySet<string> {
  if (route === '/api/campaign-auth/capabilities') return new Set()
  if (route === '/api/campaigns') return new Set(['workspace_id', 'kind', 'status'])
  if (/\/events$/.test(route)) return new Set(['workspace_id', 'after_cursor', 'limit'])
  if (/\/metrics$/.test(route)) {
    return new Set(['workspace_id', 'metric_name', 'source', 'after_step', 'limit'])
  }
  return new Set(['workspace_id'])
}

function assertJsonDepth(value: unknown, depth = 0): void {
  if (depth > 8) throw new Error('Campaign request body is too deeply nested')
  if (Array.isArray(value)) {
    value.forEach((item) => assertJsonDepth(item, depth + 1))
    return
  }
  if (isPlainRecord(value)) {
    Object.entries(value).forEach(([key, nested]) => {
      if (!key || key.length > 240 || /[\r\n]/.test(key)) {
        throw new Error('Campaign request body contains an invalid key')
      }
      assertJsonDepth(nested, depth + 1)
    })
    return
  }
  if (
    value !== null
    && !['string', 'number', 'boolean'].includes(typeof value)
  ) {
    throw new Error('Campaign request body must contain JSON values only')
  }
  if (typeof value === 'number' && !Number.isFinite(value)) {
    throw new Error('Campaign request body contains a non-finite number')
  }
}

export function validateCampaignRequest(
  method: CampaignMethod,
  route: string,
  body?: CampaignBody,
  query?: CampaignQuery,
): void {
  if (!['GET', 'POST'].includes(method)) throw new Error('Unsupported campaign method')
  if (
    !route
    || route.length > MAX_ROUTE_LENGTH
    || route.includes('?')
    || route.includes('#')
    || route.includes('..')
    || route.includes('\\')
  ) {
    throw new Error('Invalid campaign route')
  }
  const patterns = method === 'GET' ? GET_ROUTES : POST_ROUTES
  if (!patterns.some((pattern) => pattern.test(route))) {
    throw new Error('Campaign route is not allowlisted')
  }
  if (method === 'GET' && body !== undefined) {
    throw new Error('GET campaign requests cannot include a body')
  }
  if (method === 'POST') {
    if (!isPlainRecord(body)) throw new Error('POST campaign requests require an object body')
    assertJsonDepth(body)
    const encoded = JSON.stringify(body)
    if (Buffer.byteLength(encoded, 'utf8') > MAX_BODY_BYTES) {
      throw new Error('Campaign request body is too large')
    }
  }
  const entries = Object.entries(query ?? {})
  if (entries.length > MAX_QUERY_ENTRIES) throw new Error('Too many campaign query fields')
  if (method === 'POST' && entries.length > 0) {
    throw new Error('POST campaign requests cannot include query fields')
  }
  const allowedQueryKeys = queryKeysForRoute(route)
  entries.forEach(([key, value]) => {
    if (!allowedQueryKeys.has(key)) {
      throw new Error('Campaign query field is not allowlisted')
    }
    if (
      !['string', 'number', 'boolean'].includes(typeof value)
      || (typeof value === 'number' && !Number.isFinite(value))
      || String(value).length > MAX_QUERY_VALUE_LENGTH
      || /[\r\n]/.test(String(value))
    ) {
      throw new Error('Invalid campaign query value')
    }
  })
}

export function createDesktopBootstrapToken(): string {
  const launchId = randomUUID().replaceAll('-', '')
  return `bgcb.electron-${launchId}.${randomBytes(32).toString('base64url')}`
}

export function buildBackendChildEnvironment(
  base: NodeJS.ProcessEnv,
  bootstrapToken: string,
): NodeJS.ProcessEnv {
  return {
    ...base,
    BASHGYM_DESKTOP_BOOTSTRAP_SECRET: bootstrapToken,
    BASHGYM_MODE: 'desktop',
    PYTHONUTF8: '1',
    PYTHONIOENCODING: 'utf-8',
  }
}

export function resolveCampaignApiOrigin(value?: string): string {
  const url = new URL(value || 'http://127.0.0.1:8003')
  if (url.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) {
    throw new Error('Campaign API must use a loopback HTTP origin')
  }
  return url.origin
}

async function responsePayload(response: Response): Promise<unknown> {
  const text = await response.text()
  if (text.length > MAX_RESPONSE_BYTES) throw new Error('Campaign response is too large')
  if (!text) return undefined
  try {
    return JSON.parse(text)
  } catch {
    return undefined
  }
}

function safeError(status: number): string {
  return `Campaign request failed with HTTP ${status}`
}

export class CampaignBridgeClient {
  private accessToken: string | null = null
  private exchangeInFlight: Promise<string> | null = null

  constructor(
    private readonly apiOrigin: string,
    private readonly bootstrapToken: string,
    private readonly fetchImpl: FetchLike = fetch,
    private readonly idFactory: () => string = randomUUID,
  ) {}

  private async exchange(force = false): Promise<string> {
    if (this.accessToken && !force) return this.accessToken
    if (this.exchangeInFlight) return this.exchangeInFlight
    this.exchangeInFlight = (async () => {
      const response = await this.fetchImpl(`${this.apiOrigin}/api/campaign-auth/exchange`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.bootstrapToken}` },
      })
      const payload = await responsePayload(response)
      if (
        !response.ok
        || !isPlainRecord(payload)
        || typeof payload.raw_token !== 'string'
        || !payload.raw_token.startsWith('bgca.')
        || payload.raw_token.length > 1000
      ) {
        throw new Error('Campaign desktop authentication is unavailable')
      }
      this.accessToken = payload.raw_token
      return payload.raw_token
    })().finally(() => {
      this.exchangeInFlight = null
    })
    return this.exchangeInFlight
  }

  async initialize(): Promise<void> {
    await this.exchange()
  }

  async request(
    method: CampaignMethod,
    route: string,
    body?: CampaignBody,
    query?: CampaignQuery,
  ): Promise<CampaignResponse> {
    validateCampaignRequest(method, route, body, query)
    const url = new URL(route, this.apiOrigin)
    Object.entries(query ?? {}).forEach(([key, value]) => {
      url.searchParams.set(key, String(value))
    })
    const mutationId = method === 'POST' ? this.idFactory() : null
    const execute = async (token: string) => {
      const headers: Record<string, string> = { Authorization: `Bearer ${token}` }
      if (method === 'POST') {
        headers['Content-Type'] = 'application/json'
        headers['Idempotency-Key'] = `desktop-${mutationId}`
        headers['X-Correlation-ID'] = `desktop-${mutationId}`
      }
      return this.fetchImpl(url, {
        method,
        headers,
        ...(method === 'POST' ? { body: JSON.stringify(body) } : {}),
      })
    }

    let response = await execute(await this.exchange())
    if (response.status === 401) {
      this.accessToken = null
      response = await execute(await this.exchange(true))
    }
    const data = await responsePayload(response)
    return response.ok
      ? { ok: true, status: response.status, data }
      : { ok: false, status: response.status, error: safeError(response.status) }
  }

  dispose(): void {
    this.accessToken = null
  }
}
