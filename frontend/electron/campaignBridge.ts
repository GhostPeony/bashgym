import { randomBytes, randomUUID } from 'node:crypto'
import type { CampaignAgentHostRegistrationBody } from './campaignAgentHost'

export type CampaignMethod = 'GET' | 'POST'
export type CampaignQueryValue = string | number | boolean
export type CampaignBody = Record<string, unknown>
export type CampaignQuery = Record<string, CampaignQueryValue>

export interface CampaignRequestAuthority {
  idempotencyKey?: string
}

export interface CampaignResponse {
  ok: boolean
  status: number
  data?: unknown
  error?: string
  code?: string
  details?: unknown
}

export interface CampaignAgentHeartbeatBody {
  scope: { workspaceId: string; campaignId: string }
  agentFamily: 'codex' | 'hermes'
  agentOrigin: string
  agentPrincipalId: string
  sessionId: string
  resumeCursor?: string
  resumeSequence?: number
  expectedResumeCursor?: string
}

export interface CampaignAgentArtifactQuery {
  afterCursor?: string
  limit?: number
}

type FetchLike = (input: string | URL, init?: RequestInit) => Promise<Response>

const IDENTIFIER = '[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}'
const GET_ROUTES = [
  /^\/api\/campaign-auth\/capabilities$/,
  /^\/api\/campaigns$/,
  /^\/api\/campaigns\/templates$/,
  /^\/api\/campaigns\/setup\/context$/,
  new RegExp(`^/api/campaigns/${IDENTIFIER}$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/autoresearch$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/control-room-snapshot$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/human-work$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/agent-attachment$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/recovery$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/artifacts/${IDENTIFIER}/preview$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/(?:events|artifacts|attempts|comparisons|proposals|studies|evidence|ledger)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/manifest/[1-9][0-9]*$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/studies/${IDENTIFIER}$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/attempts/${IDENTIFIER}/metrics$`),
]
const POST_ROUTES = [
  /^\/api\/campaigns$/,
  /^\/api\/campaigns\/from-template$/,
  /^\/api\/campaigns\/live-ticket$/,
  /^\/api\/campaigns\/setup\/(?:session|doctor|validate|create)$/,
  new RegExp(`^/api/campaigns/${IDENTIFIER}/(?:start|pause|resume|cancel|conclude|advance|protected-lease|protected-result|promotion|export)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/human-work/${IDENTIFIER}/(?:claim|submit)$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/human-promotion$`),
  new RegExp(`^/api/campaigns/${IDENTIFIER}/recovery/(?:resume|repair|takeover)$`),
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
const CAMPAIGN_REQUEST_TIMEOUT_MS = 5_000
const MAX_QUERY_ENTRIES = 8
const MAX_QUERY_VALUE_LENGTH = 512
const CAMPAIGN_AGENT_ARTIFACT_CURSOR = /^a1\.[A-Za-z0-9_-]{11}$/
const CAMPAIGN_AGENT_CREDENTIAL = /^bgag\.[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\.[A-Za-z0-9_-]{32,512}$/
const CAMPAIGN_AGENT_CREDENTIAL_FAILURE = 'Campaign agent credential request failed'
const PUBLIC_IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const SETUP_SESSION_ID = /^setupsess_[0-9a-f]{32}$/
const SETUP_INSTALLATION_ID = /^ins_[0-9a-f]{32}$/
const SETUP_VALIDATION_RECEIPT_ID = /^setuprcpt_[0-9a-f]{32}$/
const SETUP_STEPS = new Set(['template', 'installation', 'model', 'data', 'compute', 'evaluation'])
const SETUP_BINDING_KEYS = ['model', 'data', 'compute', 'evaluation'] as const

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
  if (route === '/api/campaigns/setup/context') return new Set(['workspace_id', 'session_id'])
  if (/\/events$/.test(route)) return new Set(['workspace_id', 'after_cursor', 'limit'])
  if (/\/artifacts$/.test(route)) return new Set(['workspace_id', 'after_cursor', 'limit'])
  if (/\/human-work$/.test(route)) return new Set(['workspace_id', 'limit'])
  if (/\/agent-attachment$/.test(route)) return new Set(['workspace_id', 'after_sequence', 'limit'])
  if (/\/recovery$/.test(route)) return new Set(['workspaceId'])
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

function exactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value)
  return actual.length === keys.length && keys.every((key) => key in value)
}

function validSetupDraft(body: CampaignBody): boolean {
  if (!exactKeys(body, ['workspace_id', 'template_id', 'installation_id', 'bindings'])
    || typeof body.workspace_id !== 'string' || !PUBLIC_IDENTIFIER.test(body.workspace_id)
    || typeof body.template_id !== 'string' || !PUBLIC_IDENTIFIER.test(body.template_id)
    || typeof body.installation_id !== 'string' || !SETUP_INSTALLATION_ID.test(body.installation_id)
    || !isPlainRecord(body.bindings) || !exactKeys(body.bindings, SETUP_BINDING_KEYS)) return false
  const bindings = body.bindings
  return SETUP_BINDING_KEYS.every((key) => typeof bindings[key] === 'string' && PUBLIC_IDENTIFIER.test(bindings[key] as string))
}

function validateGuidedSetupRequest(method: CampaignMethod, route: string, body?: CampaignBody, query?: CampaignQuery): void {
  if (!route.startsWith('/api/campaigns/setup/')) return
  let valid = false
  if (method === 'GET' && route === '/api/campaigns/setup/context') {
    const keys = Object.keys(query ?? {})
    valid = keys.length >= 1 && keys.length <= 2
      && typeof query?.workspace_id === 'string' && PUBLIC_IDENTIFIER.test(query.workspace_id)
      && (query.session_id === undefined || (typeof query.session_id === 'string' && SETUP_SESSION_ID.test(query.session_id)))
  } else if (method === 'POST' && route === '/api/campaigns/setup/session' && body) {
    valid = exactKeys(body, ['workspace_id', 'session_id', 'expected_version', 'step', 'selection_id'])
      && typeof body.workspace_id === 'string' && PUBLIC_IDENTIFIER.test(body.workspace_id)
      && typeof body.session_id === 'string' && SETUP_SESSION_ID.test(body.session_id)
      && Number.isSafeInteger(body.expected_version) && Number(body.expected_version) >= 0 && Number(body.expected_version) <= 6
      && typeof body.step === 'string' && SETUP_STEPS.has(body.step)
      && typeof body.selection_id === 'string' && PUBLIC_IDENTIFIER.test(body.selection_id)
  } else if (method === 'POST' && (route === '/api/campaigns/setup/doctor' || route === '/api/campaigns/setup/validate') && body) {
    valid = validSetupDraft(body)
  } else if (method === 'POST' && route === '/api/campaigns/setup/create' && body) {
    valid = exactKeys(body, ['workspace_id', 'campaign_id', 'title', 'validation_receipt_id'])
      && typeof body.workspace_id === 'string' && PUBLIC_IDENTIFIER.test(body.workspace_id)
      && typeof body.campaign_id === 'string' && PUBLIC_IDENTIFIER.test(body.campaign_id)
      && typeof body.title === 'string' && body.title.length >= 1 && body.title.length <= 240 && !/[\r\n]/.test(body.title)
      && typeof body.validation_receipt_id === 'string' && SETUP_VALIDATION_RECEIPT_ID.test(body.validation_receipt_id)
  }
  if (!valid) throw new Error('Invalid guided setup request contract')
}

export function validateCampaignRequest(
  method: CampaignMethod,
  route: string,
  body?: CampaignBody,
  query?: CampaignQuery,
  authority?: CampaignRequestAuthority,
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
  validateGuidedSetupRequest(method, route, body, query)
  if (authority !== undefined) {
    if (!isPlainRecord(authority) || Object.keys(authority).some((key) => key !== 'idempotencyKey')) {
      throw new Error('Invalid campaign request authority')
    }
    if (method !== 'POST') throw new Error('Campaign idempotency authority requires POST')
    const idempotencyKey = authority.idempotencyKey
    if (
      idempotencyKey !== undefined
      && (typeof idempotencyKey !== 'string' || !/^idem_[A-Za-z0-9_-]{16,120}$/.test(idempotencyKey))
    ) {
      throw new Error('Invalid campaign idempotency key')
    }
  }
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

class CampaignAgentAuthorityError extends Error {
  readonly status: 401 | 403

  constructor(status: 401 | 403) {
    super(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
    this.name = 'CampaignAgentAuthorityError'
    this.status = status
  }
}

function sanitizedCampaignAgentError(error: unknown): Error {
  return error instanceof CampaignAgentAuthorityError
    ? error
    : new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
}

async function campaignAgentCredentialPayload(response: Response): Promise<unknown> {
  const declaredLength = response.headers.get('content-length')
  if (declaredLength !== null) {
    const bytes = Number(declaredLength)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > MAX_RESPONSE_BYTES) {
      throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
    }
  }

  const chunks: Buffer[] = []
  let totalBytes = 0
  const reader = response.body?.getReader()
  if (reader) {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      totalBytes += value.byteLength
      if (totalBytes > MAX_RESPONSE_BYTES) {
        await reader.cancel()
        throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
      }
      chunks.push(Buffer.from(value))
    }
  }
  const combined = Buffer.concat(chunks, totalBytes)
  try {
    const text = combined.toString('utf8')
    if (text.includes('bgag.')) {
      throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
    }
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        throw new CampaignAgentAuthorityError(response.status)
      }
      throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
    }
    if (!text) return undefined
    try {
      const payload = JSON.parse(text) as unknown
      if (JSON.stringify(payload).includes('bgag.')) {
        throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
      }
      return payload
    } catch (error) {
      if (error instanceof Error && error.message === CAMPAIGN_AGENT_CREDENTIAL_FAILURE) {
        throw error
      }
      return undefined
    }
  } finally {
    combined.fill(0)
    chunks.forEach((chunk) => chunk.fill(0))
  }
}

function campaignAgentCredentialToken(credential: Buffer): string {
  if (!Buffer.isBuffer(credential) || credential.byteLength > 1000) {
    throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
  }
  const token = credential.toString('utf8')
  if (!CAMPAIGN_AGENT_CREDENTIAL.test(token)) {
    throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
  }
  return token
}

function safeError(status: number): string {
  return `Campaign request failed with HTTP ${status}`
}

function structuredErrorMetadata(payload: unknown): Pick<CampaignResponse, 'code' | 'details'> {
  if (!isPlainRecord(payload) || !isPlainRecord(payload.detail)) return {}
  const code = typeof payload.detail.code === 'string'
    && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(payload.detail.code)
    ? payload.detail.code
    : undefined
  if (!code) return {}
  if (code !== 'campaign_event_cursor_expired' || !isPlainRecord(payload.detail.details)) {
    return { code }
  }
  const resumeCursor = payload.detail.details.resume_cursor
  if (!Number.isSafeInteger(resumeCursor) || Number(resumeCursor) < 0) return { code }
  return { code, details: { resume_cursor: Number(resumeCursor) } }
}

export class CampaignBridgeClient {
  private accessToken: string | null = null
  private exchangeInFlight: Promise<string> | null = null

  constructor(
    private readonly apiOrigin: string,
    private readonly bootstrapToken: string,
    private readonly fetchImpl: FetchLike = fetch,
    private readonly idFactory: () => string = randomUUID,
    private readonly requestTimeoutMs: number = CAMPAIGN_REQUEST_TIMEOUT_MS,
  ) {}

  private async exchange(force = false): Promise<string> {
    if (this.accessToken && !force) return this.accessToken
    if (this.exchangeInFlight) return this.exchangeInFlight
    this.exchangeInFlight = (async () => {
      const response = await this.fetchImpl(`${this.apiOrigin}/api/campaign-auth/exchange`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${this.bootstrapToken}` },
        signal: AbortSignal.timeout(this.requestTimeoutMs),
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

  private async mainOnlyCampaignAgentPost(route: string, body?: CampaignBody): Promise<unknown> {
    const mutationId = this.idFactory()
    const execute = async (token: string) => {
      const headers: Record<string, string> = {
        Authorization: `Bearer ${token}`,
        'X-Correlation-ID': `desktop-agent-${mutationId}`,
      }
      if (body !== undefined) headers['Content-Type'] = 'application/json'
      return this.fetchImpl(new URL(route, this.apiOrigin), {
        method: 'POST',
        headers,
        signal: AbortSignal.timeout(5_000),
        ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
      })
    }
    let response = await execute(await this.exchange())
    if (response.status === 401) {
      this.accessToken = null
      response = await execute(await this.exchange(true))
    }
    const payload = await responsePayload(response)
    if (!response.ok) throw new Error(`Campaign agent host request failed with HTTP ${response.status}`)
    return payload
  }

  registerCampaignAgentHostSession(body: CampaignAgentHostRegistrationBody): Promise<unknown> {
    return this.mainOnlyCampaignAgentPost('/api/campaign-agent/sessions', body as unknown as CampaignBody)
  }

  issueCampaignAgentGrant(campaignId: string, body: CampaignBody): Promise<unknown> {
    if (!PUBLIC_IDENTIFIER.test(campaignId)) throw new Error('Invalid campaign id')
    return this.mainOnlyCampaignAgentPost(`/api/campaigns/${campaignId}/agent-grant`, body)
  }

  attachCampaignAgent(campaignId: string, body: CampaignBody): Promise<unknown> {
    if (!PUBLIC_IDENTIFIER.test(campaignId)) throw new Error('Invalid campaign id')
    return this.mainOnlyCampaignAgentPost(`/api/campaigns/${campaignId}/agent-attachment`, body)
  }

  revokeCampaignAgentAttachment(
    campaignId: string,
    attachmentId: string,
    body: CampaignBody,
  ): Promise<unknown> {
    if (!PUBLIC_IDENTIFIER.test(campaignId) || !PUBLIC_IDENTIFIER.test(attachmentId)) {
      throw new Error('Invalid campaign attachment identity')
    }
    return this.mainOnlyCampaignAgentPost(
      `/api/campaigns/${campaignId}/agent-attachment/${attachmentId}/revoke`,
      body,
    )
  }

  claimCampaignAgentDelivery(registrationId: string): Promise<unknown> {
    if (!PUBLIC_IDENTIFIER.test(registrationId)) throw new Error('Invalid campaign agent registration id')
    return this.mainOnlyCampaignAgentPost(
      `/api/campaign-agent/sessions/${registrationId}/deliveries/claim`,
    )
  }

  revokeCampaignAgentHostSession(registrationId: string): Promise<unknown> {
    if (!PUBLIC_IDENTIFIER.test(registrationId)) throw new Error('Invalid campaign agent registration id')
    return this.mainOnlyCampaignAgentPost(`/api/campaign-agent/sessions/${registrationId}/revoke`)
  }

  async heartbeatCampaignAgent(
    credential: Buffer,
    body: CampaignAgentHeartbeatBody,
  ): Promise<unknown> {
    try {
      const token = campaignAgentCredentialToken(credential)
      const encodedBody = JSON.stringify(body)
      if (encodedBody.includes('bgag.') || Buffer.byteLength(encodedBody, 'utf8') > MAX_BODY_BYTES) {
        throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
      }
      const response = await this.fetchImpl(
        new URL('/api/campaign-agent/heartbeat', this.apiOrigin),
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: encodedBody,
          signal: AbortSignal.timeout(5_000),
        },
      )
      return await campaignAgentCredentialPayload(response)
    } catch (error) {
      throw sanitizedCampaignAgentError(error)
    }
  }

  async observeCampaignAsAgent(credential: Buffer): Promise<unknown> {
    try {
      const token = campaignAgentCredentialToken(credential)
      const response = await this.fetchImpl(
        new URL('/api/campaign-agent/actions/observe', this.apiOrigin),
        {
          method: 'GET',
          headers: { Authorization: `Bearer ${token}` },
          signal: AbortSignal.timeout(5_000),
        },
      )
      return await campaignAgentCredentialPayload(response)
    } catch (error) {
      throw sanitizedCampaignAgentError(error)
    }
  }

  async listCampaignArtifactsAsAgent(
    credential: Buffer,
    query: CampaignAgentArtifactQuery = {},
  ): Promise<unknown> {
    try {
      const token = campaignAgentCredentialToken(credential)
      if (
        (query.afterCursor !== undefined && !CAMPAIGN_AGENT_ARTIFACT_CURSOR.test(query.afterCursor))
        || (query.limit !== undefined && (!Number.isSafeInteger(query.limit) || query.limit < 1 || query.limit > 50))
      ) {
        throw new Error(CAMPAIGN_AGENT_CREDENTIAL_FAILURE)
      }
      const url = new URL('/api/campaign-agent/actions/artifacts', this.apiOrigin)
      if (query.afterCursor !== undefined) url.searchParams.set('after_cursor', query.afterCursor)
      if (query.limit !== undefined) url.searchParams.set('limit', String(query.limit))
      const response = await this.fetchImpl(url, {
        method: 'GET',
        headers: { Authorization: `Bearer ${token}` },
        signal: AbortSignal.timeout(5_000),
      })
      return await campaignAgentCredentialPayload(response)
    } catch (error) {
      throw sanitizedCampaignAgentError(error)
    }
  }

  async request(
    method: CampaignMethod,
    route: string,
    body?: CampaignBody,
    query?: CampaignQuery,
    authority?: CampaignRequestAuthority,
  ): Promise<CampaignResponse> {
    validateCampaignRequest(method, route, body, query, authority)
    try {
      const url = new URL(route, this.apiOrigin)
      Object.entries(query ?? {}).forEach(([key, value]) => {
        url.searchParams.set(key, String(value))
      })
      const mutationId = method === 'POST' ? this.idFactory() : null
      const idempotencyKey = authority?.idempotencyKey || (mutationId ? `desktop-${mutationId}` : null)
      const execute = async (token: string) => {
        const headers: Record<string, string> = { Authorization: `Bearer ${token}` }
        if (method === 'POST') {
          headers['Content-Type'] = 'application/json'
          headers['Idempotency-Key'] = idempotencyKey!
          headers['X-Correlation-ID'] = `desktop-${mutationId}`
        }
        return this.fetchImpl(url, {
          method,
          headers,
          signal: AbortSignal.timeout(this.requestTimeoutMs),
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
        : {
            ok: false,
            status: response.status,
            error: safeError(response.status),
            ...structuredErrorMetadata(data),
          }
    } catch {
      return {
        ok: false,
        status: 503,
        code: 'campaign_backend_unavailable',
        error: 'The campaign service did not respond in time.',
      }
    }
  }

  dispose(): void {
    this.accessToken = null
  }
}
