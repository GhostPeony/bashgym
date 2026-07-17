import {
  createDecipheriv,
  createHash,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  randomUUID,
  type KeyObject,
} from 'node:crypto'

export type CampaignAgentHostFamily = 'codex' | 'hermes'
export type CampaignAgentCapability =
  | 'campaign_observe'
  | 'training_launch'
  | 'training_pause_self'
  | 'artifact_read'
  | 'artifact_propose'

export interface MainOwnedCampaignAgentIdentity {
  terminalId: string
  generation: string
  family: CampaignAgentHostFamily
  origin: string
  principalId: string
  sessionId: string
  live: boolean
}

export interface CampaignAgentHostAttachRequest {
  terminalId: string
  workspaceId: string
  campaignId: string
}

export interface CampaignAgentHostRegistrationBody {
  scope: { workspaceId: string; campaignId: string }
  agentFamily: CampaignAgentHostFamily
  agentOrigin: string
  agentPrincipalId: string
  sessionId: string
  ephemeralPublicKey: string
  ttlSeconds: number
  idempotencyKey: string
}

export interface CampaignAgentHostHeartbeatBody {
  scope: { workspaceId: string; campaignId: string }
  agentFamily: CampaignAgentHostFamily
  agentOrigin: string
  agentPrincipalId: string
  sessionId: string
}

export interface CampaignAgentHostArtifactQuery {
  afterCursor?: string
  limit?: number
}

export interface CampaignAgentHostTransport {
  register(body: CampaignAgentHostRegistrationBody): Promise<unknown>
  claim(registrationId: string): Promise<unknown>
  revoke(registrationId: string): Promise<unknown>
  grant?(campaignId: string, body: Record<string, unknown>): Promise<unknown>
  attach?(campaignId: string, body: Record<string, unknown>): Promise<unknown>
  revokeAttachment?(campaignId: string, attachmentId: string, body: Record<string, unknown>): Promise<unknown>
  heartbeat?(credential: Buffer, body: CampaignAgentHostHeartbeatBody): Promise<unknown>
  observe?(credential: Buffer): Promise<unknown>
  artifacts?(credential: Buffer, query: CampaignAgentHostArtifactQuery): Promise<unknown>
}

export interface CampaignAgentHostAuthorizeRequest extends CampaignAgentHostAttachRequest {
  requestedCapabilities: CampaignAgentCapability[]
  grantedCapabilities: CampaignAgentCapability[]
  idempotencyKey: string
}

export interface CampaignAgentHostAuthorizationStatus {
  receiptId: string
  attachmentId: string
  attachmentVersion: number
  grantedCapabilities: CampaignAgentCapability[]
  credentialExpiresAt: string
}

export interface CampaignAgentHostPublicStatus {
  terminalId: string
  workspaceId: string
  campaignId: string
  family: CampaignAgentHostFamily
  registrationId: string
  state: 'registered' | 'authorized' | 'credential_ready' | 'active' | 'credential_consumed' | 'failed'
  expiresAt: string
  credentialReady: boolean
  actionsEnabled?: boolean
  authorization?: CampaignAgentHostAuthorizationStatus
}

export interface CampaignAgentHostEligibleSession {
  terminalId: string
  family: CampaignAgentHostFamily
  state: 'eligible' | CampaignAgentHostPublicStatus['state']
}

interface SessionReceipt {
  registrationId: string
  registeredAt: number
  expiresAt: number
}

interface HostState {
  identity: MainOwnedCampaignAgentIdentity
  scope: { workspaceId: string; campaignId: string }
  registration: SessionReceipt
  publicKeyDigest: string
  privateKey: KeyObject | null
  credential: Buffer | null
  state: CampaignAgentHostPublicStatus['state']
  claimAttempted: boolean
  expiryTimer: unknown | null
  heartbeatTimer: unknown | null
  heartbeatDeadlineTimer: unknown | null
  actionsEnabled: boolean
  activating: boolean
  lastSuccessfulHeartbeatAt: number | null
  authorization?: CampaignAgentHostAuthorizationStatus
  authorizationAuthority?: { humanPrincipalId: string; idempotencyDigest: string }
}

export type CampaignAgentHostTeardownReason =
  | 'pty_exit'
  | 'pty_replacement'
  | 'renderer_reload'
  | 'app_shutdown'
  | 'explicit_revoke'

export type CampaignAgentHostLifecycleEvent =
  | { terminalId: string; kind: 'activated' }
  | {
    terminalId: string
    kind: 'actions_locked' | 'torn_down'
    reason: 'heartbeat_timeout' | 'authority_rejected' | 'credential_expired' | 'identity_changed'
  }

export interface CampaignAgentHostScheduler {
  set(delayMs: number, action: () => void | Promise<void>): unknown
  clear(handle: unknown): void
}

const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const REGISTRATION_ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/
const BASE64URL_32 = /^[A-Za-z0-9_-]{43}$/
const SHA256_DIGEST = /^sha256:[0-9a-f]{64}$/
const ALGORITHM = 'X25519-HKDF-SHA256+CHACHA20-POLY1305'
const HOST_TTL_SECONDS = 300
const CLOCK_SKEW_MS = 5_000
const MAX_TOKEN_BYTES = 2_048
const HEARTBEAT_INTERVAL_MS = 30_000
const HEARTBEAT_RETRY_MS = 5_000
const HEARTBEAT_STALE_MS = 90_000
const ARTIFACT_CURSOR = /^a1\.[A-Za-z0-9_-]{11}$/
const CAPABILITIES = new Set<CampaignAgentCapability>([
  'campaign_observe', 'training_launch', 'training_pause_self', 'artifact_read', 'artifact_propose',
])

class CampaignAgentAuthorityClosedError extends Error {}

export function createMainOwnedCampaignAgentIdentity(input: {
  terminalId: string
  generation: string
  family: CampaignAgentHostFamily
  hostInstanceId: string
}): MainOwnedCampaignAgentIdentity {
  const terminalId = publicIdentifier(input.terminalId, 'terminal id')
  const generation = publicIdentifier(input.generation, 'PTY generation')
  if (input.family !== 'codex' && input.family !== 'hermes') {
    throw new Error('Unsupported campaign agent family')
  }
  if (typeof input.hostInstanceId !== 'string' || input.hostInstanceId.length < 8 || input.hostInstanceId.length > 256) {
    throw new Error('Campaign agent desktop host identity is invalid')
  }
  const digest = (domain: string, parts: readonly string[]) => createHash('sha256')
    .update(`${domain}\0${parts.join('\0')}`, 'utf8')
    .digest('hex')
    .slice(0, 32)
  return {
    terminalId,
    generation,
    family: input.family,
    origin: `desktop_${digest('origin', [input.hostInstanceId])}`,
    principalId: `${input.family}_pty_${digest('principal', [input.hostInstanceId, terminalId, generation, input.family])}`,
    sessionId: `pty_${digest('session', [input.hostInstanceId, terminalId, generation, input.family])}`,
    live: true,
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value)
  return actual.length === expected.length && expected.every((key) => key in value)
}

function publicIdentifier(value: unknown, label: string): string {
  if (typeof value !== 'string' || !IDENTIFIER.test(value)) {
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  return value
}

function publicDate(value: unknown, label: string): number {
  if (typeof value !== 'string' || value.length > 64) {
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  const parsed = Date.parse(value)
  if (!Number.isFinite(parsed)) throw new Error(`Campaign agent host ${label} is invalid`)
  return parsed
}

function decodeBase64Url(value: unknown, expectedBytes: number | null, label: string): Buffer {
  if (typeof value !== 'string' || value.length > 4_096 || !/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  const decoded = Buffer.from(value, 'base64url')
  if (decoded.toString('base64url') !== value || (expectedBytes !== null && decoded.length !== expectedBytes)) {
    decoded.fill(0)
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  return decoded
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (isRecord(value)) {
    return `{${Object.entries(value)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, nested]) => `${JSON.stringify(key)}:${canonical(nested)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function parseCanonicalObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== 'string' || value.length > 16_384) {
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(value)
  } catch {
    throw new Error(`Campaign agent host ${label} is invalid`)
  }
  if (!isRecord(parsed) || canonical(parsed) !== value) {
    throw new Error(`Campaign agent host ${label} is not canonical`)
  }
  return parsed
}

function publicKeyRaw(publicKey: KeyObject): Buffer {
  const jwk = publicKey.export({ format: 'jwk' })
  if (jwk.kty !== 'OKP' || jwk.crv !== 'X25519' || typeof jwk.x !== 'string' || !BASE64URL_32.test(jwk.x)) {
    throw new Error('Campaign agent host could not create an X25519 public key')
  }
  return Buffer.from(jwk.x, 'base64url')
}

function publicKeyObject(encoded: string): KeyObject {
  if (!BASE64URL_32.test(encoded)) throw new Error('Campaign agent delivery public key is invalid')
  return createPublicKey({ key: { kty: 'OKP', crv: 'X25519', x: encoded }, format: 'jwk' })
}

function assertAttachRequest(value: CampaignAgentHostAttachRequest): CampaignAgentHostAttachRequest {
  if (!isRecord(value) || !exactKeys(value, ['terminalId', 'workspaceId', 'campaignId'])) {
    throw new Error('Campaign agent host attach request contract is invalid')
  }
  const terminalId = publicIdentifier(value.terminalId, 'terminal id')
  return {
    terminalId,
    workspaceId: publicIdentifier(value.workspaceId, 'workspace id'),
    campaignId: publicIdentifier(value.campaignId, 'campaign id'),
  }
}

function capabilityList(value: unknown, label: string, allowEmpty: boolean): CampaignAgentCapability[] {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0) || value.length > CAPABILITIES.size) {
    throw new Error(`Campaign agent ${label} capabilities are invalid`)
  }
  const capabilities = value.map((entry) => {
    if (typeof entry !== 'string' || !CAPABILITIES.has(entry as CampaignAgentCapability)) {
      throw new Error(`Campaign agent ${label} capability is invalid`)
    }
    return entry as CampaignAgentCapability
  })
  if (new Set(capabilities).size !== capabilities.length) {
    throw new Error(`Campaign agent ${label} capabilities must be unique`)
  }
  return capabilities.sort()
}

function assertAuthorizeRequest(value: CampaignAgentHostAuthorizeRequest): CampaignAgentHostAuthorizeRequest {
  if (!isRecord(value) || !exactKeys(value, [
    'terminalId', 'workspaceId', 'campaignId', 'requestedCapabilities',
    'grantedCapabilities', 'idempotencyKey',
  ])) throw new Error('Campaign agent authorize request contract is invalid')
  const scope = assertAttachRequest({
    terminalId: value.terminalId,
    workspaceId: value.workspaceId,
    campaignId: value.campaignId,
  })
  const requestedCapabilities = capabilityList(value.requestedCapabilities, 'requested', false)
  const grantedCapabilities = capabilityList(value.grantedCapabilities, 'granted', true)
  if (!grantedCapabilities.every((capability) => requestedCapabilities.includes(capability))) {
    throw new Error('Campaign agent granted capabilities must be requested')
  }
  return {
    ...scope,
    requestedCapabilities,
    grantedCapabilities,
    idempotencyKey: publicIdentifier(value.idempotencyKey, 'authorization idempotency key'),
  }
}

function assertArtifactQuery(value: unknown): CampaignAgentHostArtifactQuery {
  if (!isRecord(value)) throw new Error('Campaign agent artifact query is invalid')
  const keys = Reflect.ownKeys(value)
  if (keys.length > 2 || keys.some((key) => typeof key !== 'string'
    || (key !== 'afterCursor' && key !== 'limit'))) {
    throw new Error('Campaign agent artifact query is invalid')
  }
  const descriptors = Object.getOwnPropertyDescriptors(value)
  if (keys.some((key) => {
    const descriptor = descriptors[key as string]
    return !descriptor || !('value' in descriptor)
  })) throw new Error('Campaign agent artifact query is invalid')
  const afterCursor = descriptors.afterCursor?.value
  const limit = descriptors.limit?.value
  if ((afterCursor !== undefined && (typeof afterCursor !== 'string' || !ARTIFACT_CURSOR.test(afterCursor)))
    || (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1 || limit > 50))) {
    throw new Error('Campaign agent artifact query is invalid')
  }
  return {
    ...(afterCursor !== undefined ? { afterCursor } : {}),
    ...(limit !== undefined ? { limit } : {}),
  }
}

function capabilityDigest(
  principalId: string,
  requested: readonly CampaignAgentCapability[],
  granted: readonly CampaignAgentCapability[],
): string {
  const payload = `principal=${principalId};requested=${[...requested].sort().join(',')};granted=${[...granted].sort().join(',')}`
  let digest = 0x811C9DC5
  for (const value of Buffer.from(payload, 'ascii')) {
    digest ^= value
    digest = Math.imul(digest, 0x01000193) >>> 0
  }
  return `fnv1a32:${digest.toString(16).padStart(8, '0')}`
}

export interface CampaignAgentHostControllerOptions {
  transport: CampaignAgentHostTransport
  resolveIdentity(terminalId: string): MainOwnedCampaignAgentIdentity | null
  clock?: () => number
  idFactory?: () => string
  scheduler?: CampaignAgentHostScheduler
  onLifecycle?(event: CampaignAgentHostLifecycleEvent): void
}

export class CampaignAgentHostController {
  private readonly sessions = new Map<string, HostState>()
  private readonly clock: () => number
  private readonly idFactory: () => string
  private readonly scheduler: CampaignAgentHostScheduler

  constructor(private readonly options: CampaignAgentHostControllerOptions) {
    this.clock = options.clock ?? Date.now
    this.idFactory = options.idFactory ?? (() => randomUUID().replaceAll('-', ''))
    this.scheduler = options.scheduler ?? {
      set: (delayMs, action) => {
        const timer = setTimeout(() => { void action() }, Math.max(0, delayMs))
        timer.unref?.()
        return timer
      },
      clear: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    }
  }

  private replaceTimer(state: HostState, delayMs: number, action: () => Promise<unknown>): void {
    if (state.expiryTimer !== null) this.scheduler.clear(state.expiryTimer)
    state.expiryTimer = this.scheduler.set(Math.max(0, delayMs), async () => {
      await action().catch(() => undefined)
    })
  }

  private scheduleExpiry(state: HostState): void {
    this.replaceTimer(
      state,
      state.registration.expiresAt - this.clock(),
      () => this.teardownTerminal(state.identity.terminalId, 'explicit_revoke'),
    )
  }

  private scheduleRenewal(state: HostState): void {
    const credentialExpiresAt = state.authorization
      ? Date.parse(state.authorization.credentialExpiresAt)
      : 0
    const renewAt = state.registration.expiresAt - 60_000
    if (credentialExpiresAt > 0 && credentialExpiresAt <= renewAt) {
      this.replaceTimer(
        state,
        credentialExpiresAt - this.clock(),
        () => this.teardownTerminal(state.identity.terminalId, 'explicit_revoke'),
      )
      return
    }
    this.replaceTimer(
      state,
      renewAt - this.clock(),
      () => this.renewRegistration(state.identity.terminalId),
    )
  }

  private publicStatus(state: HostState): CampaignAgentHostPublicStatus {
    return {
      terminalId: state.identity.terminalId,
      workspaceId: state.scope.workspaceId,
      campaignId: state.scope.campaignId,
      family: state.identity.family,
      registrationId: state.registration.registrationId,
      state: state.state,
      expiresAt: new Date(state.registration.expiresAt).toISOString(),
      credentialReady: state.credential !== null,
      ...(state.credential !== null ? { actionsEnabled: state.actionsEnabled } : {}),
      ...(state.authorization ? { authorization: { ...state.authorization } } : {}),
    }
  }

  private validateReceipt(
    value: unknown,
    body: CampaignAgentHostRegistrationBody,
    expectedStatus: 'live' | 'revoked',
  ): SessionReceipt {
    if (!isRecord(value) || !exactKeys(value, [
      'schemaVersion', 'registrationId', 'scope', 'agentFamily', 'agentOrigin',
      'agentPrincipalId', 'sessionId', 'publicKeyDigest', 'registeredAt', 'expiresAt', 'status',
    ])) throw new Error('Campaign agent host receipt contract is invalid')
    if (value.schemaVersion !== 'campaign_agent_host_session.v1' || value.status !== expectedStatus) {
      throw new Error('Campaign agent host receipt state is invalid')
    }
    if (!isRecord(value.scope) || !exactKeys(value.scope, ['workspaceId', 'campaignId'])) {
      throw new Error('Campaign agent host receipt scope is invalid')
    }
    const registrationId = publicIdentifier(value.registrationId, 'registration id')
    if (!REGISTRATION_ID.test(registrationId)
      || value.scope.workspaceId !== body.scope.workspaceId
      || value.scope.campaignId !== body.scope.campaignId
      || value.agentFamily !== body.agentFamily
      || value.agentOrigin !== body.agentOrigin
      || value.agentPrincipalId !== body.agentPrincipalId
      || value.sessionId !== body.sessionId
      || typeof value.publicKeyDigest !== 'string'
      || !SHA256_DIGEST.test(value.publicKeyDigest)) {
      throw new Error('Campaign agent host receipt tuple is invalid')
    }
    const rawPublicKey = Buffer.from(body.ephemeralPublicKey, 'base64url')
    const expectedDigest = `sha256:${createHash('sha256').update(rawPublicKey).digest('hex')}`
    rawPublicKey.fill(0)
    if (value.publicKeyDigest !== expectedDigest) throw new Error('Campaign agent host receipt public key is invalid')
    const registeredAt = publicDate(value.registeredAt, 'registered time')
    const expiresAt = publicDate(value.expiresAt, 'expiry time')
    const now = this.clock()
    if (registeredAt > now + CLOCK_SKEW_MS || expiresAt <= registeredAt || expiresAt <= now
      || expiresAt > registeredAt + (body.ttlSeconds * 1_000) + CLOCK_SKEW_MS) {
      throw new Error('Campaign agent host receipt time bounds are invalid')
    }
    return { registrationId, registeredAt, expiresAt }
  }

  async attach(input: CampaignAgentHostAttachRequest): Promise<CampaignAgentHostPublicStatus> {
    const request = assertAttachRequest(input)
    if (this.sessions.has(request.terminalId)) {
      throw new Error('Campaign agent host terminal is already registered')
    }
    const identity = this.options.resolveIdentity(request.terminalId)
    if (!identity || !identity.live || identity.terminalId !== request.terminalId) {
      throw new Error('Campaign agent host requires a live PTY')
    }
    if (identity.family !== 'codex' && identity.family !== 'hermes') {
      throw new Error('Unsupported campaign agent family')
    }
    publicIdentifier(identity.generation, 'PTY generation')
    publicIdentifier(identity.origin, 'agent origin')
    publicIdentifier(identity.principalId, 'agent principal')
    publicIdentifier(identity.sessionId, 'agent session')
    const keypair = generateKeyPairSync('x25519')
    const rawPublicKey = publicKeyRaw(keypair.publicKey)
    const encodedPublicKey = rawPublicKey.toString('base64url')
    const publicKeyDigest = `sha256:${createHash('sha256').update(rawPublicKey).digest('hex')}`
    rawPublicKey.fill(0)
    const body: CampaignAgentHostRegistrationBody = {
      scope: { workspaceId: request.workspaceId, campaignId: request.campaignId },
      agentFamily: identity.family,
      agentOrigin: identity.origin,
      agentPrincipalId: identity.principalId,
      sessionId: identity.sessionId,
      ephemeralPublicKey: encodedPublicKey,
      ttlSeconds: HOST_TTL_SECONDS,
      idempotencyKey: `hostreg_${this.idFactory()}`,
    }
    let registration: SessionReceipt | null = null
    try {
      registration = this.validateReceipt(await this.options.transport.register(body), body, 'live')
      const currentIdentity = this.options.resolveIdentity(request.terminalId)
      if (!currentIdentity || !currentIdentity.live || currentIdentity.generation !== identity.generation) {
        throw new Error('Campaign agent host PTY changed during registration')
      }
      const state: HostState = {
        identity: { ...identity },
        scope: body.scope,
        registration,
        publicKeyDigest,
        privateKey: keypair.privateKey,
        credential: null,
        state: 'registered',
        claimAttempted: false,
        expiryTimer: null,
        heartbeatTimer: null,
        heartbeatDeadlineTimer: null,
        actionsEnabled: false,
        activating: false,
        lastSuccessfulHeartbeatAt: null,
      }
      this.sessions.set(request.terminalId, state)
      this.scheduleExpiry(state)
      return this.publicStatus(state)
    } catch (error) {
      if (registration) void this.options.transport.revoke(registration.registrationId).catch(() => undefined)
      throw error
    }
  }

  async authorize(input: CampaignAgentHostAuthorizeRequest): Promise<CampaignAgentHostPublicStatus> {
    const request = assertAuthorizeRequest(input)
    const state = this.sessions.get(request.terminalId)
    if (!state || !state.privateKey || state.claimAttempted) {
      throw new Error('Campaign agent host registration is not ready for authorization')
    }
    if (request.workspaceId !== state.scope.workspaceId || request.campaignId !== state.scope.campaignId) {
      throw new Error('Campaign agent authorization scope changed')
    }
    const currentIdentity = this.options.resolveIdentity(request.terminalId)
    if (!currentIdentity || !currentIdentity.live || currentIdentity.generation !== state.identity.generation) {
      await this.teardownTerminal(request.terminalId, 'pty_replacement').catch(() => undefined)
      throw new Error('Campaign agent host PTY changed before authorization')
    }
    if (!this.options.transport.grant || !this.options.transport.attach) {
      throw new Error('Campaign agent authorization transport is unavailable')
    }
    const tuple = {
      scope: { workspaceId: state.scope.workspaceId, campaignId: state.scope.campaignId },
      agentFamily: state.identity.family,
      agentOrigin: state.identity.origin,
      agentPrincipalId: state.identity.principalId,
      sessionId: state.identity.sessionId,
      requestedCapabilities: request.requestedCapabilities,
      grantedCapabilities: request.grantedCapabilities,
    }
    const idempotencyDigest = createHash('sha256')
      .update(`${request.idempotencyKey}\0${request.terminalId}\0${request.workspaceId}\0${request.campaignId}`, 'utf8')
      .digest('hex')
      .slice(0, 32)
    const grantBody = {
      ...tuple,
      idempotencyKey: `hostgrant_${idempotencyDigest}`,
    }
    const receipt = await this.options.transport.grant(request.campaignId, grantBody)
    if (!isRecord(receipt) || !exactKeys(receipt, [
      'schemaVersion', 'issuer', 'receiptId', 'receiptDigest', 'humanPrincipal', 'scope',
      'agentFamily', 'agentOrigin', 'agentPrincipalId', 'sessionId', 'requestedCapabilities',
      'grantedCapabilities', 'capabilityDigest', 'grantRevision', 'issuedAt', 'expiresAt',
    ])
      || receipt.schemaVersion !== 'campaign_agent_grant_confirmation.v1'
      || receipt.issuer !== 'campaign_authority'
      || !isRecord(receipt.humanPrincipal)
      || !exactKeys(receipt.humanPrincipal, ['principalId', 'principalType'])
      || receipt.humanPrincipal.principalType !== 'human'
      || !IDENTIFIER.test(String(receipt.humanPrincipal.principalId))
      || !isRecord(receipt.scope)
      || !exactKeys(receipt.scope, ['workspaceId', 'campaignId'])
      || receipt.scope.workspaceId !== state.scope.workspaceId
      || receipt.scope.campaignId !== state.scope.campaignId
      || receipt.agentFamily !== state.identity.family
      || receipt.agentOrigin !== state.identity.origin
      || receipt.agentPrincipalId !== state.identity.principalId
      || receipt.sessionId !== state.identity.sessionId
      || canonical(receipt.requestedCapabilities) !== canonical(request.requestedCapabilities)
      || canonical(receipt.grantedCapabilities) !== canonical(request.grantedCapabilities)
      || receipt.capabilityDigest !== capabilityDigest(
        state.identity.principalId, request.requestedCapabilities, request.grantedCapabilities,
      )
      || !Number.isSafeInteger(receipt.grantRevision) || Number(receipt.grantRevision) < 1
      || typeof receipt.receiptDigest !== 'string' || !SHA256_DIGEST.test(receipt.receiptDigest)) {
      throw new Error('Campaign agent grant receipt tuple is invalid')
    }
    const receiptId = publicIdentifier(receipt.receiptId, 'grant receipt id')
    const issuedAt = publicDate(receipt.issuedAt, 'grant issue time')
    const expiresAt = publicDate(receipt.expiresAt, 'grant expiry time')
    if (issuedAt > this.clock() + CLOCK_SKEW_MS || expiresAt <= issuedAt || expiresAt <= this.clock()
      || expiresAt > issuedAt + (30 * 60_000)) {
      throw new Error('Campaign agent grant receipt time bounds are invalid')
    }
    const attachment = await this.options.transport.attach(request.campaignId, {
      action: 'attach',
      ...tuple,
      confirmationReceipt: receipt,
      baseAttachmentVersion: null,
      idempotencyKey: `hostattach_${idempotencyDigest}`,
    })
    if (!isRecord(attachment) || !exactKeys(attachment, [
      'schema_version', 'observed_at', 'scope', 'attachment', 'audit_events',
    ])
      || attachment.schema_version !== 'campaign_agent_public_view.v1'
      || !isRecord(attachment.scope)
      || attachment.scope.workspace_id !== state.scope.workspaceId
      || attachment.scope.campaign_id !== state.scope.campaignId
      || !isRecord(attachment.attachment)) {
      throw new Error('Campaign agent attachment view is invalid')
    }
    const view = attachment.attachment
    if (view.status !== 'attached'
      || !Number.isSafeInteger(view.attachment_version) || Number(view.attachment_version) < 1
      || canonical(view.requested_capabilities) !== canonical(request.requestedCapabilities)
      || canonical(view.granted_capabilities) !== canonical(request.grantedCapabilities)
      || !isRecord(view.provenance)
      || view.provenance.agent_family !== state.identity.family
      || view.provenance.agent_origin !== state.identity.origin
      || view.provenance.agent_principal_id !== state.identity.principalId
      || view.provenance.session_id !== state.identity.sessionId
      || view.provenance.grant_receipt_id !== receiptId
      || view.provenance.grant_receipt_digest !== receipt.receiptDigest) {
      throw new Error('Campaign agent attachment tuple is invalid')
    }
    const attachmentId = publicIdentifier(view.attachment_id, 'attachment id')
    const credentialExpiresAt = publicDate(view.provenance.credential_expires_at, 'credential expiry time')
    if (credentialExpiresAt <= this.clock()) throw new Error('Campaign agent credential already expired')
    state.authorization = {
      receiptId,
      attachmentId,
      attachmentVersion: Number(view.attachment_version),
      grantedCapabilities: [...request.grantedCapabilities],
      credentialExpiresAt: new Date(credentialExpiresAt).toISOString(),
    }
    state.authorizationAuthority = {
      humanPrincipalId: String(receipt.humanPrincipal.principalId),
      idempotencyDigest,
    }
    state.state = 'authorized'
    return this.publicStatus(state)
  }

  private validateEnvelope(value: unknown, state: HostState): {
    credentialId: string
    ciphertext: Buffer
    nonce: Buffer
    salt: Buffer
    info: string
    aadJson: string
    peerPublicKey: KeyObject
  } {
    if (!isRecord(value) || !exactKeys(value, [
      'schemaVersion', 'envelopeId', 'registrationId', 'credentialId', 'algorithm',
      'ephemeralPublicKey', 'hkdfSalt', 'hkdfInfo', 'nonce', 'ciphertext', 'aadJson', 'createdAt',
    ])) throw new Error('Campaign agent delivery envelope contract is invalid')
    const envelopeId = publicIdentifier(value.envelopeId, 'envelope id')
    const credentialId = publicIdentifier(value.credentialId, 'credential id')
    if (value.schemaVersion !== 'campaign_agent_delivery_envelope.v1'
      || value.registrationId !== state.registration.registrationId
      || value.algorithm !== ALGORITHM) {
      throw new Error('Campaign agent delivery envelope identity is invalid')
    }
    const expectedInfo = `bashgym.campaign-agent-delivery-key.v1:${state.registration.registrationId}:${credentialId}`
    if (value.hkdfInfo !== expectedInfo) throw new Error('Campaign agent delivery KDF context is invalid')
    const createdAt = publicDate(value.createdAt, 'delivery creation time')
    if (createdAt < state.registration.registeredAt - CLOCK_SKEW_MS
      || createdAt > this.clock() + CLOCK_SKEW_MS
      || createdAt >= state.registration.expiresAt) {
      throw new Error('Campaign agent delivery time bounds are invalid')
    }
    const aad = parseCanonicalObject(value.aadJson, 'delivery AAD')
    if (!exactKeys(aad, [
      'schema_version', 'envelope_id', 'registration_id', 'credential_id', 'attachment_id',
      'workspace_id', 'campaign_id', 'agent_family', 'agent_origin', 'session_id',
      'agent_principal_id', 'public_key_digest', 'issued_at', 'expires_at',
    ])
      || aad.schema_version !== 'campaign_agent_delivery_aad.v1'
      || aad.envelope_id !== envelopeId
      || aad.registration_id !== state.registration.registrationId
      || aad.credential_id !== credentialId
      || aad.workspace_id !== state.scope.workspaceId
      || aad.campaign_id !== state.scope.campaignId
      || aad.agent_family !== state.identity.family
      || aad.agent_origin !== state.identity.origin
      || aad.session_id !== state.identity.sessionId
      || aad.agent_principal_id !== state.identity.principalId
      || aad.public_key_digest !== state.publicKeyDigest) {
      throw new Error('Campaign agent delivery AAD tuple is invalid')
    }
    publicIdentifier(aad.attachment_id, 'attachment id')
    const issuedAt = publicDate(aad.issued_at, 'credential issue time')
    const expiresAt = publicDate(aad.expires_at, 'credential expiry time')
    if (issuedAt !== createdAt || expiresAt <= issuedAt || expiresAt <= this.clock()) {
      throw new Error('Campaign agent credential time bounds are invalid')
    }
    return {
      credentialId,
      ciphertext: decodeBase64Url(value.ciphertext, null, 'ciphertext'),
      nonce: decodeBase64Url(value.nonce, 12, 'nonce'),
      salt: decodeBase64Url(value.hkdfSalt, 16, 'HKDF salt'),
      info: expectedInfo,
      aadJson: value.aadJson as string,
      peerPublicKey: publicKeyObject(value.ephemeralPublicKey as string),
    }
  }

  async claim(terminalId: string): Promise<CampaignAgentHostPublicStatus> {
    publicIdentifier(terminalId, 'terminal id')
    const state = this.sessions.get(terminalId)
    if (!state) throw new Error('Campaign agent host registration is unavailable')
    if (state.claimAttempted) throw new Error('Campaign agent credential was already claimed')
    if (!state.authorization || state.state !== 'authorized') {
      throw new Error('Campaign agent human authorization is not complete')
    }
    if (!state.privateKey || state.registration.expiresAt <= this.clock()) {
      throw new Error('Campaign agent host registration expired')
    }
    const currentIdentity = this.options.resolveIdentity(terminalId)
    if (!currentIdentity || !currentIdentity.live || currentIdentity.generation !== state.identity.generation) {
      await this.teardownTerminal(terminalId, 'pty_replacement')
      throw new Error('Campaign agent host PTY is no longer live')
    }
    state.claimAttempted = true
    let shared: Buffer | null = null
    let key: Buffer | null = null
    let plaintext: Buffer | null = null
    let encrypted: ReturnType<CampaignAgentHostController['validateEnvelope']> | null = null
    try {
      encrypted = this.validateEnvelope(
        await this.options.transport.claim(state.registration.registrationId),
        state,
      )
      if (encrypted.ciphertext.length < 17 || encrypted.ciphertext.length > MAX_TOKEN_BYTES + 2_048) {
        throw new Error('Campaign agent delivery ciphertext is invalid')
      }
      shared = diffieHellman({ privateKey: state.privateKey, publicKey: encrypted.peerPublicKey })
      key = Buffer.from(hkdfSync('sha256', shared, encrypted.salt, encrypted.info, 32))
      const authTag = encrypted.ciphertext.subarray(encrypted.ciphertext.length - 16)
      const ciphertext = encrypted.ciphertext.subarray(0, encrypted.ciphertext.length - 16)
      const decipher = createDecipheriv('chacha20-poly1305', key, encrypted.nonce, { authTagLength: 16 })
      decipher.setAAD(Buffer.from(encrypted.aadJson), { plaintextLength: ciphertext.length })
      decipher.setAuthTag(authTag)
      plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()])
      if (plaintext.length > MAX_TOKEN_BYTES + 512) throw new Error('Campaign agent credential is too large')
      const payload = parseCanonicalObject(plaintext.toString('utf8'), 'credential payload')
      if (!exactKeys(payload, ['schema_version', 'registration_id', 'credential_id', 'raw_token'])
        || payload.schema_version !== 'campaign_agent_credential_delivery.v1'
        || payload.registration_id !== state.registration.registrationId
        || payload.credential_id !== encrypted.credentialId
        || typeof payload.raw_token !== 'string'
        || !/^bgag\.[A-Za-z0-9_.:-]+\.[A-Za-z0-9_-]{32,512}$/.test(payload.raw_token)) {
        throw new Error('Campaign agent credential payload is invalid')
      }
      state.credential = Buffer.from(payload.raw_token, 'utf8')
      state.privateKey = null
      state.state = 'credential_ready'
      this.scheduleRenewal(state)
      return this.publicStatus(state)
    } catch (error) {
      state.privateKey = null
      state.state = 'failed'
      await this.teardownTerminal(terminalId, 'explicit_revoke').catch(() => undefined)
      throw error
    } finally {
      encrypted?.ciphertext.fill(0)
      encrypted?.nonce.fill(0)
      encrypted?.salt.fill(0)
      shared?.fill(0)
      key?.fill(0)
      plaintext?.fill(0)
    }
  }

  async renewRegistration(terminalId: string): Promise<CampaignAgentHostPublicStatus> {
    publicIdentifier(terminalId, 'terminal id')
    const state = this.sessions.get(terminalId)
    if (!state || !state.claimAttempted || !state.authorization
      || !['credential_ready', 'active', 'credential_consumed'].includes(state.state)) {
      throw new Error('Campaign agent host registration is not eligible for renewal')
    }
    const identity = this.options.resolveIdentity(terminalId)
    if (!identity || !identity.live || identity.generation !== state.identity.generation) {
      await this.teardownTerminal(terminalId, 'pty_replacement').catch(() => undefined)
      throw new Error('Campaign agent host PTY changed before renewal')
    }
    const keypair = generateKeyPairSync('x25519')
    const rawPublicKey = publicKeyRaw(keypair.publicKey)
    const ephemeralPublicKey = rawPublicKey.toString('base64url')
    const publicKeyDigest = `sha256:${createHash('sha256').update(rawPublicKey).digest('hex')}`
    rawPublicKey.fill(0)
    const body: CampaignAgentHostRegistrationBody = {
      scope: state.scope,
      agentFamily: state.identity.family,
      agentOrigin: state.identity.origin,
      agentPrincipalId: state.identity.principalId,
      sessionId: state.identity.sessionId,
      ephemeralPublicKey,
      ttlSeconds: HOST_TTL_SECONDS,
      idempotencyKey: `hostrenew_${this.idFactory()}`,
    }
    try {
      const registration = this.validateReceipt(
        await this.options.transport.register(body), body, 'live',
      )
      const current = this.options.resolveIdentity(terminalId)
      if (!current || !current.live || current.generation !== state.identity.generation) {
        throw new Error('Campaign agent host PTY changed during renewal')
      }
      state.registration = registration
      state.publicKeyDigest = publicKeyDigest
      state.privateKey = null
      this.scheduleRenewal(state)
      return this.publicStatus(state)
    } catch (error) {
      await this.teardownTerminal(terminalId, 'explicit_revoke').catch(() => undefined)
      throw error
    }
  }

  private emitLifecycle(event: CampaignAgentHostLifecycleEvent): void {
    try {
      this.options.onLifecycle?.(event)
    } catch {
      // A local observer must never alter credential authority.
    }
  }

  private clearHeartbeatTimers(state: HostState): void {
    if (state.heartbeatTimer !== null) this.scheduler.clear(state.heartbeatTimer)
    if (state.heartbeatDeadlineTimer !== null) this.scheduler.clear(state.heartbeatDeadlineTimer)
    state.heartbeatTimer = null
    state.heartbeatDeadlineTimer = null
  }

  private authorityFailure(state: HostState): 'credential_expired' | 'identity_changed' | null {
    const identity = this.options.resolveIdentity(state.identity.terminalId)
    if (!identity || !identity.live
      || identity.terminalId !== state.identity.terminalId
      || identity.generation !== state.identity.generation
      || identity.family !== state.identity.family
      || identity.origin !== state.identity.origin
      || identity.principalId !== state.identity.principalId
      || identity.sessionId !== state.identity.sessionId) {
      return 'identity_changed'
    }
    if (!state.authorization
      || Date.parse(state.authorization.credentialExpiresAt) <= this.clock()) {
      return 'credential_expired'
    }
    return null
  }

  private isAuthorityRejection(error: unknown): boolean {
    if (!error || (typeof error !== 'object' && typeof error !== 'function')) return false
    const value = error as { status?: unknown; statusCode?: unknown; code?: unknown }
    return value.status === 401 || value.status === 403
      || value.statusCode === 401 || value.statusCode === 403
      || value.code === 'UNAUTHORIZED' || value.code === 'FORBIDDEN'
  }

  private heartbeatBody(state: HostState): CampaignAgentHostHeartbeatBody {
    return {
      scope: { workspaceId: state.scope.workspaceId, campaignId: state.scope.campaignId },
      agentFamily: state.identity.family,
      agentOrigin: state.identity.origin,
      agentPrincipalId: state.identity.principalId,
      sessionId: state.identity.sessionId,
    }
  }

  private async teardownAuthority(
    state: HostState,
    reason: 'authority_rejected' | 'credential_expired' | 'identity_changed',
  ): Promise<never> {
    const terminalId = state.identity.terminalId
    await this.teardownTerminal(terminalId, 'explicit_revoke').catch(() => undefined)
    this.emitLifecycle({ terminalId, kind: 'torn_down', reason })
    throw new CampaignAgentAuthorityClosedError(reason === 'authority_rejected'
      ? 'Campaign agent authority rejected the credential'
      : 'Campaign agent credential is no longer authorized')
  }

  private scheduleHeartbeatDeadline(state: HostState): void {
    if (state.heartbeatDeadlineTimer !== null) this.scheduler.clear(state.heartbeatDeadlineTimer)
    const lastSuccess = state.lastSuccessfulHeartbeatAt
    if (lastSuccess === null) return
    state.heartbeatDeadlineTimer = this.scheduler.set(HEARTBEAT_STALE_MS, async () => {
      state.heartbeatDeadlineTimer = null
      if (this.sessions.get(state.identity.terminalId) !== state
        || state.lastSuccessfulHeartbeatAt !== lastSuccess
        || this.clock() - lastSuccess < HEARTBEAT_STALE_MS) return
      state.actionsEnabled = false
      state.state = 'credential_ready'
      if (state.heartbeatTimer !== null) this.scheduler.clear(state.heartbeatTimer)
      state.heartbeatTimer = null
      this.emitLifecycle({
        terminalId: state.identity.terminalId,
        kind: 'actions_locked',
        reason: 'heartbeat_timeout',
      })
    })
  }

  private scheduleHeartbeat(state: HostState, delayMs: number, retry: boolean): void {
    if (state.heartbeatTimer !== null) this.scheduler.clear(state.heartbeatTimer)
    state.heartbeatTimer = this.scheduler.set(delayMs, async () => {
      state.heartbeatTimer = null
      await this.runHeartbeat(state, retry).catch(() => undefined)
    })
  }

  private async runHeartbeat(state: HostState, retry: boolean): Promise<void> {
    if (this.sessions.get(state.identity.terminalId) !== state || !state.credential || !state.actionsEnabled) return
    const authorityFailure = this.authorityFailure(state)
    if (authorityFailure) await this.teardownAuthority(state, authorityFailure)
    const heartbeat = this.options.transport.heartbeat
    if (!heartbeat) {
      state.actionsEnabled = false
      state.state = 'credential_ready'
      this.emitLifecycle({
        terminalId: state.identity.terminalId,
        kind: 'actions_locked',
        reason: 'heartbeat_timeout',
      })
      return
    }
    try {
      await heartbeat(state.credential, this.heartbeatBody(state))
      if (this.sessions.get(state.identity.terminalId) !== state) return
      const currentFailure = this.authorityFailure(state)
      if (currentFailure) await this.teardownAuthority(state, currentFailure)
      state.lastSuccessfulHeartbeatAt = this.clock()
      this.scheduleHeartbeatDeadline(state)
      this.scheduleHeartbeat(state, HEARTBEAT_INTERVAL_MS, false)
    } catch (error) {
      if (error instanceof CampaignAgentAuthorityClosedError) throw error
      if (this.isAuthorityRejection(error)) await this.teardownAuthority(state, 'authority_rejected')
      this.scheduleHeartbeat(state, retry ? HEARTBEAT_INTERVAL_MS : HEARTBEAT_RETRY_MS, !retry)
    }
  }

  async activate(terminalId: string): Promise<CampaignAgentHostPublicStatus> {
    publicIdentifier(terminalId, 'terminal id')
    const state = this.sessions.get(terminalId)
    if (!state?.credential || state.state !== 'credential_ready') {
      throw new Error('Campaign agent credential is not ready for activation')
    }
    if (state.actionsEnabled || state.activating) {
      throw new Error('Campaign agent credential is already active')
    }
    const authorityFailure = this.authorityFailure(state)
    if (authorityFailure) await this.teardownAuthority(state, authorityFailure)
    const heartbeat = this.options.transport.heartbeat
    if (!heartbeat) throw new Error('Campaign agent heartbeat transport is unavailable')
    state.activating = true
    try {
      await heartbeat(state.credential, this.heartbeatBody(state))
      if (this.sessions.get(terminalId) !== state) {
        await this.teardownAuthority(state, 'identity_changed')
      }
      const currentFailure = this.authorityFailure(state)
      if (currentFailure) await this.teardownAuthority(state, currentFailure)
      state.lastSuccessfulHeartbeatAt = this.clock()
      state.actionsEnabled = true
      state.state = 'active'
      this.scheduleHeartbeatDeadline(state)
      this.scheduleHeartbeat(state, HEARTBEAT_INTERVAL_MS, false)
      this.emitLifecycle({ terminalId, kind: 'activated' })
      return this.publicStatus(state)
    } catch (error) {
      if (error instanceof CampaignAgentAuthorityClosedError) throw error
      if (this.isAuthorityRejection(error)) await this.teardownAuthority(state, 'authority_rejected')
      throw new Error('Campaign agent heartbeat failed')
    } finally {
      state.activating = false
    }
  }

  private async fixedAction(
    terminalId: string,
    capability: 'campaign_observe' | 'artifact_read',
    action: ((credential: Buffer) => Promise<unknown>) | undefined,
    label: string,
  ): Promise<unknown> {
    publicIdentifier(terminalId, 'terminal id')
    const state = this.sessions.get(terminalId)
    if (!state?.credential || !state.actionsEnabled) {
      throw new Error('Campaign agent credential is not active; activate it first')
    }
    if (!state.authorization?.grantedCapabilities.includes(capability)) {
      throw new Error(`Campaign agent ${label} capability is not granted`)
    }
    const authorityFailure = this.authorityFailure(state)
    if (authorityFailure) await this.teardownAuthority(state, authorityFailure)
    if (!action) throw new Error(`Campaign agent ${label} transport is unavailable`)
    try {
      return await action(state.credential)
    } catch (error) {
      if (error instanceof CampaignAgentAuthorityClosedError) throw error
      if (this.isAuthorityRejection(error)) {
        await this.teardownAuthority(state, 'authority_rejected')
      }
      throw new Error(`Campaign agent ${label} action failed`)
    }
  }

  observe(terminalId: string): Promise<unknown> {
    return this.fixedAction(
      terminalId, 'campaign_observe', this.options.transport.observe, 'observe',
    )
  }

  async artifacts(terminalId: string, query: CampaignAgentHostArtifactQuery): Promise<unknown> {
    const safeQuery = assertArtifactQuery(query)
    return await this.fixedAction(
      terminalId,
      'artifact_read',
      this.options.transport.artifacts
        ? (credential) => this.options.transport.artifacts!(credential, safeQuery)
        : undefined,
      'artifact read',
    )
  }

  status(terminalId: string): CampaignAgentHostPublicStatus | null {
    if (typeof terminalId !== 'string' || !IDENTIFIER.test(terminalId)) return null
    const state = this.sessions.get(terminalId)
    return state ? this.publicStatus(state) : null
  }

  eligibleSessions(
    identities: readonly MainOwnedCampaignAgentIdentity[],
  ): CampaignAgentHostEligibleSession[] {
    return identities
      .filter((identity) => identity.live && (identity.family === 'codex' || identity.family === 'hermes'))
      .map((identity): CampaignAgentHostEligibleSession => ({
        terminalId: identity.terminalId,
        family: identity.family,
        state: this.sessions.get(identity.terminalId)?.state ?? 'eligible',
      }))
      .sort((left, right) => left.terminalId.localeCompare(right.terminalId))
  }

  async teardownTerminal(
    terminalId: string,
    _reason: CampaignAgentHostTeardownReason,
  ): Promise<void> {
    const state = this.sessions.get(terminalId)
    if (!state) return
    this.sessions.delete(terminalId)
    if (state.expiryTimer !== null) this.scheduler.clear(state.expiryTimer)
    state.expiryTimer = null
    this.clearHeartbeatTimers(state)
    state.actionsEnabled = false
    state.activating = false
    state.lastSuccessfulHeartbeatAt = null
    state.privateKey = null
    state.credential?.fill(0)
    state.credential = null
    let attachmentFailure: unknown = null
    if (state.authorization) {
      try {
        if (!state.authorizationAuthority || !this.options.transport.revokeAttachment) {
          throw new Error('Campaign agent attachment revoke transport is unavailable')
        }
        const response = await this.options.transport.revokeAttachment(
          state.scope.campaignId,
          state.authorization.attachmentId,
          {
            action: 'revoke',
            attachmentId: state.authorization.attachmentId,
            attachmentVersion: state.authorization.attachmentVersion,
            scope: { workspaceId: state.scope.workspaceId, campaignId: state.scope.campaignId },
            actorId: state.authorizationAuthority.humanPrincipalId,
            idempotencyKey: `hostrevoke_${state.authorizationAuthority.idempotencyDigest}`,
          },
        )
        if (!isRecord(response) || response.schema_version !== 'campaign_agent_public_view.v1'
          || !isRecord(response.scope)
          || response.scope.workspace_id !== state.scope.workspaceId
          || response.scope.campaign_id !== state.scope.campaignId
          || !isRecord(response.attachment)
          || response.attachment.attachment_id !== state.authorization.attachmentId
          || response.attachment.status !== 'revoked'
          || !Number.isSafeInteger(response.attachment.attachment_version)
          || Number(response.attachment.attachment_version) <= state.authorization.attachmentVersion
          || !isRecord(response.attachment.provenance)
          || response.attachment.provenance.revoked_by !== state.authorizationAuthority.humanPrincipalId) {
          throw new Error('Campaign agent attachment revoke view is invalid')
        }
      } catch (error) {
        attachmentFailure = error
      }
    }
    const response = await this.options.transport.revoke(state.registration.registrationId)
    if (!isRecord(response) || !exactKeys(response, [
      'schemaVersion', 'registrationId', 'scope', 'agentFamily', 'agentOrigin',
      'agentPrincipalId', 'sessionId', 'publicKeyDigest', 'registeredAt', 'expiresAt', 'status',
    ])
      || response.schemaVersion !== 'campaign_agent_host_session.v1'
      || response.registrationId !== state.registration.registrationId
      || response.status !== 'revoked'
      || !isRecord(response.scope)
      || !exactKeys(response.scope, ['workspaceId', 'campaignId'])
      || response.scope.workspaceId !== state.scope.workspaceId
      || response.scope.campaignId !== state.scope.campaignId
      || response.agentFamily !== state.identity.family
      || response.agentOrigin !== state.identity.origin
      || response.agentPrincipalId !== state.identity.principalId
      || response.sessionId !== state.identity.sessionId
      || response.publicKeyDigest !== state.publicKeyDigest
      || publicDate(response.registeredAt, 'revoke registered time') !== state.registration.registeredAt
      || publicDate(response.expiresAt, 'revoke expiry time') !== state.registration.expiresAt) {
      throw new Error('Campaign agent revoke receipt tuple is invalid')
    }
    if (attachmentFailure) throw attachmentFailure
  }

  async teardownAll(reason: CampaignAgentHostTeardownReason): Promise<void> {
    const terminalIds = [...this.sessions.keys()]
    const results = await Promise.allSettled(terminalIds.map((terminalId) => this.teardownTerminal(terminalId, reason)))
    const failure = results.find((result): result is PromiseRejectedResult => result.status === 'rejected')
    if (failure) throw failure.reason
  }

  disposeLocal(): void {
    for (const state of this.sessions.values()) {
      if (state.expiryTimer !== null) this.scheduler.clear(state.expiryTimer)
      state.expiryTimer = null
      this.clearHeartbeatTimers(state)
      state.actionsEnabled = false
      state.activating = false
      state.lastSuccessfulHeartbeatAt = null
      state.privateKey = null
      state.credential?.fill(0)
      state.credential = null
    }
    this.sessions.clear()
  }
}
