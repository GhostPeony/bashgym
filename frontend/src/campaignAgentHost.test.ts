import assert from 'node:assert/strict'
import {
  createCipheriv,
  createHash,
  createPublicKey,
  diffieHellman,
  generateKeyPairSync,
  hkdfSync,
  randomBytes,
} from 'node:crypto'
import test from 'node:test'
import {
  CampaignAgentHostController,
  createMainOwnedCampaignAgentIdentity,
  type CampaignAgentHostLifecycleEvent,
  type CampaignAgentHostScheduler,
  type CampaignAgentHostHeartbeatBody,
  type CampaignAgentHostArtifactQuery,
  type CampaignAgentHostAuthorizeRequest,
  type CampaignAgentHostRegistrationBody,
  type CampaignAgentHostTransport,
  type MainOwnedCampaignAgentIdentity,
} from '../electron/campaignAgentHost'

const NOW = Date.parse('2026-07-17T12:00:00Z')
const TOKEN = `bgag.credential-1.${'s'.repeat(48)}`

class ManualScheduler implements CampaignAgentHostScheduler {
  private nextId = 1
  private readonly tasks = new Map<number, { dueAt: number; action: () => void | Promise<void> }>()

  constructor(private readonly now: () => number, private readonly setNow: (value: number) => void) {}

  set(delayMs: number, action: () => void | Promise<void>): number {
    const id = this.nextId++
    this.tasks.set(id, { dueAt: this.now() + delayMs, action })
    return id
  }

  clear(handle: unknown): void {
    if (typeof handle === 'number') this.tasks.delete(handle)
  }

  async advance(delayMs: number): Promise<void> {
    const target = this.now() + delayMs
    while (true) {
      const next = [...this.tasks.entries()]
        .filter(([, task]) => task.dueAt <= target)
        .sort((left, right) => left[1].dueAt - right[1].dueAt || left[0] - right[0])[0]
      if (!next) break
      this.tasks.delete(next[0])
      this.setNow(next[1].dueAt)
      await next[1].action()
    }
    this.setNow(target)
  }
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => `${JSON.stringify(key)}:${canonical(nested)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

function b64(value: Uint8Array): string {
  return Buffer.from(value).toString('base64url')
}

function liveIdentity(overrides: Partial<MainOwnedCampaignAgentIdentity> = {}): MainOwnedCampaignAgentIdentity {
  return {
    terminalId: 'terminal-1',
    generation: 'generation-1',
    family: 'codex',
    origin: 'desktop_origin_1',
    principalId: 'codex_pty_1',
    sessionId: 'pty_session_1',
    live: true,
    ...overrides,
  }
}

function receipt(body: CampaignAgentHostRegistrationBody, overrides: Record<string, unknown> = {}) {
  return {
    schemaVersion: 'campaign_agent_host_session.v1',
    registrationId: 'cahs_registration_1',
    scope: body.scope,
    agentFamily: body.agentFamily,
    agentOrigin: body.agentOrigin,
    agentPrincipalId: body.agentPrincipalId,
    sessionId: body.sessionId,
    publicKeyDigest: `sha256:${createHash('sha256').update(Buffer.from(body.ephemeralPublicKey, 'base64url')).digest('hex')}`,
    registeredAt: '2026-07-17T12:00:00Z',
    expiresAt: '2026-07-17T12:05:00Z',
    status: 'live',
    ...overrides,
  }
}

function envelope(body: CampaignAgentHostRegistrationBody, registrationId: string, rawToken = TOKEN) {
  const credentialId = 'credential-1'
  const envelopeId = 'cage_envelope_1'
  const publicKeyDigest = `sha256:${createHash('sha256').update(Buffer.from(body.ephemeralPublicKey, 'base64url')).digest('hex')}`
  const aad = {
    schema_version: 'campaign_agent_delivery_aad.v1',
    envelope_id: envelopeId,
    registration_id: registrationId,
    credential_id: credentialId,
    attachment_id: 'attachment-1',
    workspace_id: body.scope.workspaceId,
    campaign_id: body.scope.campaignId,
    agent_family: body.agentFamily,
    agent_origin: body.agentOrigin,
    session_id: body.sessionId,
    agent_principal_id: body.agentPrincipalId,
    public_key_digest: publicKeyDigest,
    issued_at: '2026-07-17T12:00:01Z',
    expires_at: '2026-07-17T13:00:01Z',
  }
  const aadJson = canonical(aad)
  const plaintext = canonical({
    schema_version: 'campaign_agent_credential_delivery.v1',
    registration_id: registrationId,
    credential_id: credentialId,
    raw_token: rawToken,
  })
  const serverKeys = generateKeyPairSync('x25519')
  const recipient = createPublicKey({
    key: { kty: 'OKP', crv: 'X25519', x: body.ephemeralPublicKey },
    format: 'jwk',
  })
  const shared = diffieHellman({ privateKey: serverKeys.privateKey, publicKey: recipient })
  const salt = randomBytes(16)
  const nonce = randomBytes(12)
  const hkdfInfo = `bashgym.campaign-agent-delivery-key.v1:${registrationId}:${credentialId}`
  const key = Buffer.from(hkdfSync('sha256', shared, salt, hkdfInfo, 32))
  const cipher = createCipheriv('chacha20-poly1305', key, nonce, { authTagLength: 16 })
  cipher.setAAD(Buffer.from(aadJson), { plaintextLength: Buffer.byteLength(plaintext) })
  const ciphertext = Buffer.concat([cipher.update(plaintext), cipher.final(), cipher.getAuthTag()])
  const serverJwk = serverKeys.publicKey.export({ format: 'jwk' })
  if (!serverJwk.x) throw new Error('test X25519 key is unavailable')
  return {
    schemaVersion: 'campaign_agent_delivery_envelope.v1',
    envelopeId,
    registrationId,
    credentialId,
    algorithm: 'X25519-HKDF-SHA256+CHACHA20-POLY1305',
    ephemeralPublicKey: serverJwk.x,
    hkdfSalt: b64(salt),
    hkdfInfo,
    nonce: b64(nonce),
    ciphertext: b64(ciphertext),
    aadJson,
    createdAt: '2026-07-17T12:00:01Z',
  }
}

interface HarnessOptions {
  clock?: () => number
  scheduler?: CampaignAgentHostScheduler
  resolveIdentity?: (terminalId: string) => MainOwnedCampaignAgentIdentity | null
  heartbeat?: (credential: Buffer, body: CampaignAgentHostHeartbeatBody) => Promise<unknown>
  observe?: (credential: Buffer) => Promise<unknown>
  artifacts?: (credential: Buffer, query: CampaignAgentHostArtifactQuery) => Promise<unknown>
  onLifecycle?: (event: CampaignAgentHostLifecycleEvent) => void
}

function harness(identity = liveIdentity(), options: HarnessOptions = {}) {
  let registeredBody: CampaignAgentHostRegistrationBody | null = null
  const registeredBodies: CampaignAgentHostRegistrationBody[] = []
  const revoked: string[] = []
  const attachmentRevoked: string[] = []
  let claimOverride: unknown
  const transport: CampaignAgentHostTransport = {
    register: async (body) => {
      registeredBody = body
      registeredBodies.push(body)
      return receipt(body, { registrationId: `cahs_registration_${registeredBodies.length}` })
    },
    claim: async (registrationId) => {
      if (claimOverride !== undefined) return claimOverride
      if (!registeredBody) throw new Error('registration missing')
      return envelope(registeredBody, registrationId)
    },
    revoke: async (registrationId) => {
      revoked.push(registrationId)
      if (!registeredBody) throw new Error('registration missing')
      return receipt(registeredBody, { registrationId, status: 'revoked' })
    },
    grant: async (_campaignId, grantBody) => grantReceipt(grantBody),
    attach: async (_campaignId, attachBody) => attachmentView(attachBody),
    revokeAttachment: async (_campaignId, attachmentId, revokeBody) => {
      attachmentRevoked.push(attachmentId)
      return revokedAttachmentView(revokeBody)
    },
    heartbeat: options.heartbeat ?? (async () => ({})),
    observe: options.observe ?? (async () => ({})),
    artifacts: options.artifacts ?? (async () => ({})),
  }
  const controller = new CampaignAgentHostController({
    transport,
    resolveIdentity: options.resolveIdentity
      ?? ((terminalId) => terminalId === identity.terminalId ? identity : null),
    clock: options.clock ?? (() => NOW),
    scheduler: options.scheduler,
    onLifecycle: options.onLifecycle,
    idFactory: () => '0123456789abcdef0123456789abcdef',
  })
  return {
    controller,
    revoked,
    attachmentRevoked,
    setClaimOverride(value: unknown) { claimOverride = value },
    registeredBody: () => registeredBody,
    registeredBodies,
  }
}

function fnvDigest(principalId: string, requested: readonly string[], granted: readonly string[]): string {
  const payload = `principal=${principalId};requested=${[...requested].sort().join(',')};granted=${[...granted].sort().join(',')}`
  let digest = 0x811C9DC5
  for (const value of Buffer.from(payload, 'ascii')) {
    digest ^= value
    digest = Math.imul(digest, 0x01000193) >>> 0
  }
  return `fnv1a32:${digest.toString(16).padStart(8, '0')}`
}

function grantReceipt(body: Record<string, unknown>) {
  const requested = body.requestedCapabilities as string[]
  const granted = body.grantedCapabilities as string[]
  return {
    schemaVersion: 'campaign_agent_grant_confirmation.v1', issuer: 'campaign_authority',
    receiptId: 'cagr_receipt_1', receiptDigest: `sha256:${'a'.repeat(64)}`,
    humanPrincipal: { principalId: 'desktop-human-1', principalType: 'human' },
    scope: body.scope, agentFamily: body.agentFamily, agentOrigin: body.agentOrigin,
    agentPrincipalId: body.agentPrincipalId, sessionId: body.sessionId,
    requestedCapabilities: requested, grantedCapabilities: granted,
    capabilityDigest: fnvDigest(body.agentPrincipalId as string, requested, granted),
    grantRevision: 1, issuedAt: '2026-07-17T12:00:01Z', expiresAt: '2026-07-17T12:10:01Z',
  }
}

function attachmentView(body: Record<string, unknown>) {
  const receiptValue = body.confirmationReceipt as Record<string, unknown>
  return {
    schema_version: 'campaign_agent_public_view.v1', observed_at: '2026-07-17T12:00:02Z',
    scope: { workspace_id: 'workspace-1', campaign_id: 'campaign-1' },
    attachment: {
      attachment_id: 'caa_attachment_1', attachment_version: 1, status: 'attached',
      requested_capabilities: body.requestedCapabilities,
      granted_capabilities: body.grantedCapabilities,
      receipt_window: { from_version: 1, through_version: 1, has_earlier: false },
      provenance: {
        agent_family: body.agentFamily, agent_origin: body.agentOrigin,
        session_id: body.sessionId, agent_principal_id: body.agentPrincipalId,
        grant_receipt_id: receiptValue.receiptId,
        grant_receipt_digest: receiptValue.receiptDigest,
        credential_expires_at: '2026-07-17T13:00:02Z',
      },
      receipts: [],
    },
    audit_events: [],
  }
}

function revokedAttachmentView(body: Record<string, unknown>) {
  return {
    schema_version: 'campaign_agent_public_view.v1', observed_at: '2026-07-17T12:00:03Z',
    scope: { workspace_id: 'workspace-1', campaign_id: 'campaign-1' },
    attachment: {
      attachment_id: body.attachmentId, attachment_version: 2, status: 'revoked',
      provenance: { revoked_by: body.actorId },
    },
    audit_events: [],
  }
}

async function authorizeDefault(controller: CampaignAgentHostController) {
  return controller.authorize({
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    requestedCapabilities: ['campaign_observe'], grantedCapabilities: ['campaign_observe'],
    idempotencyKey: 'authorize-visible-choice-1',
  })
}

test('host derives the exact agent tuple from a live main-owned PTY and exposes no key material', async () => {
  const { controller, registeredBody } = harness()
  const status = await controller.attach({
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
  })

  assert.deepEqual(status, {
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    family: 'codex', registrationId: 'cahs_registration_1', state: 'registered',
    expiresAt: '2026-07-17T12:05:00.000Z', credentialReady: false,
  })
  const registered = registeredBody()
  assert.ok(registered)
  assert.match(registered.ephemeralPublicKey, /^[A-Za-z0-9_-]{43}$/)
  assert.deepEqual({ ...registered, ephemeralPublicKey: '<public-key>' }, {
    scope: { workspaceId: 'workspace-1', campaignId: 'campaign-1' },
    agentFamily: 'codex', agentOrigin: 'desktop_origin_1', agentPrincipalId: 'codex_pty_1',
    sessionId: 'pty_session_1', ephemeralPublicKey: '<public-key>',
    ttlSeconds: 300, idempotencyKey: 'hostreg_0123456789abcdef0123456789abcdef',
  })
  assert.equal(JSON.stringify(status).includes('ephemeral'), false)
  assert.equal(JSON.stringify(status).includes('agentOrigin'), false)
})

test('main-owned PTY identity is stable per generation and contains no renderer provenance', () => {
  const first = createMainOwnedCampaignAgentIdentity({
    terminalId: 'terminal-1', generation: 'generation-1', family: 'codex', hostInstanceId: 'host-instance-1',
  })
  const repeated = createMainOwnedCampaignAgentIdentity({
    terminalId: 'terminal-1', generation: 'generation-1', family: 'codex', hostInstanceId: 'host-instance-1',
  })
  const replacement = createMainOwnedCampaignAgentIdentity({
    terminalId: 'terminal-1', generation: 'generation-2', family: 'codex', hostInstanceId: 'host-instance-1',
  })
  assert.deepEqual(first, repeated)
  assert.notEqual(first.sessionId, replacement.sessionId)
  assert.deepEqual(
    { ...first, origin: '<opaque>', principalId: '<opaque>', sessionId: '<opaque>' },
    {
      terminalId: 'terminal-1', generation: 'generation-1', family: 'codex', live: true,
      origin: '<opaque>', principalId: '<opaque>', sessionId: '<opaque>',
    },
  )
  assert.match(first.origin, /^desktop_[0-9a-f]{32}$/)
  assert.match(first.principalId, /^codex_pty_[0-9a-f]{32}$/)
  assert.match(first.sessionId, /^pty_[0-9a-f]{32}$/)
  assert.equal(JSON.stringify(first).includes('host-instance-1'), false)
})

test('host rejects absent, exited, unsupported, and renderer-selected family identities', async () => {
  await assert.rejects(
    () => harness(liveIdentity({ live: false })).controller.attach({ terminalId: 'terminal-1', workspaceId: 'w', campaignId: 'c' }),
    /live PTY/i,
  )
  await assert.rejects(
    () => harness(liveIdentity({ family: 'claude' as 'codex' })).controller.attach({ terminalId: 'terminal-1', workspaceId: 'w', campaignId: 'c' }),
    /unsupported/i,
  )
  await assert.rejects(
    () => harness().controller.attach({ terminalId: 'terminal-1', workspaceId: 'w', campaignId: 'c', agentFamily: 'hermes' } as never),
    /request contract/i,
  )
})

test('eligible-session projection exposes only opaque live terminal descriptors', async () => {
  const { controller } = harness()
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  const descriptors = controller.eligibleSessions([
    liveIdentity(),
    liveIdentity({ terminalId: 'terminal-2', generation: 'generation-2', family: 'hermes', origin: 'private-origin', principalId: 'private-principal', sessionId: 'private-session' }),
    liveIdentity({ terminalId: 'terminal-dead', generation: 'generation-3', live: false }),
    liveIdentity({ terminalId: 'terminal-claude', generation: 'generation-4', family: 'claude' as 'codex' }),
  ])
  assert.deepEqual(descriptors, [
    { terminalId: 'terminal-1', family: 'codex', state: 'registered' },
    { terminalId: 'terminal-2', family: 'hermes', state: 'eligible' },
  ])
  assert.equal(JSON.stringify(descriptors).includes('private-origin'), false)
  assert.equal(JSON.stringify(descriptors).includes('private-principal'), false)
  assert.equal(JSON.stringify(descriptors).includes('private-session'), false)
})

test('claim keeps the credential controller-owned and activate gates fixed actions behind an immediate heartbeat', async () => {
  const credentials: Buffer[] = []
  const heartbeatBodies: CampaignAgentHostHeartbeatBody[] = []
  const artifactQueries: CampaignAgentHostArtifactQuery[] = []
  const { controller } = harness(liveIdentity(), {
    heartbeat: async (credential, body) => {
      credentials.push(credential)
      heartbeatBodies.push(body)
      return { ok: true }
    },
    observe: async (credential) => {
      credentials.push(credential)
      return { campaign: 'bounded-observation' }
    },
    artifacts: async (credential, query) => {
      credentials.push(credential)
      artifactQueries.push(query)
      return { artifacts: ['bounded-artifact'] }
    },
  })
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await controller.authorize({
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    requestedCapabilities: ['campaign_observe', 'artifact_read'],
    grantedCapabilities: ['campaign_observe', 'artifact_read'],
    idempotencyKey: 'authorize-visible-choice-1',
  })
  const publicStatus = await controller.claim('terminal-1')
  assert.equal(publicStatus.credentialReady, true)
  assert.equal(publicStatus.actionsEnabled, false)
  assert.equal(JSON.stringify(publicStatus).includes('bgag.'), false)
  assert.equal(JSON.stringify(publicStatus).includes('ciphertext'), false)
  await assert.rejects(() => controller.observe('terminal-1'), /activate/i)

  const active = await controller.activate('terminal-1')
  assert.equal(active.actionsEnabled, true)
  assert.equal(active.state, 'active')
  assert.deepEqual(await controller.observe('terminal-1'), { campaign: 'bounded-observation' })
  assert.deepEqual(await controller.artifacts('terminal-1', {
    afterCursor: 'a1.ABCDEFGHIJK', limit: 10,
  }), { artifacts: ['bounded-artifact'] })
  assert.equal(credentials.length, 3)
  assert.strictEqual(credentials[0], credentials[1])
  assert.strictEqual(credentials[0], credentials[2])
  assert.equal(credentials[0].toString('utf8'), TOKEN)
  assert.deepEqual(heartbeatBodies, [{
    scope: { workspaceId: 'workspace-1', campaignId: 'campaign-1' },
    agentFamily: 'codex',
    agentOrigin: 'desktop_origin_1',
    agentPrincipalId: 'codex_pty_1',
    sessionId: 'pty_session_1',
  }])
  assert.deepEqual(artifactQueries, [{ afterCursor: 'a1.ABCDEFGHIJK', limit: 10 }])
  const serialized = JSON.stringify(active)
  for (const privateValue of [
    TOKEN, 'desktop_origin_1', 'codex_pty_1', 'pty_session_1',
    'header', 'endpoint', 'private error',
  ]) assert.equal(serialized.includes(privateValue), false)
  await assert.rejects(() => controller.claim('terminal-1'), /already claimed/i)
})

test('claim fails closed until the main-owned human authorization flow is complete', async () => {
  const { controller } = harness()
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await assert.rejects(() => controller.claim('terminal-1'), /authorization/i)
  assert.equal(controller.status('terminal-1')?.state, 'registered')
})

test('artifact queries reject extra, hostile, and unbounded values before credential transport', async () => {
  let artifactCalls = 0
  const { controller } = harness(liveIdentity(), {
    artifacts: async () => {
      artifactCalls += 1
      return { artifacts: [] }
    },
  })
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await controller.authorize({
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    requestedCapabilities: ['artifact_read'], grantedCapabilities: ['artifact_read'],
    idempotencyKey: 'authorize-artifact-read-1',
  })
  await controller.claim('terminal-1')
  await controller.activate('terminal-1')

  const hostileCursor = `a1.${TOKEN}`
  const accessorQuery = Object.defineProperty({}, 'limit', {
    enumerable: true,
    get: () => { throw new Error('private accessor must not run') },
  })
  const symbolQuery = { [Symbol('private')]: 10 }
  const invalidQueries: unknown[] = [
    null,
    [],
    accessorQuery,
    symbolQuery,
    { extra: true },
    { afterCursor: 'not-a-cursor' },
    { afterCursor: hostileCursor },
    { limit: 0 },
    { limit: 51 },
    { limit: 1.5 },
    { limit: '10' },
  ]
  for (const query of invalidQueries) {
    await assert.rejects(
      () => controller.artifacts('terminal-1', query as CampaignAgentHostArtifactQuery),
      (error: unknown) => error instanceof Error
        && /artifact query is invalid/i.test(error.message)
        && !error.message.includes(TOKEN),
    )
  }
  assert.equal(artifactCalls, 0)
  assert.deepEqual(await controller.artifacts('terminal-1', {}), { artifacts: [] })
  assert.equal(artifactCalls, 1)
})

test('active credentials heartbeat every 30 seconds, retry one transient failure, and lock at 90 seconds stale', async () => {
  let now = NOW
  const scheduler = new ManualScheduler(() => now, (value) => { now = value })
  const lifecycle: CampaignAgentHostLifecycleEvent[] = []
  let heartbeatCount = 0
  const { controller } = harness(liveIdentity(), {
    clock: () => now,
    scheduler,
    onLifecycle: (event) => lifecycle.push(event),
    heartbeat: async () => {
      heartbeatCount += 1
      if (heartbeatCount === 1 || heartbeatCount === 3) return { ok: true }
      throw new Error('temporary network failure')
    },
  })
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await authorizeDefault(controller)
  await controller.claim('terminal-1')
  await controller.activate('terminal-1')
  assert.equal(heartbeatCount, 1)

  await scheduler.advance(30_000)
  assert.equal(heartbeatCount, 2)
  await scheduler.advance(4_999)
  assert.equal(heartbeatCount, 2)
  await scheduler.advance(1)
  assert.equal(heartbeatCount, 3)
  assert.equal(controller.status('terminal-1')?.actionsEnabled, true)

  await scheduler.advance(89_999)
  assert.equal(controller.status('terminal-1')?.actionsEnabled, true)
  assert.deepEqual(lifecycle, [{ terminalId: 'terminal-1', kind: 'activated' }])
  await scheduler.advance(1)
  assert.equal(controller.status('terminal-1')?.actionsEnabled, false)
  assert.equal(controller.status('terminal-1')?.state, 'credential_ready')
  assert.deepEqual(lifecycle, [
    { terminalId: 'terminal-1', kind: 'activated' },
    { terminalId: 'terminal-1', kind: 'actions_locked', reason: 'heartbeat_timeout' },
  ])
  await assert.rejects(() => controller.observe('terminal-1'), /not active/i)
})

test('401 and 403 fixed-action failures fail closed, erase the credential, and expose no private error', async () => {
  for (const status of [401, 403]) {
    const lifecycle: CampaignAgentHostLifecycleEvent[] = []
    const credentials: Buffer[] = []
    const { controller, revoked } = harness(liveIdentity(), {
      onLifecycle: (event) => lifecycle.push(event),
      heartbeat: async (value) => {
        credentials.push(value)
        return { ok: true }
      },
      observe: async () => {
        throw Object.assign(new Error(`private backend ${status} detail`), { status })
      },
    })
    await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
    await authorizeDefault(controller)
    await controller.claim('terminal-1')
    await controller.activate('terminal-1')
    await assert.rejects(
      () => controller.observe('terminal-1'),
      (error: unknown) => error instanceof Error
        && /authority rejected/i.test(error.message)
        && !error.message.includes('private backend'),
    )
    assert.equal(controller.status('terminal-1'), null)
    assert.deepEqual(revoked, ['cahs_registration_1'])
    assert.equal(credentials.length, 1)
    assert.equal(credentials[0].every((value) => value === 0), true)
    assert.deepEqual(lifecycle.at(-1), {
      terminalId: 'terminal-1', kind: 'torn_down', reason: 'authority_rejected',
    })
  }
})

test('credential expiry and main-owned identity changes fail closed before fixed actions run', async () => {
  for (const failure of ['credential_expired', 'identity_changed'] as const) {
    let now = NOW
    let identity = liveIdentity()
    const scheduler = new ManualScheduler(() => now, (value) => { now = value })
    const lifecycle: CampaignAgentHostLifecycleEvent[] = []
    let observed = false
    const { controller } = harness(identity, {
      clock: () => now,
      scheduler,
      resolveIdentity: () => identity,
      onLifecycle: (event) => lifecycle.push(event),
      observe: async () => {
        observed = true
        return { campaign: 'must-not-run' }
      },
    })
    await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
    await authorizeDefault(controller)
    await controller.claim('terminal-1')
    await controller.activate('terminal-1')
    if (failure === 'credential_expired') now = Date.parse('2026-07-17T13:00:03Z')
    else identity = liveIdentity({ generation: 'generation-2' })

    await assert.rejects(() => controller.observe('terminal-1'), /no longer authorized/i)
    assert.equal(observed, false)
    assert.equal(controller.status('terminal-1'), null)
    assert.deepEqual(lifecycle.at(-1), {
      terminalId: 'terminal-1', kind: 'torn_down', reason: failure,
    })
  }
})

test('identity replacement during the activation heartbeat cannot activate stale authority', async () => {
  let identity = liveIdentity()
  const lifecycle: CampaignAgentHostLifecycleEvent[] = []
  const { controller } = harness(identity, {
    resolveIdentity: () => identity,
    onLifecycle: (event) => lifecycle.push(event),
    heartbeat: async () => {
      identity = liveIdentity({ generation: 'generation-2' })
      return { ok: true }
    },
  })
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await authorizeDefault(controller)
  await controller.claim('terminal-1')

  await assert.rejects(() => controller.activate('terminal-1'), /no longer authorized/i)
  assert.equal(controller.status('terminal-1'), null)
  assert.deepEqual(lifecycle, [
    { terminalId: 'terminal-1', kind: 'torn_down', reason: 'identity_changed' },
  ])
})

test('claimed sessions renew with a fresh X25519 key without exposing or redelivering credentials', async () => {
  const { controller, registeredBodies } = harness()
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await authorizeDefault(controller)
  await controller.claim('terminal-1')
  const before = controller.status('terminal-1')
  await controller.renewRegistration('terminal-1')
  const after = controller.status('terminal-1')
  assert.equal(registeredBodies.length, 2)
  assert.notEqual(registeredBodies[0].ephemeralPublicKey, registeredBodies[1].ephemeralPublicKey)
  assert.equal(before?.registrationId, 'cahs_registration_1')
  assert.equal(after?.registrationId, 'cahs_registration_2')
  assert.equal(after?.credentialReady, true)
})

test('authorize owns grant confirmation and attachment tuple fields and returns only a stripped summary', async () => {
  const { controller } = harness()
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  const status = await controller.authorize({
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    requestedCapabilities: ['training_launch', 'campaign_observe'],
    grantedCapabilities: ['campaign_observe'],
    idempotencyKey: 'authorize-visible-choice-1',
  })
  assert.deepEqual(status.authorization, {
    receiptId: 'cagr_receipt_1', attachmentId: 'caa_attachment_1', attachmentVersion: 1,
    grantedCapabilities: ['campaign_observe'], credentialExpiresAt: '2026-07-17T13:00:02.000Z',
  })
  const serialized = JSON.stringify(status)
  for (const privateField of ['agentOrigin', 'agentPrincipalId', 'sessionId', 'desktop_origin_1', 'codex_pty_1', 'pty_session_1']) {
    assert.equal(serialized.includes(privateField), false)
  }
})

test('authorize rejects renderer identity fields, unknown capabilities, and scope changes', async () => {
  const { controller } = harness()
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  const valid: CampaignAgentHostAuthorizeRequest = {
    terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1',
    requestedCapabilities: ['campaign_observe'], grantedCapabilities: ['campaign_observe'],
    idempotencyKey: 'authorize-visible-choice-1',
  }
  await assert.rejects(() => controller.authorize({ ...valid, agentOrigin: 'renderer-spoof' } as never), /contract/i)
  await assert.rejects(() => controller.authorize({ ...valid, requestedCapabilities: ['campaign_write'] } as never), /capabilit/i)
  await assert.rejects(() => controller.authorize({ ...valid, campaignId: 'campaign-other' }), /scope/i)
})

test('claim fails closed on cross-scope, replayed registration, public-key, and credential mismatches', async () => {
  const cases: Array<[string, (value: ReturnType<typeof envelope>) => void]> = [
    ['registration', (value) => { value.registrationId = 'cahs_other' }],
    ['scope', (value) => {
      const aad = JSON.parse(value.aadJson)
      aad.workspace_id = 'workspace-other'
      value.aadJson = canonical(aad)
    }],
    ['key', (value) => {
      const aad = JSON.parse(value.aadJson)
      aad.public_key_digest = `sha256:${'0'.repeat(64)}`
      value.aadJson = canonical(aad)
    }],
    ['credential', (value) => { value.credentialId = 'credential-other' }],
  ]
  for (const [label, mutate] of cases) {
    const current = harness()
    await current.controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
    await authorizeDefault(current.controller)
    const body = current.registeredBody()
    assert.ok(body)
    const value = envelope(body, 'cahs_registration_1')
    mutate(value)
    current.setClaimOverride(value)
    await assert.rejects(() => current.controller.claim('terminal-1'), /campaign agent/i, label)
    assert.deepEqual(current.revoked, ['cahs_registration_1'])
    assert.equal(current.controller.status('terminal-1'), null)
  }
})

test('teardown stays locally revoked when the backend revoke receipt is cross-scoped', async () => {
  let body: CampaignAgentHostRegistrationBody | null = null
  const controller = new CampaignAgentHostController({
    resolveIdentity: () => liveIdentity(),
    clock: () => NOW,
    idFactory: () => '0123456789abcdef0123456789abcdef',
    transport: {
      register: async (value) => {
        body = value
        return receipt(value)
      },
      claim: async () => { throw new Error('not used') },
      revoke: async (registrationId) => {
        if (!body) throw new Error('missing body')
        return receipt(body, {
          registrationId,
          status: 'revoked',
          scope: { workspaceId: 'workspace-other', campaignId: body.scope.campaignId },
        })
      },
    },
  })
  await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
  await assert.rejects(() => controller.teardownTerminal('terminal-1', 'pty_exit'), /revoke receipt/i)
  assert.equal(controller.status('terminal-1'), null)
})

test('PTY exit, replacement, renderer reload, and shutdown revoke remotely and drop local authority', async () => {
  for (const reason of ['pty_exit', 'pty_replacement', 'renderer_reload', 'app_shutdown'] as const) {
    const { controller, revoked, attachmentRevoked } = harness()
    await controller.attach({ terminalId: 'terminal-1', workspaceId: 'workspace-1', campaignId: 'campaign-1' })
    await authorizeDefault(controller)
    await controller.claim('terminal-1')
    await controller.teardownTerminal('terminal-1', reason)
    assert.deepEqual(attachmentRevoked, ['caa_attachment_1'])
    assert.deepEqual(revoked, ['cahs_registration_1'])
    assert.equal(controller.status('terminal-1'), null)
  }
})
