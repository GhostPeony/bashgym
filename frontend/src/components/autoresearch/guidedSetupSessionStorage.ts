import type { GuidedSetupStep } from './guidedSetupModel'

export interface GuidedSetupStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export interface GuidedSetupMutationScope {
  workspaceId: string
  sessionId: string
  version: number
  step: GuidedSetupStep
  selectionId: string
}

const publicId = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/
const sessionId = /^setupsess_[0-9a-f]{32}$/
const hex32 = /^[0-9a-f]{32}$/
const idempotencyKey = /^idem_[0-9a-f]{32}$/

function safeRandomHex(factory: () => string): string {
  const value = factory()
  if (!hex32.test(value)) throw new Error('Guided setup random authority is invalid')
  return value
}

function sessionStorageKey(workspaceId: string): string {
  if (!publicId.test(workspaceId)) throw new Error('Guided setup workspace scope is invalid')
  return `bashgym.autoresearch.setup-session.v1:${workspaceId}`
}

function mutationStorageKey(scope: GuidedSetupMutationScope): string {
  if (!publicId.test(scope.workspaceId) || !sessionId.test(scope.sessionId)
    || !Number.isInteger(scope.version) || scope.version < 0 || scope.version > 6
    || !publicId.test(scope.selectionId)) throw new Error('Guided setup mutation scope is invalid')
  return `bashgym.autoresearch.setup-idempotency.v1:${scope.workspaceId}:${scope.sessionId}:${scope.version}:${scope.step}:${scope.selectionId}`
}

export function getOrCreateGuidedSetupSessionId(
  storage: GuidedSetupStorage,
  workspaceId: string,
  randomHex: () => string,
): string {
  const key = sessionStorageKey(workspaceId)
  const existing = storage.getItem(key)
  if (existing && sessionId.test(existing)) return existing
  const created = `setupsess_${safeRandomHex(randomHex)}`
  storage.setItem(key, created)
  return created
}

export function readGuidedSetupSessionId(
  storage: GuidedSetupStorage,
  workspaceId: string,
): string | null {
  const existing = storage.getItem(sessionStorageKey(workspaceId))
  return existing && sessionId.test(existing) ? existing : null
}

export function getOrCreateGuidedSetupIdempotencyKey(
  storage: GuidedSetupStorage,
  scope: GuidedSetupMutationScope,
  randomHex: () => string,
): string {
  const key = mutationStorageKey(scope)
  const existing = storage.getItem(key)
  if (existing && idempotencyKey.test(existing)) return existing
  const created = `idem_${safeRandomHex(randomHex)}`
  storage.setItem(key, created)
  return created
}

export function clearGuidedSetupIdempotencyKey(
  storage: GuidedSetupStorage,
  scope: GuidedSetupMutationScope,
): void {
  storage.removeItem(mutationStorageKey(scope))
}
