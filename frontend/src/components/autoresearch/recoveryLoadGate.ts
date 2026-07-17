/** Deduplicate recovery loads per scope while allowing a new selection to proceed. */
export function createRecoveryLoadGate() {
  let active: { scope: string; promise: Promise<void> } | null = null
  return (scope: string, operation: () => Promise<void>): Promise<void> => {
    if (active?.scope === scope) return active.promise
    const token = { scope, promise: Promise.resolve() }
    token.promise = Promise.resolve()
      .then(operation)
      .finally(() => {
        if (active === token) active = null
      })
    active = token
    return token.promise
  }
}
