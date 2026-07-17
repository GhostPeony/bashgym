import path from 'node:path'

interface BackendRootResolutionInput {
  configuredRoot?: string
  cwd: string
  appPath: string
  resourcesPath: string
  executablePath: string
  markerExists: (markerPath: string) => boolean
}

interface RetryableInitializer {
  ensureReady: () => Promise<void>
  invalidate: () => void
}

function ancestors(value: string, maximumDepth = 5): string[] {
  const result: string[] = []
  let current = path.extname(value) ? path.dirname(value) : value
  for (let depth = 0; depth <= maximumDepth; depth += 1) {
    result.push(current)
    const parent = path.dirname(current)
    if (parent === current) break
    current = parent
  }
  return result
}

export function resolveBackendRoot(input: BackendRootResolutionInput): string | undefined {
  if (input.configuredRoot) {
    const configuredRoot = path.resolve(input.configuredRoot)
    if (!input.markerExists(path.join(configuredRoot, 'bashgym', 'api', 'routes.py'))) {
      throw new Error('The configured BashGym backend root is invalid')
    }
    return configuredRoot
  }

  const inferred = [
    ...ancestors(input.cwd, 2),
    ...ancestors(input.appPath),
    ...ancestors(input.resourcesPath),
    ...ancestors(input.executablePath),
  ]
  const candidates = inferred.map((candidate) => path.resolve(candidate))
  const uniqueCandidates = Array.from(new Set(candidates))
  return uniqueCandidates.find((candidate) => (
    input.markerExists(path.join(candidate, 'bashgym', 'api', 'routes.py'))
  ))
}

export function managedBackendStartAction(
  reachable: boolean,
  ownsLiveChild: boolean,
): 'reuse' | 'spawn' {
  if (reachable && ownsLiveChild) return 'reuse'
  if (reachable) throw new Error('BashGym managed backend port is already in use')
  return 'spawn'
}

export function createRetryableInitializer(
  initialize: () => Promise<void>,
): RetryableInitializer {
  let inFlightOrReady: Promise<void> | null = null

  return {
    ensureReady: () => {
      if (inFlightOrReady) return inFlightOrReady
      const attempt = Promise.resolve().then(initialize)
      inFlightOrReady = attempt
      void attempt.catch(() => {
        if (inFlightOrReady === attempt) inFlightOrReady = null
      })
      return attempt
    },
    invalidate: () => {
      inFlightOrReady = null
    },
  }
}
