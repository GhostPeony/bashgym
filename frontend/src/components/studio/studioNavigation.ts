export const studioViews = ['home', 'setup', 'experiments', 'resources'] as const
export type StudioView = (typeof studioViews)[number]

export function readStudioView(hash: string): StudioView {
  const value = hash.replace(/^#\/?/, '')
  return studioViews.includes(value as StudioView) ? (value as StudioView) : 'home'
}

export function resolveStudioWorkspace(
  allowed: readonly string[],
  requested: string | null
): string | null {
  const ids = allowed.filter(
    (id) => typeof id === 'string' && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(id)
  )
  return requested && ids.includes(requested) ? requested : ids[0] || null
}
