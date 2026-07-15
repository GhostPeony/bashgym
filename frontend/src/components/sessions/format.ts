export function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  if (count >= 1_000) return `${Math.round(count / 1_000)}k`
  return String(count)
}

export function folderNameFromPath(path: string | undefined, fallback = 'Session'): string {
  if (!path || path === '~') return fallback
  const normalized = path.replace(/\\/g, '/').replace(/\/+$/, '')
  const folder = normalized.split('/').pop()
  return folder && !/^[A-Za-z]:$/.test(folder) ? folder : fallback
}
