const DEFAULT_API_BASE = 'http://127.0.0.1:8003/api'
const DEFAULT_DEV_SERVER_URL = 'http://127.0.0.1:5173'

export interface DesktopRuntimeEndpoints {
  apiBase: string
  apiOrigin: string
  webSocketUrl: string
  devServerUrl: string
}

function loopbackHostname(hostname: string): boolean {
  return ['localhost', '127.0.0.1', '[::1]'].includes(hostname.toLowerCase())
}

function normalizeApiBase(value: string): Pick<DesktopRuntimeEndpoints, 'apiBase' | 'apiOrigin' | 'webSocketUrl'> {
  const url = new URL(value)
  if (url.protocol !== 'http:' || !loopbackHostname(url.hostname)) {
    throw new Error('Desktop BashGym API must use a loopback HTTP URL')
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error('Desktop BashGym API URL cannot contain credentials, a query, or a fragment')
  }
  const path = url.pathname.replace(/\/+$/, '')
  if (path && path !== '/api') {
    throw new Error('Desktop BashGym API URL path must be /api')
  }
  const apiBase = `${url.origin}/api`
  const webSocket = new URL('/ws', url.origin)
  webSocket.protocol = 'ws:'
  return { apiBase, apiOrigin: url.origin, webSocketUrl: webSocket.toString().replace(/\/$/, '') }
}

function normalizeDevServerUrl(value: string): string {
  const url = new URL(value)
  if (!['http:', 'https:'].includes(url.protocol) || !loopbackHostname(url.hostname)) {
    throw new Error('BashGym development server must use a loopback HTTP(S) origin')
  }
  if (url.username || url.password || url.search || url.hash || url.pathname !== '/') {
    throw new Error('BashGym development server must be a credential-free origin')
  }
  return url.origin
}

export function resolveDesktopRuntimeEndpoints(
  environment: Record<string, string | undefined>,
): DesktopRuntimeEndpoints {
  const api = normalizeApiBase(
    environment.BASHGYM_API_BASE
      || environment.BASHGYM_API_URL
      || DEFAULT_API_BASE,
  )
  return {
    ...api,
    devServerUrl: normalizeDevServerUrl(
      environment.BASHGYM_DEV_SERVER_URL || DEFAULT_DEV_SERVER_URL,
    ),
  }
}

export const DEFAULT_DESKTOP_RUNTIME_ENDPOINTS = resolveDesktopRuntimeEndpoints({})
