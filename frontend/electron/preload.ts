import { contextBridge, ipcRenderer, type IpcRendererEvent } from 'electron'
import type {
  CampaignBody,
  CampaignMethod,
  CampaignQuery,
  CampaignRequestAuthority,
  CampaignResponse,
} from './campaignBridge'
import type {
  CampaignAgentHostAttachRequest,
  CampaignAgentHostAuthorizeRequest,
  CampaignAgentHostEligibleSession,
  CampaignAgentHostPublicStatus,
} from './campaignAgentHost'

export type CampaignRoute =
  | '/api/campaign-auth/capabilities'
  | '/api/campaigns'
  | '/api/campaigns/from-template'
  | `/api/campaigns/${string}`
  | `/api/campaigns/${string}/events`
  | `/api/campaigns/${string}/artifacts`
  | `/api/campaigns/${string}/human-work`
  | `/api/campaigns/${string}/human-work/${string}/claim`
  | `/api/campaigns/${string}/human-work/${string}/submit`
  | `/api/campaigns/${string}/human-promotion`
  | `/api/campaigns/${string}/agent-attachment`
  | `/api/campaigns/${string}/attempts`
  | `/api/campaigns/${string}/comparisons`
  | `/api/campaigns/${string}/attempts/${string}/metrics`
  | `/api/campaigns/${string}/start`
  | `/api/campaigns/${string}/pause`
  | `/api/campaigns/${string}/resume`
  | `/api/campaigns/${string}/cancel`

// Type definitions for the exposed API
export interface TerminalCreateResult {
  success: boolean
  id?: string
  /** true if an existing live PTY was re-attached instead of spawned */
  attached?: boolean
  /** scrollback replay payload, present when attached */
  buffer?: string
  cwd?: string
  error?: string
}

export interface TerminalSessionInfo {
  id: string
  cwd: string
  createdAt: number
  exited: boolean
  exitCode: number | null
}

export interface TerminalSnapshotResult {
  success: boolean
  /** Raw scrollback tail (may contain ANSI sequences) */
  data?: string
  cwd?: string
  exited?: boolean
  error?: string
}

export interface TerminalOutputFrame {
  data: string
  frameId: number
}

export interface TerminalAPI {
  create: (id: string, cwd?: string) => Promise<TerminalCreateResult>
  write: (id: string, data: string) => Promise<boolean>
  resize: (id: string, cols: number, rows: number) => Promise<boolean>
  kill: (id: string) => Promise<boolean>
  list: () => Promise<TerminalSessionInfo[]>
  snapshot: (id: string, maxBytes?: number) => Promise<TerminalSnapshotResult>
  setOutputFlowControl: (id: string, owner: string, enabled: boolean) => void
  ackOutput: (id: string, owner: string, frameId: number) => void
  onData: (id: string, callback: (output: string | TerminalOutputFrame) => void) => () => void
  onExit: (id: string, callback: (exitCode: number) => void) => () => void
}

export interface ThemeAPI {
  get: () => Promise<'light' | 'dark'>
  set: (theme: 'light' | 'dark') => Promise<'light' | 'dark'>
}

export interface SystemAPI {
  info: () => Promise<{
    platform: string
    arch: string
    nodeVersion: string
    electronVersion: string
    cwd: string
  }>
}

export interface RuntimeAPI {
  apiBase: string
  webSocketUrl: string
}

export type ApiStreamEvent =
  | { type: 'headers'; ok: boolean; status: number }
  | { type: 'chunk'; data: string }
  | { type: 'done' }
  | { type: 'aborted' }
  | { type: 'error'; error: string }

export interface ApiProxy {
  fetch: (url: string, options?: RequestInit) => Promise<{
    ok: boolean
    status?: number
    data?: any
    error?: string
  }>
  stream: (
    url: string,
    options: RequestInit | undefined,
    callback: (event: ApiStreamEvent) => void
  ) => () => void
}

export interface FileInfo {
  name: string
  path: string
  type: 'file' | 'directory'
  size?: number
  modified?: number
}

export interface FilesAPI {
  readDirectory: (path: string) => Promise<{
    success: boolean
    files?: FileInfo[]
    error?: string
  }>
  getHomeDirectory: () => Promise<string>
  getParentDirectory: (path: string) => Promise<string>
  readFile: (path: string) => Promise<{
    success: boolean
    content?: string
    error?: string
  }>
  exists: (path: string) => Promise<boolean>
  stat: (path: string) => Promise<{
    success: boolean
    stats?: {
      isFile: boolean
      isDirectory: boolean
      size: number
      modified: number
      created: number
    }
    error?: string
  }>
  writeTempFile: (dataUrl: string, ext: string, basename?: string) => Promise<{
    success: boolean
    path?: string
    error?: string
  }>
}

export interface SessionFileInfo {
  path: string
  size: number
  modified: number
}

export interface SessionAccountInfo {
  emailAddress: string | null
  displayName: string | null
  organizationName: string | null
  organizationRole: string | null
  billingType: string | null
  seatTier: string | null
  userRateLimitTier: string | null
  organizationRateLimitTier: string | null
}

/** Read-only access to local agent-CLI session journals (Claude Code / Codex) */
export interface SessionsAPI {
  scan: (lookbackDays?: number) => Promise<{
    success: boolean
    claude?: SessionFileInfo[]
    codex?: SessionFileInfo[]
    error?: string
  }>
  readTail: (filePath: string, fromOffset: number, maxBytes?: number) => Promise<{
    success: boolean
    data?: string
    newOffset?: number
    size?: number
    reset?: boolean
    error?: string
  }>
  readHead: (filePath: string, maxBytes?: number) => Promise<{
    success: boolean
    data?: string
    error?: string
  }>
  readAccount: () => Promise<{
    success: boolean
    account?: SessionAccountInfo
    error?: string
  }>
}

export interface WindowAPI {
  minimize: () => Promise<void>
  maximize: () => Promise<void>
  close: () => Promise<void>
  isMaximized: () => Promise<boolean>
  onAppKeydown: (callback: (data: { key: string; ctrlKey: boolean; shiftKey: boolean }) => void) => () => void
}

export interface BrowserAPI {
  screenshot: (webContentsId: number, rect?: { x: number; y: number; width: number; height: number; vpW: number; vpH: number }) => Promise<{
    success: boolean
    dataUrl?: string
    error?: string
  }>
}

export interface ClipboardAPI {
  writeImage: (dataUrl: string) => Promise<{ success: boolean; error?: string }>
  writeText: (text: string) => Promise<{ success: boolean; error?: string }>
}

export interface CredentialsAPI {
  store: (key: string, value: string) => Promise<{ success: boolean; error?: string }>
  read: (key: string) => Promise<{ success: boolean; value?: string; error?: string }>
  delete: (key: string) => Promise<{ success: boolean; error?: string }>
}

export interface AgentBridgeAPI {
  prepareLaunch: (request: {
    kind: 'claude' | 'codex'
    workspaceId: string
    terminalId: string
    panelId?: string
    apiBase?: string
  }) => Promise<{ success: boolean; command?: string; error?: string }>
}

export type CampaignAgentHostResult<T extends Record<string, unknown> = Record<string, never>> =
  | ({ success: true } & T)
  | { success: false; error: string }

export interface CampaignAgentHostLaunchRequest {
  workspaceId: string
  campaignId: string
  cwd?: string
}

export type CampaignAgentHostScopeRequest = Pick<
  CampaignAgentHostLaunchRequest,
  'workspaceId' | 'campaignId'
>

export interface CampaignAgentHostTerminalScopeRequest extends CampaignAgentHostScopeRequest {
  terminalId: string
}

export interface CampaignAgentHostAPI {
  launch: (request: CampaignAgentHostLaunchRequest) => Promise<CampaignAgentHostResult<{ terminalId: string; cwd: string }>>
  eligible: (request: CampaignAgentHostScopeRequest) => Promise<CampaignAgentHostResult<{ sessions: CampaignAgentHostEligibleSession[] }>>
  attach: (request: CampaignAgentHostAttachRequest) => Promise<CampaignAgentHostResult<{ status: CampaignAgentHostPublicStatus }>>
  authorize: (request: CampaignAgentHostAuthorizeRequest) => Promise<CampaignAgentHostResult<{ status: CampaignAgentHostPublicStatus }>>
  activate: (request: CampaignAgentHostTerminalScopeRequest) => Promise<CampaignAgentHostResult<{ status: CampaignAgentHostPublicStatus }>>
  revoke: (request: CampaignAgentHostTerminalScopeRequest) => Promise<CampaignAgentHostResult>
}

export interface BashGymAPI {
  terminal: TerminalAPI
  theme: ThemeAPI
  system: SystemAPI
  runtime: RuntimeAPI
  api: ApiProxy
  files: FilesAPI
  sessions: SessionsAPI
  window: WindowAPI
  browser: BrowserAPI
  clipboard: ClipboardAPI
  credentials: CredentialsAPI
  agentBridge: AgentBridgeAPI
  campaignAgentHost: CampaignAgentHostAPI
  campaignRequest: (
    method: CampaignMethod,
    route: CampaignRoute,
    body?: CampaignBody,
    query?: CampaignQuery,
    authority?: CampaignRequestAuthority,
  ) => Promise<CampaignResponse>
}

function runtimeArgument(name: string, fallback: string): string {
  const prefix = `--${name}=`
  return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length)
    || fallback
}

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('bashgym', {
  terminal: {
    create: (id: string, cwd?: string) => ipcRenderer.invoke('terminal:create', id, cwd),
    write: (id: string, data: string) => {
      ipcRenderer.send('terminal:write', id, data)
      return Promise.resolve(true)
    },
    resize: (id: string, cols: number, rows: number) => ipcRenderer.invoke('terminal:resize', id, cols, rows),
    kill: (id: string) => ipcRenderer.invoke('terminal:kill', id),
    list: () => ipcRenderer.invoke('terminal:list'),
    snapshot: (id: string, maxBytes?: number) => ipcRenderer.invoke('terminal:snapshot', id, maxBytes),
    setOutputFlowControl: (id: string, owner: string, enabled: boolean) => {
      ipcRenderer.send('terminal:output-flow', id, owner, enabled)
    },
    ackOutput: (id: string, owner: string, frameId: number) => {
      ipcRenderer.send('terminal:output-ack', id, owner, frameId)
    },
    onData: (id: string, callback: (output: string | TerminalOutputFrame) => void) => {
      const channel = `terminal:data:${id}`
      const listener = (_: any, output: string | TerminalOutputFrame) => callback(output)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    },
    onExit: (id: string, callback: (exitCode: number) => void) => {
      const channel = `terminal:exit:${id}`
      const listener = (_: any, exitCode: number) => callback(exitCode)
      ipcRenderer.on(channel, listener)
      return () => ipcRenderer.removeListener(channel, listener)
    }
  },
  theme: {
    get: () => ipcRenderer.invoke('theme:get'),
    set: (theme: 'light' | 'dark') => ipcRenderer.invoke('theme:set', theme)
  },
  system: {
    info: () => ipcRenderer.invoke('system:info')
  },
  runtime: {
    apiBase: runtimeArgument('bashgym-api-base', 'http://127.0.0.1:8003/api'),
    webSocketUrl: runtimeArgument('bashgym-websocket-url', 'ws://127.0.0.1:8003/ws'),
  },
  api: {
    fetch: (url: string, options?: RequestInit) => ipcRenderer.invoke('api:fetch', url, options),
    stream: (
      url: string,
      options: RequestInit | undefined,
      callback: (event: ApiStreamEvent) => void
    ) => {
      const requestId = crypto.randomUUID()
      let closed = false
      const listener = (_event: IpcRendererEvent, id: string, payload: ApiStreamEvent) => {
        if (id !== requestId || closed) return
        callback(payload)
        if (payload.type === 'done' || payload.type === 'aborted' || payload.type === 'error') {
          closed = true
          ipcRenderer.removeListener('api:stream:event', listener)
        }
      }
      ipcRenderer.on('api:stream:event', listener)
      ipcRenderer.send('api:stream:start', requestId, url, options)
      return () => {
        if (closed) return
        closed = true
        ipcRenderer.removeListener('api:stream:event', listener)
        ipcRenderer.send('api:stream:cancel', requestId)
      }
    }
  },
  files: {
    readDirectory: (path: string) => ipcRenderer.invoke('files:readDirectory', path),
    getHomeDirectory: () => ipcRenderer.invoke('files:getHomeDirectory'),
    getParentDirectory: (path: string) => ipcRenderer.invoke('files:getParentDirectory', path),
    readFile: (path: string) => ipcRenderer.invoke('files:readFile', path),
    exists: (path: string) => ipcRenderer.invoke('files:exists', path),
    stat: (path: string) => ipcRenderer.invoke('files:stat', path),
    writeTempFile: (dataUrl: string, ext: string, basename?: string) => ipcRenderer.invoke('files:writeTempFile', dataUrl, ext, basename)
  },
  sessions: {
    scan: (lookbackDays?: number) => ipcRenderer.invoke('sessions:scan', lookbackDays),
    readTail: (filePath: string, fromOffset: number, maxBytes?: number) => ipcRenderer.invoke('sessions:readTail', filePath, fromOffset, maxBytes),
    readHead: (filePath: string, maxBytes?: number) => ipcRenderer.invoke('sessions:readHead', filePath, maxBytes),
    readAccount: () => ipcRenderer.invoke('sessions:readAccount')
  },
  browser: {
    screenshot: (webContentsId: number, rect?: { x: number; y: number; width: number; height: number; vpW: number; vpH: number }) =>
      ipcRenderer.invoke('browser:screenshot', webContentsId, rect)
  },
  clipboard: {
    writeImage: (dataUrl: string) => ipcRenderer.invoke('clipboard:writeImage', dataUrl),
    writeText: (text: string) => ipcRenderer.invoke('clipboard:writeText', text)
  },
  credentials: {
    store: (key: string, value: string) => ipcRenderer.invoke('credentials:store', key, value),
    read: (key: string) => ipcRenderer.invoke('credentials:read', key),
    delete: (key: string) => ipcRenderer.invoke('credentials:delete', key),
  },
  agentBridge: {
    prepareLaunch: (request) => ipcRenderer.invoke('agent-bridge:prepare-launch', request),
  },
  campaignAgentHost: {
    launch: (request) => ipcRenderer.invoke('campaign-agent-host:launch', request),
    eligible: (request) => ipcRenderer.invoke('campaign-agent-host:eligible', request),
    attach: (request) => ipcRenderer.invoke('campaign-agent-host:attach', request),
    authorize: (request) => ipcRenderer.invoke('campaign-agent-host:authorize', request),
    activate: (request) => ipcRenderer.invoke('campaign-agent-host:activate', request),
    revoke: (request) => ipcRenderer.invoke('campaign-agent-host:revoke', request),
  },
  campaignRequest: (method, route, body, query, authority) => (
    ipcRenderer.invoke('campaign:request', method, route, body, query, authority)
  ),
  window: {
    minimize: () => ipcRenderer.invoke('window:minimize'),
    maximize: () => ipcRenderer.invoke('window:maximize'),
    close: () => ipcRenderer.invoke('window:close'),
    isMaximized: () => ipcRenderer.invoke('window:isMaximized'),
    onAppKeydown: (callback: (data: { key: string; ctrlKey: boolean; shiftKey: boolean }) => void) => {
      const listener = (_: any, data: { key: string; ctrlKey: boolean; shiftKey: boolean }) => callback(data)
      ipcRenderer.on('app-keydown', listener)
      return () => ipcRenderer.removeListener('app-keydown', listener)
    }
  }
} satisfies BashGymAPI)

// Declare the global type
declare global {
  interface Window {
    bashgym: BashGymAPI
  }
}
