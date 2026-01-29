import { contextBridge, ipcRenderer } from 'electron'

// Type definitions for the exposed API
export interface TerminalAPI {
  create: (id: string, cwd?: string) => Promise<{ success: boolean; id?: string; error?: string }>
  write: (id: string, data: string) => Promise<boolean>
  resize: (id: string, cols: number, rows: number) => Promise<boolean>
  kill: (id: string) => Promise<boolean>
  onData: (id: string, callback: (data: string) => void) => () => void
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

export interface ApiProxy {
  fetch: (url: string, options?: RequestInit) => Promise<{
    ok: boolean
    status?: number
    data?: any
    error?: string
  }>
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
}

export interface BashGymAPI {
  terminal: TerminalAPI
  theme: ThemeAPI
  system: SystemAPI
  api: ApiProxy
  files: FilesAPI
}

// Expose protected methods to renderer
contextBridge.exposeInMainWorld('bashgym', {
  terminal: {
    create: (id: string, cwd?: string) => ipcRenderer.invoke('terminal:create', id, cwd),
    write: (id: string, data: string) => ipcRenderer.invoke('terminal:write', id, data),
    resize: (id: string, cols: number, rows: number) => ipcRenderer.invoke('terminal:resize', id, cols, rows),
    kill: (id: string) => ipcRenderer.invoke('terminal:kill', id),
    onData: (id: string, callback: (data: string) => void) => {
      const channel = `terminal:data:${id}`
      const listener = (_: any, data: string) => callback(data)
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
  api: {
    fetch: (url: string, options?: RequestInit) => ipcRenderer.invoke('api:fetch', url, options)
  },
  files: {
    readDirectory: (path: string) => ipcRenderer.invoke('files:readDirectory', path),
    getHomeDirectory: () => ipcRenderer.invoke('files:getHomeDirectory'),
    getParentDirectory: (path: string) => ipcRenderer.invoke('files:getParentDirectory', path),
    readFile: (path: string) => ipcRenderer.invoke('files:readFile', path),
    exists: (path: string) => ipcRenderer.invoke('files:exists', path),
    stat: (path: string) => ipcRenderer.invoke('files:stat', path)
  }
} satisfies BashGymAPI)

// Declare the global type
declare global {
  interface Window {
    bashgym: BashGymAPI
  }
}
