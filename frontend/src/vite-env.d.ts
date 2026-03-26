/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_WS_URL: string
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface ImportMeta {
  readonly env: ImportMetaEnv
}

// Extend Window interface for bashgym API
declare global {
  interface Window {
    bashgym: import('../electron/preload').BashGymAPI
  }
}

export {}
