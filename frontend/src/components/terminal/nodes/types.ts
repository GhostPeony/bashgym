import type { LucideIcon } from 'lucide-react'

export interface ContextPayload {
  summary: string
  content: string
  tokenEstimate: number
}

export interface NodeAction {
  id: string
  label: string
  icon: LucideIcon
  handler: () => Promise<void> | void
}

export interface ConfigField {
  key: string
  label: string
  type: 'text' | 'password' | 'select' | 'textarea' | 'toggle'
  placeholder?: string
  options?: { label: string; value: string }[]
  value?: unknown
}

export interface NodeAdapter {
  type: string
  icon: LucideIcon
  label: string
  getContext(): ContextPayload
  getActions(): NodeAction[]
  getConfigFields(): ConfigField[]
  onReceive?(data: unknown): void
}

export interface IntegrationNodeData {
  panelId: string
  title: string
  adapterType: 'context' | 'neon' | 'vercel'
  adapterConfig: Record<string, unknown>
  hasConnections?: boolean
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
}
