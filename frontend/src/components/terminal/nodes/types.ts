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

export interface IntegrationNodeData extends Record<string, unknown> {
  panelId: string
  title: string
  adapterType: 'context' | 'neon' | 'vercel'
  adapterConfig: Record<string, unknown>
  hasConnections?: boolean
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
}

export interface DataNodeData extends Record<string, unknown> {
  panelId: string
  title: string
  workspaceId?: string
  adapterConfig?: Record<string, unknown>
  hasConnections?: boolean
  hasTerminalConnections?: boolean
  linkedHuggingFace?: Array<{
    panelId: string
    title: string
    adapterConfig?: Record<string, unknown>
  }>
  linkedKnowledgeBases?: Array<{
    panelId: string
    title: string
    adapterConfig?: Record<string, unknown>
  }>
  linkedEvals?: Array<{
    panelId: string
    title: string
  }>
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
}
