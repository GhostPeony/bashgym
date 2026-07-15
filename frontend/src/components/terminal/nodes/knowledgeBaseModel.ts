export type KnowledgeProvider = 'workspace' | 'gbrain'

export interface KnowledgeTreeNode {
  name: string
  path: string
  type: 'folder' | 'file'
  knowledge?: boolean
  size_bytes?: number
  children?: KnowledgeTreeNode[]
}

export interface KnowledgeStatus {
  workspace_root: string
  gbrain: {
    installed: boolean
    executable?: string | null
    configured: boolean
    sources: Array<Record<string, unknown>>
  }
  adapters: Array<{
    id: KnowledgeProvider
    label: string
    available: boolean
    mode: string
  }>
}

export interface KnowledgeSnapshot {
  provider: KnowledgeProvider
  root: string
  label: string
  tree: KnowledgeTreeNode[]
  counts: { files: number; folders: number; knowledge_files: number }
  truncated: boolean
  capabilities: string[]
}

export interface KnowledgeSearchResponse {
  provider: KnowledgeProvider
  root: string
  query: string
  results: Array<{ path: string; snippet: string }>
  scanned_files: number
  truncated: boolean
}

export interface KnowledgePreview {
  path: string
  content: string
  truncated: boolean
}

export interface KnowledgeNodeConfig {
  provider: KnowledgeProvider
  root: string
  label: string
  selectedPath?: string
  selectedSection?: 'browse' | 'search' | 'config'
}

export const DEFAULT_KNOWLEDGE_CONFIG: KnowledgeNodeConfig = {
  provider: 'workspace',
  root: '',
  label: 'Knowledge Base',
  selectedSection: 'browse'
}

export function sanitizeKnowledgeConfig(value?: Record<string, unknown>): KnowledgeNodeConfig {
  const provider = value?.provider === 'gbrain' ? 'gbrain' : 'workspace'
  const section = value?.selectedSection
  return {
    provider,
    root: typeof value?.root === 'string' ? value.root.slice(0, 4096) : '',
    label: typeof value?.label === 'string' && value.label.trim()
      ? value.label.trim().slice(0, 160)
      : provider === 'gbrain' ? 'GBrain' : 'Knowledge Base',
    selectedPath: typeof value?.selectedPath === 'string' ? value.selectedPath.slice(0, 4096) : undefined,
    selectedSection: section === 'search' || section === 'config' ? section : 'browse'
  }
}

export function countVisibleKnowledgeNodes(nodes: KnowledgeTreeNode[]): number {
  return nodes.reduce((count, node) => count + 1 + countVisibleKnowledgeNodes(node.children ?? []), 0)
}

export function buildKnowledgeContext(
  config: KnowledgeNodeConfig,
  snapshot: KnowledgeSnapshot | null,
  search?: KnowledgeSearchResponse | null
): string {
  const lines = [
    `## ${config.label || 'Knowledge Base'}`,
    '',
    `Provider: ${config.provider === 'gbrain' ? 'GBrain' : 'Workspace files'}`,
    `Source: ${snapshot?.root || config.root || 'not configured'}`,
    `Knowledge files: ${snapshot?.counts.knowledge_files ?? 0}`,
    `Folders: ${snapshot?.counts.folders ?? 0}`,
    '',
    'Use this source as read-only context. Preserve file paths as citations.'
  ]
  if (search?.results.length) {
    lines.push('', `### Search: ${search.query}`)
    for (const result of search.results.slice(0, 8)) {
      lines.push(`- ${result.path}: ${result.snippet}`)
    }
  }
  return lines.join('\n')
}
