import { memo, useCallback, useEffect, useMemo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  BookOpenText,
  ChevronDown,
  ChevronRight,
  FileText,
  Folder,
  FolderOpen,
  Loader2,
  RefreshCw,
  Search,
  Settings2
} from 'lucide-react'
import { clsx } from 'clsx'
import { knowledgeApi } from '../../../services/api'
import { getActiveWorkspaceId } from '../../../stores/workspacePersistence'
import { useTerminalStore } from '../../../stores/terminalStore'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import {
  buildKnowledgeContext,
  sanitizeKnowledgeConfig,
  type KnowledgeNodeConfig,
  type KnowledgePreview,
  type KnowledgeSearchResponse,
  type KnowledgeSnapshot,
  type KnowledgeStatus,
  type KnowledgeTreeNode
} from './knowledgeBaseModel'
import type { DataNodeData } from './types'

export type KnowledgeBaseNodeType = Node<DataNodeData, 'knowledge'>
type KnowledgeSection = 'browse' | 'search' | 'config'

function TreeRow({
  node,
  depth,
  expanded,
  selectedPath,
  onToggle,
  onSelect
}: {
  node: KnowledgeTreeNode
  depth: number
  expanded: Set<string>
  selectedPath?: string
  onToggle: (path: string) => void
  onSelect: (node: KnowledgeTreeNode) => void
}) {
  const folder = node.type === 'folder'
  const open = expanded.has(node.path)
  const Icon = folder ? (open ? FolderOpen : Folder) : FileText
  return (
    <>
      <button
        type="button"
        className={clsx('knowledge-tree-row', selectedPath === node.path && 'knowledge-tree-row-selected')}
        style={{ paddingLeft: `${8 + depth * 16}px` }}
        onClick={() => folder ? onToggle(node.path) : onSelect(node)}
        title={node.path}
      >
        {folder
          ? open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />
          : <span className="w-3" />}
        <Icon className={clsx('h-3.5 w-3.5 flex-shrink-0', node.knowledge && 'text-accent')} />
        <span className="truncate">{node.name}</span>
      </button>
      {folder && open ? node.children?.map((child) => (
        <TreeRow
          key={child.path}
          node={child}
          depth={depth + 1}
          expanded={expanded}
          selectedPath={selectedPath}
          onToggle={onToggle}
          onSelect={onSelect}
        />
      )) : null}
    </>
  )
}

export const KnowledgeBaseNode = memo(function KnowledgeBaseNode({ data, selected }: NodeProps<KnowledgeBaseNodeType>) {
  const initialConfig = useMemo(() => sanitizeKnowledgeConfig(data.adapterConfig), [data.adapterConfig])
  const [config, setConfig] = useState(initialConfig)
  const [draft, setDraft] = useState(initialConfig)
  const [status, setStatus] = useState<KnowledgeStatus | null>(null)
  const [snapshot, setSnapshot] = useState<KnowledgeSnapshot | null>(null)
  const [preview, setPreview] = useState<KnowledgePreview | null>(null)
  const [searchResult, setSearchResult] = useState<KnowledgeSearchResponse | null>(null)
  const [query, setQuery] = useState('')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [labOpen, setLabOpen] = useState(false)
  const [section, setSection] = useState<KnowledgeSection>(initialConfig.selectedSection ?? 'browse')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const workspaceId = getActiveWorkspaceId()
  const updatePanelConfig = useTerminalStore((state) => state.updatePanelConfig)
  const renamePanel = useTerminalStore((state) => state.renamePanel)

  const discoveredGbrainSources = useMemo(() => (status?.gbrain.sources ?? []).flatMap((source) => {
    const path = source.path ?? source.root
    if (typeof path !== 'string') return []
    const label = typeof source.name === 'string' ? source.name : typeof source.id === 'string' ? source.id : path
    return [{ path, label }]
  }), [status])

  const load = useCallback(async (target: KnowledgeNodeConfig = config) => {
    setLoading(true)
    setError(null)
    try {
      const response = await knowledgeApi.inspect({
        workspace_id: workspaceId,
        provider: target.provider,
        root: target.root || undefined,
        max_depth: 5,
        max_entries: 220
      })
      if (!response.ok || !response.data) throw new Error(response.error || 'Knowledge source unavailable')
      setSnapshot(response.data)
      setExpanded(new Set(response.data.tree.filter((node) => node.type === 'folder').slice(0, 3).map((node) => node.path)))
      if (!target.root) {
        const resolved = { ...target, root: response.data.root }
        setConfig(resolved)
        setDraft(resolved)
        updatePanelConfig(data.panelId, resolved as unknown as Record<string, unknown>)
      }
    } catch (caught) {
      setSnapshot(null)
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLoading(false)
    }
  }, [config, data.panelId, updatePanelConfig, workspaceId])

  useEffect(() => {
    let active = true
    const discover = async () => {
      const response = await knowledgeApi.status(workspaceId)
      if (!active) return
      if (response.ok && response.data) setStatus(response.data)
      await load(initialConfig)
    }
    void discover()
    return () => { active = false }
    // Discovery intentionally runs once for this workspace.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workspaceId])

  const saveConfig = async () => {
    const safe = sanitizeKnowledgeConfig({ ...draft, selectedSection: section })
    setConfig(safe)
    setDraft(safe)
    updatePanelConfig(data.panelId, safe as unknown as Record<string, unknown>)
    renamePanel(data.panelId, safe.label)
    await load(safe)
  }

  const selectFile = async (node: KnowledgeTreeNode) => {
    if (!node.knowledge) {
      setError('Preview is available for safe text and source files only.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const response = await knowledgeApi.preview({
        workspace_id: workspaceId,
        provider: config.provider,
        root: snapshot?.root || config.root || undefined,
        path: node.path
      })
      if (!response.ok || !response.data) throw new Error(response.error || 'Preview unavailable')
      setPreview(response.data)
      const next = { ...config, selectedPath: node.path }
      setConfig(next)
      updatePanelConfig(data.panelId, next as unknown as Record<string, unknown>)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLoading(false)
    }
  }

  const runSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const response = await knowledgeApi.search({
        workspace_id: workspaceId,
        provider: config.provider,
        root: snapshot?.root || config.root || undefined,
        query: query.trim(),
        limit: 20
      })
      if (!response.ok || !response.data) throw new Error(response.error || 'Search unavailable')
      setSearchResult(response.data)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLoading(false)
    }
  }

  const changeSection = (next: KnowledgeSection) => {
    setSection(next)
    const safe = { ...config, selectedSection: next }
    setConfig(safe)
    updatePanelConfig(data.panelId, safe as unknown as Record<string, unknown>)
  }

  const ready = Boolean(snapshot && !error)
  const title = config.label || data.title || 'Knowledge Base'

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={title}
        flowerVariant="knowledge"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? () => buildKnowledgeContext(config, snapshot, searchResult) : undefined}
        statusBarClass={error ? 'bg-status-error' : ready ? 'bg-status-success' : 'bg-status-warning'}
        hue={hueFor('knowledge')}
        headerRight={(
          <button type="button" className="node-btn node-btn-accent" onClick={(event) => { event.stopPropagation(); void load() }} disabled={loading} title="Refresh knowledge source">
            {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
          </button>
        )}
        onFocus={data.onFocus}
        onClose={data.onClose}
      >
        <div className="grid grid-cols-3 gap-1.5">
          <div className="node-section !p-2"><div className="node-field-label">Source</div><div className="mt-1 truncate text-[10px] font-mono text-text-primary">{config.provider === 'gbrain' ? 'GBrain' : 'Workspace'}</div></div>
          <div className="node-section !p-2"><div className="node-field-label">Docs shown</div><div className="mt-1 text-sm font-mono font-bold text-text-primary">{snapshot?.counts.knowledge_files ?? 0}{snapshot?.truncated ? '+' : ''}</div></div>
          <div className="node-section !p-2"><div className="node-field-label">Folders</div><div className="mt-1 text-sm font-mono font-bold text-text-primary">{snapshot?.counts.folders ?? 0}{snapshot?.truncated ? '+' : ''}</div></div>
        </div>
        <div className="mt-2 flex items-center gap-2 rounded-brutal border-brutal border-border-subtle bg-background-card p-2">
          <BookOpenText className="h-4 w-4 flex-shrink-0 text-accent" />
          <div className="min-w-0 flex-1">
            <div className="truncate text-[10px] font-mono font-bold text-text-primary">{snapshot?.label || 'No source loaded'}</div>
            <div className="mt-0.5 truncate text-[9px] font-mono text-text-muted" title={snapshot?.root || config.root}>{snapshot?.root || config.root || 'Choose a source'}</div>
          </div>
          <ConfigPill tone={ready ? 'success' : error ? 'error' : 'warning'}>{ready ? 'ready' : error ? 'error' : 'loading'}</ConfigPill>
        </div>
        {error ? <div className="mt-2 line-clamp-2 text-[9px] font-mono text-status-error">{error}</div> : null}
        <button type="button" className="node-btn node-btn-wide node-btn-accent mt-2 w-full justify-center" onClick={() => setLabOpen(true)}>
          <BookOpenText className="h-3 w-3" /> Open knowledge
        </button>
      </DataNodeShell>

      <NodeConfigModal isOpen={labOpen} onClose={() => setLabOpen(false)} title={`${title} Lab`} description="Browse, search, and publish cited context to connected canvas nodes" size="xl" layout="workspace">
        <div className="knowledge-lab-layout">
          <nav className="mcp-lab-nav" aria-label="Knowledge Base sections">
            {(['browse', 'search', 'config'] as KnowledgeSection[]).map((item) => (
              <button key={item} type="button" className={clsx('mcp-lab-nav-button', section === item ? 'border-accent bg-accent/10 text-accent-dark' : 'border-transparent text-text-secondary hover:border-border-subtle')} onClick={() => changeSection(item)}>{item}</button>
            ))}
          </nav>

          <div className="knowledge-lab-content">
            {section === 'browse' ? (
              <div className="knowledge-browser">
                <ConfigSection title={snapshot?.truncated ? 'Source tree · representative view' : 'Source tree'} className="knowledge-tree-panel">
                  <div className="knowledge-tree-scroll">
                    {snapshot?.tree.map((node) => <TreeRow key={node.path} node={node} depth={0} expanded={expanded} selectedPath={config.selectedPath} onToggle={(path) => setExpanded((current) => { const next = new Set(current); if (next.has(path)) next.delete(path); else next.add(path); return next })} onSelect={(node) => void selectFile(node)} />)}
                  </div>
                </ConfigSection>
                <ConfigSection title={preview?.path || 'Preview'} className="knowledge-preview-panel">
                  {preview ? <pre className="knowledge-preview-content">{preview.content}</pre> : <div className="knowledge-empty-state">Select a text or source file to inspect it with its provenance path.</div>}
                </ConfigSection>
              </div>
            ) : null}

            {section === 'search' ? (
              <div className="space-y-3">
                <ConfigSection title="Search this source">
                  <div className="flex gap-2">
                    <label className="relative min-w-0 flex-1">
                      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" />
                      <input className="input-brutal min-h-9 !pl-8 text-[11px] font-mono" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void runSearch() }} placeholder="What should connected agents know?" />
                    </label>
                    <button type="button" className="node-btn node-btn-wide node-btn-accent" onClick={() => void runSearch()} disabled={loading || !query.trim()}>{loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Search className="h-3 w-3" />} Search</button>
                  </div>
                </ConfigSection>
                <ConfigSection title={searchResult ? `${searchResult.results.length} cited results` : 'Results'}>
                  {searchResult?.results.length ? searchResult.results.map((result) => (
                    <button key={result.path} type="button" className="knowledge-search-result" onClick={() => void selectFile({ name: result.path.split('/').pop() || result.path, path: result.path, type: 'file', knowledge: true })}>
                      <span className="font-mono text-[10px] font-bold text-accent-dark">{result.path}</span>
                      <span className="mt-1 line-clamp-3 text-[11px] leading-5 text-text-secondary">{result.snippet}</span>
                    </button>
                  )) : <div className="py-10 text-center text-[10px] text-text-muted">Search results keep source paths so downstream agents can cite them.</div>}
                </ConfigSection>
              </div>
            ) : null}

            {section === 'config' ? (
              <div className="space-y-3">
                <ConfigSection title="Knowledge source">
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <label className="node-field"><span className="node-field-label">Node name</span><input className="input-brutal min-h-9 text-[11px] font-mono" value={draft.label} onChange={(event) => setDraft((current) => ({ ...current, label: event.target.value }))} /></label>
                    <label className="node-field"><span className="node-field-label">Provider</span><select className="input-brutal min-h-9 text-[11px] font-mono" value={draft.provider} onChange={(event) => setDraft((current) => ({ ...current, provider: event.target.value === 'gbrain' ? 'gbrain' : 'workspace', root: '' }))}><option value="workspace">Workspace files</option><option value="gbrain">GBrain source</option></select></label>
                    <label className="node-field md:col-span-2"><span className="node-field-label">Source directory</span><input className="input-brutal min-h-9 text-[11px] font-mono" value={draft.root} onChange={(event) => setDraft((current) => ({ ...current, root: event.target.value }))} placeholder={status?.workspace_root || 'Defaults to this workspace'} /></label>
                  </div>
                  {draft.provider === 'gbrain' && !status?.gbrain.installed ? <div className="flex gap-2 rounded-brutal border-brutal border-status-warning/60 bg-status-warning/10 p-2 text-[10px] leading-4 text-text-secondary"><AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-status-warning" /><span>GBrain is not installed on this machine. Workspace browsing remains available now.</span></div> : null}
                  {discoveredGbrainSources.length ? <div className="space-y-1"><div className="node-field-label">Discovered GBrain sources</div>{discoveredGbrainSources.map((source) => <button key={source.path} type="button" className="knowledge-source-option" onClick={() => setDraft((current) => ({ ...current, provider: 'gbrain', root: source.path, label: source.label }))}><span>{source.label}</span><span className="truncate text-text-muted">{source.path}</span></button>)}</div> : null}
                  <div className="node-config-action-row"><button type="button" className="node-btn node-btn-wide node-btn-accent" onClick={() => void saveConfig()}><Settings2 className="h-3 w-3" /> Save and inspect</button><button type="button" className="node-btn node-btn-wide" onClick={() => void load()} disabled={loading}><RefreshCw className={clsx('h-3 w-3', loading && 'animate-spin')} /> Refresh</button></div>
                </ConfigSection>
                <ConfigSection title="Context contract">
                  <p className="text-[11px] leading-5 text-text-secondary">Connected apps and nodes receive a read-only context packet with provider, source root, counts, and cited search results. Hidden environment files are excluded.</p>
                  <ConfigRows>
                    <ConfigRow label="Flow" value="Sources → files → concepts/relations → cited context" />
                    <ConfigRow label="Available now" value="Browse, safe preview, keyword search, canvas context" />
                    <ConfigRow label="Adapter targets" value="GBrain, Graphiti, Cognee, LightRAG" />
                    <ConfigRow label="Canvas links" value={data.hasConnections ? 'Published to connected nodes' : 'Connect this node to publish context'} />
                  </ConfigRows>
                </ConfigSection>
              </div>
            ) : null}
            {error ? <div className="knowledge-error">{error}</div> : null}
          </div>
        </div>
      </NodeConfigModal>
    </>
  )
})
