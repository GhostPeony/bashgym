import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  BrainCircuit,
  Database,
  ExternalLink,
  FlaskConical,
  Loader2,
  Lock,
  Package,
  Pin,
  RefreshCw,
  Search,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { clsx } from 'clsx'
import {
  hfApi,
  workspaceApi,
  type HFContextBundle,
  type HFContextEvalPreview,
  type HFInventoryDataset,
  type HFInventoryModel,
  type HFInventoryResponse,
} from '../../../services/api'
import { useHFContextStore } from '../../../stores/hfContextStore'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type HFInventoryNodeType = Node<DataNodeData, 'huggingface'>

const POLL_MS = 30_000
const INVENTORY_LIMIT = 24

function formatDate(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function formatNumber(value?: number): string {
  if (value == null) return '0'
  return Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

function formatBytes(value?: number | null): string {
  if (value == null) return '-'
  if (value < 1024) return `${value} B`
  const units = ['KB', 'MB', 'GB', 'TB']
  let amount = value / 1024
  let unit = units[0]
  for (let index = 1; index < units.length && amount >= 1024; index += 1) {
    amount /= 1024
    unit = units[index]
  }
  return `${amount >= 10 ? amount.toFixed(0) : amount.toFixed(1)} ${unit}`
}

function CountTile({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className="min-w-0 border-brutal border-border-subtle rounded-brutal bg-background-card px-2 py-1.5">
      <div className="flex items-center gap-1.5 font-mono text-[9px] uppercase text-text-muted">
        <Icon className="h-3 w-3 flex-shrink-0" />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-0.5 truncate font-mono text-sm font-semibold text-text-primary">{value}</div>
    </div>
  )
}

function ResourceRow({
  id,
  meta,
  isPrivate,
  tone = 'default',
  url,
}: {
  id: string
  meta?: string
  isPrivate?: boolean | null
  tone?: 'default' | 'local'
  url?: string
}) {
  const content = (
    <>
      <span
        className={clsx(
          'h-1.5 w-1.5 flex-shrink-0 rounded-full',
          tone === 'local' ? 'bg-status-success' : 'bg-accent',
        )}
      />
      {isPrivate ? <Lock className="h-2.5 w-2.5 flex-shrink-0 text-text-muted" /> : null}
      <span className="min-w-0 flex-1 truncate text-text-secondary">{id}</span>
      {meta ? <span className="flex-shrink-0 text-text-muted">{meta}</span> : null}
    </>
  )

  if (url) {
    return (
      <button
        type="button"
        className="nodrag flex w-full min-w-0 items-center gap-1.5 py-0.5 text-left font-mono text-[10px] hover:text-text-primary"
        onClick={(event) => {
          event.stopPropagation()
          window.open(url, '_blank')
        }}
        title={`${id} - open on Hugging Face`}
      >
        {content}
      </button>
    )
  }

  return <div className="flex min-w-0 items-center gap-1.5 py-0.5 font-mono text-[10px]">{content}</div>
}

function DatasetListRow({ dataset }: { dataset: HFInventoryDataset }) {
  return (
    <button
      type="button"
      className="flex w-full items-center gap-3 border-b border-border-subtle px-1 py-3 text-left last:border-b-0 hover:bg-background-tertiary/60"
      onClick={() => window.open(dataset.url, '_blank')}
    >
      <Database className="h-4 w-4 flex-shrink-0 text-accent" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          {dataset.private ? <Lock className="h-3 w-3 flex-shrink-0 text-text-muted" /> : null}
          <span className="truncate font-mono text-xs font-semibold text-text-primary">{dataset.id}</span>
        </div>
        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] text-text-muted">
          <span>{dataset.private ? 'Private' : 'Public'}</span>
          {dataset.used_storage != null ? <span>{formatBytes(dataset.used_storage)}</span> : null}
          {dataset.last_modified ? <span>Updated {formatDate(dataset.last_modified)}</span> : null}
          {dataset.downloads ? <span>{formatNumber(dataset.downloads)} downloads</span> : null}
        </div>
      </div>
      <ExternalLink className="h-4 w-4 flex-shrink-0 text-text-muted" />
    </button>
  )
}

function modelMeta(model: HFInventoryModel): string {
  const parts: string[] = []
  if (model.pipeline_tag) parts.push(model.pipeline_tag)
  if (model.downloads) parts.push(`${formatNumber(model.downloads)} dl`)
  return parts.join(' | ')
}

function datasetMeta(dataset: HFInventoryDataset): string {
  if (dataset.used_storage != null) return formatBytes(dataset.used_storage)
  const date = formatDate(dataset.last_modified)
  if (date) return date
  return dataset.downloads ? `${formatNumber(dataset.downloads)} dl` : ''
}

function buildInventoryContext(inventory: HFInventoryResponse | null, error: string | null): string {
  if (error) return `## Hugging Face storage\n\nInventory load failed: ${error}`
  if (!inventory) return '## Hugging Face storage\n\nNo inventory snapshot loaded yet.'
  if (!inventory.status.enabled) {
    return '## Hugging Face storage\n\nHugging Face is not connected. Add an API key in Settings.'
  }

  return [
    '## Hugging Face storage',
    '',
    `Namespace: ${inventory.namespace || '(default)'}`,
    `Datasets: ${inventory.counts.datasets}`,
    `Models: ${inventory.counts.models}`,
    '',
    '### Datasets',
    ...(inventory.datasets.length
      ? inventory.datasets.map((dataset) => `- ${dataset.id}${dataset.private ? ' (private)' : ''}`)
      : ['- none']),
  ].join('\n')
}

export const HFInventoryNode = memo(function HFInventoryNode({ data, selected }: NodeProps<HFInventoryNodeType>) {
  const [inventory, setInventory] = useState<HFInventoryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [storageOpen, setStorageOpen] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const [contextIntent, setContextIntent] = useState('')
  const [contextTask, setContextTask] = useState('')
  const [contextBusy, setContextBusy] = useState(false)
  const [contextSelection, setContextSelection] = useState<string[]>([])
  const [contextBundle, setContextBundle] = useState<HFContextBundle | null>(null)
  const [evalPreview, setEvalPreview] = useState<HFContextEvalPreview | null>(null)
  const [contextSendError, setContextSendError] = useState<string | null>(null)
  const workspaceId = data.workspaceId || 'default'
  const contextWorkspace = useHFContextStore((state) => state.workspaces[workspaceId])
  const loadContext = useHFContextStore((state) => state.load)
  const discoverContext = useHFContextStore((state) => state.discover)
  const pinContext = useHFContextStore((state) => state.pin)
  const refreshContext = useHFContextStore((state) => state.refresh)
  const cancelContext = useHFContextStore((state) => state.cancel)
  const activateContext = useHFContextStore((state) => state.activate)
  const deactivateContext = useHFContextStore((state) => state.deactivate)
  const contextProjection = useHFContextStore((state) => state.projection)
  const prepareContextEval = useHFContextStore((state) => state.prepareEval)
  const mountedRef = useRef(true)
  const refreshRequestRef = useRef(data.adapterConfig?.refreshRequestedAt)

  const loadInventory = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true)
    try {
      const res = await hfApi.inventory({ prefix: '', limit: INVENTORY_LIMIT, refresh })
      if (!mountedRef.current) return
      setLoaded(true)
      if (res.ok && res.data) {
        setInventory(res.data)
        setError(null)
      } else {
        setError(res.error || 'API unreachable')
      }
    } catch (caught) {
      if (mountedRef.current) setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      if (mountedRef.current) setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    void loadInventory(false)
    const timer = setInterval(() => void loadInventory(false), POLL_MS)
    return () => {
      mountedRef.current = false
      clearInterval(timer)
    }
  }, [loadInventory])

  useEffect(() => {
    const requestedAt = data.adapterConfig?.refreshRequestedAt
    if (!requestedAt || requestedAt === refreshRequestRef.current) return
    refreshRequestRef.current = requestedAt
    void loadInventory(true)
  }, [data.adapterConfig?.refreshRequestedAt, loadInventory])

  const openStorage = () => {
    setStorageOpen(true)
  }

  const closeStorage = () => {
    setStorageOpen(false)
  }

  const activeBundle = useMemo(() => {
    const active = contextWorkspace?.active
    if (!active) return null
    return contextWorkspace.bundles.find(
      (bundle) => bundle.bundle_id === active.bundle_id && bundle.version === active.version,
    ) || null
  }, [contextWorkspace])
  const visibleContextBundle = contextWorkspace?.collecting || contextBundle

  const openContext = () => {
    setContextOpen(true)
    void loadContext(workspaceId)
  }

  const runContextDiscovery = async () => {
    const intent = contextIntent.trim()
    if (!intent || contextBusy) return
    setContextBusy(true)
    setEvalPreview(null)
    try {
      const bundle = await discoverContext(
        workspaceId,
        intent,
        contextTask.trim() || undefined,
        { panel_id: data.panelId },
      )
      if (bundle) {
        setContextBundle(bundle)
        setContextSelection(bundle.evidence.map((item) => item.evidence_id))
      }
    } finally {
      setContextBusy(false)
    }
  }

  const pinCurrentContext = async () => {
    if (!contextBundle || contextBusy) return
    setContextBusy(true)
    try {
      const pinned = await pinContext(workspaceId, contextBundle, contextSelection)
      if (pinned) setContextBundle(pinned)
    } finally {
      setContextBusy(false)
    }
  }

  const refreshCurrentContext = async () => {
    if (!contextBundle || contextBundle.lifecycle !== 'ready' || contextBusy) return
    setContextBusy(true)
    setEvalPreview(null)
    try {
      const refreshed = await refreshContext(workspaceId, contextBundle)
      if (refreshed) {
        setContextBundle(refreshed)
        setContextSelection(refreshed.selected_evidence_ids.length
          ? refreshed.selected_evidence_ids
          : refreshed.evidence.map((item) => item.evidence_id))
      }
    } finally {
      setContextBusy(false)
    }
  }

  const buildNodeContext = useCallback(async () => {
    if (activeBundle) {
      return contextProjection(workspaceId, activeBundle)
    }
    return buildInventoryContext(inventory, error)
  }, [activeBundle, contextProjection, error, inventory, workspaceId])

  const topDatasets = useMemo(() => inventory?.datasets.slice(0, 3) ?? [], [inventory])
  const topModels = useMemo(
    () => inventory?.models.slice(0, Math.max(0, 3 - topDatasets.length)) ?? [],
    [inventory, topDatasets.length],
  )
  const privateDatasetCount = inventory?.datasets.filter((dataset) => dataset.private).length ?? 0
  const primaryWarnings = inventory?.warnings.filter(
    (warning) => warning.section === 'datasets' || warning.section === 'models',
  ) ?? []
  const disabled = loaded && inventory?.status.enabled === false
  const statusBarClass = error
    ? 'bg-status-error'
    : disabled
      ? 'bg-status-warning'
      : primaryWarnings.length
        ? 'bg-status-warning'
        : inventory?.status.enabled
          ? 'bg-status-success'
          : 'bg-background-tertiary'

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={data.title}
        flowerVariant="huggingface"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? buildNodeContext : undefined}
        contextBasename={activeBundle ? `hf_context_${activeBundle.bundle_id}_v${activeBundle.version}` : undefined}
        onContextRouted={activeBundle ? async (result) => {
          if (result.routed <= 0) return
          setContextSendError(null)
          await workspaceApi.emitEvent({
            type: 'hf-context:sent',
            workspace_id: workspaceId,
            source: { kind: 'canvas', panel_id: data.panelId },
            title: 'Hugging Face context sent',
            summary: `Sent ${activeBundle.bundle_id} v${activeBundle.version} to ${result.routed} terminal${result.routed === 1 ? '' : 's'}`,
            entity: {
              bundle_id: activeBundle.bundle_id,
              version: activeBundle.version,
              evidence_count: activeBundle.evidence.length,
            },
          })
        } : undefined}
        onContextRouteError={activeBundle ? (caught) => {
          setContextSendError(caught instanceof Error ? caught.message : 'Unable to send active Hugging Face context')
        } : undefined}
        statusBarClass={statusBarClass}
        hue={hueFor('huggingface')}
        headerRight={(
          <>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                openContext()
              }}
              className="node-btn node-btn-accent"
              title="Build Hugging Face context"
            >
              <BrainCircuit className="h-3 w-3" />
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void loadInventory(true)
              }}
              disabled={refreshing}
              className="node-btn node-btn-accent"
              title="Refresh Hugging Face storage"
            >
              {refreshing ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                openStorage()
              }}
              disabled={!inventory}
              className="nodrag node-btn node-btn-accent"
              title="Browse Hugging Face storage"
            >
              <Database className="h-3 w-3" />
            </button>
          </>
        )}
        onFocus={data.onFocus}
        onClose={data.onClose}
      >
        {!loaded ? (
          <div className="flex justify-center py-3">
            <Loader2 className="h-4 w-4 animate-spin text-text-muted" />
          </div>
        ) : error ? (
          <div className="flex items-start gap-1.5 py-2 font-mono text-[10px] text-status-error">
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
            <span className="min-w-0 break-words">{error}</span>
          </div>
        ) : disabled ? (
          <div className="py-3 text-center font-mono text-[10px] text-text-muted">
            Hugging Face not connected
            <span className="mt-1 block text-text-secondary">Add an API key in Settings</span>
          </div>
        ) : inventory ? (
          <div className="space-y-2">
            {activeBundle ? (
              <div className="border-l-2 border-accent bg-accent/10 px-2 py-1.5">
                <div className="truncate font-mono text-[10px] font-semibold text-text-primary">
                  {activeBundle.intent}
                </div>
                <div className="mt-0.5 font-mono text-[9px] text-text-muted">
                  Context v{activeBundle.version} · {activeBundle.evidence.length} evidence · {activeBundle.freshness}
                </div>
              </div>
            ) : null}
            {contextSendError ? (
              <div className="flex items-start gap-1.5 border-brutal border-status-error/50 bg-status-error/10 px-2 py-1.5 font-mono text-[9px] text-status-error">
                <AlertCircle className="mt-0.5 h-3 w-3 flex-shrink-0" />
                <span>{contextSendError}</span>
              </div>
            ) : null}
            <div className="flex items-center gap-1.5 font-mono text-[10px] text-text-muted">
              <span className="truncate" title={inventory.namespace}>{inventory.namespace || 'default namespace'}</span>
              {inventory.cached ? (
                <span className="rounded-brutal border-brutal border-border-subtle px-1 py-px text-[8px] uppercase">cache</span>
              ) : null}
              <span className="ml-auto flex-shrink-0">{formatDate(inventory.last_refreshed)}</span>
            </div>

            <div className="grid grid-cols-3 gap-1.5">
              <CountTile icon={Database} label="datasets" value={formatNumber(inventory.counts.datasets)} />
              <CountTile icon={Package} label="models" value={formatNumber(inventory.counts.models)} />
              <CountTile icon={Lock} label="private" value={formatNumber(privateDatasetCount)} />
            </div>

            {primaryWarnings.length ? (
              <div className="flex items-start gap-1.5 border-brutal border-status-warning/50 bg-status-warning/10 px-2 py-1.5 font-mono text-[9px] text-status-warning">
                <AlertCircle className="mt-0.5 h-3 w-3 flex-shrink-0" />
                <span className="min-w-0 truncate" title={primaryWarnings[0].message}>
                  {primaryWarnings[0].section}: {primaryWarnings[0].message}
                </span>
              </div>
            ) : null}

            <div className="space-y-1">
              {topDatasets.map((dataset) => (
                <ResourceRow
                  key={dataset.id}
                  id={dataset.id}
                  meta={datasetMeta(dataset)}
                  isPrivate={dataset.private}
                  url={dataset.url}
                />
              ))}
              {topModels.map((model) => (
                <ResourceRow
                  key={model.id}
                  id={model.id}
                  meta={modelMeta(model)}
                  isPrivate={model.private}
                  tone={model.local ? 'local' : 'default'}
                  url={model.url}
                />
              ))}
            </div>

            {topDatasets.length + topModels.length === 0 ? (
              <div className="py-2 text-center font-mono text-[10px] text-text-muted">No datasets or models found</div>
            ) : null}
          </div>
        ) : null}
      </DataNodeShell>

      <NodeConfigModal
        isOpen={storageOpen}
        onClose={closeStorage}
        title="Hugging Face Storage"
        description={`${inventory?.namespace || 'Connected account'} datasets and models`}
        size="lg"
        footer={(
          <div className="flex w-full items-center justify-between gap-3">
            <span className="font-mono text-[10px] text-text-muted">API key is managed in Settings</span>
            <button type="button" className="btn-secondary text-xs" onClick={closeStorage}>Close</button>
          </div>
        )}
      >
        <ConfigSection title={`Datasets (${inventory?.datasets.length || 0})`}>
          {inventory?.datasets.length ? (
            <div>
              {inventory.datasets.map((dataset) => (
                <DatasetListRow key={dataset.id} dataset={dataset} />
              ))}
            </div>
          ) : (
            <p className="py-5 text-center font-mono text-xs text-text-muted">No datasets in this account</p>
          )}
        </ConfigSection>

        {inventory?.models.length ? (
          <ConfigSection title={`Models (${inventory.models.length})`}>
            <div>
              {inventory.models.map((model) => (
                <button
                  key={model.id}
                  type="button"
                  className="flex w-full items-center gap-3 border-b border-border-subtle px-1 py-3 text-left last:border-b-0 hover:bg-background-tertiary/60"
                  onClick={() => window.open(model.url, '_blank')}
                >
                  <Package className="h-4 w-4 flex-shrink-0 text-accent" />
                  <span className="min-w-0 flex-1 truncate font-mono text-xs text-text-primary">{model.id}</span>
                  <span className="font-mono text-[10px] text-text-muted">{modelMeta(model)}</span>
                  <ExternalLink className="h-3.5 w-3.5 flex-shrink-0 text-text-muted" />
                </button>
              ))}
            </div>
          </ConfigSection>
        ) : null}

        {primaryWarnings.length ? (
          <ConfigSection title="Unable to load">
            <ConfigRows>
              {primaryWarnings.map((warning, index) => (
                <ConfigRow key={`${warning.section}-${index}`} label={warning.section} value={warning.message} />
              ))}
            </ConfigRows>
          </ConfigSection>
        ) : null}
      </NodeConfigModal>

      <NodeConfigModal
        isOpen={contextOpen}
        onClose={() => setContextOpen(false)}
        title="Hugging Face Evidence Desk"
        description="Find, pin, and send source-linked model, dataset, and evaluation context"
        size="xl"
        layout="workspace"
        footer={(
          <div className="flex w-full flex-wrap items-center justify-between gap-2">
            <span className="font-mono text-[10px] text-text-muted">
              {contextWorkspace?.active
                ? `Active: ${contextWorkspace.active.bundle_id} v${contextWorkspace.active.version}`
                : 'No active context bundle'}
            </span>
            <div className="flex items-center gap-2">
              {contextWorkspace?.active ? (
                <button type="button" className="btn-ghost text-xs" onClick={() => void deactivateContext(workspaceId)}>
                  Deactivate
                </button>
              ) : null}
              <button type="button" className="btn-secondary text-xs" onClick={() => setContextOpen(false)}>Close</button>
            </div>
          </div>
        )}
      >
        <div className="grid min-h-0 gap-4 lg:grid-cols-[minmax(260px,0.85fr)_minmax(0,1.4fr)]">
          <div className="min-h-0 space-y-4 overflow-y-auto pr-1">
            <ConfigSection title="Ask Hugging Face">
              <div className="space-y-3">
                <label className="block">
                  <span className="mb-1 block font-mono text-[10px] uppercase tracking-wider text-text-muted">What do you need?</span>
                  <textarea
                    value={contextIntent}
                    onChange={(event) => setContextIntent(event.target.value)}
                    className="input min-h-24 w-full resize-y text-sm"
                    placeholder="Find comparable small code models and evaluation datasets for this checkpoint"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block font-mono text-[10px] uppercase tracking-wider text-text-muted">Task (optional)</span>
                  <input
                    value={contextTask}
                    onChange={(event) => setContextTask(event.target.value)}
                    className="input w-full text-sm"
                    placeholder="text-generation, tool use, code generation"
                  />
                </label>
                <button
                  type="button"
                  className="btn-primary btn-compact w-full"
                  disabled={!contextIntent.trim() || contextBusy}
                  onClick={() => void runContextDiscovery()}
                >
                  {contextBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                  Search Hub evidence
                </button>
              </div>
            </ConfigSection>

            {contextWorkspace?.bundles.length ? (
              <ConfigSection title="Recent context">
                <div className="space-y-1">
                  {contextWorkspace.bundles.slice(0, 8).map((bundle) => (
                    <button
                      key={`${bundle.bundle_id}-${bundle.version}`}
                      type="button"
                      className="w-full border-b border-border-subtle px-1 py-2 text-left last:border-b-0 hover:bg-background-tertiary"
                      onClick={() => {
                        setContextBundle(bundle)
                        setContextSelection(bundle.selected_evidence_ids.length ? bundle.selected_evidence_ids : bundle.evidence.map((item) => item.evidence_id))
                        setEvalPreview(null)
                      }}
                    >
                      <div className="truncate font-mono text-xs text-text-primary">{bundle.intent}</div>
                      <div className="font-mono text-[9px] text-text-muted">v{bundle.version} · {bundle.evidence.length} evidence · {bundle.freshness}</div>
                    </button>
                  ))}
                </div>
              </ConfigSection>
            ) : null}
          </div>

          <div className="min-h-0 overflow-y-auto pr-1">
            {visibleContextBundle ? (
              <div className="space-y-4">
                {visibleContextBundle.lifecycle === 'collecting' ? (
                  <div className="flex items-center justify-between gap-3 border-brutal border-accent/50 bg-accent/10 p-3">
                    <div>
                      <div className="flex items-center gap-2 font-mono text-xs font-semibold text-text-primary">
                        <Loader2 className="h-4 w-4 animate-spin text-accent" /> Collecting Hub evidence
                      </div>
                      <p className="mt-1 font-mono text-[10px] text-text-muted">Completed source checkpoints are preserved if you cancel.</p>
                    </div>
                    <button type="button" className="btn-secondary btn-compact" onClick={() => void cancelContext(workspaceId, visibleContextBundle)}>
                      Cancel
                    </button>
                  </div>
                ) : null}
                {visibleContextBundle.freshness === 'stale' ? (
                  <div className="border-brutal border-status-warning/60 bg-status-warning/10 p-3 text-xs text-status-warning">
                    This exact bundle is stale. Refresh it to create a new immutable version, or deactivate it.
                  </div>
                ) : null}
                {contextWorkspace?.error ? (
                  <div className="border-brutal border-status-error/60 bg-status-error/10 p-3 text-xs text-status-error">
                    {contextWorkspace.error}
                  </div>
                ) : null}
                <ConfigSection title={`Evidence (${contextSelection.length}/${visibleContextBundle.evidence.length} selected)`}>
                  <div className="divide-y divide-border-subtle">
                    {visibleContextBundle.evidence.map((item) => {
                      const checked = contextSelection.includes(item.evidence_id)
                      return (
                        <label key={item.evidence_id} className="flex cursor-pointer gap-3 py-3">
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => setContextSelection((current) => checked
                              ? current.filter((id) => id !== item.evidence_id)
                              : [...current, item.evidence_id])}
                            className="mt-1"
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                              <span className="break-all font-mono text-xs font-semibold text-text-primary">{item.resource_id}</span>
                              <span className="font-mono text-[9px] uppercase text-accent">{item.kind}</span>
                              {item.visibility === 'workspace_private' ? (
                                <span className="flex items-center gap-1 font-mono text-[9px] text-status-warning"><Lock className="h-3 w-3" /> Private/gated metadata</span>
                              ) : null}
                              {item.kind === 'evaluation' ? (
                                <span className="font-mono text-[9px] text-status-warning">{item.assessment.comparability.replace('_', ' ')}</span>
                              ) : null}
                            </div>
                            <p className="mt-1 text-xs leading-relaxed text-text-secondary">{item.summary}</p>
                            {item.assessment.rationale ? <p className="mt-1 font-mono text-[9px] text-text-muted">Why: {item.assessment.rationale}</p> : null}
                            {item.cautions.map((caution) => <p key={caution} className="mt-1 text-[10px] text-status-warning">{caution}</p>)}
                            <button type="button" className="mt-1 font-mono text-[10px] text-accent hover:underline" onClick={(event) => { event.preventDefault(); window.open(item.canonical_url, '_blank') }}>
                              Open source
                            </button>
                          </div>
                        </label>
                      )
                    })}
                  </div>
                </ConfigSection>

                <div className="flex flex-wrap gap-2">
                  <button type="button" className="btn-secondary btn-compact" disabled={contextBusy || visibleContextBundle.lifecycle !== 'ready'} onClick={() => void pinCurrentContext()}>
                    <Pin className="h-4 w-4" /> Pin selection
                  </button>
                  <button type="button" className="btn-secondary btn-compact" disabled={contextBusy || visibleContextBundle.lifecycle !== 'ready'} onClick={() => void refreshCurrentContext()}>
                    <RefreshCw className="h-4 w-4" /> Refresh evidence
                  </button>
                  <button type="button" className="btn-primary btn-compact" disabled={contextBusy || visibleContextBundle.lifecycle !== 'ready'} onClick={() => void activateContext(workspaceId, visibleContextBundle)}>
                    <BrainCircuit className="h-4 w-4" /> Activate context
                  </button>
                  <button type="button" className="btn-secondary btn-compact" disabled={contextBusy || visibleContextBundle.lifecycle !== 'ready'} onClick={async () => setEvalPreview(await prepareContextEval(workspaceId, visibleContextBundle))}>
                    <FlaskConical className="h-4 w-4" /> Prepare Eval
                  </button>
                </div>

                {evalPreview ? (
                  <ConfigSection title="Eval preview (does not execute)">
                    <ConfigRows>
                      <ConfigRow label="Model" value={evalPreview.model_id || 'Choose in Eval'} />
                      <ConfigRow label="Tasks" value={evalPreview.tasks.join(', ') || 'No structured tasks found'} />
                      <ConfigRow label="Destination" value={data.linkedEvals?.map((node) => node.title).join(', ') || 'Eval node'} />
                      <ConfigRow label="Needs confirmation" value={evalPreview.unknowns.join(' ')} />
                    </ConfigRows>
                  </ConfigSection>
                ) : null}
              </div>
            ) : (
              <div className="flex min-h-64 items-center justify-center border border-dashed border-border-subtle p-8 text-center">
                <div>
                  <BrainCircuit className="mx-auto h-8 w-8 text-accent" />
                  <p className="mt-3 font-brand text-lg text-text-primary">Ask a concrete model or evaluation question</p>
                  <p className="mt-1 max-w-md text-xs leading-relaxed text-text-muted">BashGym will return source-linked models, datasets, and published evaluation evidence with explicit confidence and comparability warnings.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </NodeConfigModal>
    </>
  )
})
