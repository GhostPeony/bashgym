import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  Cloud,
  Database,
  HardDrive,
  Loader2,
  Lock,
  Package,
  RefreshCw,
  SlidersHorizontal
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { clsx } from 'clsx'
import {
  API_BASE,
  hfApi,
  type HFInventoryBucket,
  type HFInventoryDataset,
  type HFInventoryModel,
  type HFInventoryResponse
} from '../../../services/api'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type HFInventoryNodeType = Node<DataNodeData, 'huggingface'>

const POLL_MS = 30_000
const INVENTORY_LIMIT = 8

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

function CountTile({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: number }) {
  return (
    <div className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5 bg-background-card min-w-0">
      <div className="flex items-center gap-1.5 text-[9px] font-mono text-text-muted uppercase">
        <Icon className="w-3 h-3 flex-shrink-0" />
        <span className="truncate">{label}</span>
      </div>
      <div className="text-sm font-mono font-semibold text-text-primary mt-0.5">
        {formatNumber(value)}
      </div>
    </div>
  )
}

function ResourceRow({
  id,
  meta,
  isPrivate,
  tone = 'default',
  url
}: {
  id: string
  meta?: string
  isPrivate?: boolean | null
  tone?: 'default' | 'local'
  url?: string
}) {
  return (
    <div
      className={clsx(
        'flex items-center gap-1.5 text-[10px] font-mono min-w-0',
        url && 'nodrag cursor-pointer hover:text-text-primary'
      )}
      onClick={url ? (e) => {
        e.stopPropagation()
        window.open(url, '_blank')
      } : undefined}
      title={url ? `${id} — open on Hugging Face` : id}
    >
      <span
        className={clsx(
          'w-1.5 h-1.5 rounded-full flex-shrink-0',
          tone === 'local' ? 'bg-status-success' : 'bg-accent'
        )}
      />
      {isPrivate && <Lock className="w-2.5 h-2.5 text-text-muted flex-shrink-0" />}
      <span className="flex-1 truncate text-text-secondary">{id}</span>
      {meta && <span className="text-text-muted flex-shrink-0">{meta}</span>}
    </div>
  )
}

function modelMeta(model: HFInventoryModel): string {
  const parts: string[] = []
  if (model.pipeline_tag) parts.push(model.pipeline_tag)
  if (model.downloads) parts.push(`${formatNumber(model.downloads)} dl`)
  return parts.join(' | ')
}

function datasetMeta(dataset: HFInventoryDataset): string {
  const parts: string[] = []
  if (dataset.downloads) parts.push(`${formatNumber(dataset.downloads)} dl`)
  const date = formatDate(dataset.last_modified)
  if (date) parts.push(date)
  return parts.join(' | ')
}

function bucketMeta(bucket: HFInventoryBucket): string {
  return formatDate(bucket.updated_at || bucket.created_at)
}

function buildInventoryContext(inventory: HFInventoryResponse | null, error: string | null): string {
  if (error) {
    return ['## Hugging Face inventory', '', `Inventory load failed: ${error}`].join('\n')
  }
  if (!inventory) {
    return '## Hugging Face inventory\n\nNo inventory snapshot loaded yet.'
  }
  if (!inventory.status.enabled) {
    return [
      '## Hugging Face inventory',
      '',
      'Hugging Face is not configured for this BashGym backend.',
      `- status: GET ${API_BASE}/hf/status`,
      `- configure: POST ${API_BASE}/hf/configure`
    ].join('\n')
  }

  const lines: string[] = [
    '## Hugging Face inventory',
    '',
    `Snapshot: ${inventory.last_refreshed}${inventory.cached ? ' (served from BashGym cache)' : ''}`,
    `Namespace: ${inventory.namespace || '(default)'}`,
    `Dataset prefix: ${inventory.prefix || '(none)'}`,
    '',
    '### Counts',
    `- models: ${inventory.counts.models}`,
    `- datasets: ${inventory.counts.datasets}`,
    `- trace datasets: ${inventory.counts.trace_datasets}`,
    `- storage buckets: ${inventory.counts.buckets}`
  ]

  if (inventory.models.length) {
    lines.push('', '### Models')
    for (const m of inventory.models.slice(0, 6)) {
      const local = m.local ? ` local=${m.local.model_id}` : ''
      lines.push(`- ${m.id}${local}${m.private ? ' private' : ''}`)
    }
  }

  if (inventory.datasets.length || inventory.trace_datasets.length) {
    lines.push('', '### Datasets')
    for (const d of inventory.datasets.slice(0, 5)) lines.push(`- ${d.id}`)
    for (const d of inventory.trace_datasets.slice(0, 5)) lines.push(`- ${d.id} (traces)`)
  }

  if (inventory.buckets.length) {
    lines.push('', '### Buckets')
    for (const b of inventory.buckets.slice(0, 5)) lines.push(`- ${b.id}${b.private ? ' private' : ''}`)
  }

  if (inventory.warnings.length) {
    lines.push('', '### Section warnings')
    for (const w of inventory.warnings) lines.push(`- ${w.section}: ${w.message}`)
  }

  lines.push('', '### Live handles')
  lines.push(`- inventory: GET ${API_BASE}/hf/inventory?prefix=${encodeURIComponent(inventory.prefix)}&limit=${inventory.limit}`)
  lines.push(`- models: GET ${API_BASE}/hf/models/mine`)
  lines.push(`- datasets: GET ${API_BASE}/hf/datasets?prefix=${encodeURIComponent(inventory.prefix)}`)
  lines.push(`- buckets: GET ${API_BASE}/hf/buckets${inventory.namespace ? `?namespace=${encodeURIComponent(inventory.namespace)}` : ''}`)

  return lines.join('\n')
}

export const HFInventoryNode = memo(function HFInventoryNode({ data, selected }: NodeProps<HFInventoryNodeType>) {
  const [inventory, setInventory] = useState<HFInventoryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const mountedRef = useRef(true)

  const loadInventory = useCallback(async (refresh = false) => {
    if (refresh) setRefreshing(true)
    try {
      const res = await hfApi.inventory({ limit: INVENTORY_LIMIT, refresh })
      if (!mountedRef.current) return
      setLoaded(true)
      if (res.ok && res.data) {
        setInventory(res.data)
        setError(null)
      } else {
        setError(res.error || 'API unreachable')
      }
    } catch (e) {
      if (mountedRef.current) setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (mountedRef.current) setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    void loadInventory(false)
    const timer = setInterval(() => {
      void loadInventory(false)
    }, POLL_MS)
    return () => {
      mountedRef.current = false
      clearInterval(timer)
    }
  }, [loadInventory])

  const topModels = useMemo(() => inventory?.models.slice(0, 3) ?? [], [inventory])
  const topDatasets = useMemo(() => inventory?.datasets.slice(0, 2) ?? [], [inventory])
  const topTraces = useMemo(() => inventory?.trace_datasets.slice(0, 2) ?? [], [inventory])
  const topBuckets = useMemo(() => inventory?.buckets.slice(0, 2) ?? [], [inventory])

  const disabled = loaded && inventory?.status.enabled === false
  const statusBarClass = error
    ? 'bg-status-error'
    : disabled
      ? 'bg-status-warning'
      : inventory?.warnings.length
        ? 'bg-status-warning'
        : inventory?.status.enabled
          ? 'bg-status-success'
          : 'bg-background-tertiary'

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Cloud}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={() => buildInventoryContext(inventory, error)}
      statusBarClass={statusBarClass}
      hue={hueFor('huggingface')}
      headerRight={
        <>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              void loadInventory(true)
            }}
            disabled={refreshing}
            className="node-btn node-btn-accent"
            title="Refresh Hugging Face inventory"
          >
            {refreshing ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setConfigOpen(true)
            }}
            className="nodrag node-btn node-btn-accent"
            title="Configure Hugging Face node"
          >
            <SlidersHorizontal className="w-3 h-3" />
          </button>
        </>
      }
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      {!loaded ? (
        <div className="flex justify-center py-3">
          <Loader2 className="w-4 h-4 animate-spin text-text-muted" />
        </div>
      ) : error ? (
        <div className="flex items-start gap-1.5 text-[10px] font-mono text-status-error py-2">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      ) : disabled ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-3">
          Hugging Face not configured
          <span className="block mt-1 text-text-secondary">Set HF_TOKEN or connect a stored token</span>
        </div>
      ) : inventory ? (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-[10px] font-mono text-text-muted">
            <span className="truncate" title={inventory.namespace}>
              {inventory.namespace || 'default namespace'}
            </span>
            {inventory.cached && (
              <span className="px-1 py-px border-brutal border-border-subtle rounded-brutal text-[8px] uppercase">
                cache
              </span>
            )}
            <span className="ml-auto flex-shrink-0">{formatDate(inventory.last_refreshed)}</span>
          </div>

          <div className="grid grid-cols-4 gap-1.5">
            <CountTile icon={Package} label="models" value={inventory.counts.models} />
            <CountTile icon={Database} label="data" value={inventory.counts.datasets} />
            <CountTile icon={Database} label="traces" value={inventory.counts.trace_datasets} />
            <CountTile icon={HardDrive} label="buckets" value={inventory.counts.buckets} />
          </div>

          {inventory.warnings.length > 0 && (
            <div className="flex items-start gap-1.5 px-2 py-1.5 border-brutal border-status-warning/50 bg-status-warning/10 rounded-brutal text-[9px] font-mono text-status-warning">
              <AlertCircle className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <span className="min-w-0 truncate" title={inventory.warnings[0].message}>
                {inventory.warnings[0].section}: {inventory.warnings[0].message}
              </span>
            </div>
          )}

          <div className="space-y-1.5">
            {topModels.map((model) => (
              <ResourceRow
                key={`model-${model.id}`}
                id={model.id}
                meta={modelMeta(model)}
                isPrivate={model.private}
                tone={model.local ? 'local' : 'default'}
                url={model.url}
              />
            ))}
            {topDatasets.map((dataset) => (
              <ResourceRow
                key={`dataset-${dataset.id}`}
                id={dataset.id}
                meta={datasetMeta(dataset)}
                isPrivate={dataset.private}
                url={dataset.url}
              />
            ))}
            {topTraces.map((trace) => (
              <ResourceRow
                key={`trace-${trace.id}`}
                id={`${trace.id} / traces`}
                meta={datasetMeta(trace)}
                isPrivate={trace.private}
                url={trace.url}
              />
            ))}
            {topBuckets.map((bucket) => (
              <ResourceRow
                key={`bucket-${bucket.id}`}
                id={bucket.id}
                meta={bucketMeta(bucket)}
                isPrivate={bucket.private}
              />
            ))}
          </div>

          {topModels.length + topDatasets.length + topTraces.length + topBuckets.length === 0 && (
            <div className="text-[10px] font-mono text-text-muted text-center py-2">
              No matching HF resources found
            </div>
          )}
        </div>
      ) : null}
    </DataNodeShell>
    <NodeConfigModal
      isOpen={configOpen}
      onClose={() => setConfigOpen(false)}
      title={`${data.title} Config`}
      description="Hugging Face inventory"
      size="lg"
    >
      <ConfigSection title="Inventory State">
        <div className="flex flex-wrap gap-1.5">
          <ConfigPill tone={error ? 'error' : disabled ? 'warning' : inventory?.status.enabled ? 'success' : 'neutral'}>
            {error ? 'error' : disabled ? 'not configured' : inventory?.status.enabled ? 'enabled' : 'idle'}
          </ConfigPill>
          {inventory?.cached ? <ConfigPill tone="neutral">cache</ConfigPill> : null}
          {inventory?.warnings.length ? <ConfigPill tone="warning">{inventory.warnings.length} warnings</ConfigPill> : null}
        </div>
        <ConfigRows>
          <ConfigRow label="Poll interval" value={`${POLL_MS / 1000}s`} />
          <ConfigRow label="Limit" value={INVENTORY_LIMIT} />
          <ConfigRow label="Namespace" value={inventory?.namespace || '(default)'} />
          <ConfigRow label="Prefix" value={inventory?.prefix || '(none)'} />
          <ConfigRow label="Last refresh" value={inventory?.last_refreshed ? new Date(inventory.last_refreshed).toLocaleString() : undefined} />
          <ConfigRow label="Models" value={inventory?.counts.models} />
          <ConfigRow label="Datasets" value={inventory?.counts.datasets} />
          <ConfigRow label="Trace data" value={inventory?.counts.trace_datasets} />
          <ConfigRow label="Buckets" value={inventory?.counts.buckets} />
          <ConfigRow label="Error" value={error} />
        </ConfigRows>
      </ConfigSection>

      {inventory?.warnings.length ? (
        <ConfigSection title="Warnings">
          <ConfigRows>
            {inventory.warnings.map((warning, index) => (
              <ConfigRow
                key={`${warning.section}-${index}`}
                label={warning.section}
                value={warning.message}
              />
            ))}
          </ConfigRows>
        </ConfigSection>
      ) : null}

      <ConfigSection title="Live Handles">
        <ConfigRows>
          <ConfigRow
            label="Inventory"
            value={`${API_BASE}/hf/inventory?limit=${INVENTORY_LIMIT}`}
          />
          <ConfigRow label="Models" value={`${API_BASE}/hf/models/mine`} />
          <ConfigRow label="Datasets" value={`${API_BASE}/hf/datasets`} />
          <ConfigRow label="Buckets" value={`${API_BASE}/hf/buckets`} />
          <ConfigRow label="Workspace API" value={`${API_BASE}/workspace/context?format=json`} />
        </ConfigRows>
      </ConfigSection>
    </NodeConfigModal>
    </>
  )
})
