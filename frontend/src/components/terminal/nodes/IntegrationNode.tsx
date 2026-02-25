import { memo, useCallback, useMemo, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import {
  ChevronDown,
  ChevronRight,
  Link2,
  Loader2,
  Send,
  X
} from 'lucide-react'
import { clsx } from 'clsx'
import type { ConfigField, IntegrationNodeData, NodeAdapter } from './types'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'

// ---------------------------------------------------------------------------
// Adapter registry
// ---------------------------------------------------------------------------

type AdapterFactory = (
  config: Record<string, unknown>,
  onChange: (key: string, value: unknown) => void
) => NodeAdapter

const adapterRegistry = new Map<string, AdapterFactory>()

export function registerAdapter(type: string, factory: AdapterFactory): void {
  adapterRegistry.set(type, factory)
}

// ---------------------------------------------------------------------------
// Config field renderers
// ---------------------------------------------------------------------------

function ConfigFieldInput({
  field,
  value,
  onChange
}: {
  field: ConfigField
  value: unknown
  onChange: (key: string, value: unknown) => void
}) {
  const inputClasses =
    'w-full text-xs font-mono bg-background border-brutal border-border px-2 py-1.5 focus:border-accent focus:outline-none'

  switch (field.type) {
    case 'text':
    case 'password':
      return (
        <input
          type={field.type}
          className={inputClasses}
          placeholder={field.placeholder}
          value={(value as string) ?? ''}
          onChange={e => onChange(field.key, e.target.value)}
        />
      )
    case 'textarea':
      return (
        <textarea
          className={clsx(inputClasses, 'resize-y min-h-[60px]')}
          placeholder={field.placeholder}
          value={(value as string) ?? ''}
          onChange={e => onChange(field.key, e.target.value)}
          rows={3}
        />
      )
    case 'select':
      return (
        <select
          className={inputClasses}
          value={(value as string) ?? ''}
          onChange={e => onChange(field.key, e.target.value)}
        >
          <option value="">-- select --</option>
          {field.options?.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      )
    case 'toggle':
      return (
        <button
          type="button"
          className={clsx(
            'px-3 py-1 text-[10px] font-mono uppercase tracking-wider border-brutal transition-press',
            value
              ? 'bg-accent text-white border-accent'
              : 'bg-background text-text-secondary border-border'
          )}
          onClick={() => onChange(field.key, !value)}
        >
          {value ? 'On' : 'Off'}
        </button>
      )
    default:
      return null
  }
}

// ---------------------------------------------------------------------------
// IntegrationNode
// ---------------------------------------------------------------------------

export type IntegrationNodeType = NodeProps<IntegrationNodeData>

export const IntegrationNode = memo(function IntegrationNode({
  data,
  selected
}: IntegrationNodeType) {
  const {
    panelId,
    title,
    adapterType,
    adapterConfig,
    hasConnections,
    onFocus,
    onClose
  } = data

  // Local UI state
  const [expanded, setExpanded] = useState(false)
  const [config, setConfig] = useState<Record<string, unknown>>(
    () => ({ ...adapterConfig })
  )
  const [sending, setSending] = useState(false)

  // Config change handler fed into the adapter factory
  const handleConfigChange = useCallback(
    (key: string, value: unknown) => {
      setConfig(prev => ({ ...prev, [key]: value }))
    },
    []
  )

  // Build adapter (or null if type not registered yet)
  const adapter = useMemo(() => {
    const factory = adapterRegistry.get(adapterType)
    if (!factory) return null
    return factory(config, handleConfigChange)
  }, [adapterType, config, handleConfigChange])

  // ---------------------------------------------------------------------------
  // Event handlers
  // ---------------------------------------------------------------------------

  const handleClose = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      onClose?.(panelId)
    },
    [panelId, onClose]
  )

  const handleFocus = useCallback(() => {
    onFocus?.(panelId)
  }, [panelId, onFocus])

  const handleToggleExpand = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation()
      setExpanded(prev => !prev)
    },
    []
  )

  const handleSendContext = useCallback(
    async (e: React.MouseEvent) => {
      e.stopPropagation()
      if (!adapter || sending) return
      setSending(true)
      try {
        const ctx = adapter.getContext()
        await routeToLinkedTerminals(panelId, ctx.content, 'md')
      } finally {
        setSending(false)
      }
    },
    [adapter, panelId, sending]
  )

  // ---------------------------------------------------------------------------
  // Derived data
  // ---------------------------------------------------------------------------

  const context = adapter?.getContext()
  const actions = adapter?.getActions() ?? []
  const configFields = adapter?.getConfigFields() ?? []

  const AdapterIcon = adapter?.icon

  return (
    <div
      className={clsx(
        'w-[300px] bg-surface border-brutal border-border shadow-brutal-sm cursor-pointer',
        selected && 'border-accent shadow-brutal'
      )}
      onClick={handleFocus}
    >
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />

      {/* Header bar */}
      <div className="flex items-center gap-2 px-3 py-2 bg-background border-b border-border">
        {AdapterIcon && (
          <AdapterIcon className="w-4 h-4 text-accent flex-shrink-0" />
        )}
        <span className="text-xs font-mono font-bold uppercase tracking-wider text-primary flex-1 truncate">
          {title}
        </span>

        <div className="flex items-center gap-0.5">
          {hasConnections && (
            <div
              className="flex items-center gap-0.5 px-1 py-0.5 border-brutal border-accent/60 bg-accent/10 rounded-brutal text-accent"
              title="Connected to terminal"
            >
              <Link2 className="w-2.5 h-2.5" />
              <span className="text-[8px] font-mono font-bold uppercase tracking-wide">
                Linked
              </span>
            </div>
          )}
          <button
            type="button"
            onClick={handleToggleExpand}
            className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
            title={expanded ? 'Collapse config' : 'Expand config'}
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
          </button>
          <button
            type="button"
            onClick={handleClose}
            className="p-1 hover:bg-status-error/20 text-text-muted hover:text-status-error transition-press"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Summary line */}
      {context && (
        <div className="flex items-center justify-between px-3 py-1.5 text-[10px] font-mono text-text-muted border-b border-border">
          <span className="truncate flex-1">{context.summary}</span>
          <span className="flex-shrink-0 ml-2">~{context.tokenEstimate}t</span>
        </div>
      )}

      {/* Action buttons */}
      {(hasConnections || actions.length > 0) && (
        <div className="flex flex-wrap items-center gap-1.5 px-3 py-2">
          {hasConnections && (
            <button
              type="button"
              onClick={handleSendContext}
              disabled={sending}
              className={clsx(
                'flex items-center gap-1 px-2 py-1 text-[10px] font-mono uppercase tracking-wider border-brutal border-accent text-accent transition-press',
                'hover:bg-accent hover:text-white',
                sending && 'opacity-50 cursor-not-allowed'
              )}
            >
              {sending ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Send className="w-3 h-3" />
              )}
              Send Context
            </button>
          )}
          {actions.map(action => {
            const ActionIcon = action.icon
            return (
              <button
                key={action.id}
                type="button"
                onClick={e => {
                  e.stopPropagation()
                  void action.handler()
                }}
                className="flex items-center gap-1 px-2 py-1 text-[10px] font-mono uppercase tracking-wider border-brutal border-border text-text-secondary hover:border-accent hover:text-accent transition-press"
              >
                <ActionIcon className="w-3 h-3" />
                {action.label}
              </button>
            )
          })}
        </div>
      )}

      {/* Expanded config panel */}
      {expanded && configFields.length > 0 && (
        <div className="px-3 py-2 space-y-2 border-t border-border bg-background">
          {configFields.map(field => (
            <div key={field.key}>
              <label className="block text-[9px] font-mono uppercase tracking-wider text-secondary mb-1">
                {field.label}
              </label>
              <ConfigFieldInput
                field={field}
                value={config[field.key]}
                onChange={handleConfigChange}
              />
            </div>
          ))}
        </div>
      )}

      {/* Fallback when adapter not yet registered */}
      {!adapter && (
        <div className="px-3 py-4 text-center">
          <span className="text-[10px] font-mono text-text-muted uppercase tracking-wider">
            Adapter &quot;{adapterType}&quot; not registered
          </span>
        </div>
      )}
    </div>
  )
})
