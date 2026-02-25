import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import {
  Check,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Link2,
  Loader2,
  Send,
  X
} from 'lucide-react'
import { clsx } from 'clsx'
import type { ConfigField, IntegrationNodeData, NodeAdapter } from './types'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'
import { useTerminalStore } from '../../../stores/terminalStore'

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
    'w-full text-xs font-mono bg-background-card border-brutal border-border rounded-brutal px-2 py-1.5 focus:border-accent focus:outline-none'

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
            'px-3 py-1 text-[10px] font-mono uppercase tracking-wider border-brutal rounded-brutal transition-press',
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
// Credential field definitions per adapter type
// ---------------------------------------------------------------------------

interface CredentialFieldDef {
  key: string
  label: string
}

const CREDENTIAL_FIELDS: Record<string, CredentialFieldDef[]> = {
  neon: [
    { key: 'neon-api-key', label: 'API Key' },
    { key: 'neon-connection-string', label: 'Connection String' }
  ],
  vercel: [
    { key: 'vercel-token', label: 'Vercel Token' },
    { key: 'v0-api-key', label: 'v0 API Key' }
  ]
}

// ---------------------------------------------------------------------------
// CredentialField - a single credential input row
// ---------------------------------------------------------------------------

function CredentialField({
  field,
  isSaved,
  onSave
}: {
  field: CredentialFieldDef
  isSaved: boolean
  onSave: (key: string, value: string) => Promise<void>
}) {
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)

  const handleSave = useCallback(async () => {
    if (!value.trim() || saving) return
    setSaving(true)
    try {
      await onSave(field.key, value)
      setValue('')
    } finally {
      setSaving(false)
    }
  }, [field.key, value, saving, onSave])

  return (
    <div>
      <label className="block text-[9px] font-mono uppercase tracking-wider text-text-secondary mb-1">
        {field.label}
        {isSaved && (
          <span className="inline-flex items-center gap-0.5 ml-2 text-status-success">
            <Check className="w-2.5 h-2.5" />
            Saved
          </span>
        )}
      </label>
      <div className="flex items-center gap-1">
        <input
          type="password"
          className="flex-1 text-xs font-mono bg-background-card border-brutal border-border rounded-brutal px-2 py-1.5 focus:border-accent focus:outline-none"
          placeholder={isSaved ? '••••••••' : `Enter ${field.label.toLowerCase()}…`}
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') void handleSave()
          }}
        />
        <button
          type="button"
          onClick={() => void handleSave()}
          disabled={!value.trim() || saving}
          className={clsx(
            'px-2 py-1.5 text-[10px] font-mono uppercase tracking-wider border-brutal rounded-brutal transition-press',
            value.trim()
              ? 'border-accent text-accent hover:bg-accent hover:text-white'
              : 'border-border text-text-muted cursor-not-allowed'
          )}
        >
          {saving ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Save'}
        </button>
      </div>
    </div>
  )
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

  // Credential management state
  const [credentialStatus, setCredentialStatus] = useState<Record<string, boolean>>({})
  const credentialFields = CREDENTIAL_FIELDS[adapterType] ?? []

  // Check credential status on mount / adapter type change
  useEffect(() => {
    const fields = CREDENTIAL_FIELDS[adapterType]
    if (!fields || !window.bashgym?.credentials) return

    Promise.all(
      fields.map(f =>
        window.bashgym.credentials.read(f.key).then(
          result => [f.key, Boolean(result?.value)] as const
        ).catch(() => [f.key, false] as const)
      )
    ).then(entries => {
      setCredentialStatus(Object.fromEntries(entries))
    })
  }, [adapterType])

  // Save a credential and update status
  const handleCredentialSave = useCallback(async (key: string, value: string) => {
    await window.bashgym?.credentials.store(key, value)
    setCredentialStatus(prev => ({ ...prev, [key]: true }))
  }, [])

  // Persist config changes to store (debounced)
  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (persistTimer.current) clearTimeout(persistTimer.current)
    persistTimer.current = setTimeout(() => {
      const { _panelId, ...persistConfig } = config as Record<string, unknown> & { _panelId?: unknown }
      useTerminalStore.getState().updatePanelConfig(panelId, persistConfig)
    }, 300)
    return () => {
      if (persistTimer.current) clearTimeout(persistTimer.current)
    }
  }, [config, panelId])

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
        'w-[300px] card !rounded-brutal border-brutal cursor-pointer',
        selected ? 'border-accent shadow-brutal' : 'border-border hover:border-border'
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
      <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border rounded-t-brutal">
        <div className="p-1.5 border-brutal border-border-subtle rounded-brutal bg-background-tertiary">
          {AdapterIcon ? (
            <AdapterIcon className="w-4 h-4 text-accent" />
          ) : (
            <div className="w-4 h-4" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-mono font-semibold text-text-primary truncate block">
            {title}
          </span>
        </div>

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

      {/* Summary line with adapter status indicators */}
      {context && (
        <div className="flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono text-text-muted border-b border-brutal border-border">
          {/* Neon: connection status dot + table count */}
          {adapterType === 'neon' && (
            <div
              className={clsx(
                'w-1.5 h-1.5 rounded-full flex-shrink-0',
                config.connected ? 'bg-status-success' : 'bg-background-tertiary'
              )}
              title={config.connected ? 'Connected' : 'Disconnected'}
            />
          )}
          {/* Vercel: deploy status dot */}
          {adapterType === 'vercel' && (
            <div
              className={clsx(
                'w-1.5 h-1.5 rounded-full flex-shrink-0',
                config.deployStatus === 'READY' ? 'bg-status-success' :
                config.deployStatus === 'BUILDING' ? 'bg-status-warning animate-pulse' :
                config.deployStatus === 'ERROR' ? 'bg-status-error' :
                'bg-background-tertiary'
              )}
              title={typeof config.deployStatus === 'string' ? config.deployStatus : 'Unknown'}
            />
          )}
          <span className="truncate flex-1">{context.summary}</span>
          {/* Neon: table count badge */}
          {adapterType === 'neon' && config.tableCount && (
            <span className="text-[9px] font-mono text-text-secondary flex-shrink-0">
              {String(config.tableCount)} tables
            </span>
          )}
          {/* Vercel: deploy URL link */}
          {adapterType === 'vercel' && typeof config.deployUrl === 'string' && config.deployUrl && (
            <a
              href={config.deployUrl as string}
              target="_blank"
              rel="noopener noreferrer"
              onClick={e => e.stopPropagation()}
              className="flex items-center gap-0.5 text-[9px] text-accent hover:text-accent-dark flex-shrink-0"
              title={config.deployUrl as string}
            >
              <ExternalLink className="w-2.5 h-2.5" />
            </a>
          )}
          <span className="flex-shrink-0 ml-auto">~{context.tokenEstimate}t</span>
        </div>
      )}

      {/* Action buttons */}
      {(hasConnections || actions.length > 0) && (
        <div className="flex flex-wrap items-center gap-1.5 px-3 py-2 border-b border-brutal border-border">
          {hasConnections && (
            <button
              type="button"
              onClick={handleSendContext}
              disabled={sending}
              className={clsx(
                'flex items-center gap-1 px-2 py-1 text-[10px] font-mono uppercase tracking-wider',
                'border-brutal rounded-brutal transition-press',
                sending
                  ? 'border-border text-text-muted cursor-not-allowed opacity-50'
                  : 'border-accent text-accent hover:bg-accent hover:text-white'
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
                className="flex items-center gap-1 px-2 py-1 text-[10px] font-mono uppercase tracking-wider border-brutal rounded-brutal border-border text-text-secondary hover:border-accent hover:text-accent transition-press"
              >
                <ActionIcon className="w-3 h-3" />
                {action.label}
              </button>
            )
          })}
        </div>
      )}

      {/* Expanded config panel */}
      {expanded && (credentialFields.length > 0 || configFields.length > 0) && (
        <div className="px-3 py-2 space-y-2 bg-background-secondary">
          {/* Credential fields (shown above adapter config) */}
          {credentialFields.length > 0 && (
            <>
              {credentialFields.map(field => (
                <CredentialField
                  key={field.key}
                  field={field}
                  isSaved={credentialStatus[field.key] ?? false}
                  onSave={handleCredentialSave}
                />
              ))}
              {configFields.length > 0 && (
                <div className="border-t border-brutal border-border-subtle my-1" />
              )}
            </>
          )}
          {/* Adapter config fields */}
          {configFields.map(field => (
            <div key={field.key}>
              <label className="block text-[9px] font-mono uppercase tracking-wider text-text-secondary mb-1">
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
        <div className="px-3 py-4 text-center rounded-b-brutal">
          <span className="text-[10px] font-mono text-text-muted uppercase tracking-wider">
            Adapter &quot;{adapterType}&quot; not registered
          </span>
        </div>
      )}
    </div>
  )
})
