import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import { Check, ExternalLink, Link2, Loader2, Send, Settings2, X } from 'lucide-react'
import { clsx } from 'clsx'
import type { ConfigField, IntegrationNodeData, NodeAdapter } from './types'
import { routeToLinkedTerminals } from '../../../utils/edgeRouting'
import { useTerminalStore } from '../../../stores/terminalStore'
import { ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import { useNodeSurface } from './nodeSurface'

// ---------------------------------------------------------------------------
// Adapter registry
// ---------------------------------------------------------------------------

type AdapterFactory = (
  config: Record<string, unknown>,
  onChange: (key: string, value: unknown) => void
) => NodeAdapter

const adapterRegistry = new Map<string, AdapterFactory>()

// eslint-disable-next-line react-refresh/only-export-components
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
          onChange={(e) => onChange(field.key, e.target.value)}
        />
      )
    case 'textarea':
      return (
        <textarea
          className={clsx(inputClasses, 'nodrag nowheel resize-y min-h-[60px]')}
          placeholder={field.placeholder}
          value={(value as string) ?? ''}
          onChange={(e) => onChange(field.key, e.target.value)}
          rows={3}
        />
      )
    case 'select':
      return (
        <select
          className={inputClasses}
          value={(value as string) ?? ''}
          onChange={(e) => onChange(field.key, e.target.value)}
        >
          <option value="">-- select --</option>
          {field.options?.map((opt) => (
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
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
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

export type IntegrationNodeType = Node<IntegrationNodeData>

export const IntegrationNode = memo(function IntegrationNode({
  data,
  selected
}: NodeProps<IntegrationNodeType>) {
  const { panelId, title, adapterType, adapterConfig, hasConnections, onFocus, onClose } = data

  // Local UI state — open config when node is freshly created (no meaningful config yet)
  const surface = useNodeSurface()
  const isGrid = surface === 'grid'
  const hasExistingConfig = Object.keys(adapterConfig).some(
    (k) => k !== '_panelId' && adapterConfig[k] !== undefined && adapterConfig[k] !== ''
  )
  const [configOpen, setConfigOpen] = useState(!isGrid && !hasExistingConfig)
  const [config, setConfig] = useState<Record<string, unknown>>(() => ({ ...adapterConfig }))
  const [sending, setSending] = useState(false)

  // Credential management state
  const [credentialStatus, setCredentialStatus] = useState<Record<string, boolean>>({})
  const credentialFields = CREDENTIAL_FIELDS[adapterType] ?? []

  // Check credential status on mount / adapter type change
  useEffect(() => {
    const fields = CREDENTIAL_FIELDS[adapterType]
    if (!fields || !window.bashgym?.credentials) return

    Promise.all(
      fields.map((f) =>
        window.bashgym.credentials
          .read(f.key)
          .then((result) => [f.key, Boolean(result?.value)] as const)
          .catch(() => [f.key, false] as const)
      )
    ).then((entries) => {
      setCredentialStatus(Object.fromEntries(entries))
    })
  }, [adapterType])

  // Save a credential and update status
  const handleCredentialSave = useCallback(async (key: string, value: string) => {
    await window.bashgym?.credentials.store(key, value)
    setCredentialStatus((prev) => ({ ...prev, [key]: true }))
  }, [])

  // Persist config changes to store (debounced)
  const persistTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (persistTimer.current) clearTimeout(persistTimer.current)
    persistTimer.current = setTimeout(() => {
      const { _panelId: _omitPanelId, ...persistConfig } = config as Record<string, unknown> & {
        _panelId?: unknown
      }
      useTerminalStore.getState().updatePanelConfig(panelId, persistConfig)
    }, 300)
    return () => {
      if (persistTimer.current) clearTimeout(persistTimer.current)
    }
  }, [config, panelId])

  // Config change handler fed into the adapter factory
  const handleConfigChange = useCallback((key: string, value: unknown) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }, [])

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

  const handleOpenConfig = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    setConfigOpen(true)
  }, [])

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
    <>
      <div
        className={clsx(
          'card !rounded-brutal border-brutal cursor-pointer',
          isGrid ? 'flex h-full w-full min-w-0 flex-col overflow-hidden' : 'w-[300px]',
          selected ? 'border-accent shadow-brutal' : 'border-border hover:border-border'
        )}
        onClick={handleFocus}
        data-node-surface={surface}
      >
        {/* Connection handles */}
        {!isGrid && (
          <>
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
          </>
        )}

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

          <div className="flex items-center gap-1 nodrag">
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
              onClick={handleOpenConfig}
              className={clsx('node-btn', configOpen ? 'node-btn-success' : 'node-btn-accent')}
              title="Configure integration"
            >
              <Settings2 className="w-3 h-3" />
            </button>
            {!isGrid && (
              <button
                type="button"
                onClick={handleClose}
                className="node-btn node-btn-danger"
                title="Close"
              >
                <X className="w-3 h-3" />
              </button>
            )}
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
                  config.deployStatus === 'READY'
                    ? 'bg-status-success'
                    : config.deployStatus === 'BUILDING'
                      ? 'bg-status-warning animate-pulse'
                      : config.deployStatus === 'ERROR'
                        ? 'bg-status-error'
                        : 'bg-background-tertiary'
                )}
                title={typeof config.deployStatus === 'string' ? config.deployStatus : 'Unknown'}
              />
            )}
            <span className="truncate flex-1">{context.summary}</span>
            {/* Neon: table count badge */}
            {adapterType === 'neon' && Boolean(config.tableCount) && (
              <span className="text-[9px] font-mono text-text-secondary flex-shrink-0">
                {String(config.tableCount)} tables
              </span>
            )}
            {/* Vercel: deploy URL link */}
            {adapterType === 'vercel' &&
              typeof config.deployUrl === 'string' &&
              config.deployUrl && (
                <a
                  href={config.deployUrl as string}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
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
          <div className="flex flex-wrap items-center gap-1.5 px-3 py-2 border-b border-brutal border-border nodrag">
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
            {actions.map((action) => {
              const ActionIcon = action.icon
              return (
                <button
                  key={action.id}
                  type="button"
                  onClick={(e) => {
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

        {/* Setup prompt when no config has been saved yet */}
        {adapter && !hasExistingConfig && (
          <button
            type="button"
            onClick={handleOpenConfig}
            className="w-full px-3 py-3 text-center bg-background-secondary hover:bg-background-tertiary transition-press cursor-pointer border-t border-brutal border-border"
          >
            <span className="text-[10px] font-mono text-accent uppercase tracking-wider">
              Configure adapter
            </span>
          </button>
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
      <NodeConfigModal
        isOpen={configOpen}
        onClose={() => setConfigOpen(false)}
        title={`${title} Config`}
        description={context?.summary || adapterType}
        size="lg"
      >
        <ConfigSection title="Adapter State">
          <ConfigRows>
            <ConfigRow label="Adapter" value={adapterType} />
            <ConfigRow label="Summary" value={context?.summary} />
            <ConfigRow label="Token estimate" value={context?.tokenEstimate} />
            <ConfigRow label="Linked" value={hasConnections ? 'yes' : 'no'} />
            <ConfigRow label="Config fields" value={configFields.length} />
            <ConfigRow label="Credential fields" value={credentialFields.length} />
          </ConfigRows>
        </ConfigSection>

        {credentialFields.length > 0 && (
          <ConfigSection title="Credentials">
            <div className="space-y-3">
              {credentialFields.map((field) => (
                <CredentialField
                  key={field.key}
                  field={field}
                  isSaved={credentialStatus[field.key] ?? false}
                  onSave={handleCredentialSave}
                />
              ))}
            </div>
          </ConfigSection>
        )}

        {configFields.length > 0 && (
          <ConfigSection title="Adapter Config">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {configFields.map((field) => (
                <label key={field.key} className="node-field">
                  <span className="node-field-label">{field.label}</span>
                  <ConfigFieldInput
                    field={field}
                    value={config[field.key]}
                    onChange={handleConfigChange}
                  />
                </label>
              ))}
            </div>
          </ConfigSection>
        )}

        {credentialFields.length === 0 && configFields.length === 0 && (
          <ConfigSection>
            <div className="text-xs font-mono text-text-muted">
              No editable fields for this adapter.
            </div>
          </ConfigSection>
        )}
      </NodeConfigModal>
    </>
  )
})
