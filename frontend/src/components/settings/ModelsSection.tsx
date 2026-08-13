import { useState, useCallback } from 'react'
import {
  CheckCircle,
  XCircle,
  Trash2,
  Loader2,
  RefreshCw,
  HardDrive,
  Cloud,
  Cpu,
  Code,
  AlertTriangle,
  ExternalLink,
  Plus,
  ChevronDown,
  ChevronRight
} from 'lucide-react'
import { providersApi, ProviderStatus, OllamaModel } from '../../services/api'
import { ollamaStatusResource } from '../../stores/appResources'
import { providersResource } from '../../stores/opsResources'
import { useSessionResource } from '../../stores/sessionResource'
import { StudentModelPicker } from './StudentModelPicker'
import { ModelSelect } from '../common/ModelSelect'
import { clsx } from 'clsx'

function ProviderCard({ provider }: { provider: ProviderStatus }) {
  const isLocal = provider.type === 'ollama' || provider.type === 'lm_studio'

  return (
    <div
      className={clsx(
        'flex items-center gap-2 p-2 border-brutal border-border rounded-brutal transition-press',
        provider.available
          ? 'bg-background-card shadow-brutal-sm'
          : 'bg-background-secondary border-border-subtle'
      )}
    >
      <div
        className={clsx(
          'w-8 h-8 border-brutal border-border rounded-brutal flex items-center justify-center flex-shrink-0',
          provider.available
            ? 'bg-accent-light text-accent-dark'
            : 'bg-background-secondary text-text-muted'
        )}
      >
        {isLocal ? <HardDrive className="w-3.5 h-3.5" /> : <Cloud className="w-3.5 h-3.5" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-mono font-semibold text-text-primary truncate">
            {provider.name}
          </span>
          {provider.available ? (
            <CheckCircle className="w-3 h-3 text-status-success flex-shrink-0" />
          ) : (
            <XCircle className="w-3 h-3 text-text-muted flex-shrink-0" />
          )}
        </div>
        {provider.model_count !== undefined && provider.model_count > 0 && (
          <p className="text-[10px] text-text-muted font-mono">
            {provider.model_count} model{provider.model_count > 1 ? 's' : ''}
          </p>
        )}
      </div>
    </div>
  )
}

function OllamaModelCard({
  model,
  onDelete,
  isDeleting
}: {
  model: OllamaModel
  onDelete: (name: string) => void
  isDeleting: boolean
}) {
  return (
    <div className="flex items-center gap-2 p-2.5 border-brutal border-border rounded-brutal bg-background-card shadow-brutal-sm">
      <div
        className={clsx(
          'w-8 h-8 border-brutal border-border rounded-brutal flex items-center justify-center flex-shrink-0',
          model.is_code_model
            ? 'bg-accent-light text-accent-dark'
            : 'bg-background-secondary text-text-muted'
        )}
      >
        {model.is_code_model ? <Code className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-mono font-semibold text-text-primary truncate">
            {model.name}
          </span>
          {model.is_code_model && (
            <span className="tag text-[10px] py-0 px-1.5">
              <span>code</span>
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-[11px] text-text-muted font-mono">
          <span>{model.size_gb.toFixed(1)} GB</span>
          <span className="text-border-subtle">|</span>
          <span>{model.parameter_size}</span>
          <span className="text-border-subtle">|</span>
          <span>{model.quantization}</span>
        </div>
      </div>
      <button
        onClick={() => onDelete(model.name)}
        disabled={isDeleting}
        className="btn-icon w-8 h-8 text-text-muted hover:text-status-error flex-shrink-0"
        title="Delete model"
      >
        {isDeleting ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <Trash2 className="w-3.5 h-3.5" />
        )}
      </button>
    </div>
  )
}

export function ModelsSection() {
  const { data: providersData, loading: providersLoading } = useSessionResource(providersResource)
  const {
    data: ollamaStatus,
    loading: ollamaLoading,
    error: ollamaFetchError
  } = useSessionResource(ollamaStatusResource)
  const providers = providersData?.providers ?? []
  const ollamaModels = ollamaStatus?.models ?? []
  const ollamaAvailable = ollamaStatus?.available ?? false
  const isLoading = providersLoading || ollamaLoading
  const error = ollamaStatus?.error || ollamaFetchError
  const [isPulling, setIsPulling] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState<string | null>(null)
  const [customModel, setCustomModel] = useState('')
  // Connect cloud provider (OpenAI-compatible) form
  const [showConnect, setShowConnect] = useState(false)
  const [connectPlatform, setConnectPlatform] = useState('together')
  const [connectBaseUrl, setConnectBaseUrl] = useState('')
  const [connectKey, setConnectKey] = useState('')
  const [connectModel, setConnectModel] = useState('')
  const [connecting, setConnecting] = useState(false)
  const [connectMsg, setConnectMsg] = useState<string | null>(null)

  const fetchData = useCallback(() => {
    void providersResource.getState().refresh()
    void ollamaStatusResource.getState().refresh()
  }, [])

  const handlePullModel = async (modelName: string) => {
    setIsPulling(modelName)
    try {
      const result = await providersApi.pullOllamaModel(modelName)
      if (result.ok) {
        // Refresh after a short delay to allow download to start
        setTimeout(() => fetchData(), 2000)
      }
    } finally {
      setIsPulling(null)
    }
  }

  const handleConnect = async () => {
    setConnecting(true)
    setConnectMsg(null)
    try {
      const body =
        connectPlatform === '__custom__'
          ? {
              base_url: connectBaseUrl,
              api_key: connectKey || undefined,
              default_model: connectModel || undefined
            }
          : {
              platform: connectPlatform,
              api_key: connectKey || undefined,
              default_model: connectModel || undefined
            }
      const result = await providersApi.connect(body)
      if (result.ok && result.data?.ok) {
        const n = result.data.models?.length ?? 0
        setConnectMsg(
          `Connected ${result.data.provider_type} — ${result.data.available ? `${n} models` : 'registered (health check failed)'}`
        )
        setConnectKey('')
        setTimeout(() => fetchData(), 500)
      } else {
        setConnectMsg(result.data?.error || result.error || 'Connection failed')
      }
    } finally {
      setConnecting(false)
    }
  }

  const handleDeleteModel = async (modelName: string) => {
    setIsDeleting(modelName)
    try {
      const result = await providersApi.deleteOllamaModel(modelName)
      if (result.ok) {
        const { data: current, setData } = ollamaStatusResource.getState()
        if (current) {
          setData({ ...current, models: current.models.filter((m) => m.name !== modelName) })
        }
      }
    } finally {
      setIsDeleting(null)
    }
  }

  const handlePullCustom = async () => {
    if (!customModel.trim()) return
    await handlePullModel(customModel.trim())
    setCustomModel('')
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 animate-spin text-text-muted" />
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="font-brand text-lg text-text-primary">Model Providers</h3>
        <button onClick={fetchData} className="btn-icon w-8 h-8 text-text-muted" title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Providers Grid */}
      <div className="grid grid-cols-3 gap-1.5">
        {providers.map((provider) => (
          <ProviderCard key={provider.type} provider={provider} />
        ))}
      </div>

      {/* Connect Cloud Provider (OpenAI-compatible) */}
      <div>
        <button
          onClick={() => setShowConnect(!showConnect)}
          className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-widest text-text-muted hover:text-accent-dark"
        >
          {showConnect ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          <Plus className="w-3 h-3" /> Connect cloud provider
        </button>
        {showConnect && (
          <div className="mt-2 p-3 border-brutal border-border rounded-brutal bg-background-card space-y-2.5">
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                  Platform
                </label>
                <select
                  value={connectPlatform}
                  onChange={(e) => setConnectPlatform(e.target.value)}
                  className="input w-full text-xs"
                >
                  <option value="together">Together</option>
                  <option value="fireworks">Fireworks</option>
                  <option value="openrouter">OpenRouter</option>
                  <option value="groq">Groq</option>
                  <option value="deepinfra">DeepInfra</option>
                  <option value="hyperbolic">Hyperbolic</option>
                  <option value="vllm">vLLM (local, no key)</option>
                  <option value="__custom__">Custom base URL…</option>
                </select>
              </div>
              <div>
                <label className="block font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                  Default Model
                </label>
                <ModelSelect
                  value={connectModel}
                  onChange={setConnectModel}
                  className="input w-full text-xs"
                  placeholder="Select a default model..."
                  catalogOnly
                />
              </div>
            </div>
            {connectPlatform === '__custom__' && (
              <div>
                <label className="block font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                  Base URL
                </label>
                <input
                  value={connectBaseUrl}
                  onChange={(e) => setConnectBaseUrl(e.target.value)}
                  className="input w-full text-xs"
                  placeholder="https://host/v1"
                />
              </div>
            )}
            {connectPlatform !== 'vllm' && (
              <div>
                <label className="block font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                  API Key
                </label>
                <input
                  type="password"
                  value={connectKey}
                  onChange={(e) => setConnectKey(e.target.value)}
                  className="input w-full text-xs"
                  placeholder="held in memory only — never written to disk"
                />
              </div>
            )}
            <div className="flex items-center gap-2">
              <button
                onClick={handleConnect}
                disabled={connecting}
                className="btn-primary text-xs px-3 py-1.5"
              >
                {connecting ? <Loader2 className="w-3 h-3 animate-spin" /> : 'Connect'}
              </button>
              {connectMsg && (
                <span className="text-[11px] font-mono text-text-secondary">{connectMsg}</span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Ollama Section */}
      <div className="section-divider" />
      <div className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h4 className="font-brand text-sm text-text-primary">Local Models (Ollama)</h4>
          <a
            href="https://ollama.com/library"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] font-mono text-accent-dark flex items-center gap-0.5 hover:text-accent"
          >
            Browse library
            <ExternalLink className="w-2.5 h-2.5" />
          </a>
        </div>

        {!ollamaAvailable ? (
          <div className="flex items-start gap-2 p-3 border-brutal border-status-warning rounded-brutal bg-background-card">
            <AlertTriangle className="w-3.5 h-3.5 text-status-warning flex-shrink-0 mt-0.5" />
            <div className="text-[11px] text-text-secondary">
              <p className="font-mono font-semibold text-text-primary">Ollama Not Running</p>
              <p className="text-text-muted">{error || 'Start Ollama to use local models'}</p>
              <code className="inline-block mt-1 px-2 py-1 bg-background-secondary border-brutal border-border rounded-brutal text-[10px] font-mono text-text-muted">
                ollama serve
              </code>
            </div>
          </div>
        ) : (
          <>
            {/* Installed Models */}
            {ollamaModels.length > 0 ? (
              <div className="space-y-2">
                {ollamaModels.map((model) => (
                  <OllamaModelCard
                    key={model.name}
                    model={model}
                    onDelete={handleDeleteModel}
                    isDeleting={isDeleting === model.name}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-text-muted text-center py-3 font-mono">
                No models installed. Enter a model ID from your local catalog below.
              </p>
            )}

            {/* Student Model Picker */}
            {ollamaModels.length > 0 && (
              <>
                <div className="section-divider" />
                <StudentModelPicker
                  ollamaModels={ollamaModels}
                  ollamaAvailable={ollamaAvailable}
                  onRefresh={fetchData}
                />
              </>
            )}

            {/* Custom Model */}
            <div className="flex items-center gap-1.5 pt-1">
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="Model ID from your Ollama catalog"
                className="input flex-1 text-xs py-1.5 px-2 font-mono"
                onKeyDown={(e) => e.key === 'Enter' && handlePullCustom()}
              />
              <button
                onClick={handlePullCustom}
                disabled={!customModel.trim() || !!isPulling}
                className="btn-primary p-1.5"
              >
                {isPulling && isPulling === customModel ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Plus className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Info */}
      <div className="text-[11px] text-text-secondary p-3 border-brutal border-border rounded-brutal bg-background-card">
        <strong className="font-mono text-text-primary">Local models</strong> run on your machine
        for privacy and zero API costs. BashGym can fine-tune these using collected traces.
      </div>
    </div>
  )
}
