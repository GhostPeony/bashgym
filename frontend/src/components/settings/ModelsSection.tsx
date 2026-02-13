import { useState, useEffect, useCallback } from 'react'
import {
  CheckCircle,
  XCircle,
  Download,
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
import { clsx } from 'clsx'

// Popular models for quick download
const POPULAR_MODELS = [
  { name: 'qwen2.5-coder:7b', description: 'Excellent code model, 7B params', size: '4.7GB' },
  { name: 'qwen2.5-coder:1.5b', description: 'Fast, efficient coder', size: '1.0GB' },
  { name: 'deepseek-coder-v2:16b', description: 'State-of-the-art coding', size: '8.9GB' },
  { name: 'codellama:7b', description: 'Meta code model', size: '3.8GB' },
  { name: 'llama3.2:3b', description: 'General purpose, small', size: '2.0GB' },
]

function ProviderCard({ provider }: { provider: ProviderStatus }) {
  const isLocal = provider.type === 'ollama' || provider.type === 'lm_studio'

  return (
    <div className={clsx(
      'flex items-center gap-2 p-2 border-brutal border-border rounded-brutal transition-press',
      provider.available
        ? 'bg-background-card shadow-brutal-sm'
        : 'bg-background-secondary border-border-subtle'
    )}>
      <div className={clsx(
        'w-8 h-8 border-brutal border-border rounded-brutal flex items-center justify-center flex-shrink-0',
        provider.available ? 'bg-accent-light text-accent-dark' : 'bg-background-secondary text-text-muted'
      )}>
        {isLocal ? <HardDrive className="w-3.5 h-3.5" /> : <Cloud className="w-3.5 h-3.5" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-mono font-semibold text-text-primary truncate">{provider.name}</span>
          {provider.available ? (
            <CheckCircle className="w-3 h-3 text-status-success flex-shrink-0" />
          ) : (
            <XCircle className="w-3 h-3 text-text-muted flex-shrink-0" />
          )}
        </div>
        {provider.model_count !== undefined && provider.model_count > 0 && (
          <p className="text-[10px] text-text-muted font-mono">{provider.model_count} model{provider.model_count > 1 ? 's' : ''}</p>
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
      <div className={clsx(
        'w-8 h-8 border-brutal border-border rounded-brutal flex items-center justify-center flex-shrink-0',
        model.is_code_model ? 'bg-accent-light text-accent-dark' : 'bg-background-secondary text-text-muted'
      )}>
        {model.is_code_model ? <Code className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-mono font-semibold text-text-primary truncate">{model.name}</span>
          {model.is_code_model && (
            <span className="tag text-[10px] py-0 px-1.5"><span>code</span></span>
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
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([])
  const [ollamaAvailable, setOllamaAvailable] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [isPulling, setIsPulling] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState<string | null>(null)
  const [customModel, setCustomModel] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [showMoreModels, setShowMoreModels] = useState(false)

  const fetchData = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      // Fetch providers
      const providersResult = await providersApi.getProviders()
      if (providersResult.ok && providersResult.data) {
        setProviders(providersResult.data.providers)
      }

      // Fetch Ollama models
      const ollamaResult = await providersApi.getOllamaModels()
      if (ollamaResult.ok && ollamaResult.data) {
        setOllamaAvailable(ollamaResult.data.available)
        setOllamaModels(ollamaResult.data.models)
        if (!ollamaResult.data.available && ollamaResult.data.error) {
          setError(ollamaResult.data.error)
        }
      }
    } catch (err) {
      setError('Failed to connect to API')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchData()
  }, [fetchData])

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

  const handleDeleteModel = async (modelName: string) => {
    setIsDeleting(modelName)
    try {
      const result = await providersApi.deleteOllamaModel(modelName)
      if (result.ok) {
        setOllamaModels(prev => prev.filter(m => m.name !== modelName))
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
        <button
          onClick={fetchData}
          className="btn-icon w-8 h-8 text-text-muted"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Providers Grid */}
      <div className="grid grid-cols-3 gap-1.5">
        {providers.map(provider => (
          <ProviderCard key={provider.type} provider={provider} />
        ))}
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
                {ollamaModels.map(model => (
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
                No models installed. Download one below.
              </p>
            )}

            {/* Quick Download */}
            {(() => {
              const availableModels = POPULAR_MODELS.filter(m =>
                !ollamaModels.some(om => om.name === m.name || om.name.startsWith(m.name.split(':')[0]))
              )
              if (availableModels.length === 0) return null

              return (
                <div className="space-y-1.5">
                  <button
                    onClick={() => setShowMoreModels(!showMoreModels)}
                    className="flex items-center gap-1 text-xs font-mono text-text-muted hover:text-text-secondary transition-press"
                  >
                    {showMoreModels ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                    Download models ({availableModels.length} available)
                  </button>
                  {showMoreModels && (
                    <div className="space-y-1 pl-1">
                      {availableModels.map(model => (
                        <button
                          key={model.name}
                          onClick={() => handlePullModel(model.name)}
                          disabled={!!isPulling}
                          className="w-full flex items-center gap-2 p-1.5 border-brutal border-border rounded-brutal bg-background-card hover:shadow-brutal-sm transition-press text-left disabled:opacity-50"
                        >
                          {isPulling === model.name ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent flex-shrink-0" />
                          ) : (
                            <Download className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />
                          )}
                          <span className="text-xs font-mono text-text-primary flex-1">{model.name}</span>
                          <span className="text-[10px] font-mono text-text-muted">{model.size}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )
            })()}

            {/* Custom Model */}
            <div className="flex items-center gap-1.5 pt-1">
              <input
                type="text"
                value={customModel}
                onChange={(e) => setCustomModel(e.target.value)}
                placeholder="Custom model (e.g., mistral:7b)"
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
        <strong className="font-mono text-text-primary">Local models</strong> run on your machine for privacy and zero API costs.
        BashGym can fine-tune these using collected traces.
      </div>
    </div>
  )
}
