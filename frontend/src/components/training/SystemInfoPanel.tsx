import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Cpu, HardDrive, AlertCircle, CheckCircle2, XCircle, Server, Wifi } from 'lucide-react'
import { systemInfoApi, providersApi, sshApi, SystemInfo, ModelRecommendations, OllamaModel } from '../../services/api'
import { clsx } from 'clsx'

interface SystemInfoPanelProps {
  onSystemInfo?: (info: SystemInfo) => void
  onRecommendations?: (recs: ModelRecommendations) => void
  compact?: boolean
}

export function SystemInfoPanel({ onSystemInfo, onRecommendations, compact = false }: SystemInfoPanelProps) {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [recommendations, setRecommendations] = useState<ModelRecommendations | null>(null)
  const [ollamaStatus, setOllamaStatus] = useState<{ available: boolean; models: OllamaModel[]; studentModel?: string } | null>(null)
  const [sshStatus, setSshStatus] = useState<{ ok: boolean; python_version?: string; disk_free_gb?: number; error?: string; host?: string; username?: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSystemInfo = useCallback(async (refresh = false) => {
    try {
      setLoading(true)
      setError(null)

      const [infoResult, recsResult, ollamaResult] = await Promise.all([
        systemInfoApi.getInfo(refresh),
        systemInfoApi.getRecommendations(),
        providersApi.getOllamaModels()
      ])

      if (infoResult.ok && infoResult.data) {
        setSystemInfo(infoResult.data)
        onSystemInfo?.(infoResult.data)
      } else {
        setError(infoResult.error || 'Failed to detect hardware')
      }

      if (recsResult.ok && recsResult.data) {
        setRecommendations(recsResult.data)
        onRecommendations?.(recsResult.data)
      }

      if (ollamaResult.ok && ollamaResult.data) {
        setOllamaStatus({
          available: ollamaResult.data.available,
          models: ollamaResult.data.models || [],
        })
      }

      // SSH preflight (non-blocking)
      sshApi.preflight().then((result) => {
        if (result.ok && result.data) {
          setSshStatus(result.data)
        }
      }).catch(() => {
        // SSH not configured or server unavailable
      })
    } catch (err) {
      setError('Failed to connect to API')
    } finally {
      setLoading(false)
    }
  }, [onSystemInfo, onRecommendations])

  useEffect(() => {
    fetchSystemInfo()
  }, [fetchSystemInfo])

  if (loading) {
    return (
      <div className={clsx('card', compact ? 'p-3' : 'p-4')}>
        <div className="flex items-center justify-center gap-2 text-text-muted">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="font-mono text-xs uppercase tracking-widest">Detecting hardware...</span>
        </div>
      </div>
    )
  }

  if (error || !systemInfo) {
    return (
      <div className={clsx('card', compact ? 'p-3' : 'p-4')}>
        <div className="flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="font-mono text-xs flex-1">{error || 'Unknown error'}</span>
          <button
            onClick={() => fetchSystemInfo(true)}
            className="btn-secondary text-xs px-2 py-1"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const primaryGpu = systemInfo.gpus[0]
  const hasNvidiaGpu = systemInfo.gpus.some(g => g.vendor === 'NVIDIA')
  const maxVram = Math.max(...systemInfo.gpus.map(g => g.vram), 0)

  if (compact) {
    return (
      <div className="card p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={clsx('status-dot', hasNvidiaGpu ? 'status-success' : systemInfo.cuda_available ? 'status-warning' : 'status-error')} />
            <div>
              <span className="text-sm font-medium text-text-primary">{primaryGpu.model}</span>
              <span className="font-mono text-xs text-text-muted ml-2">
                {primaryGpu.vram > 0 ? `${primaryGpu.vram} GB` : 'VRAM unknown'}
              </span>
            </div>
          </div>
          <button
            onClick={() => fetchSystemInfo(true)}
            className="btn-icon w-7 h-7 flex items-center justify-center"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* GPU Section */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className={clsx('w-4 h-4', hasNvidiaGpu ? 'text-status-success' : systemInfo.cuda_available ? 'text-status-warning' : 'text-status-error')} />
          <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">GPU</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-brand text-2xl text-text-primary">{primaryGpu.model}</span>
          <span className={clsx('font-mono text-xs font-bold', primaryGpu.vram >= 8 ? 'text-status-success' : primaryGpu.vram >= 4 ? 'text-status-warning' : 'text-status-error')}>
            {primaryGpu.vram > 0 ? `${primaryGpu.vram} GB VRAM` : 'VRAM unknown'}
          </span>
        </div>

        {/* VRAM Usage Bar */}
        {primaryGpu.vram_used !== undefined && primaryGpu.vram > 0 && (
          <div className="mt-3">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${Math.min((primaryGpu.vram_used / primaryGpu.vram) * 100, 100)}%` }}
              />
            </div>
            <p className="font-mono text-xs text-text-muted mt-1">
              {primaryGpu.vram_used.toFixed(1)} / {primaryGpu.vram} GB used
            </p>
          </div>
        )}

        {/* Temperature/Utilization */}
        {(primaryGpu.temperature !== undefined || primaryGpu.utilization !== undefined) && (
          <div className="flex gap-4 mt-2">
            {primaryGpu.temperature !== undefined && (
              <span className="font-mono text-xs text-text-muted">{primaryGpu.temperature}C</span>
            )}
            {primaryGpu.utilization !== undefined && (
              <span className="font-mono text-xs text-text-muted">{primaryGpu.utilization}% util</span>
            )}
          </div>
        )}
      </div>

      {/* RAM Section */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <HardDrive className="w-4 h-4 text-accent" />
          <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">RAM</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="font-brand text-2xl text-text-primary">
            {systemInfo.available_ram.toFixed(1)} GB
          </span>
          <span className="font-mono text-xs text-text-muted">
            free / {systemInfo.total_ram.toFixed(0)} GB total
          </span>
        </div>
      </div>

      {/* Status Row */}
      <div className="flex gap-3">
        <div className="card p-3 flex items-center gap-2 flex-1">
          {systemInfo.cuda_available ? (
            <CheckCircle2 className="w-4 h-4 text-status-success" />
          ) : (
            <XCircle className="w-4 h-4 text-status-error" />
          )}
          <span className="font-mono text-xs text-text-secondary">
            CUDA {systemInfo.cuda_version || (systemInfo.cuda_available ? '' : 'N/A')}
          </span>
        </div>
        <div className="card p-3 flex items-center gap-2 flex-1">
          {systemInfo.python_available ? (
            <CheckCircle2 className="w-4 h-4 text-status-success" />
          ) : (
            <XCircle className="w-4 h-4 text-status-error" />
          )}
          <span className="font-mono text-xs text-text-secondary">
            Python {systemInfo.python_version || 'N/A'}
          </span>
        </div>
      </div>

      {/* Ollama / Remote Inference */}
      {ollamaStatus && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Server className={clsx('w-4 h-4', ollamaStatus.available ? 'text-status-success' : 'text-status-error')} />
            <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">Inference Target</span>
          </div>
          {ollamaStatus.available ? (
            <>
              <div className="flex items-center gap-2">
                <Wifi className="w-3.5 h-3.5 text-status-success" />
                <span className="text-sm font-medium text-text-primary">Ollama Connected</span>
                <span className="tag tag-accent">{ollamaStatus.models.length} model{ollamaStatus.models.length !== 1 ? 's' : ''}</span>
              </div>
              {ollamaStatus.models.length > 0 && (
                <div className="mt-2 space-y-1">
                  {ollamaStatus.models.slice(0, 3).map((m) => (
                    <div key={m.name} className="flex items-center justify-between font-mono text-xs text-text-muted">
                      <span className="flex items-center gap-1.5">
                        {m.is_code_model && <span className="text-accent">{'</>'}</span>}
                        {m.name}
                      </span>
                      <span>{m.parameter_size} · {m.size_gb.toFixed(1)}GB</span>
                    </div>
                  ))}
                  {ollamaStatus.models.length > 3 && (
                    <p className="font-mono text-xs text-text-muted">+{ollamaStatus.models.length - 3} more</p>
                  )}
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center gap-2 text-text-muted">
              <XCircle className="w-3.5 h-3.5 text-status-error" />
              <span className="text-sm">Ollama offline — start with <code className="font-mono text-xs bg-surface px-1 py-0.5">ollama serve</code></span>
            </div>
          )}
        </div>
      )}

      {/* DGX Spark Training Target */}
      {sshStatus && (
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Server className={clsx('w-4 h-4', sshStatus.ok ? 'text-status-success' : 'text-status-error')} />
            <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">Training Target</span>
          </div>
          {sshStatus.ok ? (
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Wifi className="w-3.5 h-3.5 text-status-success" />
                <span className="text-sm font-medium text-text-primary">DGX Spark Connected</span>
              </div>
              <p className="font-mono text-xs text-text-muted">
                {sshStatus.username}@{sshStatus.host}
              </p>
              <p className="font-mono text-xs text-text-muted">
                {sshStatus.python_version}
                {sshStatus.disk_free_gb && ` · ${sshStatus.disk_free_gb}GB free`}
              </p>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-text-muted">
              <XCircle className="w-3.5 h-3.5 text-status-error" />
              <span className="text-sm">{sshStatus.error || 'SSH not configured'}</span>
            </div>
          )}
        </div>
      )}

      {/* Recommendation */}
      {recommendations && (
        <div className={clsx(
          'card p-3 font-mono text-xs border-l-4',
          recommendations.warning
            ? 'border-l-status-warning text-status-warning'
            : maxVram >= 8
              ? 'border-l-status-success text-status-success'
              : 'border-l-accent text-accent'
        )}>
          {recommendations.warning || (
            maxVram >= 8
              ? 'Your GPU supports most models with QLoRA'
              : maxVram >= 4
                ? 'Your GPU can run small-medium models with QLoRA'
                : 'Limited VRAM - try smaller models'
          )}
        </div>
      )}

      {/* Refresh Button */}
      <button
        onClick={() => fetchSystemInfo(true)}
        className="btn-secondary w-full flex items-center justify-center gap-2 text-xs"
      >
        <RefreshCw className="w-3 h-3" />
        Refresh
      </button>
    </div>
  )
}
