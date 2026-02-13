import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Cpu, HardDrive, AlertCircle, CheckCircle2, XCircle } from 'lucide-react'
import { systemInfoApi, SystemInfo, ModelRecommendations } from '../../services/api'
import { clsx } from 'clsx'

interface SystemInfoPanelProps {
  onSystemInfo?: (info: SystemInfo) => void
  onRecommendations?: (recs: ModelRecommendations) => void
  compact?: boolean
}

export function SystemInfoPanel({ onSystemInfo, onRecommendations, compact = false }: SystemInfoPanelProps) {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [recommendations, setRecommendations] = useState<ModelRecommendations | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchSystemInfo = useCallback(async (refresh = false) => {
    try {
      setLoading(true)
      setError(null)

      const [infoResult, recsResult] = await Promise.all([
        systemInfoApi.getInfo(refresh),
        systemInfoApi.getRecommendations()
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
