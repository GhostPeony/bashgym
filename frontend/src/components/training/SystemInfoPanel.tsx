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
      <div className={clsx('bg-background-tertiary rounded-lg', compact ? 'p-3' : 'p-4')}>
        <div className="flex items-center justify-center gap-2 text-text-muted">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm">Detecting hardware...</span>
        </div>
      </div>
    )
  }

  if (error || !systemInfo) {
    return (
      <div className={clsx('bg-background-tertiary rounded-lg', compact ? 'p-3' : 'p-4')}>
        <div className="flex items-center gap-2 text-status-error">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm flex-1">{error || 'Unknown error'}</span>
          <button
            onClick={() => fetchSystemInfo(true)}
            className="text-xs px-2 py-1 bg-background-secondary rounded hover:bg-background-tertiary"
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

  // GPU status indicator color
  const gpuStatusColor = hasNvidiaGpu ? 'text-status-success' : systemInfo.cuda_available ? 'text-status-warning' : 'text-status-error'

  if (compact) {
    return (
      <div className="bg-background-tertiary rounded-lg p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={clsx('w-2 h-2 rounded-full', hasNvidiaGpu ? 'bg-status-success' : systemInfo.cuda_available ? 'bg-status-warning' : 'bg-status-error')} />
            <div>
              <span className="text-sm font-medium text-text-primary">{primaryGpu.model}</span>
              <span className="text-xs text-text-muted ml-2">
                {primaryGpu.vram > 0 ? `${primaryGpu.vram} GB` : 'VRAM unknown'}
              </span>
            </div>
          </div>
          <button
            onClick={() => fetchSystemInfo(true)}
            className="p-1 rounded hover:bg-background-secondary text-text-muted"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-background-tertiary rounded-lg p-4 space-y-4">
      {/* GPU Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Cpu className={clsx('w-4 h-4', gpuStatusColor)} />
          <span className="text-xs font-semibold text-text-muted uppercase tracking-wide">GPU</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-medium text-text-primary">{primaryGpu.model}</span>
          <span className={clsx('text-xs font-medium', primaryGpu.vram >= 8 ? 'text-status-success' : primaryGpu.vram >= 4 ? 'text-status-warning' : 'text-status-error')}>
            {primaryGpu.vram > 0 ? `${primaryGpu.vram} GB VRAM` : 'VRAM unknown'}
          </span>
        </div>

        {/* VRAM Usage Bar */}
        {primaryGpu.vram_used !== undefined && primaryGpu.vram > 0 && (
          <div className="mt-2">
            <div className="h-1.5 bg-background-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${Math.min((primaryGpu.vram_used / primaryGpu.vram) * 100, 100)}%` }}
              />
            </div>
            <p className="text-xs text-text-muted mt-1">
              {primaryGpu.vram_used.toFixed(1)} / {primaryGpu.vram} GB used
            </p>
          </div>
        )}

        {/* Temperature/Utilization */}
        {(primaryGpu.temperature !== undefined || primaryGpu.utilization !== undefined) && (
          <div className="flex gap-4 mt-2 text-xs text-text-muted">
            {primaryGpu.temperature !== undefined && (
              <span>{primaryGpu.temperature}°C</span>
            )}
            {primaryGpu.utilization !== undefined && (
              <span>{primaryGpu.utilization}% util</span>
            )}
          </div>
        )}
      </div>

      {/* RAM Section */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <HardDrive className="w-4 h-4 text-accent" />
          <span className="text-xs font-semibold text-text-muted uppercase tracking-wide">RAM</span>
        </div>
        <div className="flex items-baseline gap-1">
          <span className="text-sm font-medium text-text-primary">
            {systemInfo.available_ram.toFixed(1)} GB free
          </span>
          <span className="text-xs text-text-muted">
            / {systemInfo.total_ram.toFixed(0)} GB
          </span>
        </div>
      </div>

      {/* Status Row */}
      <div className="flex gap-4 pt-3 border-t border-border-subtle">
        <div className="flex items-center gap-1.5">
          {systemInfo.cuda_available ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-status-success" />
          ) : (
            <XCircle className="w-3.5 h-3.5 text-status-error" />
          )}
          <span className="text-xs text-text-secondary">
            CUDA {systemInfo.cuda_version || (systemInfo.cuda_available ? '' : 'N/A')}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {systemInfo.python_available ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-status-success" />
          ) : (
            <XCircle className="w-3.5 h-3.5 text-status-error" />
          )}
          <span className="text-xs text-text-secondary">
            Python {systemInfo.python_version || 'N/A'}
          </span>
        </div>
      </div>

      {/* Recommendation */}
      {recommendations && (
        <div className={clsx(
          'p-2 rounded text-xs',
          recommendations.warning
            ? 'bg-status-warning/10 text-status-warning'
            : maxVram >= 8
              ? 'bg-status-success/10 text-status-success'
              : 'bg-status-info/10 text-status-info'
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
        className="w-full mt-2 py-1.5 px-3 text-xs text-text-muted border border-border-subtle rounded hover:bg-background-secondary transition-colors flex items-center justify-center gap-1.5"
      >
        <RefreshCw className="w-3 h-3" />
        Refresh
      </button>
    </div>
  )
}
