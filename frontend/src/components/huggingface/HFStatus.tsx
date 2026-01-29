import { useState, useEffect } from 'react'
import { Cloud, CloudOff, Sparkles, Loader2 } from 'lucide-react'
import { hfApi, HFStatus as HFStatusType } from '../../services/api'
import { clsx } from 'clsx'

interface HFStatusProps {
  compact?: boolean
  onClick?: () => void
}

export function HFStatus({ compact = false, onClick }: HFStatusProps) {
  const [status, setStatus] = useState<HFStatusType | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStatus = async () => {
      setLoading(true)
      const result = await hfApi.getStatus()
      if (result.ok && result.data) {
        setStatus(result.data)
        setError(null)
      } else {
        setError(result.error || 'Failed to fetch HuggingFace status')
      }
      setLoading(false)
    }

    fetchStatus()
    // Refresh every 60 seconds
    const interval = setInterval(fetchStatus, 60000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className={clsx(
        'flex items-center gap-2 text-text-secondary',
        onClick && 'cursor-pointer hover:text-text-primary'
      )} onClick={onClick}>
        <Loader2 className="w-4 h-4 animate-spin" />
        {!compact && <span className="text-sm">Loading...</span>}
      </div>
    )
  }

  if (error || !status) {
    return (
      <div className={clsx(
        'flex items-center gap-2 text-text-secondary',
        onClick && 'cursor-pointer hover:text-text-primary'
      )} onClick={onClick}>
        <CloudOff className="w-4 h-4" />
        {!compact && <span className="text-sm">HF Offline</span>}
      </div>
    )
  }

  if (!status.enabled) {
    return (
      <div className={clsx(
        'flex items-center gap-2 text-text-secondary',
        onClick && 'cursor-pointer hover:text-text-primary'
      )} onClick={onClick}>
        <CloudOff className="w-4 h-4" />
        {!compact && <span className="text-sm">HF Not Configured</span>}
      </div>
    )
  }

  return (
    <div className={clsx(
      'flex items-center gap-2',
      onClick && 'cursor-pointer hover:opacity-80'
    )} onClick={onClick}>
      <Cloud className={clsx(
        'w-4 h-4',
        status.pro_enabled ? 'text-accent-primary' : 'text-status-success'
      )} />
      {!compact && (
        <>
          <span className="text-sm text-text-primary">{status.username || 'Connected'}</span>
          {status.pro_enabled && (
            <span className="flex items-center gap-1 px-1.5 py-0.5 text-xs font-medium bg-accent-primary/20 text-accent-primary rounded">
              <Sparkles className="w-3 h-3" />
              Pro
            </span>
          )}
        </>
      )}
    </div>
  )
}
