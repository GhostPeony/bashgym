import { useEffect } from 'react'
import { Sparkles, Loader2 } from 'lucide-react'
import { clsx } from 'clsx'
import { GhostPeonyIcon } from '../common/GhostPeonyIcon'
import { hfStatusResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'

interface HFStatusProps {
  compact?: boolean
  onClick?: () => void
}

export function HFStatus({ compact = false, onClick }: HFStatusProps) {
  const { data: status, loading, error } = useSessionResource(hfStatusResource)

  useEffect(() => {
    // Refresh every 60 seconds while mounted
    const interval = setInterval(() => {
      void hfStatusResource.getState().refresh()
    }, 60000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div
        className={clsx(
          'flex items-center gap-2 text-text-secondary',
          onClick && 'cursor-pointer hover:text-text-primary'
        )}
        onClick={onClick}
      >
        <Loader2 className="w-4 h-4 animate-spin" />
        {!compact && <span className="text-sm font-mono">Loading...</span>}
      </div>
    )
  }

  if (error || !status) {
    return (
      <div
        className={clsx(
          'flex items-center gap-2 text-text-secondary',
          onClick && 'cursor-pointer hover:text-text-primary'
        )}
        onClick={onClick}
      >
        <GhostPeonyIcon name="huggingface" size="xs" tone="neutral" muted />
        {!compact && <span className="text-sm font-mono">HF Offline</span>}
      </div>
    )
  }

  if (!status.enabled) {
    return (
      <div
        className={clsx(
          'flex items-center gap-2 text-text-secondary',
          onClick && 'cursor-pointer hover:text-text-primary'
        )}
        onClick={onClick}
      >
        <GhostPeonyIcon name="huggingface" size="xs" tone="neutral" muted />
        {!compact && <span className="text-sm font-mono">HF Not Configured</span>}
      </div>
    )
  }

  return (
    <div
      className={clsx('flex items-center gap-2', onClick && 'cursor-pointer hover-press')}
      onClick={onClick}
    >
      <GhostPeonyIcon
        name="huggingface"
        size="xs"
        tone={status.pro_enabled ? 'accent' : 'neutral'}
        hue={52}
        active={status.pro_enabled}
      />
      {!compact && (
        <>
          <span className="text-sm text-text-primary font-mono">
            {status.username || 'Connected'}
          </span>
          {status.pro_enabled && (
            <span className="tag">
              <span className="flex items-center gap-1">
                <Sparkles className="w-3 h-3" />
                Pro
              </span>
            </span>
          )}
        </>
      )}
    </div>
  )
}
