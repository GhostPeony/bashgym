import { Zap, MemoryStick, Cpu } from 'lucide-react'
import { clsx } from 'clsx'

interface ResourceTilesProps {
  tokensPerSecond?: number
  gpuMemoryGb?: number
  gpuUtilization?: number
  isWaiting?: boolean
}

interface TileProps {
  icon: React.ReactNode
  label: string
  value: string
  subvalue?: string
  color: 'green' | 'blue' | 'orange'
  isWaiting?: boolean
}

function Tile({ icon, label, value, subvalue, color, isWaiting }: TileProps) {
  const iconColor = {
    green: 'text-status-success',
    blue: 'text-accent',
    orange: 'text-status-warning'
  }[color]

  return (
    <div className={clsx('card p-4 flex items-center gap-4', isWaiting && 'opacity-50')}>
      <div
        className={clsx(
          'w-12 h-12 border-brutal border-border rounded-brutal flex items-center justify-center bg-background-secondary',
          iconColor
        )}
      >
        {icon}
      </div>
      <div className="flex-1">
        <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">{label}</p>
        <div className="flex items-baseline gap-2">
          <p className="font-brand text-3xl text-text-primary">{value}</p>
          {subvalue && <span className="font-mono text-xs text-text-muted">{subvalue}</span>}
        </div>
      </div>
    </div>
  )
}

export function ResourceTiles({ tokensPerSecond, gpuMemoryGb, gpuUtilization, isWaiting }: ResourceTilesProps) {
  const fmtTps = (v?: number) => (v === undefined ? '—' : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : Math.round(v).toString())
  const fmtGb = (v?: number) => (v === undefined ? '—' : v.toFixed(2))
  // GPU utilization is unavailable on CPU smokes and unified-memory GB10 (NVML unsupported).
  const fmtUtil = (v?: number) => (v === undefined ? 'N/A' : `${Math.round(v)}%`)

  return (
    <div className="grid grid-cols-3 gap-4">
      <Tile
        icon={<Zap className="w-6 h-6" />}
        label="Tokens / sec"
        value={fmtTps(tokensPerSecond)}
        subvalue={tokensPerSecond !== undefined ? 'tok/s' : undefined}
        color="green"
        isWaiting={isWaiting}
      />
      <Tile
        icon={<MemoryStick className="w-6 h-6" />}
        label="GPU Memory"
        value={fmtGb(gpuMemoryGb)}
        subvalue={gpuMemoryGb !== undefined ? 'GB peak' : undefined}
        color="blue"
        isWaiting={isWaiting}
      />
      <Tile
        icon={<Cpu className="w-6 h-6" />}
        label="GPU Util"
        value={fmtUtil(gpuUtilization)}
        color="orange"
        isWaiting={isWaiting}
      />
    </div>
  )
}
