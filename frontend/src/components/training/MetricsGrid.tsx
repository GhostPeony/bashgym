import { TrendingDown, Gauge, Activity, Clock } from 'lucide-react'
import { clsx } from 'clsx'

interface MetricsGridProps {
  loss?: number
  learningRate?: number
  gradNorm?: number
  eta?: string
  isWaiting?: boolean
}

interface MetricCardProps {
  icon: React.ReactNode
  label: string
  value: string | number
  subvalue?: string
  trend?: 'up' | 'down' | 'neutral'
  color: 'green' | 'blue' | 'orange' | 'purple'
  isWaiting?: boolean
}

function MetricCard({ icon, label, value, subvalue, trend, color, isWaiting }: MetricCardProps) {
  const iconColorClasses = {
    green: 'text-status-success',
    blue: 'text-accent',
    orange: 'text-status-warning',
    purple: 'text-accent-dark'
  }

  return (
    <div className={clsx(
      'card p-4 flex items-center gap-4',
      isWaiting && 'opacity-50'
    )}>
      <div className={clsx(
        'w-12 h-12 border-brutal border-border rounded-brutal flex items-center justify-center bg-background-secondary',
        iconColorClasses[color]
      )}>
        {icon}
      </div>
      <div className="flex-1">
        <p className="font-mono text-xs uppercase tracking-widest text-text-secondary">{label}</p>
        <div className="flex items-baseline gap-2">
          <p className="font-brand text-3xl text-text-primary">{value}</p>
          {subvalue && <span className="font-mono text-xs text-text-muted">{subvalue}</span>}
          {trend && !isWaiting && (
            <span
              className={clsx(
                'font-mono text-xs font-bold',
                trend === 'down' && 'text-status-success',
                trend === 'up' && 'text-status-error',
                trend === 'neutral' && 'text-text-muted'
              )}
            >
              {trend === 'down' && '↓'}
              {trend === 'up' && '↑'}
              {trend === 'neutral' && '→'}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}

export function MetricsGrid({ loss, learningRate, gradNorm, eta, isWaiting }: MetricsGridProps) {
  // Format learning rate in scientific notation
  const formatLR = (lr?: number) => {
    if (lr === undefined || lr === null) return '—'
    if (lr < 0.0001) {
      const exp = Math.floor(Math.log10(lr))
      const mantissa = lr / Math.pow(10, exp)
      return `${mantissa.toFixed(1)}e${exp}`
    }
    return lr.toFixed(6)
  }

  // Format values with fallback for undefined
  const formatLoss = (l?: number) => l !== undefined ? l.toFixed(3) : '—'
  const formatGradNorm = (g?: number) => g !== undefined ? g.toFixed(2) : '—'
  const formatEta = (e?: string) => e ?? '—'

  return (
    <div className="grid grid-cols-4 gap-4">
      <MetricCard
        icon={<TrendingDown className="w-6 h-6" />}
        label="Loss"
        value={formatLoss(loss)}
        trend={loss !== undefined ? 'down' : undefined}
        color="green"
        isWaiting={isWaiting}
      />
      <MetricCard
        icon={<Gauge className="w-6 h-6" />}
        label="Learning Rate"
        value={formatLR(learningRate)}
        color="blue"
        isWaiting={isWaiting}
      />
      <MetricCard
        icon={<Activity className="w-6 h-6" />}
        label="Gradient Norm"
        value={formatGradNorm(gradNorm)}
        trend={gradNorm !== undefined ? 'neutral' : undefined}
        color="orange"
        isWaiting={isWaiting}
      />
      <MetricCard
        icon={<Clock className="w-6 h-6" />}
        label="ETA"
        value={formatEta(eta)}
        color="purple"
        isWaiting={isWaiting}
      />
    </div>
  )
}
