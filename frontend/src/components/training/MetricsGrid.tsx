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
  const colorClasses = {
    green: 'bg-nvidia-green/10 text-nvidia-green',
    blue: 'bg-status-info/10 text-status-info',
    orange: 'bg-status-warning/10 text-status-warning',
    purple: 'bg-[#BF5AF2]/10 text-[#BF5AF2]'
  }

  const waitingClasses = isWaiting ? 'opacity-50' : ''

  return (
    <div className={clsx('card-elevated p-4 flex items-center gap-4', waitingClasses)}>
      <div className={clsx('w-12 h-12 rounded-xl flex items-center justify-center', colorClasses[color])}>
        {icon}
      </div>
      <div className="flex-1">
        <p className="text-sm text-text-muted">{label}</p>
        <div className="flex items-baseline gap-2">
          <p className="text-2xl font-semibold text-text-primary">{value}</p>
          {subvalue && <span className="text-sm text-text-muted">{subvalue}</span>}
          {trend && !isWaiting && (
            <span
              className={clsx(
                'text-xs',
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
