import { clsx } from 'clsx'

interface StatCardProps {
  icon: React.ReactNode
  label: string
  value: string | number
  subvalue?: string
  color?: 'green' | 'blue' | 'orange' | 'purple' | 'default'
}

export function StatCard({ icon, label, value, subvalue, color = 'default' }: StatCardProps) {
  const colorClasses: Record<string, string> = {
    green: 'bg-status-success/10 text-status-success',
    blue: 'bg-status-info/10 text-status-info',
    orange: 'bg-status-warning/10 text-status-warning',
    purple: 'bg-purple-400/10 text-purple-400',
    default: 'bg-primary/10 text-primary',
  }

  return (
    <div className="card-elevated p-4 flex items-center gap-3">
      <div className={clsx('w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0', colorClasses[color])}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs text-text-muted truncate">{label}</p>
        <div className="flex items-baseline gap-1.5">
          <p className="text-xl font-semibold text-text-primary">{value}</p>
          {subvalue && <span className="text-xs text-text-muted">{subvalue}</span>}
        </div>
      </div>
    </div>
  )
}
