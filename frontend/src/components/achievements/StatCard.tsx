import { clsx } from 'clsx'

interface StatCardProps {
  icon: React.ReactNode
  label: string
  value: string | number
  subvalue?: string
  color?: 'green' | 'blue' | 'orange' | 'purple' | 'default'
}

const COLOR_ICON_BG: Record<string, string> = {
  green: 'bg-status-success text-white',
  blue: 'bg-accent text-white',
  orange: 'bg-status-warning text-white',
  purple: 'bg-accent-dark text-white',
  default: 'bg-background-secondary text-text-primary',
}

export function StatCard({ icon, label, value, subvalue, color = 'default' }: StatCardProps) {
  return (
    <div className="card p-4 flex items-center gap-3">
      <div className={clsx(
        'w-10 h-10 flex items-center justify-center flex-shrink-0 border-brutal border-border rounded-brutal',
        COLOR_ICON_BG[color],
      )}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="font-mono text-xs uppercase tracking-widest text-text-muted truncate">{label}</p>
        <div className="flex items-baseline gap-1.5">
          <p className="font-brand text-3xl text-text-primary">{value}</p>
          {subvalue && <span className="font-mono text-xs text-text-muted">{subvalue}</span>}
        </div>
      </div>
    </div>
  )
}
