import {
  Footprints, FolderOpen, FolderArchive, Database, Crown, Gem,
  Award, BadgeCheck, Medal, ShieldCheck, Star,
  Dumbbell, GraduationCap, GitCompare, Compass, Layers, TrendingDown, Flame,
  Sparkles, Factory, Warehouse, Rocket,
  Workflow, BrainCircuit, RefreshCcw,
  Lock, Check,
  type LucideIcon,
} from 'lucide-react'
import { clsx } from 'clsx'

const ICON_MAP: Record<string, LucideIcon> = {
  Footprints, FolderOpen, FolderArchive, Database, Crown, Gem,
  Award, BadgeCheck, Medal, ShieldCheck, Star,
  Dumbbell, GraduationCap, GitCompare, Compass, Layers, TrendingDown, Flame,
  Sparkles, Factory, Warehouse, Rocket,
  Workflow, BrainCircuit, RefreshCcw,
}

const RARITY_STYLES: Record<string, { text: string; badge: string; glow?: string }> = {
  common: { text: 'text-text-muted', badge: 'bg-text-muted/10 text-text-muted' },
  uncommon: { text: 'text-status-success', badge: 'bg-status-success/10 text-status-success' },
  rare: { text: 'text-status-info', badge: 'bg-status-info/10 text-status-info' },
  epic: { text: 'text-purple-400', badge: 'bg-purple-400/10 text-purple-400' },
  legendary: { text: 'text-yellow-400', badge: 'bg-yellow-400/10 text-yellow-400', glow: 'shadow-[0_0_12px_rgba(250,204,21,0.15)]' },
}

interface AchievementCardProps {
  id: string
  name: string
  description: string
  category: string
  rarity: string
  icon: string
  points: number
  earned: boolean
  earned_at: string | null
  progress: number
}

export function AchievementCard({
  name, description, rarity, icon, points, earned, earned_at, progress,
}: AchievementCardProps) {
  const IconComponent = ICON_MAP[icon] || Award
  const style = RARITY_STYLES[rarity] || RARITY_STYLES.common

  return (
    <div className={clsx(
      'card-elevated p-4 transition-all duration-200',
      earned ? style.glow : 'opacity-60',
    )}>
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={clsx(
          'w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0',
          earned ? `${style.badge}` : 'bg-background-tertiary text-text-muted',
        )}>
          {earned ? (
            <IconComponent className="w-5 h-5" />
          ) : (
            <Lock className="w-4 h-4" />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <h3 className={clsx('text-sm font-medium truncate', earned ? 'text-text-primary' : 'text-text-secondary')}>
              {name}
            </h3>
            {earned && <Check className="w-3.5 h-3.5 text-status-success flex-shrink-0" />}
          </div>
          <p className="text-xs text-text-muted mb-2">{description}</p>

          {/* Bottom row: rarity badge + points + earned date */}
          <div className="flex items-center gap-2">
            <span className={clsx('text-[10px] uppercase font-semibold tracking-wider', style.text)}>
              {rarity}
            </span>
            <span className="text-[10px] text-text-muted">{points} pts</span>
            {earned && earned_at && (
              <span className="text-[10px] text-text-muted ml-auto">
                {new Date(earned_at).toLocaleDateString()}
              </span>
            )}
          </div>

          {/* Progress bar (locked only) */}
          {!earned && progress > 0 && (
            <div className="mt-2">
              <div className="h-1.5 rounded-full bg-background-tertiary overflow-hidden">
                <div
                  className={clsx('h-full rounded-full transition-all', style.text.replace('text-', 'bg-'))}
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
              <p className="text-[10px] text-text-muted mt-0.5">{Math.round(progress * 100)}%</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
