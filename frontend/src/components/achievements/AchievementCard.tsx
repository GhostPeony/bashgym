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

const RARITY_STYLES: Record<string, { text: string; iconBg: string; tagBg: string }> = {
  common: {
    text: 'text-text-muted',
    iconBg: 'bg-background-secondary text-text-muted',
    tagBg: 'bg-background-secondary text-text-muted',
  },
  uncommon: {
    text: 'text-status-success',
    iconBg: 'bg-status-success text-white',
    tagBg: 'bg-status-success text-white',
  },
  rare: {
    text: 'text-accent',
    iconBg: 'bg-accent text-white',
    tagBg: 'bg-accent text-white',
  },
  epic: {
    text: 'text-accent-dark',
    iconBg: 'bg-accent-dark text-white',
    tagBg: 'bg-accent-dark text-white',
  },
  legendary: {
    text: 'text-status-warning',
    iconBg: 'bg-status-warning text-white',
    tagBg: 'bg-status-warning text-white',
  },
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
  name, description, category, rarity, icon, points, earned, earned_at, progress,
}: AchievementCardProps) {
  const IconComponent = ICON_MAP[icon] || Award
  const style = RARITY_STYLES[rarity] || RARITY_STYLES.common

  return (
    <div className={clsx(
      'card card-accent p-4',
      !earned && 'border-border-subtle',
    )}>
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={clsx(
          'w-10 h-10 flex items-center justify-center flex-shrink-0 border-brutal border-border rounded-brutal',
          earned ? style.iconBg : 'bg-background-tertiary text-text-muted',
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
            <h3 className={clsx(
              'font-brand text-lg truncate',
              earned ? 'text-text-primary' : 'text-text-secondary',
            )}>
              {name}
            </h3>
            {earned && <Check className="w-3.5 h-3.5 text-status-success flex-shrink-0" />}
          </div>
          <p className="text-xs text-text-muted mb-2">{description}</p>

          {/* Bottom row: rarity tag + points + earned date */}
          <div className="flex items-center gap-2">
            <span className="tag">
              <span>{rarity}</span>
            </span>
            <span className="font-mono text-xs text-text-muted">{points} pts</span>
            {earned && earned_at && (
              <span className="font-mono text-xs text-text-muted ml-auto">
                {new Date(earned_at).toLocaleDateString()}
              </span>
            )}
          </div>

          {/* Progress bar (locked only) */}
          {!earned && progress > 0 && (
            <div className="mt-2">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.round(progress * 100)}%` }}
                />
              </div>
              <p className="font-mono text-xs text-text-muted mt-1">{Math.round(progress * 100)}%</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
