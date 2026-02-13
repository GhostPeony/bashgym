import { useEffect } from 'react'
import { Trophy, ChevronRight } from 'lucide-react'
import { useAchievementStore } from '../../stores/achievementStore'
import { useUIStore } from '../../stores'
import { clsx } from 'clsx'

const RARITY_STYLES: Record<string, string> = {
  common: 'bg-text-muted border-text-muted',
  uncommon: 'bg-status-success border-status-success',
  rare: 'bg-status-info border-status-info',
  epic: 'bg-accent border-accent',
  legendary: 'bg-status-warning border-status-warning',
}

export function AchievementSummary() {
  const { recentUnlocks, earnedCount, totalCount, totalPoints, fetchRecent } = useAchievementStore()
  const { openOverlay } = useUIStore()

  useEffect(() => {
    fetchRecent()
  }, [fetchRecent])

  return (
    <button
      onClick={() => openOverlay('achievements')}
      className="card w-full mt-6 p-4 text-left group"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Trophy className="w-4 h-4 text-status-warning" />
          <span className="font-brand text-sm text-text-primary">Achievements</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-text-muted">{earnedCount}/{totalCount}</span>
          <span className="tag text-[10px] py-0 px-1.5"><span>{totalPoints} pts</span></span>
          <ChevronRight className="w-4 h-4 text-text-muted" />
        </div>
      </div>

      {/* Recent unlocks */}
      {recentUnlocks.length > 0 ? (
        <div className="flex items-center gap-3 mt-1">
          {recentUnlocks.slice(0, 3).map(a => (
            <div key={a.id} className="flex items-center gap-1.5">
              <span className={clsx(
                'w-0 h-0',
              )} style={{
                borderLeft: '4px solid transparent',
                borderRight: '4px solid transparent',
                borderBottom: `7px solid var(--${a.rarity === 'common' ? 'text-muted' : a.rarity === 'uncommon' ? 'status-success' : a.rarity === 'rare' ? 'status-info' : a.rarity === 'epic' ? 'accent' : 'status-warning'})`,
              }} />
              <span className="text-xs font-mono text-text-secondary truncate max-w-[120px]">{a.name}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-text-muted mt-1 font-mono">Start collecting traces to earn achievements</p>
      )}
    </button>
  )
}
