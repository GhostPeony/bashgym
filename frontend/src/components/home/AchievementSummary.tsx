import { useEffect } from 'react'
import { Trophy, ChevronRight } from 'lucide-react'
import { useAchievementStore } from '../../stores/achievementStore'
import { useUIStore } from '../../stores'
import { clsx } from 'clsx'

const RARITY_DOT: Record<string, string> = {
  common: 'bg-text-muted',
  uncommon: 'bg-status-success',
  rare: 'bg-status-info',
  epic: 'bg-purple-400',
  legendary: 'bg-yellow-400',
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
      className="w-full mt-6 p-4 rounded-xl bg-background-secondary border border-border-subtle hover:border-border-color transition-all text-left group"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Trophy className="w-4 h-4 text-yellow-400" />
          <span className="text-sm font-medium text-text-primary">Achievements</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-text-muted">{earnedCount}/{totalCount}</span>
          <span className="text-xs text-yellow-400 font-medium">{totalPoints} pts</span>
          <ChevronRight className="w-4 h-4 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
        </div>
      </div>

      {/* Recent unlocks */}
      {recentUnlocks.length > 0 ? (
        <div className="flex items-center gap-3 mt-1">
          {recentUnlocks.slice(0, 3).map(a => (
            <div key={a.id} className="flex items-center gap-1.5">
              <span className={clsx('w-2 h-2 rounded-full', RARITY_DOT[a.rarity] || RARITY_DOT.common)} />
              <span className="text-xs text-text-secondary truncate max-w-[120px]">{a.name}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-text-muted mt-1">Start collecting traces to earn achievements</p>
      )}
    </button>
  )
}
