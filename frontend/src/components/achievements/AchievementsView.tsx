import { useState, useEffect } from 'react'
import {
  Trophy, FileStack, Award, Dumbbell, Sparkles, Puzzle,
  RefreshCw,
} from 'lucide-react'
import { useAchievementStore } from '../../stores/achievementStore'
import { AchievementCard } from './AchievementCard'
import { StatCard } from './StatCard'
import { clsx } from 'clsx'

type CategoryFilter = 'all' | 'collection' | 'quality' | 'training' | 'factory' | 'mastery'

const CATEGORY_TABS: { id: CategoryFilter; label: string; icon: React.ReactNode }[] = [
  { id: 'all', label: 'All', icon: <Trophy className="w-3.5 h-3.5" /> },
  { id: 'collection', label: 'Collection', icon: <FileStack className="w-3.5 h-3.5" /> },
  { id: 'quality', label: 'Quality', icon: <Award className="w-3.5 h-3.5" /> },
  { id: 'training', label: 'Training', icon: <Dumbbell className="w-3.5 h-3.5" /> },
  { id: 'factory', label: 'Factory', icon: <Sparkles className="w-3.5 h-3.5" /> },
  { id: 'mastery', label: 'Mastery', icon: <Puzzle className="w-3.5 h-3.5" /> },
]

export function AchievementsView() {
  const {
    stats, achievements, earnedCount, totalCount, totalPoints,
    loading, fetchStats, fetchAchievements, refresh,
  } = useAchievementStore()
  const [category, setCategory] = useState<CategoryFilter>('all')
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => {
    fetchStats()
    fetchAchievements()
  }, [fetchStats, fetchAchievements])

  const handleRefresh = async () => {
    setRefreshing(true)
    await refresh()
    setRefreshing(false)
  }

  const filtered = category === 'all'
    ? achievements
    : achievements.filter(a => a.category === category)

  // Sort: earned first (by date desc), then locked (by progress desc)
  const sorted = [...filtered].sort((a, b) => {
    if (a.earned && !b.earned) return -1
    if (!a.earned && b.earned) return 1
    if (a.earned && b.earned) {
      return (b.earned_at || '').localeCompare(a.earned_at || '')
    }
    return b.progress - a.progress
  })

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-text-primary">Achievements</h1>
          <p className="text-sm text-text-secondary mt-1">
            {earnedCount} / {totalCount} unlocked &middot; {totalPoints} points
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg bg-background-secondary border border-border-subtle hover:border-border-color transition-colors disabled:opacity-50"
        >
          <RefreshCw className={clsx('w-4 h-4', refreshing && 'animate-spin')} />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-12 gap-6">
        {/* Left: Lifetime Stats */}
        <div className="col-span-12 lg:col-span-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted mb-3">Lifetime Stats</h2>

          {stats ? (
            <div className="space-y-6">
              {/* Traces */}
              <div>
                <h3 className="text-xs text-text-muted mb-2 px-1">Traces</h3>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard icon={<FileStack className="w-4 h-4" />} label="Total Traces" value={stats.traces.total} color="blue" />
                  <StatCard icon={<Award className="w-4 h-4" />} label="Gold" value={stats.traces.gold} color="green" />
                  <StatCard icon={<Trophy className="w-4 h-4" />} label="Best Quality" value={stats.traces.highest_quality.toFixed(2)} color="purple" />
                  <StatCard icon={<Puzzle className="w-4 h-4" />} label="Repos" value={stats.traces.unique_repos} color="default" />
                </div>
              </div>

              {/* Training */}
              <div>
                <h3 className="text-xs text-text-muted mb-2 px-1">Training</h3>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard icon={<Dumbbell className="w-4 h-4" />} label="Runs" value={stats.training.runs_completed} color="green" />
                  <StatCard
                    icon={<Sparkles className="w-4 h-4" />}
                    label="Lowest Loss"
                    value={stats.training.lowest_loss != null ? stats.training.lowest_loss.toFixed(3) : '---'}
                    color="purple"
                  />
                  <StatCard icon={<FileStack className="w-4 h-4" />} label="Examples" value={stats.training.total_examples_generated} color="blue" />
                  <StatCard icon={<Award className="w-4 h-4" />} label="Exported" value={stats.training.models_exported} color="orange" />
                </div>
              </div>

              {/* Activity */}
              <div>
                <h3 className="text-xs text-text-muted mb-2 px-1">Activity</h3>
                <div className="grid grid-cols-2 gap-2">
                  <StatCard icon={<Trophy className="w-4 h-4" />} label="Points" value={totalPoints} color="purple" />
                  <StatCard icon={<FileStack className="w-4 h-4" />} label="Days Active" value={stats.days_active} color="blue" />
                </div>
              </div>
            </div>
          ) : (
            <div className="card-elevated p-8 text-center">
              <p className="text-sm text-text-muted">{loading ? 'Loading stats...' : 'No data yet'}</p>
            </div>
          )}
        </div>

        {/* Right: Achievements */}
        <div className="col-span-12 lg:col-span-8">
          {/* Category Tabs */}
          <div className="flex items-center gap-1 mb-4 overflow-x-auto pb-1">
            {CATEGORY_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setCategory(tab.id)}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors whitespace-nowrap',
                  category === tab.id
                    ? 'bg-primary/10 text-primary'
                    : 'text-text-muted hover:text-text-secondary hover:bg-background-tertiary'
                )}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* Achievement Grid */}
          {loading && achievements.length === 0 ? (
            <div className="card-elevated p-8 text-center">
              <p className="text-sm text-text-muted">Loading achievements...</p>
            </div>
          ) : sorted.length === 0 ? (
            <div className="card-elevated p-8 text-center">
              <p className="text-sm text-text-muted">No achievements in this category</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {sorted.map(a => (
                <AchievementCard key={a.id} {...a} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
