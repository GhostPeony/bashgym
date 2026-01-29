import { useState, useEffect } from 'react'
import { Terminal, Sparkles, BarChart3, ArrowRight } from 'lucide-react'
import { useUIStore } from '../../stores'
import { useTutorialStore } from '../../stores/tutorialStore'
import { systemApi, tracesApi, modelsApi } from '../../services/api'
import { AchievementSummary } from './AchievementSummary'
import { clsx } from 'clsx'

interface SpaceCardProps {
  icon: React.ReactNode
  title: string
  description: string
  stats?: string
  onClick: () => void
  primary?: boolean
  accentColor: string
}

function SpaceCard({ icon, title, description, stats, onClick, primary, accentColor }: SpaceCardProps) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'group relative flex flex-col p-6 rounded-2xl border transition-all duration-200 text-left',
        'hover:scale-[1.02] hover:shadow-lg',
        primary
          ? 'bg-gradient-to-br from-primary/10 to-primary/5 border-primary/30 hover:border-primary/50'
          : 'bg-background-secondary border-border-subtle hover:border-border-color'
      )}
    >
      <div
        className={clsx(
          'w-12 h-12 rounded-xl flex items-center justify-center mb-4',
          `bg-gradient-to-br ${accentColor}`
        )}
      >
        {icon}
      </div>
      <h3 className="text-lg font-semibold text-text-primary mb-1">{title}</h3>
      <p className="text-sm text-text-secondary mb-4 flex-1">{description}</p>
      {stats && (
        <p className="text-xs text-text-muted">{stats}</p>
      )}
      <div className="absolute bottom-6 right-6 opacity-0 group-hover:opacity-100 transition-opacity">
        <ArrowRight className="w-5 h-5 text-text-muted" />
      </div>
    </button>
  )
}

export function HomeScreen() {
  const { openOverlay, closeOverlay } = useUIStore()
  const { hasSeenIntro, startTutorial, skipTutorial } = useTutorialStore()
  const [stats, setStats] = useState({
    sessions: 0,
    traces: 0,
    examples: 0,
    models: 0,
    trainingInProgress: false
  })

  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Get system stats
        const statsResult = await systemApi.stats()
        if (statsResult.ok && statsResult.data) {
          setStats(prev => ({
            ...prev,
            traces: statsResult.data.gold_traces_count || 0,
            models: statsResult.data.models_count || 0
          }))
        }

        // Get trace count for examples estimate
        const tracesResult = await tracesApi.listRepos()
        if (tracesResult.ok && tracesResult.data) {
          const totalTraces = tracesResult.data.reduce((sum, r) => sum + (r.trace_count || 0), 0)
          setStats(prev => ({
            ...prev,
            traces: totalTraces,
            examples: totalTraces * 3 // rough estimate
          }))
        }
      } catch (e) {
        // Silently fail
      }
    }

    fetchStats()
  }, [])

  const handleEnterWorkspace = () => {
    closeOverlay() // This will show the terminal grid
  }

  return (
    <div className="h-full overflow-auto bg-background-primary">
      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="mx-auto mb-6 h-32 w-36 overflow-hidden rounded-2xl">
            <img
              src="/bb.png"
              alt="Bash Gym"
              className="h-full w-auto object-cover object-right"
            />
          </div>
          <h1 className="text-3xl font-bold text-text-primary mb-3">
            Turn your coding sessions into smarter AI assistants
          </h1>
          <p className="text-text-secondary max-w-2xl mx-auto">
            Bash Gym captures your work with Claude Code, transforms it into training data,
            and fine-tunes models that learn your coding style.
          </p>
        </div>

        {/* First-time user prompt */}
        {!hasSeenIntro && (
          <div className="mb-10 p-6 rounded-2xl bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/20">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h2 className="text-lg font-semibold text-text-primary mb-1">
                  First time here?
                </h2>
                <p className="text-sm text-text-secondary">
                  Let us show you around and help you set up your first training run.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={skipTutorial}
                  className="px-4 py-2 text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
                >
                  Skip
                </button>
                <button
                  onClick={startTutorial}
                  className="px-5 py-2.5 text-sm font-semibold rounded-lg bg-primary text-white hover:bg-primary/90 transition-colors"
                >
                  Let's Go
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Space Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <SpaceCard
            icon={<Terminal className="w-6 h-6 text-white" />}
            title="Workspace"
            description="Code with AI assistance. Your sessions are automatically captured for training."
            stats={stats.sessions > 0 ? `${stats.sessions} active sessions` : 'Start coding'}
            onClick={handleEnterWorkspace}
            primary
            accentColor="from-blue-500 to-blue-600"
          />
          <SpaceCard
            icon={<Sparkles className="w-6 h-6 text-white" />}
            title="Data Factory"
            description="Transform captured traces into high-quality training examples."
            stats={stats.traces > 0 ? `${stats.traces} traces → ${stats.examples} examples` : 'No traces yet'}
            onClick={() => openOverlay('factory')}
            accentColor="from-purple-500 to-purple-600"
          />
          <SpaceCard
            icon={<BarChart3 className="w-6 h-6 text-white" />}
            title="Training"
            description="Fine-tune models on your data using SFT, DPO, or GRPO strategies."
            stats={stats.models > 0 ? `${stats.models} models trained` : 'Ready to train'}
            onClick={() => openOverlay('training')}
            accentColor="from-green-500 to-green-600"
          />
        </div>

        {/* Quick Stats Bar */}
        {(stats.traces > 0 || stats.models > 0) && (
          <div className="flex items-center justify-center gap-8 p-4 rounded-xl bg-background-secondary border border-border-subtle">
            <div className="text-center">
              <p className="text-2xl font-bold text-text-primary">{stats.traces}</p>
              <p className="text-xs text-text-muted">Traces Captured</p>
            </div>
            <div className="w-px h-8 bg-border-subtle" />
            <div className="text-center">
              <p className="text-2xl font-bold text-text-primary">{stats.examples}</p>
              <p className="text-xs text-text-muted">Training Examples</p>
            </div>
            <div className="w-px h-8 bg-border-subtle" />
            <div className="text-center">
              <p className="text-2xl font-bold text-text-primary">{stats.models}</p>
              <p className="text-xs text-text-muted">Models Trained</p>
            </div>
          </div>
        )}

        {/* Achievement Summary */}
        <AchievementSummary />

      </div>
    </div>
  )
}
