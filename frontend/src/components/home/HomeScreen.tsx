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
        'card card-accent group relative flex flex-col p-6 text-left',
        primary && 'card-elevated'
      )}
    >
      <div
        className="w-12 h-12 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center mb-4"
      >
        {icon}
      </div>
      <h3 className="font-brand text-lg text-text-primary mb-1">{title}</h3>
      <p className="text-sm text-text-secondary mb-4 flex-1">{description}</p>
      {stats && (
        <p className="font-mono text-xs uppercase tracking-widest text-text-muted">{stats}</p>
      )}
      {/* CSS triangle arrow — always visible */}
      <div className="absolute bottom-6 right-6">
        <div
          className="w-0 h-0"
          style={{
            borderTop: '6px solid transparent',
            borderBottom: '6px solid transparent',
            borderLeft: '8px solid var(--text-muted)',
          }}
        />
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
          <div className="mx-auto mb-6 h-32 w-36 overflow-hidden border-brutal border-border rounded-brutal shadow-brutal">
            <img
              src="/bb.png"
              alt="Bash Gym"
              className="h-full w-auto object-cover object-right"
            />
          </div>
          <h1 className="font-brand text-3xl text-text-primary mb-3">
            Turn your coding sessions into smarter AI assistants
          </h1>
          <p className="text-text-secondary max-w-2xl mx-auto">
            Bash Gym captures your work with Claude Code, transforms it into training data,
            and fine-tunes models that learn your coding style.
          </p>
        </div>

        {/* First-time user prompt */}
        {!hasSeenIntro && (
          <div className="card mb-10 p-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <div className="mb-2">
                  <span className="tag"><span>New Here</span></span>
                </div>
                <h2 className="font-brand text-lg text-text-primary mb-1">
                  First time here?
                </h2>
                <p className="text-sm text-text-secondary">
                  Let us show you around and help you set up your first training run.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={skipTutorial}
                  className="btn btn-ghost"
                >
                  Skip
                </button>
                <button
                  onClick={startTutorial}
                  className="btn btn-primary"
                >
                  Let's Go
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Section label */}
        <div className="mb-4">
          <span className="tag"><span>Spaces</span></span>
        </div>
        <h2 className="font-brand text-2xl text-text-primary mb-6">Choose Your Workspace</h2>

        {/* Space Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <SpaceCard
            icon={<Terminal className="w-6 h-6 text-accent-dark" />}
            title="Workspace"
            description="Code with AI assistance. Your sessions are automatically captured for training."
            stats={stats.sessions > 0 ? `${stats.sessions} active sessions` : 'Start coding'}
            onClick={handleEnterWorkspace}
            primary
            accentColor=""
          />
          <SpaceCard
            icon={<Sparkles className="w-6 h-6 text-accent-dark" />}
            title="Data Factory"
            description="Transform captured traces into high-quality training examples."
            stats={stats.traces > 0 ? `${stats.traces} traces / ${stats.examples} examples` : 'No traces yet'}
            onClick={() => openOverlay('factory')}
            accentColor=""
          />
          <SpaceCard
            icon={<BarChart3 className="w-6 h-6 text-accent-dark" />}
            title="Training"
            description="Fine-tune models on your data using SFT, DPO, or GRPO strategies."
            stats={stats.models > 0 ? `${stats.models} models trained` : 'Ready to train'}
            onClick={() => openOverlay('training')}
            accentColor=""
          />
        </div>

        {/* Quick Stats Bar */}
        {(stats.traces > 0 || stats.models > 0) && (
          <div className="card flex items-center justify-center gap-8 p-4 mb-8">
            <div className="text-center">
              <p className="font-brand text-3xl text-text-primary">{stats.traces}</p>
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Traces Captured</p>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <p className="font-brand text-3xl text-text-primary">{stats.examples}</p>
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Training Examples</p>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <p className="font-brand text-3xl text-text-primary">{stats.models}</p>
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Models Trained</p>
            </div>
          </div>
        )}

        {/* Achievement Summary */}
        <AchievementSummary />

      </div>
    </div>
  )
}
