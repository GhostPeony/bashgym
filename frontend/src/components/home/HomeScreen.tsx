import { useState, useEffect } from 'react'
import { Terminal, Sparkles, BarChart3, GitBranch, Trophy, MessageSquare, Database, ArrowRight } from 'lucide-react'
import { useUIStore } from '../../stores'
import { useAchievementStore } from '../../stores/achievementStore'
import { systemApi, tracesApi } from '../../services/api'
import { clsx } from 'clsx'

function SpaceCard({ icon, title, desc, stat, onClick, primary }: {
  icon: React.ReactNode; title: string; desc: string; stat?: string; onClick: () => void; primary?: boolean
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'card card-accent group relative flex flex-col p-6 text-left',
        primary && 'card-elevated'
      )}
    >
      <div className="w-10 h-10 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center mb-3">
        {icon}
      </div>
      <h3 className="font-brand text-lg text-text-primary mb-1">{title}</h3>
      <p className="text-sm text-text-secondary mb-3 flex-1 leading-relaxed">{desc}</p>
      {stat && (
        <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">{stat}</p>
      )}
      <div className="absolute bottom-5 right-5 opacity-0 group-hover:opacity-100 transition-opacity">
        <ArrowRight className="w-4 h-4 text-accent" />
      </div>
    </button>
  )
}

const AGENT_CAPABILITIES = [
  {
    label: 'Plan training runs',
    detail: 'Recommends strategy, model, and hyperparameters based on your traces and GPU',
  },
  {
    label: 'Schedule data generation',
    detail: 'Knows your trace inventory and suggests what to synthesize next',
  },
  {
    label: 'Check bandwidth',
    detail: 'Reads live VRAM, RAM, and GPU utilization to advise on capacity',
  },
  {
    label: 'Orchestrate work',
    detail: 'Write specs, review task graphs, manage multi-agent jobs',
  },
]

export function HomeScreen() {
  const { openOverlay, closeOverlay, toggleAgentChat } = useUIStore()
  const { earnedCount, totalCount, totalPoints, fetchRecent } = useAchievementStore()
  const [stats, setStats] = useState({ traces: 0, examples: 0, models: 0 })

  useEffect(() => {
    fetchRecent()
    const fetchStats = async () => {
      try {
        const statsResult = await systemApi.stats()
        if (statsResult.ok && statsResult.data) {
          setStats(prev => ({
            ...prev,
            traces: statsResult.data.gold_traces_count || 0,
            models: statsResult.data.models_count || 0
          }))
        }
        const tracesResult = await tracesApi.listRepos()
        if (tracesResult.ok && tracesResult.data) {
          const total = tracesResult.data.reduce((sum: number, r: any) => sum + (r.trace_count || 0), 0)
          setStats(prev => ({ ...prev, traces: total, examples: total * 3 }))
        }
      } catch { /* silent */ }
    }
    fetchStats()
  }, [fetchRecent])

  return (
    <div className="h-full flex flex-col bg-background-primary overflow-hidden">
      {/* Floating stat buttons — top right */}
      <div className="flex items-center justify-end px-8 pt-4 shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => openOverlay('traces')}
            className="card flex items-center gap-2 px-3 py-1.5 hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-none transition-all"
          >
            <Database className="w-3.5 h-3.5 text-accent" />
            <span className="font-mono text-xs text-text-primary">{stats.traces}</span>
            <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Traces</span>
          </button>
          <button
            onClick={() => openOverlay('achievements')}
            className="card flex items-center gap-2 px-3 py-1.5 hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-none transition-all"
          >
            <Trophy className="w-3.5 h-3.5 text-status-warning" />
            <span className="font-mono text-xs text-text-primary">{earnedCount}/{totalCount}</span>
            <span className="tag text-[9px] py-0 px-1"><span>{totalPoints} pts</span></span>
          </button>
        </div>
      </div>

      {/* Centered content block */}
      <div className="flex-1 flex flex-col items-center justify-center px-8 pb-6">
        <div className="w-full max-w-[1080px]">

          {/* Brand header — centered */}
          <div className="flex items-center gap-5 mb-8">
            <div className="h-16 w-18 overflow-hidden border-brutal border-border rounded-brutal shadow-brutal shrink-0">
              <img src="/bb.png" alt="Bash Gym" className="h-full w-auto object-cover object-right" />
            </div>
            <div>
              <h1 className="font-brand text-2xl text-text-primary leading-tight">Bash Gym</h1>
              <p className="text-sm text-text-secondary mt-1">Self-improving agentic development</p>
            </div>
          </div>

          {/* Agent + Space cards row */}
          <div className="flex gap-6 items-stretch">

            {/* Left: Agent panel */}
            <button
              onClick={toggleAgentChat}
              className="card card-elevated w-[320px] shrink-0 flex flex-col p-6 text-left group"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center">
                  <MessageSquare className="w-5 h-5 text-accent-dark" />
                </div>
                <div>
                  <h2 className="font-brand text-lg text-text-primary leading-tight">Gym Agent</h2>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">System-Aware</p>
                </div>
              </div>

              <p className="text-sm text-text-secondary mb-5 leading-relaxed">
                Your assistant for the entire pipeline — planning, data, training, and orchestration.
              </p>

              <div className="space-y-3 flex-1">
                {AGENT_CAPABILITIES.map((cap) => (
                  <div key={cap.label} className="flex items-start gap-2.5">
                    <div className="w-1 h-1 rounded-full bg-accent mt-[7px] shrink-0" />
                    <p className="text-xs text-text-secondary leading-relaxed">
                      <span className="text-text-primary font-medium">{cap.label}</span>
                      <span className="text-text-muted"> — </span>
                      {cap.detail}
                    </p>
                  </div>
                ))}
              </div>

              <div className="mt-5 pt-4 border-t border-border flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">
                  Open Chat
                </span>
                <ArrowRight className="w-4 h-4 text-accent opacity-60 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
              </div>
            </button>

            {/* Right: 2x2 space cards */}
            <div className="flex-1 grid grid-cols-2 gap-5">
              <SpaceCard
                icon={<Terminal className="w-5 h-5 text-accent-dark" />}
                title="Workspace"
                desc="Code with AI assistance. Sessions are automatically captured as training data."
                stat="Terminal grid"
                onClick={closeOverlay}
                primary
              />
              <SpaceCard
                icon={<Sparkles className="w-5 h-5 text-accent-dark" />}
                title="Data Factory"
                desc="Transform captured traces into high-quality training examples for fine-tuning."
                stat={stats.traces > 0 ? `${stats.traces} traces / ${stats.examples} examples` : 'No traces yet'}
                onClick={() => openOverlay('factory')}
              />
              <SpaceCard
                icon={<BarChart3 className="w-5 h-5 text-accent-dark" />}
                title="Training"
                desc="Fine-tune models on your data using SFT, DPO, or GRPO strategies."
                stat={stats.models > 0 ? `${stats.models} models trained` : 'Ready to train'}
                onClick={() => openOverlay('training')}
              />
              <SpaceCard
                icon={<GitBranch className="w-5 h-5 text-accent-dark" />}
                title="Orchestrator"
                desc="Decompose specs into parallel tasks. Multi-agent execution with git worktrees."
                stat="Multi-agent pipeline"
                onClick={() => openOverlay('orchestrator')}
              />
            </div>

          </div>
        </div>
      </div>
    </div>
  )
}
