import { useState, useEffect } from 'react'
import {
  Terminal, Sparkles, BarChart3, GitBranch, Trophy, Database, ArrowRight,
  FileStack, Layers, FlaskConical, Shield, Activity, Network, Link2, Cloud,
  Rocket
} from 'lucide-react'
import { useUIStore } from '../../stores'
import { useAchievementStore } from '../../stores/achievementStore'
import { systemApi, tracesApi } from '../../services/api'
import { clsx } from 'clsx'

function SpaceCard({ icon, title, desc, stat, onClick, primary, size = 'normal' }: {
  icon: React.ReactNode; title: string; desc: string; stat?: string; onClick: () => void; primary?: boolean; size?: 'normal' | 'compact'
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'card card-accent group relative flex flex-col text-left',
        primary && 'card-elevated',
        size === 'compact' ? 'p-4' : 'p-5'
      )}
    >
      <div className={clsx(
        'border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center mb-2.5',
        size === 'compact' ? 'w-8 h-8' : 'w-9 h-9'
      )}>
        {icon}
      </div>
      <h3 className={clsx(
        'font-brand text-text-primary mb-1',
        size === 'compact' ? 'text-sm' : 'text-base'
      )}>{title}</h3>
      <p className={clsx(
        'text-text-secondary flex-1 leading-relaxed',
        size === 'compact' ? 'text-xs mb-2' : 'text-sm mb-3'
      )}>{desc}</p>
      {stat && (
        <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">{stat}</p>
      )}
      <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <ArrowRight className="w-3.5 h-3.5 text-accent" />
      </div>
    </button>
  )
}

export function HomeScreen() {
  const { openOverlay, closeOverlay } = useUIStore()
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
    <div className="h-full bg-background-primary overflow-auto">
      <div className="max-w-[1100px] mx-auto px-8 py-8">

        {/* Header row — brand + stats */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <div className="h-14 w-16 overflow-hidden border-brutal border-border rounded-brutal shadow-brutal shrink-0">
              <img src="/bb.png" alt="Bash Gym" className="h-full w-auto object-cover object-right" />
            </div>
            <div>
              <h1 className="font-brand text-2xl text-text-primary leading-tight">Bash Gym</h1>
              <p className="text-sm text-text-secondary mt-0.5">Self-improving agentic development gym</p>
            </div>
          </div>
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

        {/* Empty-state onboarding banner */}
        {stats.traces === 0 && (
          <div className="mb-8 p-5 border-brutal border-accent rounded-brutal bg-accent-light/30">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 border-brutal border-accent rounded-brutal bg-accent-light flex items-center justify-center shrink-0">
                <Rocket className="w-5 h-5 text-accent-dark" />
              </div>
              <div className="flex-1">
                <h3 className="font-brand text-base text-text-primary mb-1">New here? Get started in minutes</h3>
                <p className="text-sm text-text-secondary mb-3">
                  Install hooks and start using Claude Code to capture your first traces. Each session automatically becomes training data for your personal coding assistant.
                </p>
                <button
                  onClick={() => useUIStore.getState().setOnboardingOpen(true)}
                  className="btn-primary font-mono text-xs px-4 py-2"
                >
                  Open Getting Started Guide
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Section: Core Spaces */}
        <div className="mb-8">
          <h2 className="tag mb-4"><span>CORE</span></h2>
          <div className="grid grid-cols-2 gap-5">
            <SpaceCard
              icon={<Terminal className="w-5 h-5 text-accent-dark" />}
              title="Workspace"
              desc="Multi-terminal grid with live status detection, canvas view, and file browser. Sessions are automatically captured as training traces."
              stat="Terminal grid + canvas"
              onClick={closeOverlay}
              primary
            />
            <SpaceCard
              icon={<Sparkles className="w-5 h-5 text-accent-dark" />}
              title="Data Factory"
              desc="Import traces from Claude Code sessions, segment them into training examples, augment with synthetic data, and export to NeMo JSONL format."
              stat={stats.traces > 0 ? `${stats.traces} traces / ~${stats.examples} examples` : 'No traces yet'}
              onClick={() => openOverlay('factory')}
            />
            <SpaceCard
              icon={<BarChart3 className="w-5 h-5 text-accent-dark" />}
              title="Training"
              desc="Fine-tune Qwen, Llama, or custom base models with SFT, DPO, or GRPO. LoRA/QLoRA with Unsloth acceleration. Live loss curves and checkpoint management."
              stat={stats.models > 0 ? `${stats.models} models trained` : 'Ready to train'}
              onClick={() => openOverlay('training')}
            />
            <SpaceCard
              icon={<Network className="w-5 h-5 text-accent-dark" />}
              title="Orchestrator"
              desc="Write a spec, decompose into a dependency graph, and execute tasks across parallel Claude agents in isolated git worktrees. Real-time progress tracking."
              stat="Multi-agent pipeline"
              onClick={() => openOverlay('orchestrator')}
            />
          </div>
        </div>

        {/* Section: Library & Tools */}
        <div className="mb-8">
          <h2 className="tag mb-4"><span>LIBRARY & TOOLS</span></h2>
          <div className="grid grid-cols-3 gap-4">
            <SpaceCard
              icon={<FileStack className="w-4 h-4 text-accent-dark" />}
              title="Traces"
              desc="Browse captured sessions by repo. View quality scores, tool usage breakdowns, and promote traces to gold for training."
              stat={stats.traces > 0 ? `${stats.traces} total` : undefined}
              onClick={() => openOverlay('traces')}
              size="compact"
            />
            <SpaceCard
              icon={<Layers className="w-4 h-4 text-accent-dark" />}
              title="Models"
              desc="Browse trained models and checkpoints. Compare performance across runs, view training curves, and export to HuggingFace."
              stat={stats.models > 0 ? `${stats.models} models` : undefined}
              onClick={() => openOverlay('models')}
              size="compact"
            />
            <SpaceCard
              icon={<FlaskConical className="w-4 h-4 text-accent-dark" />}
              title="Evaluator"
              desc="Run evaluation suites against your models. Benchmark on coding tasks, measure pass rates, and compare teacher vs student."
              onClick={() => openOverlay('evaluator')}
              size="compact"
            />
            <SpaceCard
              icon={<GitBranch className="w-4 h-4 text-accent-dark" />}
              title="Router"
              desc="Configure routing between teacher (Claude) and student (fine-tuned) models. Confidence-based, round-robin, or manual strategies."
              onClick={() => openOverlay('router')}
              size="compact"
            />
            <SpaceCard
              icon={<Shield className="w-4 h-4 text-accent-dark" />}
              title="Guardrails"
              desc="Define safety boundaries with Colang configs. Content filtering, tool restrictions, and output validation for student models."
              onClick={() => openOverlay('guardrails')}
              size="compact"
            />
            <SpaceCard
              icon={<Activity className="w-4 h-4 text-accent-dark" />}
              title="Profiler"
              desc="Monitor GPU utilization, VRAM usage, and training throughput. Identify bottlenecks and optimize batch sizes for your hardware."
              onClick={() => openOverlay('profiler')}
              size="compact"
            />
          </div>
        </div>

        {/* Section: Connections */}
        <div>
          <h2 className="tag mb-4"><span>CONNECTIONS</span></h2>
          <div className="grid grid-cols-3 gap-4">
            <SpaceCard
              icon={<Cloud className="w-4 h-4 text-accent-dark" />}
              title="HuggingFace"
              desc="Push trained models and datasets to HuggingFace Hub. Manage repos, model cards, and versioning."
              onClick={() => openOverlay('huggingface')}
              size="compact"
            />
            <SpaceCard
              icon={<Link2 className="w-4 h-4 text-accent-dark" />}
              title="Integration"
              desc="Connect to NVIDIA NIM for inference, configure API endpoints, and manage Claude API keys for the teacher model."
              onClick={() => openOverlay('integration')}
              size="compact"
            />
            <SpaceCard
              icon={<Trophy className="w-4 h-4 text-accent-dark" />}
              title="Achievements"
              desc="Track milestones across the flywheel — traces captured, models trained, evaluations passed. Earn points as you progress."
              stat={earnedCount > 0 ? `${earnedCount}/${totalCount} earned` : undefined}
              onClick={() => openOverlay('achievements')}
              size="compact"
            />
          </div>
        </div>

      </div>
    </div>
  )
}
