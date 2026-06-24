import { useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  ClipboardCheck,
  ExternalLink,
  Gauge,
  GitCompare,
  Network,
  ShieldCheck,
  TerminalSquare,
} from 'lucide-react'
import { clsx } from 'clsx'

type GuideSectionId = 'start' | 'strategies' | 'metrics' | 'world-models' | 'sources'

const guideSections: { id: GuideSectionId; label: string; icon: typeof BookOpen }[] = [
  { id: 'start', label: 'Start', icon: ClipboardCheck },
  { id: 'strategies', label: 'Strategies', icon: GitCompare },
  { id: 'metrics', label: 'Metrics', icon: Gauge },
  { id: 'world-models', label: 'World Models', icon: Network },
  { id: 'sources', label: 'Sources', icon: BookOpen },
]

const setupCards = [
  {
    title: 'Use SFT to teach format first',
    body: 'Start with gold traces when the model needs your repo conventions, tool-call shape, and terminal etiquette. Keep prompts, observations, and final answers intact.',
    icon: TerminalSquare,
  },
  {
    title: 'Use DPO only with real contrasts',
    body: 'Preference pairs need the same prompt with a clearly better chosen response. Weak or mismatched pairs produce reward margins that look busy but do not move pass@k.',
    icon: GitCompare,
  },
  {
    title: 'Use GRPO when rewards vary',
    body: 'GRPO needs multiple attempts per prompt and non-zero reward variance. If every group all passes or all fails, adjust task difficulty before scaling compute.',
    icon: Activity,
  },
  {
    title: 'Keep release gates separate',
    body: 'ECHO/RWML, reward, and loss curves are diagnostics. Shipping confidence still comes from held-out pass@k, contamination checks, tamper checks, and benchmark evidence.',
    icon: ShieldCheck,
  },
]

const strategyRows = [
  {
    strategy: 'SFT',
    use: 'Gold traces, repo style, first training run',
    setup: 'QLoRA, LoRA r16-r32, 1-3 epochs, sequence length long enough for tool output',
    failure: 'Loss improves but held-out pass@k stays flat',
  },
  {
    strategy: 'DPO',
    use: 'Chosen/rejected answers for the same prompt',
    setup: 'Beta around 0.1, conservative LR, preserve both completions in context',
    failure: 'Preference accuracy stuck near 0.5 or reward margin collapses',
  },
  {
    strategy: 'GRPO',
    use: 'Verifier-backed terminal tasks with sampled attempts',
    setup: 'Group size 8-32, DAPO for long rollouts, active sampling, zero-std filtering',
    failure: 'High frac_reward_zero_std, KL spikes, timeouts, or reward hacking',
  },
  {
    strategy: 'Cascade RL',
    use: 'Domain staging from easy terminal skills to harder workflows',
    setup: 'Gate each stage, then merge with distillation or release evidence',
    failure: 'Early-stage forgetting or a domain verifier that is too easy to exploit',
  },
]

const metricRows = [
  {
    metric: 'pass@1 / pass@k',
    meaning: 'The model solves held-out terminal tasks within one or more attempts.',
    action: 'Treat this as the main progress signal for agent behavior.',
  },
  {
    metric: 'reward_std',
    meaning: 'GRPO groups have useful contrast between attempts.',
    action: 'If it is near zero, adjust task difficulty, sampling, or verifier granularity.',
  },
  {
    metric: 'frac_reward_zero_std',
    meaning: 'Share of GRPO groups where every sampled attempt received the same reward.',
    action: 'High values mean the RL signal is weak. Filter those groups or rebalance tasks.',
  },
  {
    metric: 'KL / divergence',
    meaning: 'How far the trained policy moves from its reference behavior.',
    action: 'Use beta, Binary-TV, or Binary-KL controls when the model starts drifting.',
  },
  {
    metric: 'tamper / leakage',
    meaning: 'Whether the rollout modified protected files or overlaps held-out content.',
    action: 'Block release evidence until contamination and tamper reports are clean.',
  },
]

const worldModelRows = [
  {
    name: 'ECHO',
    role: 'Auxiliary observation prediction',
    setup: 'Start at lambda 0.05. Watch observation loss beside pass@k rather than optimizing it alone.',
  },
  {
    name: 'RWML',
    role: 'Latent next-state reward',
    setup: 'Keep hard transitions, subsample easy dynamics, and track embedding-distance distributions.',
  },
  {
    name: 'DPPO replay telemetry',
    role: 'Coverage check for world-model payloads',
    setup: 'Use replay summaries to confirm masks, rewards, and observation pairs are present before trainer integration.',
  },
]

const sourceRows = [
  {
    name: 'Hugging Face TRL - SFT',
    href: 'https://huggingface.co/docs/trl/en/sft_trainer',
    leverage: 'Dataset formats, chat-template behavior, completion-only training, and PEFT adapter setup.',
  },
  {
    name: 'Hugging Face TRL - DPO',
    href: 'https://huggingface.co/docs/trl/en/dpo_trainer',
    leverage: 'Preference dataset shape, prompt/chosen/rejected requirements, and conversational formatting.',
  },
  {
    name: 'Hugging Face TRL - GRPO',
    href: 'https://huggingface.co/docs/trl/en/grpo_trainer',
    leverage: 'Reward, KL, entropy, clip, reward_std, and frac_reward_zero_std metric definitions.',
  },
  {
    name: 'Unsloth RL Guide',
    href: 'https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide',
    leverage: 'Agent-environment-reward framing and reward-design cautions for verifiable tasks.',
  },
  {
    name: 'Unsloth Advanced RL',
    href: 'https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation',
    leverage: 'GRPO loss variants, beta/KL behavior, and truncated-completion caveats.',
  },
  {
    name: 'OpenAI RFT',
    href: 'https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning',
    leverage: 'Operator loop: grader, train/validation split, job monitoring, checkpoint evaluation, and iteration.',
  },
  {
    name: 'verl',
    href: 'https://github.com/verl-project/verl',
    leverage: 'Production RL dataflows for PPO, GRPO, and backend integration patterns.',
  },
  {
    name: 'SkyRL',
    href: 'https://skyrl.readthedocs.io/',
    leverage: 'Full-stack RL training orchestration and GRPO quick-start patterns.',
  },
]

function GuideTable({
  headers,
  rows,
}: {
  headers: string[]
  rows: string[][]
}) {
  return (
    <div className="overflow-x-auto border-brutal border-border rounded-brutal bg-background-card">
      <table className="w-full min-w-[720px] text-left">
        <thead>
          <tr className="border-b border-border bg-background-tertiary">
            {headers.map((header) => (
              <th key={header} className="px-4 py-3 font-mono text-[10px] uppercase tracking-widest text-text-muted">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.join('|')} className="border-b border-border-subtle last:border-b-0">
              {row.map((cell, index) => (
                <td
                  key={`${row[0]}-${index}`}
                  className={clsx(
                    'px-4 py-3 align-top text-sm text-text-secondary',
                    index === 0 && 'font-mono text-xs font-bold uppercase tracking-wide text-text-primary'
                  )}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SectionHeader({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string
  title: string
  body: string
}) {
  return (
    <div className="border-b border-border pb-4">
      <p className="font-mono text-[10px] uppercase tracking-widest text-accent mb-2">{eyebrow}</p>
      <h2 className="font-brand text-2xl text-text-primary">{title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-text-secondary max-w-3xl">{body}</p>
    </div>
  )
}

export function TrainingGuidance() {
  const [activeSection, setActiveSection] = useState<GuideSectionId>('start')

  return (
    <div className="space-y-6">
      <div className="border-brutal border-border rounded-brutal bg-background-secondary p-5 shadow-brutal-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-accent" />
              <p className="font-mono text-xs uppercase tracking-widest text-text-muted">Training Guides</p>
            </div>
            <h2 className="font-brand text-3xl text-text-primary mt-2">Operator guidance for better training runs</h2>
            <p className="mt-2 text-sm leading-relaxed text-text-secondary max-w-3xl">
              Use these pages when choosing a strategy, setting up a run, reading metrics, or deciding whether a
              checkpoint has enough held-out evidence to trust.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:w-[440px]">
            {['SFT first', 'DPO pairs', 'GRPO variance', 'Holdout gates'].map((item) => (
              <div key={item} className="border-brutal border-border rounded-brutal bg-background-card px-3 py-2">
                <CheckCircle2 className="w-4 h-4 text-accent mb-1" />
                <p className="font-mono text-[10px] uppercase tracking-widest text-text-secondary">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <aside className="col-span-12 lg:col-span-3">
          <div className="sticky top-4 border-brutal border-border rounded-brutal bg-background-card p-2 shadow-brutal-sm">
            {guideSections.map((section) => (
              <button
                key={section.id}
                type="button"
                onClick={() => setActiveSection(section.id)}
                className={clsx(
                  'w-full flex items-center gap-2 px-3 py-2 text-left font-mono text-xs uppercase tracking-widest rounded-brutal transition-press',
                  activeSection === section.id
                    ? 'bg-accent-light text-accent-dark border-brutal border-border shadow-brutal-sm'
                    : 'text-text-secondary hover:bg-background-secondary hover:text-text-primary'
                )}
              >
                <section.icon className="w-4 h-4 flex-shrink-0" />
                <span>{section.label}</span>
              </button>
            ))}
          </div>
        </aside>

        <main className="col-span-12 lg:col-span-9 space-y-5">
          {activeSection === 'start' && (
            <>
              <SectionHeader
                eyebrow="First run"
                title="What matters before pressing Start Training"
                body="The best setup guidance belongs where decisions happen. These are the four checks most likely to prevent wasted runs."
              />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {setupCards.map((card) => (
                  <div key={card.title} className="card p-4">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 border-brutal border-border rounded-brutal bg-background-secondary flex items-center justify-center flex-shrink-0">
                        <card.icon className="w-5 h-5 text-accent" />
                      </div>
                      <div>
                        <h3 className="font-brand text-lg text-text-primary">{card.title}</h3>
                        <p className="mt-1 text-sm leading-relaxed text-text-secondary">{card.body}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="border-l-4 border-l-accent border-brutal border-border rounded-brutal bg-background-card p-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-mono text-xs uppercase tracking-widest text-text-primary">Best default path</h3>
                    <p className="mt-1 text-sm leading-relaxed text-text-secondary">
                      Run SFT on clean gold traces, evaluate held-out pass@k, then use DPO or GRPO only when the
                      dataset and verifier produce a real contrast signal. World-model objectives should explain
                      dynamics, not replace release evidence.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}

          {activeSection === 'strategies' && (
            <>
              <SectionHeader
                eyebrow="Strategy selection"
                title="Pick the objective that matches your evidence"
                body="Each training method needs a different kind of data. The platform should steer users toward the objective their evidence can actually support."
              />
              <GuideTable
                headers={['Strategy', 'Use when', 'Setup bias', 'Failure smell']}
                rows={strategyRows.map((row) => [row.strategy, row.use, row.setup, row.failure])}
              />
            </>
          )}

          {activeSection === 'metrics' && (
            <>
              <SectionHeader
                eyebrow="Run reading"
                title="Metrics that deserve UI space"
                body="Loss is useful, but agent training succeeds or fails on held-out behavior, reward diversity, divergence, and environment integrity."
              />
              <GuideTable
                headers={['Metric', 'What it means', 'Operator action']}
                rows={metricRows.map((row) => [row.metric, row.meaning, row.action])}
              />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[
                  ['Green light', 'pass@k improves, reward_std stays healthy, KL remains bounded.'],
                  ['Yellow light', 'loss improves but held-out behavior is flat or truncation rises.'],
                  ['Red light', 'tamper, leakage, zero reward variance, or verifier exploit evidence.'],
                ].map(([title, body]) => (
                  <div key={title} className="card p-4">
                    <p className="font-mono text-xs uppercase tracking-widest text-text-primary">{title}</p>
                    <p className="mt-2 text-sm leading-relaxed text-text-secondary">{body}</p>
                  </div>
                ))}
              </div>
            </>
          )}

          {activeSection === 'world-models' && (
            <>
              <SectionHeader
                eyebrow="ECHO / RWML"
                title="Treat world models as diagnostics until validated"
                body="World-model objectives can help terminal agents learn state transitions, but they are auxiliary. They need replay coverage and held-out correlation before becoming release criteria."
              />
              <GuideTable
                headers={['Signal', 'Role', 'Setup']}
                rows={worldModelRows.map((row) => [row.name, row.role, row.setup])}
              />
              <div className="border-brutal border-border rounded-brutal bg-background-secondary p-4">
                <p className="font-mono text-xs uppercase tracking-widest text-text-primary">Surface in setup</p>
                <p className="mt-2 text-sm leading-relaxed text-text-secondary">
                  Show ECHO lambda, RWML distance, easy-transition filtering, history window, and replay coverage near
                  the GRPO settings because those values only make sense in the rollout context.
                </p>
              </div>
            </>
          )}

          {activeSection === 'sources' && (
            <>
              <SectionHeader
                eyebrow="Source-backed defaults"
                title="External guidance worth leveraging"
                body="The in-product guidance follows patterns from official or primary training platform resources, adapted to BashGym's terminal-agent workflow."
              />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {sourceRows.map((source) => (
                  <a
                    key={source.href}
                    href={source.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="card p-4 transition-press hover:shadow-none"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <h3 className="font-brand text-lg text-text-primary">{source.name}</h3>
                        <p className="mt-1 text-sm leading-relaxed text-text-secondary">{source.leverage}</p>
                      </div>
                      <ExternalLink className="w-4 h-4 text-accent flex-shrink-0 mt-1" />
                    </div>
                  </a>
                ))}
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  )
}
