import { useState, useEffect } from 'react'
import {
  Newspaper,
  Github,
  FileText,
  Lightbulb,
  ExternalLink,
  Loader2,
  AlertCircle,
  KeyRound,
} from 'lucide-react'
import { clsx } from 'clsx'
import { researchApi, type ResearchNewsItem, type ResearchAdvice } from '../../services/api'

const STRATEGIES = ['sft', 'dpo', 'grpo', 'rlvr', 'distillation']

function NeedsKey() {
  return (
    <div className="p-4 border-brutal border-border-subtle rounded-brutal bg-background-secondary flex items-start gap-3">
      <KeyRound className="w-5 h-5 text-text-muted shrink-0 mt-0.5" />
      <div>
        <p className="font-mono text-sm text-text-primary">Research feed not configured</p>
        <p className="text-xs text-text-muted mt-0.5">
          Set <code className="font-mono">FIRECRAWL_API_KEY</code> in the backend env to pull live
          training-method changes and grounded AutoResearch advice.
        </p>
      </div>
    </div>
  )
}

function NewsRow({ item }: { item: ResearchNewsItem }) {
  const Icon = item.kind === 'github' ? Github : FileText
  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block p-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary hover:border-border transition-colors group"
    >
      <div className="flex items-start gap-2">
        <Icon className="w-4 h-4 text-accent shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
              {item.source}
            </span>
            <ExternalLink className="w-3 h-3 text-text-muted opacity-0 group-hover:opacity-100" />
          </div>
          <p className="font-mono text-sm text-text-primary mt-0.5 truncate">{item.title}</p>
          {item.summary && (
            <p className="text-xs text-text-muted mt-1 line-clamp-2">{item.summary}</p>
          )}
        </div>
      </div>
    </a>
  )
}

function AdviceView({ advice }: { advice: ResearchAdvice }) {
  const suggested = advice.prior?.suggested || {}
  return (
    <div className="space-y-3 mt-3">
      <div className="p-3 border-brutal border-border rounded-brutal bg-accent-light">
        <p className="font-mono text-xs uppercase tracking-widest text-accent-dark flex items-center gap-1.5">
          <Lightbulb className="w-3.5 h-3.5" /> Suggested prior (AutoResearch can bias toward this)
        </p>
        <div className="mt-2 space-y-1">
          {Object.entries(suggested).map(([key, val]) => (
            <p key={key} className="font-mono text-sm text-text-primary">
              {key}: <span className="text-accent-dark">{JSON.stringify(val)}</span>
            </p>
          ))}
        </div>
        {advice.prior?.notes?.map((n, i) => (
          <p key={i} className="text-xs text-text-muted mt-1">
            • {n}
          </p>
        ))}
      </div>

      {advice.techniques.length > 0 && (
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
            Relevant papers
          </p>
          <div className="space-y-1.5">
            {advice.techniques.slice(0, 5).map((t, i) => (
              <a
                key={i}
                href={t.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-sm text-text-primary hover:text-accent-dark truncate"
              >
                {t.title}
              </a>
            ))}
          </div>
        </div>
      )}

      {advice.issues.length > 0 && (
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
            Relevant GitHub activity
          </p>
          <div className="space-y-1.5">
            {advice.issues.slice(0, 5).map((iss, i) => (
              <a
                key={i}
                href={iss.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block font-mono text-sm text-text-primary hover:text-accent-dark truncate"
              >
                {iss.repository}: {iss.title || iss.snippet}
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export function ResearchNewsPanel() {
  const [items, setItems] = useState<ResearchNewsItem[]>([])
  const [configured, setConfigured] = useState(true)
  const [loading, setLoading] = useState(true)

  const [baseModel, setBaseModel] = useState('google/gemma-4-31B-it')
  const [strategy, setStrategy] = useState('grpo')
  const [advice, setAdvice] = useState<ResearchAdvice | null>(null)
  const [advising, setAdvising] = useState(false)

  useEffect(() => {
    researchApi.news(20).then((r) => {
      if (r.ok && r.data) {
        setItems(r.data.items)
        setConfigured(r.data.configured)
      }
      setLoading(false)
    })
  }, [])

  const getAdvice = async () => {
    setAdvising(true)
    const r = await researchApi.advise({ base_model: baseModel, strategy })
    if (r.ok && r.data) setAdvice(r.data)
    setAdvising(false)
  }

  return (
    <div className="mt-8 grid grid-cols-2 gap-6">
      {/* News feed */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Newspaper className="w-5 h-5 text-accent" />
          <h3 className="font-brand text-lg text-text-primary">Ecosystem News</h3>
          <span className="tag"><span>RESEARCH</span></span>
        </div>
        <p className="text-xs text-text-muted mb-3">
          Latest training-method changes across unsloth / TRL / NeMo / vLLM / Liger + recent papers.
        </p>
        {loading ? (
          <div className="p-8 text-center">
            <Loader2 className="w-6 h-6 text-accent mx-auto animate-spin" />
          </div>
        ) : !configured ? (
          <NeedsKey />
        ) : items.length === 0 ? (
          <p className="font-mono text-xs text-text-muted text-center py-6">No results.</p>
        ) : (
          <div className="space-y-2 max-h-[28rem] overflow-auto">
            {items.map((it, i) => (
              <NewsRow key={`${it.url}-${i}`} item={it} />
            ))}
          </div>
        )}
      </div>

      {/* Grounded advice */}
      <div className="card p-4">
        <div className="flex items-center gap-2 mb-3">
          <Lightbulb className="w-5 h-5 text-accent" />
          <h3 className="font-brand text-lg text-text-primary">Grounded advice</h3>
        </div>
        <p className="text-xs text-text-muted mb-3">
          What the literature + GitHub suggest for a training context — a prior AutoResearch can
          bias toward.
        </p>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Base model
            </label>
            <input
              className="input text-sm w-full"
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
            />
          </div>
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
              Strategy
            </label>
            <select
              className="input text-sm w-full"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {STRATEGIES.map((s) => (
                <option key={s} value={s}>
                  {s.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>
        <button
          onClick={getAdvice}
          disabled={advising || !baseModel}
          className={clsx('btn-primary flex items-center gap-2 mt-3', advising && 'opacity-70')}
        >
          {advising ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lightbulb className="w-4 h-4" />}
          Get advice
        </button>
        {!configured && (
          <p className="font-mono text-xs text-text-muted mt-2 flex items-center gap-1.5">
            <AlertCircle className="w-3.5 h-3.5" /> Set FIRECRAWL_API_KEY for live results
          </p>
        )}
        {advice && <AdviceView advice={advice} />}
      </div>
    </div>
  )
}
