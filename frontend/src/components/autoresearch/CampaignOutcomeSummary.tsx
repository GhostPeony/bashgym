import { useMemo } from 'react'
import { AlertTriangle, CheckCircle2, CircleDashed, TrendingDown, TrendingUp } from 'lucide-react'
import { clsx } from 'clsx'

import { LossCurve } from '../training/LossCurve'
import type { CampaignOutcomeMetric, CampaignOutcomeViewModel } from './campaignOutcomeModel'

function readable(value: string): string {
  const words = value.replace(/[_-]+/g, ' ').trim()
  return words ? `${words[0]!.toUpperCase()}${words.slice(1)}` : 'Unknown'
}

function isRateMetric(metric: CampaignOutcomeMetric): boolean {
  return /accuracy|reward|rate|mrr|pass|precision|recall|f1/i.test(metric.id)
    && [metric.baseline, metric.candidate].every((value) => value == null || (value >= 0 && value <= 1))
}

function formatValue(metric: CampaignOutcomeMetric, value: number | null): string {
  if (value == null) return '—'
  return isRateMetric(metric)
    ? `${(value * 100).toFixed(1)}%`
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value)
}

function formatDelta(metric: CampaignOutcomeMetric): string {
  if (metric.delta == null) return '—'
  const sign = metric.delta > 0 ? '+' : metric.delta < 0 ? '−' : ''
  const magnitude = Math.abs(metric.delta)
  return isRateMetric(metric)
    ? `${sign}${(magnitude * 100).toFixed(1)} pp`
    : `${sign}${magnitude.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')}`
}

function improves(metric: CampaignOutcomeMetric): boolean | null {
  if (metric.delta == null || metric.direction === 'unknown') return null
  if (metric.delta === 0) return false
  return metric.direction === 'maximize' ? metric.delta > 0 : metric.delta < 0
}

function verdictClasses(verdict: CampaignOutcomeViewModel['verdict']): string {
  if (verdict === 'success') return 'border-status-success bg-status-success/10 text-status-success'
  if (verdict === 'not_improved') return 'border-status-warning bg-status-warning/10 text-status-warning'
  if (verdict === 'pending') return 'border-status-warning bg-status-warning/10 text-status-warning'
  return 'border-border-subtle bg-background-secondary text-text-muted'
}

function OutcomeIcon({ verdict, className }: { verdict: CampaignOutcomeViewModel['verdict']; className?: string }) {
  if (verdict === 'success') return <CheckCircle2 className={className} />
  if (verdict === 'not_improved' || verdict === 'pending') return <AlertTriangle className={className} />
  return <CircleDashed className={className} />
}

function LossSparkline({ model }: { model: CampaignOutcomeViewModel }) {
  const points = useMemo(() => model.loss?.points || [], [model.loss?.points])
  const coordinates = useMemo(() => {
    if (!points.length) return ''
    const values = points.map((point) => point.value)
    const minimum = Math.min(...values)
    const maximum = Math.max(...values)
    const range = maximum - minimum || 1
    return points.map((point, index) => {
      const x = points.length === 1 ? 50 : (index / (points.length - 1)) * 100
      const y = 22 - ((point.value - minimum) / range) * 18
      return `${x.toFixed(2)},${y.toFixed(2)}`
    }).join(' ')
  }, [points])
  if (!model.loss) return null
  return (
    <svg
      viewBox="0 0 100 26"
      preserveAspectRatio="none"
      className="h-8 w-full border-t border-border-subtle pt-1"
      role="img"
      aria-label={`Training loss sparkline, ${points.length} points, minimum ${model.loss.minimum.value} at step ${model.loss.minimum.step}, final ${model.loss.final.value} at step ${model.loss.final.step}`}
    >
      <polyline points={coordinates} fill="none" stroke="var(--accent)" strokeWidth="2.5" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function CompactOutcome({ model }: { model: CampaignOutcomeViewModel }) {
  const primary = model.metrics.find((metric) => metric.primary) || model.metrics[0]
  return (
    <section
      className="overflow-hidden rounded-brutal border-brutal border-border bg-background-card"
      aria-label={`${model.baselineLabel} versus ${model.candidateLabel} outcome`}
    >
      <div className="flex items-center justify-between gap-2 border-b border-border-subtle px-2 py-1.5">
        <div className={clsx('flex min-w-0 items-center gap-1.5 font-mono text-[9px] font-bold uppercase', model.verdict === 'success' ? 'text-status-success' : 'text-status-warning')}>
          <OutcomeIcon verdict={model.verdict} className="h-3 w-3 shrink-0" />
          <span className="truncate">{model.verdictLabel}</span>
        </div>
        <span className="shrink-0 font-mono text-[8px] uppercase text-text-muted">{model.lifecycleLabel}</span>
      </div>
      {primary ? (
        <div className="grid grid-cols-[1fr_auto] items-end gap-2 px-2 py-1.5">
          <div className="min-w-0">
            <div className="truncate font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">{readable(primary.id)}</div>
            <div className="font-mono text-[10px] font-bold text-text-primary">
              {formatValue(primary, primary.baseline)} → {formatValue(primary, primary.candidate)}
            </div>
          </div>
          <div className={clsx('font-mono text-[11px] font-black', improves(primary) === false ? 'text-status-error' : 'text-status-success')}>
            {formatDelta(primary)}
          </div>
        </div>
      ) : null}
      <LossSparkline model={model} />
    </section>
  )
}

function DetailedOutcome({ model }: { model: CampaignOutcomeViewModel }) {
  const chartData = useMemo(
    () => model.loss?.points.map((point) => ({ step: point.step, loss: point.value })) || [],
    [model.loss?.points],
  )
  return (
    <section
      className="rounded-brutal border-brutal border-border bg-background-card"
      aria-label={`${model.baselineLabel} versus ${model.candidateLabel} outcome`}
    >
      <header className="flex flex-col gap-3 border-b border-border-subtle p-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="font-mono text-[10px] font-bold uppercase tracking-[0.18em] text-text-muted">Campaign outcome</div>
          <h2 className="mt-1 font-brand text-xl text-text-primary">{model.baselineLabel} vs {model.candidateLabel}</h2>
          <p className="mt-1 text-xs text-text-secondary">
            {model.sameEvaluationSuite === true
              ? `Same fixed evaluation suite${model.evaluationSuiteId ? ` · ${model.evaluationSuiteId}` : ''}`
              : model.sameEvaluationSuite === false
                ? 'Evaluation suites differ; deltas are not a valid comparison.'
                : 'Evaluation-suite comparability is not yet verified.'}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className={clsx('inline-flex items-center gap-1.5 rounded-brutal border-brutal px-2.5 py-1.5 font-mono text-[10px] font-bold uppercase', verdictClasses(model.verdict))}>
            <OutcomeIcon verdict={model.verdict} className="h-3.5 w-3.5" />{model.verdictLabel}
          </span>
          <span className="rounded-brutal border-brutal border-border-subtle bg-background-secondary px-2.5 py-1.5 font-mono text-[10px] font-bold uppercase text-text-secondary">
            {model.lifecycleLabel}
          </span>
        </div>
      </header>

      <div className="grid gap-px bg-border-subtle lg:grid-cols-[minmax(0,1.2fr)_minmax(20rem,0.8fr)]">
        <div className="min-w-0 bg-background-card p-4">
          <div className="grid grid-cols-[minmax(8rem,1fr)_5.5rem_6.5rem_5.5rem] gap-2 border-b-2 border-border pb-2 font-mono text-[10px] font-bold uppercase tracking-wide text-text-muted">
            <span>Metric</span><span className="text-right">{model.baselineLabel}</span><span className="text-right">{model.candidateLabel}</span><span className="text-right">Change</span>
          </div>
          <div className="divide-y divide-border-subtle">
            {model.metrics.map((metric) => {
              const improvement = improves(metric)
              return (
                <div key={metric.id} className={clsx('grid grid-cols-[minmax(8rem,1fr)_5.5rem_6.5rem_5.5rem] items-baseline gap-2 py-3', metric.primary && 'font-bold')}>
                  <span className="truncate text-sm text-text-primary" title={readable(metric.id)}>{readable(metric.id)}{metric.primary ? <span className="ml-1 font-mono text-[10px] uppercase text-accent">Primary</span> : null}</span>
                  <span className="text-right font-mono text-xs text-text-secondary">{formatValue(metric, metric.baseline)}</span>
                  <span className="text-right font-mono text-xs text-text-primary">{formatValue(metric, metric.candidate)}</span>
                  <span className={clsx('flex items-center justify-end gap-1 text-right font-mono text-xs', improvement === true ? 'text-status-success' : improvement === false ? 'text-status-error' : 'text-text-secondary')}>
                    {improvement === true ? (metric.direction === 'minimize' ? <TrendingDown className="h-3 w-3" /> : <TrendingUp className="h-3 w-3" />) : null}
                    {formatDelta(metric)}
                  </span>
                </div>
              )
            })}
          </div>
          <div className="mt-2 grid gap-1 border-t border-border-subtle pt-3 font-mono text-[10px] text-text-muted sm:grid-cols-2">
            <div className="truncate" title={model.baselineEvaluationId || undefined}>Baseline eval · {model.baselineEvaluationId || 'not linked'}</div>
            <div className="truncate" title={model.candidateEvaluationId || undefined}>Candidate eval · {model.candidateEvaluationId || 'not linked'}</div>
          </div>
        </div>

        <div className="min-w-0 bg-background-secondary p-4">
          <div className="flex items-start justify-between gap-3">
            <div><div className="font-brand text-base text-text-primary">Training loss</div><div className="font-mono text-[10px] text-text-muted">{model.loss?.attemptId || 'No persisted series'}</div></div>
            {model.loss ? <div className="text-right font-mono text-[10px] text-text-muted"><div>Minimum {model.loss.minimum.value.toFixed(4)} at step {model.loss.minimum.step}</div><div>Final {model.loss.final.value.toFixed(4)} at step {model.loss.final.step}</div></div> : null}
          </div>
          {model.loss && chartData.length > 1 ? (
            <div
              className="mt-3 h-48 border-brutal border-border bg-background-card p-2"
              role="img"
              aria-label={`Training loss curve, ${chartData.length} points, minimum ${model.loss.minimum.value} at step ${model.loss.minimum.step}, final ${model.loss.final.value} at step ${model.loss.final.step}`}
            >
              <LossCurve data={chartData} smoothed={chartData.length > 8} />
            </div>
          ) : <div className="mt-3 border-brutal border-border-subtle bg-background-card p-4 text-xs text-text-muted">No persisted loss series is available.</div>}
          {model.checkpointWarning ? (
            <div className="mt-3 flex items-start gap-2 border-l-4 border-status-warning bg-status-warning/10 px-3 py-2 text-xs leading-5 text-text-secondary">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-status-warning" />
              <span>{model.checkpointWarning}</span>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  )
}

export function CampaignOutcomeSummary({ model, density = 'detailed' }: { model: CampaignOutcomeViewModel; density?: 'compact' | 'detailed' }) {
  return density === 'compact' ? <CompactOutcome model={model} /> : <DetailedOutcome model={model} />
}
