import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import { Activity, AlertTriangle, CheckCircle2, XOctagon } from 'lucide-react'
import { trainingApi, type TrainingRunAnalysis, type TrainingRunFinding } from '../../services/api'

interface HealthBannerProps {
  runId?: string
  isRunning?: boolean
}

// Map the run-analysis verdict level to Botanical Brutalist status styling.
const LEVEL_STYLE: Record<
  string,
  { border: string; text: string; label: string; icon: React.ReactNode }
> = {
  ok: {
    border: 'border-status-success',
    text: 'text-status-success',
    label: 'HEALTHY',
    icon: <CheckCircle2 className="w-5 h-5" />
  },
  info: {
    border: 'border-accent',
    text: 'text-accent',
    label: 'INFO',
    icon: <Activity className="w-5 h-5" />
  },
  warn: {
    border: 'border-status-warning',
    text: 'text-status-warning',
    label: 'WARNING',
    icon: <AlertTriangle className="w-5 h-5" />
  },
  blocked: {
    border: 'border-status-error',
    text: 'text-status-error',
    label: 'BLOCKED',
    icon: <XOctagon className="w-5 h-5" />
  }
}

function severityRank(finding: TrainingRunFinding): number {
  const order: Record<string, number> = { blocker: 0, blocked: 0, warn: 1, warning: 1, info: 2 }
  return order[finding.severity?.toLowerCase()] ?? 3
}

export function HealthBanner({ runId, isRunning }: HealthBannerProps) {
  const [analysis, setAnalysis] = useState<TrainingRunAnalysis | null>(null)

  useEffect(() => {
    if (!runId) {
      setAnalysis(null)
      return
    }
    let cancelled = false

    const load = async () => {
      try {
        const result = await trainingApi.getRunAnalysis(runId)
        // 404 before the first step (no metrics yet) — leave the banner hidden.
        if (!cancelled) setAnalysis(result.ok && result.data ? result.data : null)
      } catch {
        if (!cancelled) setAnalysis(null)
      }
    }

    load()
    // Re-poll while the run is active; health can change as loss/grad-norm move.
    const interval = isRunning ? window.setInterval(load, 10000) : undefined
    return () => {
      cancelled = true
      if (interval) window.clearInterval(interval)
    }
  }, [runId, isRunning])

  if (!analysis || (analysis.training_metrics?.points ?? 0) === 0) return null

  const level = analysis.verdict?.level ?? 'info'
  const style = LEVEL_STYLE[level] ?? LEVEL_STYLE.info
  const findings = [...(analysis.findings ?? [])].sort((a, b) => severityRank(a) - severityRank(b))
  const topFindings = findings.slice(0, 3)

  return (
    <div className={clsx('card border-brutal p-4 mb-5', style.border)}>
      <div className="flex items-center gap-3">
        <span className={style.text}>{style.icon}</span>
        <span className={clsx('font-mono text-xs font-bold uppercase tracking-widest', style.text)}>
          {style.label}
        </span>
        <span className="font-mono text-xs text-text-muted">
          run health · {analysis.training_metrics?.points ?? 0} steps analyzed
        </span>
      </div>
      {topFindings.length > 0 && (
        <ul className="mt-3 flex flex-col gap-2">
          {topFindings.map((finding) => (
            <li key={finding.code} className="flex flex-col gap-0.5">
              <span className="text-sm text-text-primary">{finding.message}</span>
              {finding.next && (
                <span className="font-mono text-xs text-text-secondary">→ {finding.next}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
