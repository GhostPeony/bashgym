import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertCircle, CheckCircle2, ClipboardCheck, RefreshCw, ShieldAlert } from 'lucide-react'
import { clsx } from 'clsx'
import {
  trainingApi,
  type RunCardFinding,
  type RunCardSummary,
  type RunCardValidationResponse,
} from '../../services/api'

type CardsState =
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'ready'; cards: RunCardSummary[] }

type ValidationState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'ready'; validation: RunCardValidationResponse }

const EMPTY_FINDINGS: RunCardFinding[] = []

function responseError(error: unknown): string {
  if (typeof error === 'string') return error
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

function formatModified(timestamp: number): string {
  if (!Number.isFinite(timestamp)) return 'unknown'
  return new Date(timestamp * 1000).toLocaleString()
}

function levelClass(level: string): string {
  if (level === 'fail') return 'text-status-error'
  if (level === 'warn') return 'text-status-warning'
  if (level === 'diagnostic') return 'text-accent-dark'
  return 'text-text-secondary'
}

function levelLabel(level: string): string {
  if (level === 'fail') return 'Blocker'
  if (level === 'warn') return 'Warning'
  if (level === 'diagnostic') return 'Diagnostic'
  return level
}

function findingCounts(findings: RunCardFinding[]): Record<string, number> {
  return findings.reduce<Record<string, number>>((counts, finding) => {
    counts[finding.level] = (counts[finding.level] ?? 0) + 1
    return counts
  }, {})
}

export function RunCardEvidencePanel() {
  const [cardsState, setCardsState] = useState<CardsState>({ status: 'loading' })
  const [validationState, setValidationState] = useState<ValidationState>({ status: 'idle' })
  const [selectedPath, setSelectedPath] = useState('')

  const loadCards = useCallback(async () => {
    setCardsState({ status: 'loading' })
    const response = await trainingApi.listRunCards(24)
    if (response.ok && response.data) {
      const cards = response.data.run_cards
      setCardsState({ status: 'ready', cards })
      setSelectedPath((current) => current || cards[0]?.path || '')
    } else {
      setCardsState({ status: 'error', error: responseError(response.error) })
    }
  }, [])

  const validateSelected = useCallback(async () => {
    const path = selectedPath.trim()
    if (!path) {
      setValidationState({ status: 'error', error: 'Choose or enter a RunCard path.' })
      return
    }
    setValidationState({ status: 'loading' })
    const response = await trainingApi.validateRunCard(path, true)
    if (response.ok && response.data) {
      setValidationState({ status: 'ready', validation: response.data })
    } else {
      setValidationState({ status: 'error', error: responseError(response.error) })
    }
  }, [selectedPath])

  useEffect(() => {
    void loadCards()
  }, [loadCards])

  const findings = validationState.status === 'ready' ? validationState.validation.findings : EMPTY_FINDINGS
  const counts = useMemo(() => findingCounts(findings), [findings])
  const blockingCount = counts.fail ?? 0
  const warningCount = counts.warn ?? 0
  const diagnosticCount = counts.diagnostic ?? 0
  const recentCards = cardsState.status === 'ready' ? cardsState.cards : []
  const explanation =
    validationState.status === 'ready' ? validationState.validation.promotion_explanation : null

  return (
    <div className="card p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between mb-4">
        <div>
          <h3 className="flex items-center gap-2 text-sm font-mono font-semibold text-text-primary">
            <ClipboardCheck className="w-4 h-4 text-accent" />
            RunCard Evidence
          </h3>
          <p className="text-xs text-text-muted mt-1">
            Promotion blockers, claim-tier findings, and artifact presence for serious runs.
          </p>
        </div>
        <button
          type="button"
          onClick={loadCards}
          className="btn-icon flex items-center justify-center"
          title="Refresh RunCards"
        >
          <RefreshCw className={clsx('w-4 h-4', cardsState.status === 'loading' && 'animate-spin')} />
        </button>
      </div>

      {cardsState.status === 'error' ? (
        <p className="font-mono text-xs text-status-error mb-3">{cardsState.error}</p>
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_auto] gap-3">
        <label className="block">
          <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
            RunCard path
          </span>
          <input
            value={selectedPath}
            onChange={(event) => setSelectedPath(event.target.value)}
            placeholder="data/models/run_id/run_card.json"
            className="input w-full font-mono text-xs"
          />
        </label>
        <button
          type="button"
          onClick={validateSelected}
          disabled={validationState.status === 'loading'}
          className="btn-secondary flex items-center justify-center gap-2 lg:self-end"
        >
          {validationState.status === 'loading' ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <ShieldAlert className="w-4 h-4" />
          )}
          Validate
        </button>
      </div>

      {recentCards.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-2">
          {recentCards.slice(0, 6).map((card) => (
            <button
              key={card.path}
              type="button"
              onClick={() => setSelectedPath(card.path)}
              className={clsx(
                'border-2 px-2 py-1 text-left font-mono text-[11px] transition-press',
                selectedPath === card.path
                  ? 'border-accent bg-accent-light text-accent-dark'
                  : 'border-border bg-background-card text-text-secondary'
              )}
              title={formatModified(card.modified)}
            >
              {card.run_id || card.path.split(/[\\/]/).pop()}
            </button>
          ))}
        </div>
      ) : cardsState.status === 'ready' ? (
        <p className="mt-3 text-xs text-text-muted">No RunCards found in local BashGym data roots.</p>
      ) : null}

      {validationState.status === 'error' ? (
        <div className="mt-4 border-2 border-status-error bg-status-error/10 p-3">
          <p className="font-mono text-xs text-status-error">{validationState.error}</p>
        </div>
      ) : null}

      {validationState.status === 'ready' ? (
        <div className="mt-4 border-t-2 border-border pt-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="font-mono text-xs text-text-muted">
                {validationState.validation.run_card.run_id} / {validationState.validation.run_card.claim_tier}
              </p>
              <p className="text-sm text-text-secondary mt-1">
                {validationState.validation.run_card.training_method} on{' '}
                {validationState.validation.run_card.base_model}
              </p>
            </div>
            <div className="flex items-center gap-2 font-mono text-xs">
              {validationState.validation.ok ? (
                <span className="text-status-success flex items-center gap-1">
                  <CheckCircle2 className="w-4 h-4" />
                  Promotion ready
                </span>
              ) : (
                <span className="text-status-error flex items-center gap-1">
                  <AlertCircle className="w-4 h-4" />
                  {blockingCount} blocker{blockingCount === 1 ? '' : 's'}
                </span>
              )}
              {warningCount > 0 ? <span className="text-status-warning">{warningCount} warn</span> : null}
              {diagnosticCount > 0 ? <span className="text-accent-dark">{diagnosticCount} diag</span> : null}
            </div>
          </div>

          {explanation !== null ? (
            <div
              className={clsx(
                'mt-4 border-2 p-3',
                explanation.ok
                  ? 'border-status-success bg-status-success/10'
                  : 'border-status-error bg-status-error/10'
              )}
            >
              <p
                className={clsx(
                  'font-mono text-xs',
                  explanation.ok ? 'text-status-success' : 'text-status-error'
                )}
              >
                {explanation.headline}
              </p>
              {explanation.failed_gates.length > 0 ? (
                <div className="mt-3 space-y-2">
                  {explanation.failed_gates.slice(0, 4).map((gate) => (
                    <div key={gate.gate} className="text-sm text-text-secondary">
                      <p className="font-mono text-[11px] uppercase text-text-primary">
                        {gate.gate}
                      </p>
                      <p>{gate.summary}</p>
                      <p className="text-xs text-text-muted mt-1">{gate.next_action}</p>
                    </div>
                  ))}
                </div>
              ) : null}
              {explanation.next_actions.length > 0 ? (
                <div className="mt-3 border-t-2 border-border pt-3">
                  <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-2">
                    Next actions
                  </p>
                  <ul className="space-y-1 text-sm text-text-secondary">
                    {explanation.next_actions.slice(0, 5).map((action) => (
                      <li key={action}>{action}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}

          {findings.length > 0 ? (
            <ul className="mt-4 space-y-2">
              {findings.slice(0, 8).map((finding) => (
                <li
                  key={`${finding.code}-${finding.field ?? ''}-${finding.path ?? ''}`}
                  className="border-l-4 border-border pl-3 text-sm"
                >
                  <p className={clsx('font-mono text-[11px] uppercase', levelClass(finding.level))}>
                    {levelLabel(finding.level)} / {finding.code}
                  </p>
                  <p className="text-text-secondary mt-1">{finding.message}</p>
                  {finding.path ? (
                    <p className="font-mono text-[11px] text-text-muted mt-1">{finding.path}</p>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-4 text-sm text-text-secondary">No promotion findings returned.</p>
          )}

          {validationState.validation.artifact_status.length > 0 ? (
            <div className="mt-4 border-t-2 border-border pt-3">
              <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-2">
                Artifacts
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {validationState.validation.artifact_status.slice(0, 6).map((artifact) => (
                  <div key={`${artifact.field}-${artifact.path}`} className="text-xs text-text-secondary">
                    <span className="font-mono text-text-primary">{artifact.field}</span>{' '}
                    <span className={artifact.present ? 'text-status-success' : 'text-status-error'}>
                      {artifact.present ? 'present' : 'missing'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
