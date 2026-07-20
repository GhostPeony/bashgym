import { CheckCircle2, CircleDashed, RefreshCw, ShieldAlert } from 'lucide-react'
import { clsx } from 'clsx'

import { Button } from '../common/Button'
import type {
  SetupEditorFieldId,
  SetupFlowViewModel,
  SetupStepViewModel,
} from './setupFlowModel'

export type SetupConnectionState = 'live' | 'reconciling' | 'stale' | 'offline' | 'error'

export interface AutoResearchSetupFlowProps {
  model: SetupFlowViewModel
  connectionState: SetupConnectionState
  mutationsEnabled: boolean
  onDraftChange?: (fieldId: SetupEditorFieldId, value: string) => void
  onStepAction: (
    stepId: SetupStepViewModel['id'],
    action: NonNullable<SetupStepViewModel['action']>,
    receiptId: string | null,
    mutationContract: SetupStepViewModel['mutationContract'],
  ) => void
  onStart: () => void
  onRefreshDoctor: () => void
  doctorRefreshAvailable?: boolean
}

const connectionLabels: Record<SetupConnectionState, string> = {
  live: 'Live',
  reconciling: 'Reconciling',
  stale: 'Stale',
  offline: 'Offline',
  error: 'Connection error',
}

const statusLabels: Record<SetupStepViewModel['status'], string> = {
  not_started: 'Not started',
  partial: 'Partial receipt',
  blocked: 'Blocked',
  complete: 'Complete',
  ready: 'Ready',
}

function statusClass(status: SetupStepViewModel['status']): string {
  if (status === 'complete' || status === 'ready') return 'border-status-success/60 bg-status-success/10 text-status-success'
  if (status === 'blocked' || status === 'partial') return 'border-status-warning/60 bg-status-warning/10 text-status-warning'
  return 'border-border-subtle bg-background-secondary text-text-secondary'
}

function actionLabel(action: NonNullable<SetupStepViewModel['action']>): string {
  return { continue: 'Continue', resume: 'Resume', retry: 'Retry', remediate: 'View remediation' }[action]
}

function StepRow({
  step,
  current,
  mutationsEnabled,
  draftsEnabled,
  onDraftChange,
  onStepAction,
}: {
  step: SetupStepViewModel
  current: boolean
  mutationsEnabled: boolean
  draftsEnabled: boolean
  onDraftChange?: AutoResearchSetupFlowProps['onDraftChange']
  onStepAction: AutoResearchSetupFlowProps['onStepAction']
}) {
  const actionIsLocal = step.action === 'remediate'
  const actionEnabled = actionIsLocal || (mutationsEnabled && step.actionAvailable)
  return (
    <li
      className={clsx('border-b border-border-subtle py-3 last:border-b-0', current && 'bg-accent/5 -mx-3 px-3')}
      aria-current={current ? 'step' : undefined}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            {step.status === 'complete' || step.status === 'ready'
              ? <CheckCircle2 className="h-4 w-4 shrink-0 text-status-success" aria-hidden="true" />
              : <CircleDashed className="h-4 w-4 shrink-0 text-text-muted" aria-hidden="true" />}
            <h3 className="font-brand text-base text-text-primary">{step.label}</h3>
            <span className={clsx('rounded-brutal border px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wide', statusClass(step.status))}>
              {statusLabels[step.status]}
            </span>
          </div>
          <p className="mt-1 pl-6 text-xs leading-5 text-text-secondary">{step.guidance}</p>
          {step.receipt ? (
            <div className="mt-2 ml-6 border-l-2 border-status-success/60 pl-2 text-xs text-text-secondary">
              <p>{step.receipt.summary}</p>
              <p className="mt-1 font-mono text-[10px] text-text-muted">Receipt · {step.receipt.receiptId}</p>
            </div>
          ) : null}
          {current && step.editor ? (
            <fieldset
              className="mt-3 ml-6 grid gap-2 border-l-2 border-accent/50 pl-3 sm:grid-cols-2"
              aria-label={`${step.label} editor`}
            >
              <legend className="sr-only">{step.label} editor</legend>
              {step.editor.fields.map((field) => (
                <label key={field.id} className="min-w-0 text-[11px] font-medium text-text-secondary">
                  <span className="block pb-1">{field.label}</span>
                  <input
                    className="input w-full px-2 py-1.5 font-mono text-xs"
                    name={field.id}
                    value={field.value}
                    placeholder={field.placeholder}
                    inputMode={field.inputMode}
                    maxLength={512}
                    autoComplete="off"
                    spellCheck={false}
                    readOnly={!draftsEnabled || !onDraftChange}
                    aria-invalid={field.value.length > 0 && !field.valid ? true : undefined}
                    onChange={(event) => onDraftChange?.(field.id, event.currentTarget.value)}
                  />
                  {field.value.length > 0 && field.validationMessage ? (
                    <span className="mt-1 block text-[10px] leading-4 text-status-warning">{field.validationMessage}</span>
                  ) : null}
                </label>
              ))}
              <p className="sm:col-span-2 text-[10px] leading-4 text-text-muted">
                Safe logical binding IDs only. Private host, user, key, credential, and path material stays in the local installation.
              </p>
            </fieldset>
          ) : null}
          {step.actionGuidance ? (
            <p className="mt-2 ml-6 text-[11px] leading-4 text-text-muted">{step.actionGuidance}</p>
          ) : null}
        </div>
        {step.action ? (
          <Button
            variant="secondary"
            size="sm"
            disabled={!actionEnabled}
            onClick={() => onStepAction(step.id, step.action!, step.actionReceiptId, step.mutationContract)}
            aria-label={`${actionLabel(step.action)}: ${step.label}`}
          >
            {actionLabel(step.action)}
          </Button>
        ) : null}
      </div>
    </li>
  )
}

export function AutoResearchSetupFlow({
  model,
  connectionState,
  mutationsEnabled,
  onDraftChange,
  onStepAction,
  onStart,
  onRefreshDoctor,
  doctorRefreshAvailable = false,
}: AutoResearchSetupFlowProps) {
  const isLive = connectionState === 'live'
  const effectiveMutationsEnabled = mutationsEnabled && isLive
  const startEnabled = effectiveMutationsEnabled && model.start.enabled

  return (
    <section className="mx-auto w-full max-w-5xl space-y-4" aria-labelledby="autoresearch-setup-title">
      <header className="flex flex-col gap-2 border-b border-border-subtle pb-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">Campaign intake</p>
          <h2 id="autoresearch-setup-title" className="mt-1 font-brand text-xl text-text-primary">Guided setup</h2>
          <p className="mt-1 text-sm text-text-secondary">A durable, resumable path to an operator-authorized launch.</p>
        </div>
        <span className={clsx('inline-flex w-fit items-center gap-1.5 rounded-brutal border px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-wide', isLive ? 'border-status-success/60 bg-status-success/10 text-status-success' : 'border-status-warning/60 bg-status-warning/10 text-status-warning')}>
          <span className={clsx('status-dot', isLive ? 'status-success' : 'status-warning')} aria-hidden="true" />
          {connectionLabels[connectionState]}
        </span>
      </header>

      {!isLive ? (
        <div className="flex items-start gap-2 border-l-4 border-status-warning bg-status-warning/10 px-3 py-2 text-sm text-text-primary" role="status" aria-live="polite">
          <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-status-warning" aria-hidden="true" />
          <p>Connection lost — setup is read-only. Receipts and remediation stay visible; actions unlock on reconnect.</p>
        </div>
      ) : null}

      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <section className="card p-3" aria-labelledby="setup-receipts-title">
          <div className="flex items-center justify-between gap-3 border-b border-border-subtle pb-2">
            <div>
              <h3 id="setup-receipts-title" className="font-brand text-lg text-text-primary">Setup receipts</h3>
              <p className="text-xs text-text-secondary">Completed work remains sealed; partial work resumes from its receipt.</p>
            </div>
            <span className="font-mono text-[10px] uppercase tracking-wide text-text-muted">Next · {model.nextStepId.replace(/_/g, ' ')}</span>
          </div>
          <ol className="divide-y divide-border-subtle" aria-label="Guided setup steps">
            {model.steps.map((step) => (
              <StepRow
                key={step.id}
                step={step}
                current={step.id === model.nextStepId}
                mutationsEnabled={effectiveMutationsEnabled}
                draftsEnabled={isLive}
                onDraftChange={onDraftChange}
                onStepAction={onStepAction}
              />
            ))}
          </ol>
        </section>

        <aside className="space-y-4" aria-label="Launch readiness">
          <section className="card p-3" aria-labelledby="launch-gate-title">
            <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">Authoritative gate</p>
            <h3 id="launch-gate-title" className="mt-1 font-brand text-lg text-text-primary">Launch readiness</h3>
            <p className="mt-2 text-xs leading-5 text-text-secondary">
              {model.start.reason ?? 'The latest server doctor verified launch readiness and execution identities.'}
            </p>
            <div className="mt-3 flex flex-col gap-2">
              <Button
                variant="secondary"
                size="sm"
                disabled={!effectiveMutationsEnabled || !doctorRefreshAvailable}
                onClick={onRefreshDoctor}
                aria-label="Refresh server doctor"
              >
                <RefreshCw className="h-3.5 w-3.5" /> Refresh doctor
              </Button>
              {!doctorRefreshAvailable ? (
                <p className="text-[10px] leading-4 text-text-muted">Run bashgym campaign doctor in the workspace terminal; a typed desktop doctor contract is not mounted yet.</p>
              ) : null}
              <Button variant="primary" size="sm" disabled={!startEnabled} onClick={onStart} aria-label="Start campaign">
                Start
              </Button>
            </div>
          </section>

          {model.blockers.length > 0 ? (
            <section className="card border-status-warning p-3" aria-labelledby="setup-blockers-title">
              <h3 id="setup-blockers-title" className="font-brand text-lg text-text-primary">Remediation</h3>
              <ol className="mt-2 space-y-2">
                {model.blockers.map((blocker) => (
                  <li key={`${blocker.stepId}-${blocker.code}`} className="border-l-2 border-status-warning pl-2">
                    <p className="font-mono text-[10px] font-bold uppercase tracking-wide text-status-warning">{blocker.code.replace(/_/g, ' ')}</p>
                    <p className="mt-0.5 text-xs leading-5 text-text-secondary">{blocker.summary}</p>
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          <section className="border-l-2 border-accent/60 pl-3 text-xs leading-5 text-text-secondary" aria-label="Compute policy">
            <strong className="font-semibold text-text-primary">Runs on hardware you choose.</strong> This machine or any SSH target you've registered — hosted integrations join only when you explicitly connect them.
            {model.optionalIntegrations.length > 0 ? (
              <ul className="mt-2 space-y-1 font-mono text-[10px] text-text-muted">
                {model.optionalIntegrations.map((integration) => <li key={integration.id}>{integration.label} · optional · {integration.status}</li>)}
              </ul>
            ) : null}
          </section>
        </aside>
      </div>
    </section>
  )
}
