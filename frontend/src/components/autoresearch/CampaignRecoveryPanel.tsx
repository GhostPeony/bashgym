import { AlertTriangle, CheckCircle2, FileClock, ShieldCheck } from 'lucide-react'

import { Button } from '../common/Button'
import {
  buildCampaignRecoveryModel,
  buildRecoveryRequest,
  parseCampaignRecovery,
  type CampaignRecoveryPublicV1,
  type RecoveryAction,
  type RecoveryFreshness,
  type RecoveryRequest,
} from './campaignRecoveryModel'

export interface CampaignRecoveryPanelProps {
  snapshot: unknown
  freshness: RecoveryFreshness
  now: string
  confirmations: Record<RecoveryAction, boolean>
  idempotencyKeys: Record<RecoveryAction, string>
  onConfirmationChange: (action: RecoveryAction, confirmed: boolean) => void
  onRequest: (request: RecoveryRequest) => void
}

const actionOrder: RecoveryAction[] = ['resume', 'repair', 'takeover']
const actionLabels: Record<RecoveryAction, string> = { resume: 'Resume campaign', repair: 'Request repair', takeover: 'Take over expired lease' }

function readable(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function freshnessNotice(freshness: RecoveryFreshness): string | null {
  if (freshness === 'offline') return 'Campaign state is offline; recovery controls are read-only until live authority returns.'
  if (freshness === 'stale') return 'Campaign state is stale; reconcile before changing recovery state.'
  if (freshness === 'reconciling') return 'Campaign state is reconciling; wait for live authority before changing recovery state.'
  if (freshness === 'error') return 'Campaign state has an authority error; resolve it before changing recovery state.'
  return null
}

function InvalidRecoveryPanel({ freshness }: { freshness: RecoveryFreshness }) {
  const notice = freshnessNotice(freshness)
  return (
    <section aria-label="Campaign recovery">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div><p className="text-sm text-text-secondary">Recovery projection unavailable. Unsafe or invalid fields are withheld.</p>{notice ? <p className="mt-2 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs text-text-secondary" role="status">{notice}</p> : null}</div>
        <div className="border-2 border-border bg-background-secondary px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-wide text-text-primary">Read-only</div>
      </header>
      <div className="mt-4 grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(14rem,1fr))]">
        <section className="border-2 border-border bg-background-secondary p-3"><h3 className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Immutable bindings unavailable</h3><p className="mt-2 text-xs text-text-secondary">No unvalidated binding identity is shown.</p></section>
        <section className="border-2 border-border bg-background-secondary p-3"><h3 className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Authoritative lineage unavailable</h3><p className="mt-2 text-xs text-text-secondary">Reconcile a valid public projection before recovery.</p></section>
      </div>
      <section className="mt-4 border-l-2 border-accent bg-background-secondary px-3 py-3"><h3 className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Doctor evidence unavailable</h3></section>
      <section className="mt-4 border-t-2 border-border pt-4"><h3 className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Explicit recovery requests</h3><div className="mt-3 grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(11rem,1fr))]">
        {actionOrder.map((action) => <article key={action} className="border-2 border-border bg-background-secondary p-3"><h4 className="font-brand text-base text-text-primary">{actionLabels[action]}</h4><label className="mt-3 flex gap-2 text-xs text-text-muted"><input type="checkbox" disabled />Confirmation unavailable</label><Button className="mt-3" type="button" size="sm" disabled>{actionLabels[action]}</Button></article>)}
      </div></section>
    </section>
  )
}

function executionGuidance(model: ReturnType<typeof buildCampaignRecoveryModel>): string {
  const execution = model.latestExecution
  if (!execution) return model.consumerGuidance
  if (execution.status === 'accepted') return 'The request is durably accepted and waiting for the resident recovery worker.'
  if (execution.status === 'executing') return 'The resident recovery worker is processing this request.'
  if (execution.status === 'completed') return `The worker completed this request with ${readable(execution.outcome_code || 'completed')}.`
  return `The worker stopped with ${readable(execution.outcome_code || execution.status)}. Operator reconciliation is required.`
}

function bindingRows(snapshot: CampaignRecoveryPublicV1): Array<[string, string]> {
  return [
    ['Model', snapshot.bindings.model_id ? snapshot.bindings.model_display_label ? `${snapshot.bindings.model_display_label} · ${snapshot.bindings.model_id}` : snapshot.bindings.model_id : 'Missing'],
    ['Data', snapshot.bindings.data_id || 'Missing'],
    ['Evaluator', snapshot.bindings.evaluator_id || 'Missing'],
    ['Private compute', snapshot.bindings.compute_id || 'Missing'],
  ]
}

export function CampaignRecoveryPanel({ snapshot: input, freshness, now, confirmations, idempotencyKeys, onConfirmationChange, onRequest }: CampaignRecoveryPanelProps) {
  const snapshot = parseCampaignRecovery(input)
  if (!snapshot) return <InvalidRecoveryPanel freshness={freshness} />
  const model = buildCampaignRecoveryModel({ snapshot, freshness, now })
  const notice = freshnessNotice(freshness)
  return (
    <section aria-label="Campaign recovery">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm leading-6 text-text-secondary">Restore a durable AutoResearch campaign from a cloned or forked BashGym installation without resetting its recorded lineage.</p>
          <p className="mt-1 font-mono text-[10px] uppercase tracking-wide text-text-muted">Workspace {snapshot.workspace_id} · Campaign {snapshot.campaign_id}</p>
        </div>
        <div className="border-2 border-border bg-background-secondary px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-wide text-text-primary">{model.consumerReady ? model.decisionLabel : 'Consumer unavailable'}</div>
      </header>

      <div className="mt-4 grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(14rem,1fr))]">
        <section className="border-2 border-border bg-background-secondary p-3" aria-labelledby="recovery-bindings-title">
          <h3 id="recovery-bindings-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Immutable bindings</h3>
          <dl className="mt-3 grid gap-2 text-sm">
            {bindingRows(snapshot).map(([label, value]) => <div key={label} className="flex flex-wrap justify-between gap-2 border-t border-border-subtle pt-2 first:border-t-0 first:pt-0"><dt className="text-text-secondary">{label}</dt><dd className="font-mono text-xs text-text-primary">{value}</dd></div>)}
          </dl>
          <p className="mt-3 text-xs leading-5 text-text-muted">{readable(snapshot.installation.lineage_mode)} installation {snapshot.installation.installation_id}. Bindings are identity references, not connection settings.</p>
        </section>

        <section className="border-2 border-border bg-background-secondary p-3" aria-labelledby="recovery-lineage-title">
          <h3 id="recovery-lineage-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Authoritative lineage</h3>
          <div className="mt-3 grid gap-2 text-sm text-text-primary">
            <p>Revision {model.lineage.campaign_revision} · Event cursor {model.lineage.event_cursor} · Aggregate {model.lineage.aggregate_version}</p>
            <p>Checkpoint {model.lineage.checkpoint_id} · Artifact {model.lineage.artifact_id}</p>
            <p className="text-xs text-text-secondary">Schema {model.lineage.schema_version}{model.lineage.parent_campaign_id ? ` · Parent ${model.lineage.parent_campaign_id}` : ' · Root campaign'}</p>
            <p className="text-xs text-text-secondary">Controller: {readable(snapshot.controller.owner_status)} · Compute: {readable(snapshot.compute.availability)}{snapshot.compute.integration_label ? ` · ${snapshot.compute.integration_label}` : ''}</p>
          </div>
        </section>
      </div>

      {notice ? <p className="mt-4 flex items-start gap-2 border-l-2 border-accent bg-background-secondary px-3 py-2 text-xs leading-5 text-text-secondary" role="status"><FileClock className="mt-0.5 h-4 w-4 shrink-0 text-accent" />{notice}</p> : null}

      <section className="mt-4 border-l-2 border-accent bg-background-secondary px-3 py-3" aria-labelledby="recovery-doctor-title">
        <div className="flex flex-wrap items-center justify-between gap-2"><h3 id="recovery-doctor-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Doctor evidence</h3><span className="text-xs text-text-secondary">Eligibility {model.eligibilityVerified ? 'Verified' : 'Unverified'}</span></div>
        <ul className="mt-2 grid gap-1 text-xs text-text-secondary">
          {snapshot.doctor.checks.map((entry) => <li key={entry.check} className="flex gap-2"><CheckCircle2 className={`h-3.5 w-3.5 shrink-0 ${entry.status === 'pass' ? 'text-status-success' : 'text-status-warning'}`} />{readable(entry.check)}: {readable(entry.status)}</li>)}
          {!snapshot.doctor.checks.length ? <li>No recovery doctor evidence is available.</li> : null}
        </ul>
        <p className="mt-2 text-xs text-text-muted">{model.latestReceipt ? `Latest recovery receipt: ${model.latestReceipt.receipt_id} · ${model.latestReceipt.emitted_at}.` : 'No recovery receipt is available.'}</p>
      </section>

      <section className="mt-3 border-2 border-border bg-background-secondary p-3" aria-labelledby="recovery-execution-title">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 id="recovery-execution-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Latest execution</h3>
          <span className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-primary">
            {model.latestExecution ? `${readable(model.latestExecution.action)} · ${readable(model.latestExecution.status)}` : model.consumerReady ? 'Consumer ready' : 'Consumer unavailable'}
          </span>
        </div>
        <p className="mt-2 text-xs leading-5 text-text-secondary">{executionGuidance(model)}</p>
        {model.latestExecution?.attempt_id ? <p className="mt-1 font-mono text-[10px] text-text-muted">Attempt {model.latestExecution.attempt_id}</p> : null}
      </section>

      <section className="mt-4" aria-labelledby="recovery-blockers-title">
        <h3 id="recovery-blockers-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Recovery decision</h3>
        {model.blockers.length ? <ul className="mt-2 grid gap-2">{model.blockers.map((item) => <li key={item.code} className="flex gap-2 border-2 border-border bg-background-secondary px-3 py-2 text-xs leading-5 text-text-secondary"><AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-status-warning" />{item.label}</li>)}</ul> : model.decision === 'resumable' && model.mutationsEnabled ? <p className="mt-2 flex items-center gap-2 text-sm text-status-success"><ShieldCheck className="h-4 w-4" />Live authority and current scope-bound eligibility permit an explicit recovery request.</p> : <p className="mt-2 text-sm text-text-secondary">No recovery mutation is currently permitted.</p>}
      </section>

      <section className="mt-4 border-t-2 border-border pt-4" aria-labelledby="recovery-actions-title">
        <h3 id="recovery-actions-title" className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">Explicit recovery requests</h3>
        {!model.consumerReady ? <p className="mt-2 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs leading-5 text-text-secondary">{model.consumerGuidance} Recovery controls remain visible but disabled.</p> : null}
        <div className="mt-3 grid gap-3 [grid-template-columns:repeat(auto-fit,minmax(11rem,1fr))]">
          {actionOrder.map((actionName) => {
            const action = model.actions[actionName]
            const confirmed = confirmations[actionName]
            const controlDisabled = !action.enabled
            return <article key={actionName} className="border-2 border-border bg-background-secondary p-3">
              <h4 className="font-brand text-base text-text-primary">{action.label}</h4>
              <p className="mt-1 min-h-10 text-xs leading-5 text-text-secondary">{action.guidance}</p>
              <label className="mt-3 flex items-start gap-2 text-xs text-text-primary"><input type="checkbox" checked={confirmed} disabled={controlDisabled} onChange={(event) => onConfirmationChange(actionName, event.target.checked)} /><span>I confirm this is an intentional {actionName} request.</span></label>
              <Button className="mt-3" type="button" size="sm" disabled={controlDisabled || !confirmed} onClick={() => {
                const request = buildRecoveryRequest({ model, action: actionName, humanConfirmed: confirmed, idempotencyKey: idempotencyKeys[actionName] })
                if (request) onRequest(request)
              }}>{action.label}</Button>
            </article>
          })}
        </div>
        <p className="mt-3 text-xs text-text-muted">Every request is bound to workspace, campaign, eligibility receipt, doctor evidence, revision, cursor, aggregate version, lease, and lineage. No action silently steals a lease.</p>
      </section>
    </section>
  )
}
