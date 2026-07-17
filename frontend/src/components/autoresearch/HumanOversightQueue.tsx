import { CheckCircle2, Clock3, ShieldAlert, UserRound } from 'lucide-react'

import { Button } from '../common/Button'
import { HUMAN_WORK_LIMITS, humanReviewResponseKey, reviewDecisionForChoice } from './humanWorkModel'
import type {
  HumanOversightViewModel,
  HumanPromotionRequest,
  HumanReviewDecision,
  HumanReviewResponseBinding,
  HumanWorkClaimRequest,
  HumanWorkItemViewModel,
  HumanWorkSubmitRequest,
} from './humanWorkModel'

export interface HumanReviewResponse extends HumanReviewResponseBinding {
  decision: HumanReviewDecision | ''
  rationale: string
}

export interface HumanOversightQueueProps {
  model: HumanOversightViewModel
  responses: Readonly<Record<string, HumanReviewResponse | undefined>>
  onResponseChange: (responseKey: string, response: HumanReviewResponse) => void
  onClaim: (request: HumanWorkClaimRequest) => void
  onSubmit: (request: HumanWorkSubmitRequest, response: HumanReviewResponse & { decision: HumanReviewDecision }) => void
  onPromotion: (request: HumanPromotionRequest) => void
}

function responseMatchesBinding(response: HumanReviewResponse | undefined, binding: HumanReviewResponseBinding): response is HumanReviewResponse {
  return Boolean(response)
    && response!.workspaceId === binding.workspaceId
    && response!.campaignId === binding.campaignId
    && response!.campaignRevision === binding.campaignRevision
    && response!.workId === binding.workId
    && response!.itemVersion === binding.itemVersion
    && response!.rubricVersion === binding.rubricVersion
}

function readableState(value: string): string {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function stateClass(state: string): string {
  if (state === 'submitted') return 'text-status-success'
  if (state === 'expired' || state === 'replaced') return 'text-status-warning'
  return 'text-text-secondary'
}

function ReviewItem({
  view,
  model,
  responses,
  onResponseChange,
  onClaim,
  onSubmit,
}: {
  view: HumanWorkItemViewModel
  model: HumanOversightViewModel
  responses: HumanOversightQueueProps['responses']
  onResponseChange: HumanOversightQueueProps['onResponseChange']
  onClaim: HumanOversightQueueProps['onClaim']
  onSubmit: HumanOversightQueueProps['onSubmit']
}) {
  const { item } = view
  const responseKey = humanReviewResponseKey(view.responseBinding)
  const candidate = responses[responseKey]
  const currentResponse = responseMatchesBinding(candidate, view.responseBinding) ? candidate : undefined
  const selected = currentResponse?.decision || ''
  const rationale = currentResponse?.rationale || ''
  const decisionValid = selected !== '' && view.offeredDecisions.includes(selected)
  const rationaleValid = rationale.trim().length > 0 && rationale.length <= HUMAN_WORK_LIMITS.maxRationale
  const canSubmit = model.mutationsEnabled && view.submit !== null && decisionValid && rationaleValid
  const titleId = `${item.work_id}-title`
  const decisionId = `${item.work_id}-decision`
  const rationaleId = `${item.work_id}-rationale`

  const updateResponse = (update: Pick<HumanReviewResponse, 'decision' | 'rationale'>) => {
    onResponseChange(responseKey, { ...view.responseBinding, ...update })
  }

  const submit = () => {
    if (!canSubmit || !view.submit || !currentResponse || currentResponse.decision === '') return
    onSubmit(view.submit, { ...currentResponse, decision: currentResponse.decision, rationale: currentResponse.rationale.trim() })
  }

  return (
    <article className="border-t border-border-subtle py-4 first:border-t-0 first:pt-0" aria-labelledby={titleId}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 id={titleId} className="font-brand text-lg text-text-primary">Blinded review</h3>
          <p className="mt-1 text-xs text-text-secondary">Version {item.version} · Rubric {item.rubric.rubric_id} v{item.rubric.version}</p>
        </div>
        <div className={`font-mono text-[10px] font-semibold uppercase tracking-wide ${stateClass(view.effectiveState)}`}>
          {readableState(view.effectiveState)}{view.invalidated ? ' · Revision changed' : ''}
        </div>
      </div>

      <p className="mt-3 text-sm leading-6 text-text-primary">{item.rubric.instructions}</p>
      <p className="mt-2 text-xs text-text-muted">{item.sample.prompt}</p>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        {[item.sample.left, item.sample.right].map((sample) => (
          <section key={sample.label} className="border border-border-subtle bg-background-secondary p-3" aria-label={sample.label}>
            <h4 className="font-mono text-[10px] font-semibold uppercase tracking-wide text-text-muted">{sample.label}</h4>
            <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-text-primary">{sample.display}</p>
          </section>
        ))}
      </div>

      {(view.claim || view.effectiveState === 'pending' || view.effectiveState === 'expired') ? (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button
            type="button"
            size="sm"
            onClick={() => { if (model.mutationsEnabled && view.claim) onClaim(view.claim) }}
            disabled={!model.mutationsEnabled || !view.claim}
          >
            Claim review
          </Button>
          {view.effectiveState === 'expired' ? <span className="flex items-center gap-1 text-xs text-status-warning"><Clock3 className="h-3.5 w-3.5" />The previous lease expired; claim the current version.</span> : null}
        </div>
      ) : null}

      {view.effectiveState === 'claimed' ? (
        <fieldset className="mt-4 border-l-2 border-accent pl-3" disabled={!model.mutationsEnabled || !view.submit}>
          <legend className="sr-only">Submit blinded review</legend>
          <div className="grid gap-3 md:grid-cols-[minmax(0,220px)_minmax(0,1fr)]">
            <div>
              <label htmlFor={decisionId} className="block text-xs font-semibold text-text-primary">Review decision</label>
              <select
                id={decisionId}
                className="mt-1 w-full border border-border bg-background-card px-2 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
                value={selected}
                onChange={(event) => {
                  const next = event.target.value
                  updateResponse({ decision: view.offeredDecisions.includes(next as HumanReviewDecision) ? next as HumanReviewDecision : '', rationale })
                }}
              >
                <option value="">Choose a rubric decision</option>
                {item.rubric.choices.map((choice) => <option key={choice.choice_id} value={reviewDecisionForChoice(choice.choice_id)!}>{choice.label}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor={rationaleId} className="block text-xs font-semibold text-text-primary">Rationale</label>
              <textarea
                id={rationaleId}
                className="mt-1 min-h-20 w-full border border-border bg-background-card px-2 py-2 text-sm leading-5 text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
                value={rationale}
                maxLength={HUMAN_WORK_LIMITS.maxRationale}
                onChange={(event) => updateResponse({ decision: selected, rationale: event.target.value.slice(0, HUMAN_WORK_LIMITS.maxRationale) })}
                placeholder="Record the rubric-based reason for this decision."
              />
            </div>
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Button type="button" size="sm" disabled={!canSubmit} onClick={submit}>Submit sealed review</Button>
            <span className="text-xs text-text-muted">The response and replay key are bound to this exact campaign, item, and rubric revision.</span>
          </div>
        </fieldset>
      ) : null}

      {item.receipt ? (
        <p className={`mt-3 flex flex-wrap items-center gap-2 text-xs ${view.receiptCurrent ? 'text-text-secondary' : 'text-status-warning'}`}>
          {view.receiptCurrent
            ? <CheckCircle2 className="h-4 w-4 text-status-success" />
            : <ShieldAlert className="h-4 w-4 text-status-warning" />}
          {view.receiptCurrent ? 'Current sealed receipt' : 'Receipt not current'} {item.receipt.receipt_id} · {item.receipt.sealed_at}
        </p>
      ) : null}
    </article>
  )
}

export function HumanOversightQueue({ model, responses, onResponseChange, onClaim, onSubmit, onPromotion }: HumanOversightQueueProps) {
  const { queue } = model
  const decidePromotion = (decision: HumanPromotionRequest['decision']) => {
    if (!model.mutationsEnabled || !model.promotion) return
    onPromotion({ ...model.promotion, decision })
  }
  return (
    <section className="card p-4" aria-labelledby="human-oversight-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="human-oversight-title" className="font-brand text-xl text-text-primary">Human oversight queue</h2>
          <p className="mt-1 text-sm leading-6 text-text-secondary">One authenticated desktop reviewer evaluates opaque samples against a versioned rubric.</p>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-wide text-text-muted">
          <UserRound className="h-4 w-4 text-accent" /> Campaign revision {model.scope.campaignRevision}
        </div>
      </div>

      <div className="mt-3 flex items-start gap-2 border-l-2 border-accent bg-background-secondary px-3 py-2 text-xs text-text-secondary" role="status" aria-live="polite">
        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
        <span>{model.remediation}</span>
      </div>
      {model.mutationMessage ? <p className="mt-2 text-xs text-status-warning" role="alert">{model.mutationMessage}</p> : null}

      <div className="mt-4">
        {model.items.length ? model.items.map((view) => (
          <ReviewItem
            key={humanReviewResponseKey(view.responseBinding)}
            view={view}
            model={model}
            responses={responses}
            onResponseChange={onResponseChange}
            onClaim={onClaim}
            onSubmit={onSubmit}
          />
        )) : <p className="text-sm text-text-muted">No blinded human review work is currently queued.</p>}
      </div>

      <footer className="mt-4 border-t border-border-subtle pt-4">
        <p className="text-sm text-text-primary">
          {model.blocksComparison ? 'Comparison remains blocked by human review.' : 'No current human-review comparison block.'}
          {' '}{model.blocksPromotion ? 'Promotion remains blocked until an explicit human decision.' : 'Promotion is not blocked by this queue.'}
        </p>
        {(model.promotion || queue.promotion.state === 'awaiting_human_decision') ? (
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <span className="text-xs text-text-secondary">A human reviewer must explicitly decide promotion from the current sealed receipt.</span>
            <Button type="button" size="sm" onClick={() => decidePromotion('promote')} disabled={!model.mutationsEnabled || !model.promotion}>Promote with human decision</Button>
            <Button type="button" variant="secondary" size="sm" onClick={() => decidePromotion('hold')} disabled={!model.mutationsEnabled || !model.promotion}>Hold for further review</Button>
          </div>
        ) : null}
      </footer>
    </section>
  )
}
