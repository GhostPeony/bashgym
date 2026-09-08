import { Check, Circle, Loader2, RefreshCw, ShieldAlert } from 'lucide-react'
import { clsx } from 'clsx'

import { Button } from '../common/Button'
import {
  buildGuidedSetupView,
  guidedSetupSteps,
  type GuidedSetupContext,
  type GuidedSetupDoctor,
  type GuidedSetupStep,
  type GuidedSetupValidation
} from './guidedSetupModel'

export type GuidedSetupConnectionState = 'live' | 'reconciling' | 'offline' | 'error'

export interface GuidedAutoResearchSetupProps {
  context: GuidedSetupContext | null
  connectionState: GuidedSetupConnectionState
  pending: boolean
  error: string | null
  doctor: GuidedSetupDoctor | null
  validation: GuidedSetupValidation | null
  selectedOptionId: string
  campaignId: string
  title: string
  budgetUnit: string
  budgetLimit: string
  maxAttempts: string
  minimumImprovement: string
  onSelectedOptionChange: (value: string) => void
  onCampaignIdChange: (value: string) => void
  onTitleChange: (value: string) => void
  onBudgetUnitChange: (value: string) => void
  onBudgetLimitChange: (value: string) => void
  onMaxAttemptsChange: (value: string) => void
  onMinimumImprovementChange: (value: string) => void
  onAdvance: () => void
  onDoctor: () => void
  onValidate: () => void
  onCreate: () => void
  onRetry: () => void
}

const labels: Record<GuidedSetupStep, string> = {
  template: 'Template',
  installation: 'Installation',
  model: 'Model',
  data: 'Data',
  compute: 'Compute',
  evaluation: 'Evaluation'
}

function reason(value: string): string {
  return value.replaceAll('_', ' ')
}

function selectionFor(context: GuidedSetupContext, step: GuidedSetupStep): string | null {
  const selections = context.session?.selections
  if (!selections) return null
  if (step === 'template') return selections.template_id
  if (step === 'installation') return selections.installation_id
  return selections.bindings[step]
}

export function GuidedAutoResearchSetup({
  context,
  connectionState,
  pending,
  error,
  doctor,
  validation,
  selectedOptionId,
  campaignId,
  title,
  budgetUnit,
  budgetLimit,
  maxAttempts,
  minimumImprovement,
  onSelectedOptionChange,
  onCampaignIdChange,
  onTitleChange,
  onBudgetUnitChange,
  onBudgetLimitChange,
  onMaxAttemptsChange,
  onMinimumImprovementChange,
  onAdvance,
  onDoctor,
  onValidate,
  onCreate,
  onRetry
}: GuidedAutoResearchSetupProps) {
  const live = connectionState === 'live'
  const view = context ? buildGuidedSetupView(context) : null
  const activeStep = view?.currentStep
  const selectedOption = view?.options.find((option) => option.id === selectedOptionId)
  const selectedTemplate = context?.templates.find(
    (item) => item.template_id === context.session?.selections.template_id
  )
  const approvedBudget = selectedTemplate?.experiment_contract.budget_limits[budgetUnit]
  const canSave = live && !pending && Boolean(selectedOption?.selectable)
  const limitsValid =
    selectedTemplate !== undefined &&
    /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$/.test(budgetUnit) &&
    Number.isFinite(Number(budgetLimit)) &&
    Number(budgetLimit) > 0 &&
    approvedBudget !== undefined &&
    Number(budgetLimit) <= approvedBudget &&
    /^[1-9][0-9]*$/.test(maxAttempts) &&
    Number(maxAttempts) <= selectedTemplate.experiment_contract.max_attempts_limit &&
    Number.isFinite(Number(minimumImprovement)) &&
    Number(minimumImprovement) >= 0
  const canCreate =
    live &&
    !pending &&
    limitsValid &&
    Boolean(validation?.ready) &&
    campaignId.length > 0 &&
    title.length > 0

  return (
    <section className="mx-auto w-full max-w-[1200px]" aria-labelledby="guided-setup-title">
      <header className="border-b border-border-subtle pb-3">
        <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">
          Campaign intake
        </p>
        <h2 id="guided-setup-title" className="mt-1 font-brand text-xl text-text-primary">
          Guided setup
        </h2>
        <p className="mt-1 max-w-2xl text-sm leading-6 text-text-secondary">
          Choose from what's registered to this installation — your hardware, your model at its
          pinned revision. Need something that isn't listed? Register it first, or ask your agent to
          set it up.
        </p>
        <p className="mt-2 max-w-2xl text-xs leading-5 text-text-secondary">
          Your research agent runs in its own host with its own provider configuration. The learner
          model, training data, evaluation and execution environment below are separate choices.
          Preparation ends at READY; Start is a later human decision.
        </p>
        <details className="mt-3 text-xs leading-5 text-text-secondary">
          <summary className="cursor-pointer font-semibold text-text-primary">
            Prepare with your agent or CLI
          </summary>
          <p className="mt-2">
            Ask your agent to resume preparation for workspace{' '}
            <code>{context?.workspace_id || 'the current project'}</code>, inspect existing assets,
            and register the selected inputs. Ask only for missing choices, then present the exact
            READY contract for a later Start decision.
          </p>
          <p className="mt-1">
            To inspect the supported preparation commands, run{' '}
            <code>bashgym research prepare --help</code> in the same installation. Return here and
            refresh registrations to resume the saved choices.
          </p>
        </details>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={!live || pending}
            onClick={onRetry}
          >
            <RefreshCw className="h-3.5 w-3.5" /> Refresh registrations
          </Button>
          <a href="#resources" className="text-xs text-accent-dark underline">
            Inspect resources and optional Data Designer
          </a>
        </div>
      </header>

      {context?.preparation_inventory ? (
        <aside
          aria-label="Existing workspace assets"
          className="mt-3 border-b border-border-subtle pb-3 text-xs text-text-secondary"
        >
          <p className="font-semibold text-text-primary">Existing workspace assets</p>
          <ul className="mt-1 grid gap-1 sm:grid-cols-2">
            {(['models', 'data', 'evaluation', 'compute'] as const).map((kind) => {
              const assets = context.preparation_inventory!.candidates[kind]
              return (
                <li key={kind}>
                  {kind === 'models' ? 'Models' : labels[kind]}: {assets.length}
                  {assets.length
                    ? `: ${assets
                        .slice(0, 3)
                        .map((asset) => asset.label)
                        .join(', ')}${assets.length > 3 ? ', ...' : ''}`
                    : ''}
                </li>
              )
            })}
          </ul>
          <p className="mt-2">
            These records describe existing assets. Each choice still needs an approved installation
            binding before it can be selected.
          </p>
          <p className="mt-1">
            Inventory checked{' '}
            <time dateTime={context.preparation_inventory.checked_at}>
              {context.preparation_inventory.checked_at}
            </time>
            . Metadata only; execution and recipe readiness still need verification.
          </p>
          {context.preparation_inventory.truncated ? (
            <p className="mt-1">Showing a bounded inventory.</p>
          ) : null}
        </aside>
      ) : null}

      {!live ? (
        <div
          className="mt-3 flex items-start justify-between gap-3 border-l-4 border-status-warning bg-status-warning/10 px-3 py-2"
          role={error ? 'alert' : 'status'}
        >
          <div className="flex min-w-0 items-start gap-2">
            <ShieldAlert
              className="mt-0.5 h-4 w-4 shrink-0 text-status-warning"
              aria-hidden="true"
            />
            <div>
              <p className="text-sm font-semibold text-text-primary">Live authority is offline</p>
              <p className="mt-0.5 text-xs leading-5 text-text-secondary">
                {error
                  ? `${error} Your registered choices reappear and unlock when the connection returns.`
                  : 'You can browse the steps; choices reappear and unlock when the connection returns.'}
              </p>
            </div>
          </div>
          <Button type="button" variant="secondary" size="sm" disabled={pending} onClick={onRetry}>
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </Button>
        </div>
      ) : null}

      <div className="mt-4 grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <main className="min-w-0 border-t-2 border-text-primary" aria-label="Guided setup choices">
          <ol aria-label="Six registered setup choices">
            {guidedSetupSteps.map((step, index) => {
              const sealed = Boolean(context && index < (context.session?.version ?? 0))
              const current = activeStep === step
              const selection = context ? selectionFor(context, step) : null
              return (
                <li
                  key={step}
                  className={clsx(
                    'border-b border-border-subtle py-3',
                    current && 'bg-accent/5 px-3'
                  )}
                  aria-current={current ? 'step' : undefined}
                >
                  <div className="flex items-start gap-3">
                    <span
                      className={clsx(
                        'mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center border-2 font-mono text-[10px] font-bold',
                        sealed
                          ? 'border-status-success bg-status-success/10 text-status-success'
                          : 'border-border-subtle text-text-muted'
                      )}
                    >
                      {sealed ? <Check className="h-3.5 w-3.5" aria-hidden="true" /> : index + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <h3 className="font-brand text-base text-text-primary">{labels[step]}</h3>
                        <span className="font-mono text-[9px] uppercase tracking-widest text-text-muted">
                          {sealed ? 'Sealed' : current ? 'Choose now' : 'Pending'}
                        </span>
                      </div>
                      {selection ? (
                        <p className="mt-1 break-all font-mono text-[11px] text-text-secondary">
                          {selection}
                        </p>
                      ) : null}
                      {current ? (
                        <div className="mt-3 grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
                          <label className="text-[11px] font-medium text-text-secondary">
                            Registered {labels[step].toLowerCase()}
                            <select
                              className="input mt-1 w-full font-mono text-xs"
                              value={selectedOptionId}
                              disabled={!live || pending}
                              onChange={(event) =>
                                onSelectedOptionChange(event.currentTarget.value)
                              }
                            >
                              <option value="">Select a registered choice</option>
                              {(view?.options ?? []).map((option) => (
                                <option
                                  key={option.id}
                                  value={option.id}
                                  disabled={!option.selectable}
                                >
                                  {option.label}
                                  {option.detail ? ` · ${option.detail}` : ''}
                                </option>
                              ))}
                            </select>
                          </label>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            disabled={!canSave}
                            onClick={onAdvance}
                          >
                            {pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null} Save
                            choice
                          </Button>
                          {selectedOption?.reasonCodes.length ? (
                            <p className="text-xs text-status-warning sm:col-span-2" role="status">
                              {selectedOption.reasonCodes.map(reason).join(', ')}.
                              {step === 'installation'
                                ? ' Saving this installation does not verify its recipe inputs.'
                                : ' Resolve these blockers before saving this choice.'}
                            </p>
                          ) : null}
                          {!view?.options.some((option) => option.selectable) ? (
                            <p className="text-xs text-status-warning sm:col-span-2">
                              No matching reachable registration is available for this contract.
                              Resume preparation with your agent or CLI using the guidance above,
                              then refresh registrations. Your saved choices are preserved.
                            </p>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  </div>
                </li>
              )
            })}
          </ol>

          {!context ? (
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border-subtle py-3 text-xs text-text-secondary">
              <span>Registered choices will appear here after reconnection.</span>
              <Button type="button" variant="secondary" size="sm" disabled>
                Save choice
              </Button>
            </div>
          ) : null}

          {activeStep === 'doctor' ? (
            <section className="py-4" aria-labelledby="guided-doctor-title">
              <h3 id="guided-doctor-title" className="font-brand text-lg text-text-primary">
                Verify materialization
              </h3>
              <p className="mt-1 text-xs leading-5 text-text-secondary">
                Choose the experiment limits, then verify the registered bindings. Validation seals
                both before campaign creation.
              </p>
              {selectedTemplate ? (
                <p className="mt-2 border-l-2 border-accent px-3 py-2 font-mono text-[10px] leading-5 text-text-secondary">
                  Primary metric · {selectedTemplate.experiment_contract.primary_metric} (
                  {selectedTemplate.experiment_contract.metric_direction})
                  <br />
                  Approved ceilings · {selectedTemplate.experiment_contract.max_attempts_limit}{' '}
                  attempts ·{' '}
                  {Object.entries(selectedTemplate.experiment_contract.budget_limits)
                    .map(([unit, limit]) => `${limit} ${unit}`)
                    .join(' · ')}
                  <br />
                  Protected metrics ·{' '}
                  {selectedTemplate.experiment_contract.protected_metrics.length
                    ? selectedTemplate.experiment_contract.protected_metrics
                        .map(
                          (gate) =>
                            `${gate.metric_name} (${gate.direction}, max regression ${gate.max_regression})`
                        )
                        .join(' · ')
                    : 'none'}
                </p>
              ) : null}
              <fieldset className="mt-3 grid gap-3 sm:grid-cols-2" aria-label="Experiment limits">
                <label className="text-[11px] font-medium text-text-secondary">
                  Maximum attempts (baseline + candidates)
                  <input
                    className="input mt-1 w-full font-mono text-xs"
                    inputMode="numeric"
                    value={maxAttempts}
                    placeholder="5"
                    onChange={(event) => onMaxAttemptsChange(event.currentTarget.value)}
                  />
                </label>
                <label className="text-[11px] font-medium text-text-secondary">
                  Budget unit
                  <input
                    className="input mt-1 w-full font-mono text-xs"
                    value={budgetUnit}
                    placeholder="gpu_hours"
                    onChange={(event) => onBudgetUnitChange(event.currentTarget.value)}
                  />
                </label>
                <label className="text-[11px] font-medium text-text-secondary">
                  Campaign budget limit
                  <input
                    className="input mt-1 w-full font-mono text-xs"
                    inputMode="decimal"
                    value={budgetLimit}
                    placeholder="10"
                    onChange={(event) => onBudgetLimitChange(event.currentTarget.value)}
                  />
                </label>
                <label className="text-[11px] font-medium text-text-secondary">
                  Minimum improvement
                  <input
                    className="input mt-1 w-full font-mono text-xs"
                    inputMode="decimal"
                    value={minimumImprovement}
                    placeholder="0.01"
                    onChange={(event) => onMinimumImprovementChange(event.currentTarget.value)}
                  />
                </label>
              </fieldset>
              {!limitsValid ? (
                <p className="mt-2 text-xs text-status-warning">
                  Enter an attempt count and budget within the approved ceilings, plus a
                  non-negative improvement threshold.
                </p>
              ) : null}
              {doctor ? (
                <div
                  className={clsx(
                    'mt-3 border-l-2 px-3 py-2 text-xs',
                    doctor.ready
                      ? 'border-status-success bg-status-success/10'
                      : 'border-status-warning bg-status-warning/10'
                  )}
                >
                  <p className="font-semibold text-text-primary">
                    {doctor.ready ? 'Doctor is ready' : 'Doctor found blockers'}
                  </p>
                  {doctor.blocking_codes.length ? (
                    <p className="mt-1 text-text-secondary">
                      {doctor.blocking_codes.map(reason).join(', ')}
                    </p>
                  ) : null}
                </div>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  size="sm"
                  disabled={!live || pending || !limitsValid}
                  onClick={onDoctor}
                >
                  Run doctor
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  size="sm"
                  disabled={!live || pending || !limitsValid || !doctor?.ready}
                  onClick={onValidate}
                >
                  Seal validation
                </Button>
              </div>
            </section>
          ) : null}

          {validation?.ready ? (
            <section
              className="border-t border-border-subtle py-4"
              aria-labelledby="guided-create-title"
            >
              <h3 id="guided-create-title" className="font-brand text-lg text-text-primary">
                Create the campaign
              </h3>
              <p className="mt-1 text-xs leading-5 text-text-secondary">
                Creation is idempotent. After it succeeds, BashGym opens the campaign’s authority
                rail for the separate human Start decision.
              </p>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <label className="text-[11px] font-medium text-text-secondary">
                  Campaign ID
                  <input
                    className="input mt-1 w-full font-mono text-xs"
                    value={campaignId}
                    maxLength={160}
                    autoComplete="off"
                    onChange={(event) => onCampaignIdChange(event.currentTarget.value)}
                  />
                </label>
                <label className="text-[11px] font-medium text-text-secondary">
                  Campaign title
                  <input
                    className="input mt-1 w-full text-xs"
                    value={title}
                    maxLength={240}
                    autoComplete="off"
                    onChange={(event) => onTitleChange(event.currentTarget.value)}
                  />
                </label>
              </div>
              <Button
                type="button"
                className="mt-3"
                variant="primary"
                size="sm"
                disabled={!canCreate}
                onClick={onCreate}
              >
                Create and review Start
              </Button>
            </section>
          ) : null}
        </main>

        <aside className="border-l-2 border-text-primary pl-4" aria-label="Setup authority">
          <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">
            Authority
          </p>
          <p className="mt-2 font-brand text-lg text-text-primary">
            {view?.completedCount ?? 0} of 6 choices sealed
          </p>
          <p className="mt-1 text-xs leading-5 text-text-secondary">
            {context?.session
              ? 'This receipt chain can resume on the same workspace.'
              : 'A workspace-scoped session begins with the first saved choice.'}
          </p>
          <dl className="mt-4 space-y-3 border-t border-border-subtle pt-3 text-xs">
            <div>
              <dt className="font-mono text-[9px] uppercase tracking-widest text-text-muted">
                Connection
              </dt>
              <dd className="mt-1 capitalize text-text-primary">{connectionState}</dd>
            </div>
            <div>
              <dt className="font-mono text-[9px] uppercase tracking-widest text-text-muted">
                Session
              </dt>
              <dd className="mt-1 break-all font-mono text-[10px] text-text-secondary">
                {context?.session?.session_id || 'Not sealed yet'}
              </dd>
            </div>
            <div>
              <dt className="font-mono text-[9px] uppercase tracking-widest text-text-muted">
                Latest receipt
              </dt>
              <dd className="mt-1 break-all font-mono text-[10px] text-text-secondary">
                {context?.session?.latest_receipt.receipt_id || 'None'}
              </dd>
            </div>
          </dl>
          {view?.truncationReasons.length ? (
            <div className="mt-4 border-l-2 border-status-warning pl-3 text-xs leading-5 text-text-secondary">
              <p className="font-semibold text-text-primary">Discovery was bounded</p>
              <p>
                {view.truncationReasons.map(reason).join(', ')}. Narrow the installation registry or
                use the CLI to inspect the full set.
              </p>
            </div>
          ) : null}
          {context?.reason_codes.length ? (
            <ul
              className="mt-4 space-y-1 text-[10px] text-text-muted"
              aria-label="Setup reason codes"
            >
              {context.reason_codes.slice(0, 6).map((code) => (
                <li key={code} className="flex items-start gap-2">
                  <Circle className="mt-1 h-2.5 w-2.5 shrink-0" />
                  {reason(code)}
                </li>
              ))}
            </ul>
          ) : null}
        </aside>
      </div>
    </section>
  )
}
