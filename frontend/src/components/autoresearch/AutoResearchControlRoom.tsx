import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  Activity,
  AlertTriangle,
  Archive,
  CheckCircle2,
  Circle,
  Database,
  Loader2,
  MoreHorizontal,
  RefreshCw,
  WifiOff,
} from 'lucide-react'
import { clsx } from 'clsx'

import { campaignApi } from '../../services/api'
import { wsService } from '../../services/websocket'
import type { CampaignArtifact, CampaignEventItem, CampaignRecord } from '../../stores/campaignStore'
import {
  retainCampaignSafetyReconcile,
  useCampaignStore,
} from '../../stores/campaignStore'
import {
  partitionCampaignsByArchive,
  selectArchivedIds,
  useCampaignArchiveStore,
} from '../../stores/campaignArchive'
import { useUIStore } from '../../stores/uiStore'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import { Button } from '../common/Button'
import { CampaignLibrary } from './CampaignLibrary'
import { CampaignOutcomeSummary } from './CampaignOutcomeSummary'
import { projectCampaignOutcome, type CampaignOutcomeViewModel } from './campaignOutcomeModel'
import { CampaignRecoveryPanel } from './CampaignRecoveryPanel'
import { GuidedAutoResearchSetup, type GuidedSetupConnectionState } from './GuidedAutoResearchSetup'
import { HumanOversightQueue, type HumanReviewResponse } from './HumanOversightQueue'
import {
  CampaignEvidenceDialog,
  CampaignEvidenceRow,
  type CampaignEvidenceSelection,
} from './CampaignEvidenceInspector'
import type { CampaignRecoveryPublicV1, RecoveryAction, RecoveryRequest } from './campaignRecoveryModel'
import { buildControlRoomModel, campaignStatusTone, resolveControlRoomFreshness, type ControlRoomViewModel, type PresentationTone } from './controlRoomModel'
import { resolveControlRoomCampaignSelection, shouldCanonicalizeControlRoomSelection } from './controlRoomSelection'
import { controlRoomSnapshotKey, useControlRoomSnapshotCache } from './controlRoomSnapshotCache'
import { createRecoveryLoadGate } from './recoveryLoadGate'
import {
  buildHumanOversightModel,
  humanReviewResponseKey,
  parseHumanWorkQueue,
  type HumanPromotionRequest,
  type HumanWorkClaimRequest,
  type HumanWorkSubmitRequest,
  type ParsedHumanWorkQueuePublicV1,
} from './humanWorkModel'
import {
  buildGuidedSetupView,
  type GuidedSetupContext,
  type GuidedSetupDoctor,
  type GuidedSetupValidation,
} from './guidedSetupModel'
import {
  clearGuidedSetupIdempotencyKey,
  getOrCreateGuidedSetupIdempotencyKey,
  getOrCreateGuidedSetupSessionId,
  readGuidedSetupSessionId,
} from './guidedSetupSessionStorage'

export type LifecycleAction = 'start' | 'pause' | 'resume' | 'cancel' | 'conclude'

/** Non-terminal statuses from which a campaign can still be cancelled. */
const CANCELLABLE_STATUSES = new Set(['validating', 'ready', 'active', 'paused', 'awaiting_authority'])

export interface ControlRoomContentProps {
  model: ControlRoomViewModel
  campaigns: CampaignRecord[]
  selectedCampaignId: string | null
  events: CampaignEventItem[]
  artifacts: CampaignArtifact[]
  outcome?: CampaignOutcomeViewModel | null
  pages: {
    eventsLoading: boolean
    eventsError: string | null
    eventsLoaded: boolean
    eventsHasMore: boolean
    artifactsLoading: boolean
    artifactsError: string | null
    artifactsLoaded: boolean
    artifactsHasMore: boolean
  }
  onSelect: (campaignId: string) => void
  onRetry: () => void
  onLoadEvents: () => void
  onLoadArtifacts: () => void
  onTransition?: (action: LifecycleAction) => void
  transitionPending?: LifecycleAction | null
  onArchive?: (() => void) | null
  humanOversight?: ReactNode
  campaignRecovery?: ReactNode
  guidedSetup?: ReactNode
  campaignLibrary?: ReactNode
}

const unloadedPages: ControlRoomContentProps['pages'] = {
  eventsLoading: false,
  eventsError: null,
  eventsLoaded: false,
  eventsHasMore: true,
  artifactsLoading: false,
  artifactsError: null,
  artifactsLoaded: false,
  artifactsHasMore: true,
}

const toneClasses: Record<PresentationTone, string> = {
  success: 'border-status-success/60 bg-status-success/10 text-status-success',
  warning: 'border-status-warning/60 bg-status-warning/10 text-status-warning',
  error: 'border-status-error/60 bg-status-error/10 text-status-error',
  info: 'border-accent/60 bg-accent/10 text-accent-dark',
  neutral: 'border-border-subtle bg-background-secondary text-text-secondary',
}

function readable(value: string): string {
  const words = value.replace(/^campaign:/, '').replace(/[_:-]+/g, ' ').trim()
  return words ? words.replace(/\b\w/g, (letter) => letter.toUpperCase()) : 'Unknown'
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value)
}

function StatusPill({ label, tone = 'neutral' }: { label: string; tone?: PresentationTone }) {
  const dotClass = tone === 'success'
    ? 'status-success'
    : tone === 'warning'
      ? 'status-warning'
      : tone === 'error'
        ? 'status-error'
        : tone === 'info'
          ? 'border-accent bg-accent'
          : 'border-text-muted bg-text-muted'
  return (
    <span className={clsx('inline-flex items-center gap-1.5 rounded-brutal border-brutal px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-wide', toneClasses[tone])}>
      <span className={clsx('status-dot', dotClass)} aria-hidden="true" />
      {label}
    </span>
  )
}

function CampaignSelector({
  campaigns,
  selectedCampaignId,
  onSelect,
}: Pick<ControlRoomContentProps, 'campaigns' | 'selectedCampaignId' | 'onSelect'>) {
  return (
    <div className="min-w-0 w-full sm:max-w-xl">
      <label htmlFor="autoresearch-campaign" className="mb-1 block font-mono text-[10px] font-bold uppercase tracking-widest text-text-muted">
        Campaign
      </label>
      <select
        id="autoresearch-campaign"
        className="input w-full font-mono text-sm"
        value={selectedCampaignId || ''}
        onChange={(event) => event.target.value && onSelect(event.target.value)}
      >
        <option value="" disabled>Select a campaign</option>
        {campaigns.map((campaign) => (
          <option key={campaign.campaign_id} value={campaign.campaign_id}>
            {campaign.title} · {readable(campaign.status)}
          </option>
        ))}
      </select>
    </div>
  )
}

function ControlRoomHeader(props: Pick<ControlRoomContentProps, 'campaigns' | 'selectedCampaignId' | 'onSelect'> & { withSelector?: boolean }) {
  const { withSelector = true, ...selectorProps } = props
  return (
    <header className="flex flex-col gap-3 border-b border-border-subtle pb-3 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        <h1 className="font-brand text-2xl text-text-primary">AutoResearch Control Room</h1>
        <p className="mt-1 text-sm text-text-secondary">Run bounded experiments on registered private compute with durable evidence and human oversight.</p>
      </div>
      {withSelector ? (
        <div className="flex w-full flex-col gap-2 sm:flex-row sm:items-end lg:max-w-2xl lg:justify-end">
          <CampaignSelector {...selectorProps} />
        </div>
      ) : null}
    </header>
  )
}

const unavailableJourney = ['Setup', 'Baseline', 'Experiments', 'Human review', 'Decision']

function UnavailableControlRoomGrid({ loading = false }: { loading?: boolean }) {
  const placeholder = loading ? 'Loading campaign state…' : 'Awaiting verified campaign state'
  return (
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_19rem]" aria-busy={loading}>
      <section className="card p-4" aria-labelledby="unavailable-journey-title">
        <div className="flex items-center justify-between gap-3">
          <h2 id="unavailable-journey-title" className="font-brand text-lg text-text-primary">Journey</h2>
          <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Five durable phases</span>
        </div>
        <ol className="mt-3 grid gap-2 md:grid-cols-5">
          {unavailableJourney.map((phase, index) => (
            <li key={phase} className="rounded-brutal border-brutal border-border-subtle bg-background-secondary p-3 text-text-muted">
              <div className="flex items-center justify-between font-mono text-[10px] font-bold uppercase tracking-wide">
                <span>{index + 1}</span><Circle className="h-4 w-4" />
              </div>
              <h3 className="mt-3 font-brand text-base text-text-primary">{phase}</h3>
              <div className="mt-1 font-mono text-[9px] uppercase tracking-wide">Awaiting verified state</div>
            </li>
          ))}
        </ol>
        <div className="mt-4 border-t border-border-subtle pt-3">
          <h3 className="font-brand text-base text-text-primary">Active work &amp; Recent activity</h3>
          <p className="mt-1 text-xs text-text-muted">{placeholder}. Existing evidence remains visible as soon as verified state is restored.</p>
        </div>
      </section>
      <aside className="card p-4" aria-label="Campaign authority summary">
        <div className="flex items-center gap-2 text-text-primary"><Database className="h-4 w-4 text-accent" /><h2 className="font-brand text-lg">Campaign overview</h2></div>
        <p className="mt-2 text-xs text-text-muted">{placeholder}</p>
        <dl className="mt-4 space-y-2 border-t border-border-subtle pt-3 text-xs text-text-secondary">
          <div className="flex justify-between gap-3"><dt>Controller</dt><dd>Unverified</dd></div>
          <div className="flex justify-between gap-3"><dt>Evidence &amp; metrics</dt><dd>Preserved</dd></div>
          <div className="flex justify-between gap-3"><dt>Budget</dt><dd>Read only</dd></div>
          <div className="flex justify-between gap-3"><dt>Collection counts</dt><dd>Read only</dd></div>
        </dl>
      </aside>
    </div>
  )
}

function ControlRoomStateNotice({
  model,
  onRetry,
  noCampaigns = false,
}: {
  model: Exclude<ControlRoomViewModel, { kind: 'snapshot' }>
  onRetry: () => void
  noCampaigns?: boolean
}) {
  const loading = model.kind === 'loading'
  const offline = model.kind === 'offline'
  const error = model.kind === 'error'
  const title = noCampaigns && model.kind === 'empty'
    ? 'No durable AutoResearch campaigns yet'
    : loading
    ? 'Loading AutoResearch control room'
    : offline
      ? 'AutoResearch service is offline'
      : error
        ? 'AutoResearch control room unavailable'
        : model.kind === 'empty'
          ? 'Select a campaign'
          : 'No durable AutoResearch campaigns yet'
  const icon = loading
    ? <Loader2 className="h-5 w-5 animate-spin" />
    : offline
      ? <WifiOff className="h-5 w-5" />
      : error
        ? <AlertTriangle className="h-5 w-5" />
        : <Database className="h-5 w-5" />

  return (
    <section className={clsx('card border-l-4 p-4', offline || error ? 'border-l-status-warning' : 'border-l-accent')} role="status" aria-live="polite">
      <div className="flex items-start gap-3">
        <span className={clsx('mt-0.5 shrink-0', offline || error ? 'text-status-warning' : 'text-accent')}>{icon}</span>
        <div className="min-w-0 flex-1">
          <h2 className="font-brand text-lg text-text-primary">{title}</h2>
          <p className="mt-1 text-sm leading-6 text-text-secondary">{model.message}</p>
          {offline || error ? (
            <p className="mt-2 font-mono text-[10px] uppercase tracking-wide text-status-warning">
              Live authority unavailable · Last-known campaign data will remain visible when available
            </p>
          ) : null}
        </div>
        {!loading && (offline || error) ? (
          <Button variant="secondary" size="sm" onClick={onRetry} leftIcon={<RefreshCw className="h-4 w-4" />}>Retry</Button>
        ) : null}
      </div>
    </section>
  )
}

type SnapshotViewModel = Extract<ControlRoomViewModel, { kind: 'snapshot' }>

function FreshnessIndicator({ model }: { model: SnapshotViewModel }) {
  const [display, setDisplay] = useState(model.freshness)
  useEffect(() => {
    if (model.freshness === 'live' || display !== 'live') {
      setDisplay(model.freshness)
      return
    }
    const timer = setTimeout(() => setDisplay(model.freshness), 2500)
    return () => clearTimeout(timer)
  }, [model.freshness, display])
  const smoothed = display === 'live' && model.freshness !== 'live'
  const shown = smoothed ? 'live' : model.freshness
  const dotClass = shown === 'live'
    ? 'status-success'
    : shown === 'reconciling' || shown === 'stale'
      ? 'status-warning'
      : 'status-error'
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 font-mono text-[10px] font-bold uppercase tracking-wide text-text-secondary"
      role="status"
      aria-live="polite"
    >
      <span className={clsx('status-dot', dotClass)} aria-hidden="true" />
      {smoothed ? 'Live' : model.freshnessLabel}
    </span>
  )
}

function CommandStrip({
  model,
  campaigns,
  selectedCampaignId,
  onSelect,
  onTransition,
  onRefresh,
  onOpenRecovery,
  onArchive = null,
  transitionPending = null,
  closeoutPending,
  library,
}: {
  model: SnapshotViewModel
  campaigns: CampaignRecord[]
  selectedCampaignId: string | null
  onSelect: (campaignId: string) => void
  onTransition?: (action: LifecycleAction) => void
  onRefresh: () => void
  onOpenRecovery: () => void
  onArchive?: (() => void) | null
  transitionPending?: LifecycleAction | null
  closeoutPending: boolean
  library?: ReactNode
}) {
  const { campaign, readiness } = model.snapshot
  const status = campaign.status
  const canMutate = Boolean(onTransition && model.authoritative && !transitionPending)
  const startEnabled = canMutate && status === 'ready' && readiness.launch_ready
  const startReason = status !== 'ready'
    ? `Start becomes available from Ready. Current state: ${readable(status)}.`
    : !model.authoritative
      ? 'Live data is required. Cached state never enables Start.'
      : !readiness.launch_ready
        ? 'Readiness is blocked. Resolve the blocker, then refresh.'
        : 'Ready to start. The server rechecks every binding and lease before launch.'
  const pendingIcon = (action: LifecycleAction) =>
    transitionPending === action ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : undefined
  return (
    <header className="card flex flex-wrap items-center gap-x-3 gap-y-2 p-3" aria-label="Campaign command bar">
      <div className="min-w-0 max-w-[24rem] flex-1 sm:flex-none">
        <label htmlFor="autoresearch-campaign" className="sr-only">Campaign</label>
        <select
          id="autoresearch-campaign"
          className="input w-full font-mono text-xs"
          value={selectedCampaignId || ''}
          onChange={(event) => event.target.value && onSelect(event.target.value)}
        >
          <option value="" disabled>Select a campaign</option>
          {campaigns.map((record) => (
            <option key={record.campaign_id} value={record.campaign_id}>
              {record.title} · {readable(record.status)}
            </option>
          ))}
        </select>
      </div>
      <StatusPill label={readable(campaign.status)} tone={campaignStatusTone(campaign.status)} />
      <FreshnessIndicator model={model} />
      <span className="shrink-0 font-mono text-[10px] uppercase tracking-wide text-text-muted">
        {readable(campaign.kind)} · manifest r{model.snapshot.manifest_revision}
      </span>
      <BudgetChip model={model} />
      <div className="hidden flex-1 sm:block" />
      {status === 'ready' ? (
        <Button variant="primary" size="sm" onClick={() => onTransition?.('start')} disabled={!startEnabled} aria-label="Start campaign" title={startReason} leftIcon={pendingIcon('start')}>
          {transitionPending === 'start' ? 'Starting' : 'Start campaign'}
        </Button>
      ) : null}
      {status === 'active' && closeoutPending ? (
        <Button variant="primary" size="sm" onClick={() => onTransition?.('conclude')} disabled={!canMutate} title="Close this finished campaign and preserve its results." leftIcon={pendingIcon('conclude')}>
          {transitionPending === 'conclude' ? 'Closing' : 'Close campaign'}
        </Button>
      ) : null}
      {status === 'active' && !closeoutPending ? (
        <Button variant="secondary" size="sm" onClick={() => onTransition?.('pause')} disabled={!canMutate} title="Pause scheduling. Running work finishes; no new actions are claimed." leftIcon={pendingIcon('pause')}>
          {transitionPending === 'pause' ? 'Pausing' : 'Pause'}
        </Button>
      ) : null}
      {status === 'paused' ? (
        <Button variant="primary" size="sm" onClick={() => onTransition?.('resume')} disabled={!canMutate} title="Resume scheduling from where the campaign paused." leftIcon={pendingIcon('resume')}>
          {transitionPending === 'resume' ? 'Resuming' : 'Resume'}
        </Button>
      ) : null}
      {CANCELLABLE_STATUSES.has(status) && !closeoutPending ? (
        <Button variant="ghost" size="sm" onClick={() => onTransition?.('cancel')} disabled={!canMutate} title="Stop the campaign. Scheduled work is halted; this cannot be undone." leftIcon={pendingIcon('cancel')}>
          {transitionPending === 'cancel' ? 'Cancelling' : 'Cancel'}
        </Button>
      ) : null}
      {library}
      <details className="relative shrink-0">
        <summary
          aria-label="More actions"
          className="btn-ghost flex h-8 w-8 cursor-pointer list-none items-center justify-center rounded-brutal"
        >
          <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
        </summary>
        <div className="absolute right-0 z-10 mt-1 w-48 rounded-brutal border-brutal border-border-subtle bg-background-card p-1 shadow-brutal-sm">
          <button
            type="button"
            onClick={onRefresh}
            className="flex w-full items-center gap-2 rounded-brutal px-2 py-1.5 text-left text-xs text-text-primary hover:bg-background-secondary"
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" /> Refresh
          </button>
          <button
            type="button"
            onClick={onOpenRecovery}
            className="flex w-full items-center gap-2 rounded-brutal px-2 py-1.5 text-left text-xs text-text-primary hover:bg-background-secondary"
          >
            <Database className="h-3.5 w-3.5" aria-hidden="true" /> Open recovery
          </button>
          {onArchive ? (
            <button
              type="button"
              onClick={onArchive}
              title="Hide this campaign from the selector and canvas. The durable record stays in the Library."
              className="flex w-full items-center gap-2 rounded-brutal px-2 py-1.5 text-left text-xs text-text-primary hover:bg-background-secondary"
            >
              <Archive className="h-3.5 w-3.5" aria-hidden="true" /> Archive campaign
            </button>
          ) : null}
        </div>
      </details>
    </header>
  )
}

function NeedsYouLine({ model }: { model: SnapshotViewModel }) {
  const needs = model.needsYou
  if (!needs) return null
  return (
    <section
      className="card flex h-12 items-center gap-3 border-l-4 border-l-status-warning p-3"
      role="status"
      aria-live="polite"
      aria-label="What needs you"
    >
      <AlertTriangle className="h-4 w-4 shrink-0 text-status-warning" aria-hidden="true" />
      <p className="min-w-0 flex-1 truncate text-sm leading-6 text-text-primary" title={needs.sentence}>
        <span className="font-semibold">Waiting on you:</span> {needs.sentence}
      </p>
      {needs.code ? (
        <span className="shrink-0 font-mono text-[10px] uppercase tracking-wide text-text-muted">{needs.code}</span>
      ) : null}
    </section>
  )
}

function JourneyStepper({ model }: { model: SnapshotViewModel }) {
  const phases = model.journey
  return (
    <section
      className="card flex h-14 items-center gap-1 overflow-x-auto p-3"
      aria-label="Campaign journey"
    >
      <ol className="flex min-w-max flex-1 items-center gap-1">
        {phases.map((phase, index) => {
          const done = phase.tone === 'success'
          const current = phase.tone === 'info' || phase.tone === 'warning'
          return (
            <li key={phase.id} className="flex flex-1 items-center gap-1">
              <div className="flex items-center gap-2" title={`${phase.label} · ${phase.stateLabel}`}>
                <span
                  className={clsx(
                    'flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 font-mono text-[10px] font-bold',
                    done
                      ? 'border-status-success bg-status-success text-white'
                      : current
                        ? 'border-accent-dark bg-accent text-text-primary'
                        : 'border-border-subtle text-text-muted',
                  )}
                >
                  {done ? <CheckCircle2 className="h-3 w-3" aria-hidden="true" /> : index + 1}
                </span>
                <span
                  className={clsx(
                    'whitespace-nowrap font-mono text-[10px] font-bold uppercase tracking-wide',
                    current ? 'text-accent-dark' : done ? 'text-text-primary' : 'text-text-muted',
                  )}
                >
                  {phase.label}
                </span>
              </div>
              {index < phases.length - 1 ? <span className="h-0.5 flex-1 bg-border-subtle" aria-hidden="true" /> : null}
            </li>
          )
        })}
      </ol>
    </section>
  )
}

function BudgetChip({ model }: { model: SnapshotViewModel }) {
  const resource = model.snapshot.budget.resources[0]
  if (!resource) return null
  const used = Math.max(0, resource.settled + resource.reserved)
  const limit = resource.limit > 0 ? resource.limit : 0
  const fraction = limit > 0 ? Math.min(1, used / limit) : 0
  return (
    <span
      className="flex shrink-0 items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-text-muted"
      aria-label="Budget"
      title={`${readable(resource.unit)} · ${compactNumber(resource.settled)} settled · ${compactNumber(resource.reserved)} reserved / ${compactNumber(resource.limit)} limit`}
    >
      {readable(resource.unit)}
      <span className="flex h-1.5 w-16 overflow-hidden rounded-brutal border border-border-subtle bg-background-secondary">
        <span className="bg-accent" style={{ width: `${fraction * 100}%` }} aria-hidden="true" />
      </span>
      <span className="tabular-nums">{compactNumber(used)}/{compactNumber(resource.limit)}</span>
    </span>
  )
}

function ReadinessCheck({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className={clsx('flex items-baseline gap-2 py-0.5 text-xs', ok ? 'text-text-secondary' : 'font-semibold text-text-primary')}>
      {ok
        ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0 translate-y-0.5 text-status-success" aria-hidden="true" />
        : <AlertTriangle className="h-3.5 w-3.5 shrink-0 translate-y-0.5 text-status-error" aria-hidden="true" />}
      <span>{label}</span>
    </div>
  )
}

function ConfigPanel({ model }: { model: SnapshotViewModel }) {
  const { snapshot } = model
  const { bindings, campaign, controller, readiness } = snapshot
  const controllerTone: PresentationTone = controller.state === 'online' ? 'success' : controller.state === 'stale' ? 'warning' : 'error'
  const bindingsResolved = [bindings.model, bindings.data, bindings.evaluator, bindings.source, bindings.compute].every(Boolean)
  const primaryMetric = model.metrics[0]
  const metricValue = primaryMetric
    ? primaryMetric.target != null
      ? `${primaryMetric.label} · ${primaryMetric.direction} ${primaryMetric.target}${primaryMetric.unit ? ` ${primaryMetric.unit}` : ''}`
      : `${primaryMetric.label} · ${primaryMetric.direction}`
    : 'None published'
  const rows: Array<[string, string]> = [
    ['Objective', campaign.objective || '—'],
    ['Model', bindings.model?.display_label || 'Not bound'],
    ['Dataset', bindings.data?.display_label || 'Not bound'],
    ['Evaluator', bindings.evaluator?.display_label || 'Not bound'],
    ['Source', bindings.source?.display_label || 'Not bound'],
    ['Compute', bindings.compute?.display_label || 'Not bound'],
    ['Primary metric', metricValue],
  ]
  return (
    <FixedPanel
      title="Configuration"
      heightClass="h-[360px]"
      sectionId="config"
      action={<StatusPill label={model.controllerLabel} tone={controllerTone} />}
    >
      <dl className="text-xs">
        <div>
          <dt className="font-mono text-[10px] uppercase tracking-wide text-text-secondary">Objective</dt>
          <dd className="mt-0.5 text-text-primary" title={campaign.objective || undefined}>{campaign.objective || '—'}</dd>
        </div>
        <div className="mt-2 grid gap-x-4 gap-y-2 sm:grid-cols-2">
          {rows.filter(([label]) => label !== 'Objective').map(([label, value]) => (
            <div key={label} className="min-w-0">
              <dt className="font-mono text-[10px] uppercase tracking-wide text-text-secondary">{label}</dt>
              <dd className="mt-0.5 truncate font-mono text-text-primary" title={value}>{value}</dd>
            </div>
          ))}
        </div>
      </dl>
      <div className="mt-3 border-t border-border-subtle pt-2">
        <ReadinessCheck ok={controller.state === 'online'} label={`Controller ${controller.state}`} />
        <ReadinessCheck ok={bindingsResolved} label="Model, data, evaluator, source & compute bindings resolved" />
        <ReadinessCheck ok={readiness.materializable} label="Campaign is materializable" />
        <ReadinessCheck ok={readiness.launch_ready} label="Launch-ready verdict" />
        {readiness.blocking_codes.map((code) => (
          <ReadinessCheck key={code} ok={false} label={readable(code)} />
        ))}
        {model.snapshot.agents.length ? (
          <p className="mt-2 font-mono text-[10px] uppercase tracking-wide text-text-secondary">
            {model.snapshot.agents.length} agent session{model.snapshot.agents.length === 1 ? '' : 's'} attached
          </p>
        ) : null}
      </div>
      {controller.safe_guidance ? <p className="mt-3 rounded-brutal border border-border-subtle bg-background-secondary p-2 text-xs text-text-secondary">{controller.safe_guidance}</p> : null}
    </FixedPanel>
  )
}

function FixedPanel({
  title,
  heightClass,
  action,
  sectionId,
  children,
}: {
  title: string
  heightClass: string
  action?: ReactNode
  sectionId?: string
  children?: ReactNode
}) {
  const titleId = sectionId ? `${sectionId}-title` : undefined
  return (
    <section id={sectionId} className={clsx('card flex flex-col p-4', heightClass)} aria-labelledby={titleId}>
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border-subtle pb-2">
        <h2 id={titleId} className="flex items-center gap-2 whitespace-nowrap font-mono text-[11px] font-bold uppercase tracking-widest text-text-primary">
          <span className="h-2 w-2 shrink-0 bg-accent" aria-hidden="true" />
          {title}
        </h2>
        {action}
      </div>
      <div className="mt-3 min-h-0 flex-1 overflow-y-auto">{children}</div>
    </section>
  )
}

function ActiveWorkIndicator({ model }: { model: SnapshotViewModel }) {
  const work = model.snapshot.active_work
  const stageLabel = work ? readable(work.stage || 'stage pending') : 'Idle'
  const processState = work?.process_identity?.state
  const lightClass = !work
    ? 'border-text-muted bg-background-secondary'
    : processState === 'failed'
      ? 'status-error'
      : processState === 'paused'
        ? 'status-warning'
        : processState === 'completed'
          ? 'status-success'
          : 'border-accent bg-accent'
  return (
    <section
      className="card flex h-14 min-w-0 items-center gap-2 px-4"
      aria-label={`Active work: ${stageLabel}`}
      role="status"
    >
      <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-muted">Active work</span>
      <span className={clsx('status-dot shrink-0', lightClass)} aria-hidden="true" />
      <strong className="truncate font-mono text-[10px] font-bold uppercase tracking-wide text-text-primary" title={stageLabel}>{stageLabel}</strong>
    </section>
  )
}

function EvidenceMetricsPanel({
  model,
  artifacts,
  pages,
  onLoadArtifacts,
  onInspect,
  outcome,
}: Pick<ControlRoomContentProps, 'artifacts' | 'pages' | 'onLoadArtifacts' | 'outcome'> & {
  model: SnapshotViewModel
  onInspect: (selection: CampaignEvidenceSelection) => void
}) {
  const { snapshot } = model
  const { collections } = snapshot
  const orderedArtifacts = [...artifacts].sort((left, right) => (
    left.created_at.localeCompare(right.created_at) || left.artifact_id.localeCompare(right.artifact_id)
  ))
  return (
    <FixedPanel title="Evidence & metrics" heightClass="h-[360px]" sectionId="evidence">
      {model.metrics.length ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {model.metrics.map((metric) => {
            const hasValue = metric.value !== 'Unavailable'
            const comparedMetric = outcome?.metrics.find((item) => item.id === metric.id)
            const isProportion = /accuracy|reward|rate|mrr|pass|precision|recall|f1/i.test(metric.id)
            const comparedValue = comparedMetric?.candidate == null
              ? null
              : isProportion
                ? `${(comparedMetric.candidate * 100).toFixed(1)}%`
                : compactNumber(comparedMetric.candidate)
            const comparedBaseline = comparedMetric?.baseline == null
              ? null
              : isProportion
                ? `${(comparedMetric.baseline * 100).toFixed(1)}%`
                : compactNumber(comparedMetric.baseline)
            const bigNumber = comparedValue ?? (hasValue ? metric.value : metric.target != null ? compactNumber(metric.target) : '—')
            return (
              <div key={metric.id} className="rounded-brutal border border-border-subtle bg-background-secondary p-3">
                <div className="truncate font-mono text-[10px] uppercase tracking-wide text-text-secondary" title={metric.label}>{metric.label}</div>
                <div className="mt-1 flex items-baseline gap-1.5">
                  <span className="font-mono text-xl font-bold text-text-primary tabular-nums">{bigNumber}</span>
                  {metric.unit ? <span className="text-[10px] text-text-secondary">{metric.unit}</span> : null}
                  {!hasValue && metric.target != null ? <span className="font-mono text-[9px] uppercase tracking-wide text-text-muted">target</span> : null}
                </div>
                <div className="mt-1 text-[11px] text-text-secondary">
                  {comparedMetric
                    ? `${metric.direction} · baseline ${comparedBaseline || '—'} · compared above`
                    : `${metric.direction}${hasValue && metric.target != null ? ` · target ${compactNumber(metric.target)}${metric.unit || ''}` : ''}${hasValue ? '' : ' · no evaluations yet'}`}
                </div>
              </div>
            )
          })}
        </div>
      ) : <p className="text-xs text-text-muted">No metric descriptors are published.</p>}
      <dl className="mt-3 space-y-1 border-t border-border-subtle pt-3 text-xs">
        <div className="flex justify-between gap-3"><dt className="text-text-secondary">Champion</dt><dd className="min-w-0 truncate font-mono text-text-primary">{snapshot.champion ? `${snapshot.champion.candidate_ref} · ${readable(snapshot.champion.gate_state)}` : '—'}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-secondary">Candidate</dt><dd className="min-w-0 truncate font-mono text-text-primary">{snapshot.candidate?.candidate_ref || '—'}{snapshot.candidate ? ` · ${readable(snapshot.candidate.gate_state)}` : ''}</dd></div>
      </dl>
      <div className="mt-3 border-t border-border-subtle pt-3 font-mono text-[11px] tabular-nums text-text-secondary">
        events {collections.events.count} · artifacts {collections.artifacts.count} · studies {collections.studies.count} · attempts {collections.attempts.count} · comparisons {collections.comparisons.count}
      </div>
      {artifacts.length ? (
        <div className="mt-3 border-t border-border-subtle pt-3" role="region" aria-label="Inspectable campaign artifacts">
          <ul className="space-y-2">
            {orderedArtifacts.map((artifact) => {
              return (
                <li key={artifact.artifact_id}>
                  <CampaignEvidenceRow selection={{ kind: 'artifact', artifact }} onInspect={onInspect} />
                </li>
              )
            })}
          </ul>
        </div>
      ) : pages.artifactsLoaded ? <p className="mt-3 text-xs text-text-muted">No artifact rows were returned.</p> : null}
      {pages.artifactsError ? <p className="mt-3 text-xs text-status-error" role="alert">{pages.artifactsError}</p> : null}
      {pages.artifactsError || pages.artifactsHasMore ? (
        <Button
          className="mt-3"
          variant="ghost"
          size="sm"
          disabled={pages.artifactsLoading}
          onClick={onLoadArtifacts}
          leftIcon={pages.artifactsLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Database className="h-3.5 w-3.5" />}
        >
          {pages.artifactsLoading ? 'Loading artifacts' : pages.artifactsError ? 'Retry artifact page' : pages.artifactsLoaded ? 'Load more artifacts' : 'Load artifact page'}
        </Button>
      ) : null}
    </FixedPanel>
  )
}

function ActivityPanel({
  events,
  pages,
  onLoadEvents,
  onInspect,
}: Pick<ControlRoomContentProps, 'events' | 'pages' | 'onLoadEvents'> & {
  onInspect: (selection: CampaignEvidenceSelection) => void
}) {
  const orderedEvents = [...events].sort((left, right) => (
    right.cursor - left.cursor || right.event.event_id.localeCompare(left.event.event_id)
  ))
  return (
    <FixedPanel
      title="Campaign log"
      heightClass="h-[240px] lg:absolute lg:inset-0 lg:h-auto"
      sectionId="activity"
      action={pages.eventsError || pages.eventsHasMore ? (
        <Button
          variant="secondary"
          size="sm"
          disabled={pages.eventsLoading}
          onClick={onLoadEvents}
          leftIcon={pages.eventsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
        >
          {pages.eventsLoading ? 'Loading events' : pages.eventsError ? 'Retry event page' : pages.eventsLoaded ? 'Load more events' : 'Load event page'}
        </Button>
      ) : undefined}
    >
      {pages.eventsError ? <p className="mb-2 text-xs text-status-error" role="alert">{pages.eventsError}</p> : null}
      {events.length ? (
        <ol className="divide-y divide-border-subtle" aria-label="Inspectable campaign events">
          {orderedEvents.map(({ cursor, event }) => {
            const item = { cursor, event }
            return (
              <li key={event.event_id} className="py-1.5">
                <CampaignEvidenceRow selection={{ kind: 'event', item }} onInspect={onInspect} />
              </li>
            )
          })}
        </ol>
      ) : <p className="text-xs text-text-muted">{pages.eventsLoaded ? 'No event rows were returned.' : 'No event detail page is loaded.'}</p>}
    </FixedPanel>
  )
}

function SnapshotControlRoom(props: ControlRoomContentProps & { model: SnapshotViewModel }) {
  const { model, onLoadEvents } = props
  const { eventsLoaded, eventsLoading, eventsError } = props.pages
  const campaignId = model.snapshot.campaign_id
  const autoLoadedEventsFor = useRef<string | null>(null)
  const [evidenceSelection, setEvidenceSelection] = useState<CampaignEvidenceSelection | null>(null)
  useEffect(() => {
    if (autoLoadedEventsFor.current === campaignId) return
    if (eventsLoaded || eventsLoading || eventsError) return
    autoLoadedEventsFor.current = campaignId
    onLoadEvents()
  }, [campaignId, eventsLoaded, eventsLoading, eventsError, onLoadEvents])
  const recoveryRef = useRef<HTMLDivElement | null>(null)
  const handleOpenRecovery = useCallback(() => {
    recoveryRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [])
  return (
    <div className="space-y-3">
      <ControlRoomHeader campaigns={props.campaigns} selectedCampaignId={props.selectedCampaignId} onSelect={props.onSelect} withSelector={false} />
      <CommandStrip
        model={model}
        campaigns={props.campaigns}
        selectedCampaignId={props.selectedCampaignId}
        onSelect={props.onSelect}
        onTransition={props.onTransition}
        onRefresh={props.onRetry}
        onOpenRecovery={handleOpenRecovery}
        onArchive={props.onArchive}
        transitionPending={props.transitionPending}
        closeoutPending={props.outcome?.lifecycleLabel === 'Closeout pending'}
        library={props.campaignLibrary}
      />
      <NeedsYouLine model={model} />
      <section className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_14rem]" aria-label="Campaign progress and active work">
        <JourneyStepper model={model} />
        <ActiveWorkIndicator model={model} />
      </section>
      {props.outcome ? <CampaignOutcomeSummary model={props.outcome} /> : null}
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="min-w-0 space-y-3">
          <div className="grid gap-3 lg:grid-cols-2">
            <ConfigPanel model={model} />
            <EvidenceMetricsPanel model={model} artifacts={props.artifacts} pages={props.pages} onLoadArtifacts={props.onLoadArtifacts} onInspect={setEvidenceSelection} outcome={props.outcome} />
          </div>
        </div>
        <div className="min-w-0 lg:relative">
          <ActivityPanel events={props.events} pages={props.pages} onLoadEvents={props.onLoadEvents} onInspect={setEvidenceSelection} />
        </div>
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <FixedPanel title="Human oversight" heightClass="h-[280px]" sectionId="oversight">
          {props.humanOversight || <p className="text-sm text-text-secondary">Nothing needs your review. Blinded samples will appear here before any promotion.</p>}
        </FixedPanel>
        <div ref={recoveryRef}>
          <FixedPanel title="Campaign recovery" heightClass="h-[280px]" sectionId="recovery">
            {props.campaignRecovery}
          </FixedPanel>
        </div>
      </div>
      <CampaignEvidenceDialog selection={evidenceSelection} onClose={() => setEvidenceSelection(null)} />
    </div>
  )
}

export function ControlRoomContent(props: ControlRoomContentProps) {
  const { model, campaigns } = props
  if (model.kind !== 'snapshot') {
    const displayModel = campaigns.length === 0 && model.kind === 'empty'
      ? { ...model, message: 'BashGym will show every campaign here once its model, data, evaluator, source, and registered private-compute bindings have been resolved.' }
      : model
    return (
      <div className="space-y-4">
        <ControlRoomHeader campaigns={campaigns} selectedCampaignId={props.selectedCampaignId} onSelect={props.onSelect} />
        <ControlRoomStateNotice model={displayModel} onRetry={props.onRetry} noCampaigns={campaigns.length === 0} />
        {props.campaignLibrary}
        {campaigns.length === 0 && !props.campaignLibrary ? (
          props.guidedSetup ?? <GuidedAutoResearchSetup
            context={null}
            connectionState={({ loading: 'reconciling', offline: 'offline', error: 'error', empty: 'offline' } as Record<typeof model.kind, GuidedSetupConnectionState>)[model.kind]}
            pending={model.kind === 'loading'}
            error={model.kind === 'error' || model.kind === 'offline' ? model.message : null}
            doctor={null}
            validation={null}
            selectedOptionId=""
            campaignId=""
            title=""
            onSelectedOptionChange={() => {}}
            onCampaignIdChange={() => {}}
            onTitleChange={() => {}}
            onAdvance={() => {}}
            onDoctor={() => {}}
            onValidate={() => {}}
            onCreate={() => {}}
            onRetry={props.onRetry}
          />
        ) : <UnavailableControlRoomGrid loading={model.kind === 'loading'} />}
      </div>
    )
  }

  return <SnapshotControlRoom {...props} model={model} />
}

function HumanOversightUnavailable({
  loading,
  message,
  onRetry,
}: {
  loading: boolean
  message: string | null
  onRetry: () => void
}) {
  return (
    <section className="card p-4" aria-labelledby="human-oversight-unavailable-title">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="human-oversight-unavailable-title" className="font-brand text-xl text-text-primary">Human oversight queue</h2>
          <p className="mt-1 text-sm leading-6 text-text-secondary">Blinded reviews and explicit promotion decisions remain visible here when campaign services reconnect.</p>
        </div>
        <Button type="button" size="sm" variant="secondary" onClick={onRetry} disabled={loading}>
          {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
          {loading ? 'Reconciling' : 'Retry queue'}
        </Button>
      </div>
      <div className="mt-3 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs text-text-secondary" role={message ? 'alert' : 'status'}>
        {message || 'Loading the authoritative human-review queue. Review controls stay disabled until reconciliation completes.'}
      </div>
    </section>
  )
}

function newRecoveryIdempotencyKeys(): Record<RecoveryAction, string> {
  const create = () => {
    const bytes = new Uint8Array(16)
    globalThis.crypto.getRandomValues(bytes)
    return `idem_${Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')}`
  }
  return { resume: create(), repair: create(), takeover: create() }
}

function randomAuthorityHex(): string {
  const bytes = new Uint8Array(16)
  globalThis.crypto.getRandomValues(bytes)
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('')
}

function guidedSetupDraft(context: GuidedSetupContext) {
  const selections = context.session?.selections
  if (!selections?.template_id || !selections.installation_id) return null
  const { model, data, compute, evaluation } = selections.bindings
  if (!model || !data || !compute || !evaluation) return null
  return {
    workspaceId: context.workspace_id,
    templateId: selections.template_id,
    installationId: selections.installation_id,
    bindings: { model, data, compute, evaluation },
  }
}

export function AutoResearchControlRoom() {
  const [transitionPending, setTransitionPending] = useState<LifecycleAction | null>(null)
  const [humanQueue, setHumanQueue] = useState<ParsedHumanWorkQueuePublicV1 | null>(null)
  const [humanQueueError, setHumanQueueError] = useState<string | null>(null)
  const [humanQueueLoading, setHumanQueueLoading] = useState(false)
  const [humanMutationPending, setHumanMutationPending] = useState(false)
  const [humanResponses, setHumanResponses] = useState<Record<string, HumanReviewResponse | undefined>>({})
  const [recoverySnapshot, setRecoverySnapshot] = useState<CampaignRecoveryPublicV1 | null>(null)
  const [recoveryError, setRecoveryError] = useState<string | null>(null)
  const [recoveryUnregistered, setRecoveryUnregistered] = useState(false)
  const [recoveryLoading, setRecoveryLoading] = useState(false)
  const [recoveryConfirmations, setRecoveryConfirmations] = useState<Record<RecoveryAction, boolean>>({ resume: false, repair: false, takeover: false })
  const [recoveryIdempotencyKeys, setRecoveryIdempotencyKeys] = useState<Record<RecoveryAction, string>>(() => newRecoveryIdempotencyKeys())
  const [setupContext, setSetupContext] = useState<GuidedSetupContext | null>(null)
  const [setupConnection, setSetupConnection] = useState<GuidedSetupConnectionState>('reconciling')
  const [setupPending, setSetupPending] = useState(false)
  const [setupError, setSetupError] = useState<string | null>(null)
  const [setupSessionId, setSetupSessionId] = useState<string | null>(null)
  const [setupSelectedOptionId, setSetupSelectedOptionId] = useState('')
  const [setupDoctor, setSetupDoctor] = useState<GuidedSetupDoctor | null>(null)
  const [setupValidation, setSetupValidation] = useState<GuidedSetupValidation | null>(null)
  const [setupCampaignId, setSetupCampaignId] = useState('')
  const [setupTitle, setSetupTitle] = useState('')
  const humanLoadGeneration = useRef(0)
  const recoveryLoadGeneration = useRef(0)
  const setupLoadGeneration = useRef(0)
  const recoveryLoadGate = useRef(createRecoveryLoadGate())
  const selection = useUIStore((state) => state.trainingSelection)
  const openTraining = useUIStore((state) => state.openTraining)
  const activeWorkspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const workspaceId = selection.workspaceId || activeWorkspaceId
  const workspace = useCampaignStore((state) => state.workspaces[workspaceId])
  const load = useCampaignStore((state) => state.load)
  const select = useCampaignStore((state) => state.select)
  const refresh = useCampaignStore((state) => state.refresh)
  const loadEventsPage = useCampaignStore((state) => state.loadEventsPage)
  const loadArtifactsPage = useCampaignStore((state) => state.loadArtifactsPage)
  const loadAttemptMetrics = useCampaignStore((state) => state.loadAttemptMetrics)
  const loadLegacyDetail = useCampaignStore((state) => state.loadLegacyDetail)
  const transition = useCampaignStore((state) => state.transition)
  const selectedCampaignId = resolveControlRoomCampaignSelection(
    selection.campaignId,
    workspace?.selectedCampaignId,
  )
  const detail = selectedCampaignId ? workspace?.details[selectedCampaignId] : undefined
  const outcomeLoadKey = selectedCampaignId && detail?.snapshot
    ? `${workspaceId}:${selectedCampaignId}:${detail.snapshot.aggregate_version}:${detail.snapshot.latest_event_cursor}`
    : null
  const outcomeLoadKeyRef = useRef<string | null>(null)
  const outcome = useMemo(() => detail ? projectCampaignOutcome(detail) : null, [detail])
  const activeAttemptId = detail?.snapshot?.active_work?.attempt_id ?? null
  const activeExecutorType = detail?.snapshot?.active_work?.executor_type ?? null
  const activeCampaignStatus = detail?.snapshot?.campaign.status ?? null
  const humanSnapshotCursor = detail?.snapshot?.latest_event_cursor ?? null
  const shouldLoadGuidedSetup = Boolean(workspace && !workspace.loading && workspace.campaigns.length === 0)
  const archivedCampaignIds = useCampaignArchiveStore((state) => selectArchivedIds(state, workspaceId))
  const workspaceCampaigns = workspace?.campaigns
  useEffect(() => {
    const archiveStore = useCampaignArchiveStore.getState()
    archiveStore.ensureLoaded(workspaceId)
    if (workspaceCampaigns?.length) archiveStore.reconcile(workspaceId, workspaceCampaigns)
  }, [workspaceCampaigns, workspaceId])
  const { visible: visibleCampaigns, archived: archivedCampaigns } = useMemo(
    () => partitionCampaignsByArchive(workspaceCampaigns || [], new Set(archivedCampaignIds)),
    [archivedCampaignIds, workspaceCampaigns],
  )

  const loadHumanWork = useCallback(async () => {
    if (!selectedCampaignId) return
    const generation = ++humanLoadGeneration.current
    setHumanQueueLoading(true)
    const response = await campaignApi.humanWork(workspaceId, selectedCampaignId)
    if (generation !== humanLoadGeneration.current) return
    if (!response.ok || !response.data) {
      setHumanQueueError(`Human oversight service is unavailable${response.code ? ` (${response.code})` : ''}. Cached campaign context remains visible.`)
      setHumanQueueLoading(false)
      return
    }
    const parsed = parseHumanWorkQueue(response.data)
    if (!parsed || parsed.workspace_id !== workspaceId || parsed.campaign_id !== selectedCampaignId) {
      setHumanQueueError('The human oversight response did not match the selected campaign. Controls remain disabled until reconciliation succeeds.')
      setHumanQueueLoading(false)
      return
    }
    setHumanQueue(parsed)
    setHumanQueueError(null)
    setHumanQueueLoading(false)
  }, [selectedCampaignId, workspaceId])

  const loadRecovery = useCallback((): Promise<void> => {
    if (!selectedCampaignId) return Promise.resolve()
    const scope = `${workspaceId}\u0000${selectedCampaignId}`
    return recoveryLoadGate.current(scope, async () => {
      const generation = ++recoveryLoadGeneration.current
      setRecoveryLoading(true)
      const response = await campaignApi.recovery(workspaceId, selectedCampaignId)
      if (generation !== recoveryLoadGeneration.current) return
      if (!response.ok || !response.data) {
        if (response.code === 'campaign_recovery_conflict') {
          setRecoveryUnregistered(true)
          setRecoveryError('No recovery authority is registered for this campaign yet — recovery becomes available once the campaign has started.')
        } else if (response.code === 'campaign_desktop_bridge_required') {
          setRecoveryUnregistered(false)
          setRecoveryError('Recovery is read-only — the desktop connection isn\'t available. The rest of the Control Room still works.')
        } else {
          setRecoveryUnregistered(false)
          setRecoveryError(`Recovery state couldn't be loaded${response.code ? ` (${response.code})` : ''}. The rest of the Control Room still works.`)
        }
        setRecoveryLoading(false)
        return
      }
      setRecoverySnapshot(response.data)
      setRecoveryError(null)
      setRecoveryUnregistered(false)
      setRecoveryLoading(false)
    })
  }, [selectedCampaignId, workspaceId])

  const loadGuidedSetup = useCallback(async (sessionIdOverride?: string): Promise<void> => {
    const generation = ++setupLoadGeneration.current
    setSetupConnection('reconciling')
    setSetupDoctor(null)
    setSetupValidation(null)
    const response = await campaignApi.guidedSetupContext(workspaceId, sessionIdOverride)
    if (generation !== setupLoadGeneration.current) return
    if (!response.ok || !response.data) {
      setSetupConnection(response.code === 'campaign_desktop_bridge_required' ? 'offline' : 'error')
      setSetupError(`Setup is read-only — the desktop connection isn't available. Changes unlock when it reconnects.${response.code ? ` (${response.code})` : ''}`)
      return
    }
    setSetupContext(response.data)
    setSetupConnection('live')
    setSetupError(null)
  }, [workspaceId])

  useEffect(() => wsService.retainCampaignWorkspace(workspaceId), [workspaceId])

  useEffect(() => {
    setupLoadGeneration.current += 1
    setSetupContext(useControlRoomSnapshotCache.getState().setupContexts[workspaceId] ?? null)
    setSetupConnection('reconciling')
    setSetupError(null)
    setSetupPending(false)
    setSetupSessionId(null)
    setSetupSelectedOptionId('')
    setSetupDoctor(null)
    setSetupValidation(null)
    setSetupCampaignId('')
    setSetupTitle('')
    if (!shouldLoadGuidedSetup) {
      return () => { setupLoadGeneration.current += 1 }
    }
    try {
      const persistedSessionId = readGuidedSetupSessionId(window.localStorage, workspaceId)
      setSetupSessionId(persistedSessionId)
      void loadGuidedSetup(persistedSessionId ?? undefined)
    } catch {
      setSetupSessionId(null)
      setSetupConnection('error')
      setSetupError('This browser could not persist the workspace-scoped setup session. Setup remains visible, but durable writes are disabled.')
    }
    return () => { setupLoadGeneration.current += 1 }
  }, [loadGuidedSetup, shouldLoadGuidedSetup, workspaceId])

  useEffect(() => {
    if (!selectedCampaignId) return
    return retainCampaignSafetyReconcile(workspaceId, selectedCampaignId)
  }, [selectedCampaignId, workspaceId])

  useEffect(() => {
    if (!selectedCampaignId || !detail || !outcomeLoadKey || outcomeLoadKeyRef.current === outcomeLoadKey) return
    outcomeLoadKeyRef.current = outcomeLoadKey
    void loadLegacyDetail(workspaceId, selectedCampaignId)
  }, [detail, loadLegacyDetail, outcomeLoadKey, selectedCampaignId, workspaceId])

  useEffect(() => {
    if (
      !selectedCampaignId
      || !activeAttemptId
      || activeExecutorType !== 'ssh_remote'
      || activeCampaignStatus !== 'active'
    ) return
    const refreshMetrics = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void loadAttemptMetrics(workspaceId, selectedCampaignId, activeAttemptId)
      }
    }
    refreshMetrics()
    const timer = window.setInterval(refreshMetrics, 2_500)
    return () => window.clearInterval(timer)
  }, [activeAttemptId, activeCampaignStatus, activeExecutorType, loadAttemptMetrics, selectedCampaignId, workspaceId])

  useEffect(() => {
    recoveryLoadGeneration.current += 1
    setRecoverySnapshot(selectedCampaignId
      ? useControlRoomSnapshotCache.getState().recoverySnapshots[controlRoomSnapshotKey(workspaceId, selectedCampaignId)] ?? null
      : null)
    setRecoveryError(null)
    setRecoveryUnregistered(false)
    setRecoveryLoading(false)
    setRecoveryConfirmations({ resume: false, repair: false, takeover: false })
    setRecoveryIdempotencyKeys(newRecoveryIdempotencyKeys())
  }, [selectedCampaignId, workspaceId])

  useEffect(() => {
    humanLoadGeneration.current += 1
    setHumanQueue(selectedCampaignId
      ? useControlRoomSnapshotCache.getState().humanQueues[controlRoomSnapshotKey(workspaceId, selectedCampaignId)] ?? null
      : null)
    setHumanQueueError(null)
    setHumanQueueLoading(false)
    setHumanMutationPending(false)
    setHumanResponses({})
  }, [selectedCampaignId, workspaceId])

  // Write-through mirrors: persist fetched snapshots in the session cache so
  // remounts render them instantly. Each snapshot self-identifies, so a value
  // belonging to a previous workspace/campaign is never cached under the new key.
  useEffect(() => {
    if (humanQueue && humanQueue.workspace_id === workspaceId) {
      useControlRoomSnapshotCache.getState().putHumanQueue(
        controlRoomSnapshotKey(workspaceId, humanQueue.campaign_id),
        humanQueue,
      )
    }
  }, [humanQueue, workspaceId])

  useEffect(() => {
    if (recoverySnapshot && recoverySnapshot.workspace_id === workspaceId) {
      useControlRoomSnapshotCache.getState().putRecoverySnapshot(
        controlRoomSnapshotKey(workspaceId, recoverySnapshot.campaign_id),
        recoverySnapshot,
      )
    }
  }, [recoverySnapshot, workspaceId])

  useEffect(() => {
    if (setupContext && setupContext.workspace_id === workspaceId) {
      useControlRoomSnapshotCache.getState().putSetupContext(workspaceId, setupContext)
    }
  }, [setupContext, workspaceId])

  useEffect(() => {
    if (!selectedCampaignId || humanSnapshotCursor === null) return
    void loadHumanWork()
    return () => { humanLoadGeneration.current += 1 }
  }, [humanSnapshotCursor, loadHumanWork, selectedCampaignId])

  useEffect(() => {
    if (!selectedCampaignId) return
    void loadRecovery()
    return () => { recoveryLoadGeneration.current += 1 }
  }, [humanSnapshotCursor, loadRecovery, selectedCampaignId])

  const recoveryExecutionStatus = recoverySnapshot?.latest_execution?.status
  useEffect(() => {
    if (recoveryExecutionStatus !== 'accepted' && recoveryExecutionStatus !== 'executing') return
    const timer = window.setInterval(() => { void loadRecovery() }, 1500)
    return () => window.clearInterval(timer)
  }, [loadRecovery, recoveryExecutionStatus])

  useEffect(() => {
    const current = useCampaignStore.getState().workspaces[workspaceId]
    if (
      selection.campaignId
      && current?.selectedCampaignId === selection.campaignId
      && current.campaigns.length > 0
    ) return
    void load(workspaceId, selection.campaignId || undefined)
  }, [load, selection.campaignId, workspaceId])

  useEffect(() => {
    if (!workspace?.selectedCampaignId) return
    if (selection.workspaceId === workspaceId && selection.campaignId === workspace.selectedCampaignId) return
    if (!shouldCanonicalizeControlRoomSelection({
      requestedCampaignId: selection.campaignId,
      selectedCampaignId: workspace.selectedCampaignId,
      campaigns: workspace.campaigns,
      loading: workspace.loading,
      error: workspace.error,
    })) return
    openTraining('autoresearch', {
      workspaceId,
      campaignId: workspace.selectedCampaignId,
    }, 'replace')
  }, [
    openTraining,
    selection.campaignId,
    selection.workspaceId,
    workspace?.campaigns,
    workspace?.error,
    workspace?.loading,
    workspace?.selectedCampaignId,
    workspaceId,
  ])

  const model = buildControlRoomModel({
    snapshot: detail?.snapshot || null,
    freshness: resolveControlRoomFreshness({
      detailFreshness: detail?.freshness,
      workspaceFreshness: workspace?.freshness,
      workspaceLoading: !workspace || workspace.loading,
      workspaceError: workspace?.error,
    }),
    error: detail?.error || workspace?.error || null,
  })

  const handleSelect = (campaignId: string) => {
    openTraining('autoresearch', { workspaceId, campaignId })
    void select(workspaceId, campaignId)
  }

  const handleArchive = () => {
    if (!selectedCampaignId) return
    useCampaignArchiveStore.getState().archive(workspaceId, selectedCampaignId)
    const next = visibleCampaigns.find((item) => item.campaign_id !== selectedCampaignId)
    if (next) handleSelect(next.campaign_id)
  }

  const handleUnarchive = (campaignId: string) => {
    useCampaignArchiveStore.getState().unarchive(workspaceId, campaignId)
  }

  const handleTransition = async (action: LifecycleAction) => {
    if (transitionPending || !selectedCampaignId || model.kind !== 'snapshot' || !model.authoritative) return
    if (action === 'cancel'
      && !window.confirm('Cancel this campaign? Scheduled work stops and this cannot be undone.')) return
    if (action === 'conclude'
      && !window.confirm('Close this campaign as completed? Its results and evidence stay available, but no more experiments can be added.')) return
    setTransitionPending(action)
    try {
      await transition(
        workspaceId,
        selectedCampaignId,
        action,
        model.snapshot.aggregate_version,
        action === 'conclude'
          ? 'Operator closed the campaign after the bounded AutoResearch stop rule was reached.'
          : undefined,
      )
    } finally {
      setTransitionPending(null)
    }
  }

  const handleSetupAdvance = () => {
    if (!setupContext || setupConnection !== 'live' || setupPending) return
    const view = buildGuidedSetupView(setupContext)
    if (view.currentStep === 'doctor' || view.currentStep === 'create' || !setupSelectedOptionId) return
    let activeSessionId: string
    try {
      activeSessionId = setupSessionId ?? getOrCreateGuidedSetupSessionId(window.localStorage, workspaceId, randomAuthorityHex)
      if (!setupSessionId) setSetupSessionId(activeSessionId)
    } catch {
      setSetupConnection('error')
      setSetupError('This browser could not persist the workspace-scoped setup session. Durable writes remain disabled.')
      return
    }
    const scope = {
      workspaceId,
      sessionId: activeSessionId,
      version: setupContext.session?.version ?? 0,
      step: view.currentStep,
      selectionId: setupSelectedOptionId,
    }
    void (async () => {
      setSetupPending(true)
      try {
        const idempotencyKey = getOrCreateGuidedSetupIdempotencyKey(window.localStorage, scope, randomAuthorityHex)
        const response = await campaignApi.advanceGuidedSetupSession({
          workspaceId,
          sessionId: activeSessionId,
          expectedVersion: scope.version,
          step: scope.step,
          selectionId: scope.selectionId,
          idempotencyKey,
        })
        if (!response.ok || !response.data) {
          if (response.status === undefined) setSetupConnection('offline')
          if (response.status === 409) await loadGuidedSetup(activeSessionId)
          setSetupError(`The ${scope.step} choice was not sealed${response.code ? ` (${response.code})` : ''}. The current receipt chain is unchanged.`)
          return
        }
        clearGuidedSetupIdempotencyKey(window.localStorage, scope)
        setSetupContext((current) => current ? {
          ...current,
          session: response.data!.session,
          reason_codes: response.data!.session.reason_codes,
        } : current)
        setSetupSelectedOptionId('')
        setSetupDoctor(null)
        setSetupValidation(null)
        setSetupError(null)
      } catch {
        setSetupError('The setup choice could not be persisted. Its idempotency authority is retained for a safe retry.')
      } finally {
        setSetupPending(false)
      }
    })()
  }

  const handleSetupDoctor = () => {
    if (!setupContext || setupConnection !== 'live' || setupPending) return
    const draft = guidedSetupDraft(setupContext)
    if (!draft) return
    void (async () => {
      setSetupPending(true)
      try {
        const response = await campaignApi.doctorGuidedSetup(draft)
        if (!response.ok || !response.data) {
          if (response.status === undefined) setSetupConnection('offline')
          setSetupError(`Setup doctor could not verify the registered bindings${response.code ? ` (${response.code})` : ''}.`)
          return
        }
        setSetupDoctor(response.data)
        setSetupValidation(null)
        setSetupError(null)
      } finally {
        setSetupPending(false)
      }
    })()
  }

  const handleSetupValidation = () => {
    if (!setupContext || !setupSessionId || !setupDoctor?.ready || setupConnection !== 'live' || setupPending) return
    const draft = guidedSetupDraft(setupContext)
    const receiptId = setupContext.session?.latest_receipt.receipt_id
    if (!draft || !receiptId) return
    const scope = {
      workspaceId,
      sessionId: setupSessionId,
      version: 6,
      step: 'evaluation' as const,
      selectionId: `validate-${receiptId}`,
    }
    void (async () => {
      setSetupPending(true)
      try {
        const idempotencyKey = getOrCreateGuidedSetupIdempotencyKey(window.localStorage, scope, randomAuthorityHex)
        const response = await campaignApi.validateGuidedSetup(draft, idempotencyKey)
        if (!response.ok || !response.data) {
          if (response.status === undefined) setSetupConnection('offline')
          if (response.status === 409) await loadGuidedSetup(setupSessionId)
          setSetupError(`Setup validation was not sealed${response.code ? ` (${response.code})` : ''}. Run doctor again after resolving any changed binding.`)
          return
        }
        clearGuidedSetupIdempotencyKey(window.localStorage, scope)
        setSetupValidation(response.data)
        setSetupError(null)
      } catch {
        setSetupError('Validation could not be persisted. Its exact idempotency authority is retained for retry.')
      } finally {
        setSetupPending(false)
      }
    })()
  }

  const handleSetupCreate = () => {
    if (!setupSessionId || !setupValidation?.ready || setupConnection !== 'live' || setupPending) return
    const scope = {
      workspaceId,
      sessionId: setupSessionId,
      version: 6,
      step: 'evaluation' as const,
      selectionId: `create-${setupCampaignId}`,
    }
    void (async () => {
      setSetupPending(true)
      try {
        const idempotencyKey = getOrCreateGuidedSetupIdempotencyKey(window.localStorage, scope, randomAuthorityHex)
        const response = await campaignApi.createGuidedSetup({
          workspaceId,
          campaignId: setupCampaignId,
          title: setupTitle,
          validationReceiptId: setupValidation.receipt_id,
          idempotencyKey,
        })
        if (!response.ok || !response.data) {
          if (response.status === undefined) setSetupConnection('offline')
          if (response.status === 409) await loadGuidedSetup(setupSessionId)
          setSetupError(`Campaign creation was rejected${response.code ? ` (${response.code})` : ''}. No Start action was taken.`)
          return
        }
        clearGuidedSetupIdempotencyKey(window.localStorage, scope)
        await load(workspaceId, response.data.campaign_id)
        openTraining('autoresearch', { workspaceId, campaignId: response.data.campaign_id }, 'replace')
        await select(workspaceId, response.data.campaign_id)
      } catch {
        setSetupError('Campaign creation could not complete. Its exact idempotency authority is retained for retry; no Start action was taken.')
      } finally {
        setSetupPending(false)
      }
    })()
  }

  const reconcileHumanMutation = async (operation: () => ReturnType<typeof campaignApi.claimHumanWork>) => {
    if (humanMutationPending || !selectedCampaignId) return
    setHumanMutationPending(true)
    try {
      const response = await operation()
      if (!response.ok) {
        setHumanQueueError(`The human oversight request was rejected${response.code ? ` (${response.code})` : ''}. Reconciled authoritative state is shown below.`)
      }
      await Promise.all([
        loadHumanWork(),
        refresh(workspaceId, selectedCampaignId),
      ])
    } finally {
      setHumanMutationPending(false)
    }
  }

  const handleHumanClaim = (request: HumanWorkClaimRequest) => {
    void reconcileHumanMutation(() => campaignApi.claimHumanWork(request))
  }

  const handleHumanSubmit = (
    request: HumanWorkSubmitRequest,
    response: HumanReviewResponse & { decision: Exclude<HumanReviewResponse['decision'], ''> },
  ) => {
    void reconcileHumanMutation(() => campaignApi.submitHumanWork(request, response.decision, response.rationale))
  }

  const handleHumanPromotion = (request: HumanPromotionRequest) => {
    void reconcileHumanMutation(() => campaignApi.decideHumanPromotion(request, request.decision))
  }

  const handleRecoveryRequest = (request: RecoveryRequest) => {
    void (async () => {
      const response = await campaignApi.requestRecovery(request)
      if (!response.ok) {
        setRecoveryError(`Recovery request was rejected${response.code ? ` (${response.code})` : ''}. Reconcile before retrying.`)
      }
      setRecoveryConfirmations((current) => ({ ...current, [request.action]: false }))
      setRecoveryIdempotencyKeys((current) => ({ ...current, [request.action]: newRecoveryIdempotencyKeys()[request.action] }))
      await Promise.all([loadRecovery(), refresh(workspaceId, request.campaignId)])
    })()
  }

  const humanSummary = model.kind === 'snapshot' ? model.snapshot.human_work : null
  const showHumanOversight = Boolean(
    humanSummary && (
      humanSummary.open_count > 0
      || humanSummary.blocking_count > 0
      || humanQueue
      || humanQueueError
      || humanQueueLoading
    ),
  )
  const humanOversightModel = humanQueue && model.kind === 'snapshot'
    ? buildHumanOversightModel({
        queue: humanQueue,
        authority: { workspaceId, campaignId: model.snapshot.campaign_id },
        freshness: humanMutationPending ? 'reconciling' : model.freshness,
        error: humanQueueError,
        now: new Date().toISOString(),
      })
    : null
  const humanOversight = showHumanOversight
    ? humanOversightModel
      ? (
          <div className="space-y-2">
            {humanQueueError ? <p className="border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs text-text-secondary" role="alert">{humanQueueError}</p> : null}
            <HumanOversightQueue
              model={humanOversightModel}
              responses={humanResponses}
              onResponseChange={(responseKey, response) => {
                if (responseKey !== humanReviewResponseKey(response)) return
                setHumanResponses((current) => ({ ...current, [responseKey]: response }))
              }}
              onClaim={handleHumanClaim}
              onSubmit={handleHumanSubmit}
              onPromotion={handleHumanPromotion}
            />
          </div>
        )
      : <HumanOversightUnavailable loading={humanQueueLoading} message={humanQueueError} onRetry={() => { void loadHumanWork() }} />
    : undefined
  const recoveryFreshness = recoveryError
    ? 'error'
    : recoveryLoading
      ? 'reconciling'
      : model.kind === 'snapshot' ? model.freshness : 'error'
  const campaignRecovery = selectedCampaignId
    ? recoverySnapshot
      ? (
          <div className="space-y-2">
            {recoveryError || recoveryLoading ? <div className="flex flex-wrap items-center justify-between gap-2 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs text-text-secondary" role={recoveryError ? 'alert' : 'status'}><span>{recoveryError || 'Refreshing recovery lifecycle…'}</span><Button type="button" size="sm" variant="secondary" onClick={() => { void loadRecovery() }} disabled={recoveryLoading}>Retry recovery</Button></div> : null}
            <CampaignRecoveryPanel
              snapshot={recoverySnapshot}
              freshness={recoveryFreshness}
              now={new Date().toISOString()}
              confirmations={recoveryConfirmations}
              idempotencyKeys={recoveryIdempotencyKeys}
              onConfirmationChange={(action, confirmed) => setRecoveryConfirmations((current) => ({ ...current, [action]: confirmed }))}
              onRequest={handleRecoveryRequest}
            />
          </div>
        )
      : recoveryUnregistered
        ? (
            <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-xs text-text-secondary" role="status"><span>{recoveryError}</span><Button type="button" size="sm" variant="secondary" onClick={() => { void loadRecovery() }} disabled={recoveryLoading}>Check again</Button></div>
          )
        : (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 border-l-2 border-status-warning bg-status-warning/10 px-3 py-2 text-xs text-text-secondary" role={recoveryError ? 'alert' : 'status'}><span>{recoveryError || (recoveryLoading ? 'Loading authoritative recovery evidence…' : 'Recovery evidence has not loaded. Controls remain disabled.')}</span><Button type="button" size="sm" variant="secondary" onClick={() => { void loadRecovery() }} disabled={recoveryLoading}>{recoveryLoading ? 'Reconciling' : 'Retry recovery'}</Button></div>
              <CampaignRecoveryPanel
                snapshot={null}
                freshness={recoveryFreshness}
                now={new Date().toISOString()}
                confirmations={recoveryConfirmations}
                idempotencyKeys={recoveryIdempotencyKeys}
                onConfirmationChange={(action, confirmed) => setRecoveryConfirmations((current) => ({ ...current, [action]: confirmed }))}
                onRequest={handleRecoveryRequest}
              />
            </div>
          )
    : undefined
  const guidedSetup = (
    <GuidedAutoResearchSetup
      context={setupContext}
      connectionState={setupConnection}
      pending={setupPending}
      error={setupError}
      doctor={setupDoctor}
      validation={setupValidation}
      selectedOptionId={setupSelectedOptionId}
      campaignId={setupCampaignId}
      title={setupTitle}
      onSelectedOptionChange={setSetupSelectedOptionId}
      onCampaignIdChange={setSetupCampaignId}
      onTitleChange={setSetupTitle}
      onAdvance={handleSetupAdvance}
      onDoctor={handleSetupDoctor}
      onValidate={handleSetupValidation}
      onCreate={handleSetupCreate}
      onRetry={() => { void loadGuidedSetup(setupSessionId ?? undefined) }}
    />
  )

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mx-auto max-w-[1240px]">
        <ControlRoomContent
          model={model}
          campaigns={visibleCampaigns}
          selectedCampaignId={selectedCampaignId || null}
          events={detail?.pages.events || []}
          artifacts={detail?.pages.artifacts || []}
          outcome={outcome}
          pages={detail?.pages || unloadedPages}
          onSelect={handleSelect}
          onArchive={
            selectedCampaignId && detail && detail.campaign.status !== 'active'
              ? handleArchive
              : null
          }
          campaignLibrary={archivedCampaigns.length > 0 ? (
            <CampaignLibrary
              campaigns={archivedCampaigns}
              onUnarchive={handleUnarchive}
              onOpen={handleSelect}
            />
          ) : undefined}
          onRetry={() => {
            if (selectedCampaignId) void refresh(workspaceId, selectedCampaignId)
            else void load(workspaceId, selection.campaignId || undefined)
          }}
          onLoadEvents={() => selectedCampaignId && void loadEventsPage(workspaceId, selectedCampaignId)}
          onLoadArtifacts={() => selectedCampaignId && void loadArtifactsPage(workspaceId, selectedCampaignId)}
          onTransition={(action) => { void handleTransition(action) }}
          transitionPending={transitionPending}
          humanOversight={humanOversight}
          campaignRecovery={campaignRecovery}
          guidedSetup={guidedSetup}
        />
      </div>
    </div>
  )
}
