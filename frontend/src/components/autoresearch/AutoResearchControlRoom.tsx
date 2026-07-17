import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Circle,
  Database,
  FileCheck2,
  Gauge,
  Loader2,
  RefreshCw,
  ShieldCheck,
  UserRound,
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
import { useUIStore } from '../../stores/uiStore'
import { useWorkspaceStore } from '../../stores/workspaceStore'
import { Button } from '../common/Button'
import { CampaignRecoveryPanel } from './CampaignRecoveryPanel'
import { GuidedAutoResearchSetup, type GuidedSetupConnectionState } from './GuidedAutoResearchSetup'
import { HumanOversightQueue, type HumanReviewResponse } from './HumanOversightQueue'
import type { CampaignRecoveryPublicV1, RecoveryAction, RecoveryRequest } from './campaignRecoveryModel'
import { buildControlRoomModel, campaignStatusTone, resolveControlRoomFreshness, type ControlRoomViewModel, type PresentationTone } from './controlRoomModel'
import { resolveControlRoomCampaignSelection, shouldCanonicalizeControlRoomSelection } from './controlRoomSelection'
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

export interface ControlRoomContentProps {
  model: ControlRoomViewModel
  campaigns: CampaignRecord[]
  selectedCampaignId: string | null
  events: CampaignEventItem[]
  artifacts: CampaignArtifact[]
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
  onStart?: () => void
  startPending?: boolean
  humanOversight?: ReactNode
  campaignRecovery?: ReactNode
  guidedSetup?: ReactNode
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

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${compactNumber(value / 1024)} KB`
  return `${compactNumber(value / 1024 ** 2)} MB`
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return 'No estimate'
  if (seconds < 60) return `${Math.round(seconds)} sec`
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`
  return `${compactNumber(seconds / 3600)} hr`
}

function eventSummaryEntries(event: CampaignEventItem['event']): Array<[string, string]> {
  if (!event.summary) return []
  return Object.entries(event.summary)
    .filter(([key, value]) => key !== 'schema_version' && value !== undefined && value !== null)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => [readable(key), typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)])
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
        <option value="" disabled>Select a durable campaign</option>
        {campaigns.map((campaign) => (
          <option key={campaign.campaign_id} value={campaign.campaign_id}>
            {campaign.title} · {readable(campaign.status)}
          </option>
        ))}
      </select>
    </div>
  )
}

function ControlRoomHeader(props: Pick<ControlRoomContentProps, 'campaigns' | 'selectedCampaignId' | 'onSelect'>) {
  return (
    <header className="flex flex-col gap-3 border-b border-border-subtle pb-3 lg:flex-row lg:items-end lg:justify-between">
      <div className="min-w-0">
        <h1 className="font-brand text-2xl text-text-primary">AutoResearch Control Room</h1>
        <p className="mt-1 text-sm text-text-secondary">Run bounded experiments on registered private compute with durable evidence and human oversight.</p>
      </div>
      <div className="flex w-full flex-col gap-2 sm:flex-row sm:items-end lg:max-w-2xl lg:justify-end">
        <CampaignSelector {...props} />
        <div className="shrink-0 pb-2 font-mono text-[9px] uppercase tracking-widest text-text-muted">Durable control plane</div>
      </div>
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

function FreshnessNotice({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const isLive = model.authoritative
  const icon = model.freshness === 'offline'
    ? <WifiOff className="h-4 w-4" />
    : model.freshness === 'live'
      ? <CheckCircle2 className="h-4 w-4" />
      : <AlertTriangle className="h-4 w-4" />
  return (
    <div
      className={clsx(
        'flex items-start gap-2 rounded-brutal border-brutal px-3 py-2 text-sm',
        isLive ? 'border-status-success/60 bg-status-success/10 text-text-primary' : 'border-status-warning/70 bg-status-warning/10 text-text-primary',
      )}
      role="status"
      aria-live="polite"
    >
      <span className={clsx('mt-0.5 shrink-0', isLive ? 'text-status-success' : 'text-status-warning')}>{icon}</span>
      <div>
        <span className="font-mono text-xs font-bold uppercase tracking-wide">{model.freshnessLabel}</span>
        <span className="ml-2 text-xs text-text-secondary">
          Verified {model.snapshot.snapshot_at} · version {model.snapshot.aggregate_version} · cursor {model.snapshot.latest_event_cursor}
        </span>
        {model.error ? <div className="mt-1 text-xs text-status-warning">{model.error}</div> : null}
      </div>
    </div>
  )
}

function BlockerPanel({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  if (!model.blocker) return null
  return (
    <section className="card border-status-warning p-4" aria-labelledby="primary-blocker-title">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-status-warning" />
        <div className="min-w-0">
          <div className="font-mono text-[10px] font-bold uppercase tracking-widest text-status-warning">Primary blocker</div>
          <h2 id="primary-blocker-title" className="mt-1 font-brand text-lg text-text-primary">{model.blocker.summary}</h2>
          <div className="mt-2 font-mono text-xs text-text-secondary">Code · {readable(model.blocker.code)}</div>
          {model.blocker.evidenceIds.length ? (
            <div className="mt-2 text-xs text-text-muted">Evidence · {model.blocker.evidenceIds.join(', ')}</div>
          ) : null}
        </div>
      </div>
    </section>
  )
}

function CampaignOverview({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const { snapshot } = model
  return (
    <section className="card col-span-12 p-4 xl:col-span-8" aria-labelledby="campaign-summary-title">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 id="campaign-summary-title" className="font-brand text-xl text-text-primary">{snapshot.campaign.title}</h2>
            <StatusPill label={readable(snapshot.campaign.status)} tone={campaignStatusTone(snapshot.campaign.status)} />
          </div>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-text-secondary">{snapshot.campaign.objective}</p>
          <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] uppercase tracking-wide text-text-muted">
            <span>{readable(snapshot.campaign.kind)}</span>
            <span>Manifest r{snapshot.manifest_revision}</span>
            <span>{snapshot.campaign.campaign_id}</span>
          </div>
          {snapshot.campaign.stop_reason ? <p className="mt-3 text-xs text-status-warning">Stop reason · {snapshot.campaign.stop_reason}</p> : null}
        </div>
        <div className="grid min-w-[250px] grid-cols-2 gap-px overflow-hidden rounded-brutal border-brutal border-border-subtle bg-border-subtle">
          <div className="bg-background-card p-3">
            <div className="font-mono text-[10px] uppercase tracking-wide text-text-muted">Execution owner</div>
            <div className="mt-1 flex items-center gap-2 text-sm font-semibold text-text-primary"><Bot className="h-4 w-4 text-accent" />{model.owner.execution}</div>
          </div>
          <div className="bg-background-card p-3">
            <div className="font-mono text-[10px] uppercase tracking-wide text-text-muted">Attention owner</div>
            <div className="mt-1 flex items-center gap-2 text-sm font-semibold text-text-primary"><UserRound className="h-4 w-4 text-accent" />{model.owner.attention}</div>
          </div>
        </div>
      </div>

      <div className="mt-4 border-t border-border-subtle pt-3">
        <div className="font-mono text-[10px] font-bold uppercase tracking-widest text-text-muted">Server-evaluated guidance</div>
        {model.actions.length ? (
          <ul className="mt-2 grid gap-2 sm:grid-cols-2">
            {model.actions.map((action) => (
              <li key={`${action.kind}-${action.id}`} className="flex items-center justify-between gap-3 rounded-brutal border border-border-subtle bg-background-secondary px-3 py-2">
                <span className="text-sm text-text-primary">{action.label}</span>
                <span className="font-mono text-[9px] uppercase tracking-wide text-text-muted">{action.kind} · read only</span>
              </li>
            ))}
          </ul>
        ) : <p className="mt-2 text-sm text-text-muted">No next or recovery action is currently published.</p>}
      </div>
    </section>
  )
}

function ControllerPanel({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const { controller, readiness } = model.snapshot
  const controllerTone: PresentationTone = controller.state === 'online' ? 'success' : controller.state === 'stale' ? 'warning' : 'error'
  return (
    <section className="card col-span-12 p-4 xl:col-span-4" aria-labelledby="controller-title">
      <div className="flex items-center justify-between gap-2">
        <h2 id="controller-title" className="font-brand text-lg text-text-primary">Controller & readiness</h2>
        <StatusPill label={model.controllerLabel} tone={controllerTone} />
      </div>
      <dl className="mt-3 space-y-2 text-sm">
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Observed</dt><dd className="font-mono text-xs text-text-primary">{controller.observed_at}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Heartbeat age</dt><dd className="font-mono text-xs text-text-primary">{controller.heartbeat_age_seconds == null ? 'Unavailable' : `${compactNumber(controller.heartbeat_age_seconds)} sec`}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Lease expiry</dt><dd className="font-mono text-xs text-text-primary">{controller.lease_expires_at || 'Unavailable'}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Instance</dt><dd className="max-w-[180px] truncate font-mono text-xs text-text-primary">{controller.controller_instance_id || 'Unavailable'}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Materializable</dt><dd className="font-mono text-xs font-bold text-text-primary">{readiness.materializable ? 'Yes' : 'No'}</dd></div>
        <div className="flex justify-between gap-3"><dt className="text-text-muted">Launch ready</dt><dd className="font-mono text-xs font-bold text-text-primary">{readiness.launch_ready ? 'Verified' : 'Blocked'}</dd></div>
      </dl>
      {controller.safe_guidance ? <p className="mt-3 rounded-brutal border border-border-subtle bg-background-secondary p-2 text-xs text-text-secondary">{controller.safe_guidance}</p> : null}
      {readiness.blocking_codes.length ? <p className="mt-2 font-mono text-[10px] text-status-warning">{readiness.blocking_codes.map(readable).join(' · ')}</p> : null}
      <div className="mt-3 border-t border-border-subtle pt-3">
        <div className="font-mono text-[10px] uppercase tracking-wide text-text-muted">Attached agents</div>
        <div className="mt-1 text-xs text-text-secondary">{model.snapshot.agents.length ? `${model.snapshot.agents.length} visible session${model.snapshot.agents.length === 1 ? '' : 's'}` : 'None attached'}</div>
      </div>
    </section>
  )
}

function LaunchAuthority({
  model,
  onStart,
  onRefresh,
  pending = false,
}: {
  model: Extract<ControlRoomViewModel, { kind: 'snapshot' }>
  onStart?: () => void
  onRefresh: () => void
  pending?: boolean
}) {
  const { campaign, readiness } = model.snapshot
  const startableState = campaign.status === 'ready'
  const enabled = Boolean(onStart && model.authoritative && readiness.launch_ready && startableState && !pending)
  const stateCopy = campaign.status === 'active'
    ? 'Campaign execution is active. BashGym continues to reconcile the durable event stream.'
    : !startableState
      ? `Start becomes available from Ready. Current state: ${readable(campaign.status)}.`
      : !readiness.launch_ready
        ? 'The current server readiness projection is blocked. Refresh after remediation.'
        : !model.authoritative
          ? 'Live authority is required. Cached state never enables Start.'
          : 'The server will recompute every binding, controller lease, and execution identity before launch.'

  return (
    <section className="card p-4" aria-labelledby="launch-authority-title">
      <p className="font-mono text-[10px] font-bold uppercase tracking-widest text-accent-dark">Authoritative gate</p>
      <h2 id="launch-authority-title" className="mt-1 font-brand text-lg text-text-primary">Launch authority</h2>
      <p className="mt-2 text-xs leading-5 text-text-secondary">{stateCopy}</p>
      {readiness.blocking_codes.length ? (
        <ul className="mt-3 space-y-1 border-l-2 border-status-warning pl-2 text-xs text-status-warning">
          {readiness.blocking_codes.map((code) => <li key={code}>{readable(code)}</li>)}
        </ul>
      ) : null}
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="secondary" size="sm" onClick={onRefresh} disabled={pending} leftIcon={<RefreshCw className="h-3.5 w-3.5" />}>
          Refresh doctor
        </Button>
        {startableState ? (
          <Button variant="primary" size="sm" onClick={onStart} disabled={!enabled} aria-label="Start campaign" leftIcon={pending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : undefined}>
            {pending ? 'Starting' : 'Start campaign'}
          </Button>
        ) : null}
      </div>
      <p className="mt-3 font-mono text-[9px] uppercase tracking-wide text-text-muted">Renderer state is advisory · server revalidation is final</p>
    </section>
  )
}

function CampaignJourney({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  return (
    <section className="card col-span-12 p-4" aria-labelledby="journey-title">
      <div className="flex items-center justify-between gap-3">
        <h2 id="journey-title" className="font-brand text-lg text-text-primary">Journey</h2>
        <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted">Five durable phases</span>
      </div>
      <ol className="mt-3 grid gap-2 md:grid-cols-5">
        {model.journey.map((phase, index) => (
          <li key={phase.id} className={clsx('relative rounded-brutal border-brutal p-3', toneClasses[phase.tone])}>
            <div className="flex items-start justify-between gap-2">
              <span className="font-mono text-[10px] font-bold uppercase tracking-wide">{index + 1}</span>
              {phase.tone === 'success' ? <CheckCircle2 className="h-4 w-4" /> : <Circle className="h-4 w-4" />}
            </div>
            <h3 className="mt-3 font-brand text-base text-text-primary">{phase.label}</h3>
            <div className="mt-1 font-mono text-[10px] uppercase tracking-wide">{phase.stateLabel}</div>
            <div className="mt-2 text-[11px] text-text-secondary">{phase.evidenceCount} evidence item{phase.evidenceCount === 1 ? '' : 's'}</div>
            {phase.blocker ? <div className="mt-2 text-[11px] text-status-warning">{phase.blocker.summary}</div> : null}
          </li>
        ))}
      </ol>
    </section>
  )
}

function ActiveWorkPanel({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const work = model.snapshot.active_work
  return (
    <section className="card col-span-12 p-4 lg:col-span-7" aria-labelledby="active-work-title">
      <div className="flex items-center justify-between gap-3">
        <h2 id="active-work-title" className="font-brand text-lg text-text-primary">Active work</h2>
        {work?.stage ? <StatusPill label={readable(work.stage)} tone="info" /> : null}
      </div>
      {!work ? <p className="mt-3 text-sm text-text-muted">No study action is active in the verified campaign state.</p> : (
        <div className="mt-3">
          <p className="text-sm leading-6 text-text-primary">{work.hypothesis_summary || 'No public hypothesis summary is available.'}</p>
          <dl className="mt-3 grid gap-2 text-xs sm:grid-cols-2">
            <div><dt className="font-mono uppercase tracking-wide text-text-muted">Primary variable</dt><dd className="mt-1 text-text-secondary">{work.primary_variable_summary || 'Unavailable'}</dd></div>
            <div><dt className="font-mono uppercase tracking-wide text-text-muted">Executor</dt><dd className="mt-1 text-text-secondary">{work.executor_type ? readable(work.executor_type) : 'Unavailable'}</dd></div>
            <div><dt className="font-mono uppercase tracking-wide text-text-muted">Attempt</dt><dd className="mt-1 font-mono text-text-secondary">{work.attempt_id || 'Unavailable'}</dd></div>
            <div><dt className="font-mono uppercase tracking-wide text-text-muted">ETA</dt><dd className="mt-1 text-text-secondary">{formatDuration(work.eta_seconds)}</dd></div>
          </dl>
          {work.progress_fraction != null ? (
            <div className="mt-4">
              <div className="mb-1 flex justify-between font-mono text-[10px] uppercase tracking-wide text-text-muted">
                <span>Observed progress</span><span>{Math.round(work.progress_fraction * 100)}%</span>
              </div>
              <progress className="h-2 w-full accent-accent" value={work.progress_fraction} max={1} aria-label={`Active work progress ${Math.round(work.progress_fraction * 100)}%`} />
              <div className="sr-only">{Math.round(work.progress_fraction * 100)}%</div>
            </div>
          ) : null}
          {work.process_identity ? <div className="mt-3 rounded-brutal border border-border-subtle bg-background-secondary p-2 font-mono text-[10px] text-text-muted">Run {work.process_identity.run_id} · {readable(work.process_identity.state)} · {work.process_identity.compute_profile_id}</div> : null}
          {work.controlled_variable_summary.length ? <div className="mt-2 text-[11px] text-text-muted">Controlled · {work.controlled_variable_summary.join(' · ')}</div> : null}
        </div>
      )}
    </section>
  )
}

function EvidencePanel({
  model,
  artifacts,
  pages,
  onLoadArtifacts,
}: Pick<ControlRoomContentProps, 'artifacts' | 'pages' | 'onLoadArtifacts'> & { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const { snapshot } = model
  const orderedArtifacts = [...artifacts].sort((left, right) => (
    left.created_at.localeCompare(right.created_at) || left.artifact_id.localeCompare(right.artifact_id)
  ))
  return (
    <section className="card col-span-12 p-4 lg:col-span-5" aria-labelledby="evidence-title">
      <div className="flex items-center justify-between gap-3">
        <h2 id="evidence-title" className="font-brand text-lg text-text-primary">Evidence & metrics</h2>
        <FileCheck2 className="h-5 w-5 text-accent" />
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
        {model.metrics.length ? model.metrics.map((metric) => (
          <div key={metric.id} className="rounded-brutal border border-border-subtle bg-background-secondary p-3">
            <div className="font-mono text-[10px] uppercase tracking-wide text-text-muted">{metric.label}</div>
            <div className="mt-1 font-mono text-lg font-bold text-text-primary">{metric.value}</div>
            <div className="mt-1 text-[11px] text-text-muted">{metric.direction}{metric.target == null ? '' : ` · target ${compactNumber(metric.target)}${metric.unit || ''}`}</div>
          </div>
        )) : <p className="text-sm text-text-muted">No metric descriptors are published.</p>}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
        <div><dt className="text-text-muted">Champion</dt><dd className="mt-1 truncate font-mono text-text-primary">{snapshot.champion?.candidate_ref || 'Unavailable'}{snapshot.champion ? ` · ${readable(snapshot.champion.gate_state)}` : ''}</dd></div>
        <div><dt className="text-text-muted">Candidate</dt><dd className="mt-1 truncate font-mono text-text-primary">{snapshot.candidate?.candidate_ref || 'Unavailable'}{snapshot.candidate ? ` · ${readable(snapshot.candidate.gate_state)}` : ''}</dd></div>
        {([
          ['Model', snapshot.bindings.model],
          ['Data', snapshot.bindings.data],
          ['Evaluator', snapshot.bindings.evaluator],
          ['Source', snapshot.bindings.source],
          ['Compute', snapshot.bindings.compute],
        ] as const).map(([label, binding]) => (
          <div key={label}><dt className="text-text-muted">{label} binding</dt><dd className="mt-1 truncate text-text-primary">{binding?.display_label || 'Unbound'}</dd></div>
        ))}
      </dl>
      {artifacts.length ? (
        <div className="mt-3 max-h-80 overflow-auto border-t border-border-subtle pt-3" role="region" aria-label="Inspectable campaign artifacts" tabIndex={0}>
          <ul className="space-y-2">
            {orderedArtifacts.map((artifact) => (
              <li key={artifact.artifact_id} className="rounded-brutal border border-border-subtle bg-background-secondary p-2 text-xs">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-mono font-bold text-text-primary">{artifact.artifact_id}</span>
                  <span className="text-text-muted">{readable(artifact.schema_name)} · {formatBytes(artifact.size_bytes)}</span>
                </div>
                <dl className="mt-2 grid gap-x-3 gap-y-1 font-mono text-[10px] sm:grid-cols-2">
                  <div><dt className="inline text-text-muted">Producer · </dt><dd className="inline text-text-secondary">{artifact.producer_action_id || 'None'}</dd></div>
                  <div><dt className="inline text-text-muted">State · </dt><dd className="inline text-text-secondary">{artifact.sealed ? 'sealed' : 'unsealed'} · {artifact.valid ? 'valid' : 'invalid'}</dd></div>
                  <div className="sm:col-span-2"><dt className="inline text-text-muted">SHA-256 · </dt><dd className="break-all text-text-secondary">{artifact.sha256}</dd></div>
                  <div className="sm:col-span-2"><dt className="inline text-text-muted">Created · </dt><dd className="inline text-text-secondary">{artifact.created_at}</dd></div>
                </dl>
              </li>
            ))}
          </ul>
        </div>
      ) : pages.artifactsLoaded ? <p className="mt-3 text-sm text-text-muted">No artifact rows were returned.</p> : null}
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
    </section>
  )
}

function BudgetPanel({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const resources = model.snapshot.budget.resources
  return (
    <section className="card col-span-12 p-4 lg:col-span-6" aria-labelledby="budget-title">
      <div className="flex items-center justify-between gap-3">
        <h2 id="budget-title" className="font-brand text-lg text-text-primary">Budget</h2>
        <Gauge className={clsx('h-5 w-5', model.snapshot.budget.blocked ? 'text-status-warning' : 'text-accent')} />
      </div>
      {resources.length ? <ul className="mt-3 space-y-3">
        {resources.map((resource) => {
          const used = Math.max(0, resource.settled + resource.reserved)
          const fraction = resource.limit > 0 ? Math.min(1, used / resource.limit) : 0
          return (
            <li key={resource.unit}>
              <div className="flex justify-between gap-3 text-xs"><span className="font-mono uppercase tracking-wide text-text-secondary">{readable(resource.unit)}</span><span className="text-text-muted">{compactNumber(resource.remaining)} remaining / {compactNumber(resource.limit)}</span></div>
              <progress className="mt-1 h-2 w-full accent-accent" value={fraction} max={1} aria-label={`${readable(resource.unit)} budget ${Math.round(fraction * 100)}% committed`} />
              <div className="mt-1 text-[10px] text-text-muted">{compactNumber(resource.settled)} settled · {compactNumber(resource.reserved)} reserved{resource.blocked ? ' · blocked' : ''}</div>
            </li>
          )
        })}
      </ul> : <p className="mt-3 text-sm text-text-muted">No bounded resources are configured.</p>}
    </section>
  )
}

function CollectionPanel({ model }: { model: Extract<ControlRoomViewModel, { kind: 'snapshot' }> }) {
  const collections = model.snapshot.collections
  const rows = [
    ['Events', collections.events.count],
    ['Proposals', collections.proposals.count],
    ['Studies', collections.studies.count],
    ['Attempts', collections.attempts.count],
    ['Artifacts', collections.artifacts.count],
    ['Comparisons', collections.comparisons.count],
    ['Human work', collections.human_work.count],
  ] as const
  return (
    <section className="card col-span-12 p-4 lg:col-span-6" aria-labelledby="collections-title">
      <h2 id="collections-title" className="font-brand text-lg text-text-primary">Collection counts</h2>
      <dl className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-brutal border-brutal border-border-subtle bg-border-subtle sm:grid-cols-4">
        {rows.map(([label, count]) => (
          <div key={label} className="bg-background-card p-3">
            <dt className="font-mono text-[9px] uppercase tracking-wide text-text-muted">{label}</dt>
            <dd className="mt-1 font-mono text-lg font-bold text-text-primary">{count}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 flex items-center gap-2 text-xs text-text-secondary">
        <ShieldCheck className="h-4 w-4 text-status-success" />
        {model.snapshot.human_work.blocking_count} blocking · {model.snapshot.human_work.open_count} open human items
      </div>
    </section>
  )
}

function ActivityPanel({
  events,
  pages,
  onLoadEvents,
}: Pick<ControlRoomContentProps, 'events' | 'pages' | 'onLoadEvents'>) {
  const orderedEvents = [...events].sort((left, right) => (
    left.cursor - right.cursor || left.event.event_id.localeCompare(right.event.event_id)
  ))
  return (
    <section className="card col-span-12 p-4" aria-labelledby="activity-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="activity-title" className="font-brand text-lg text-text-primary">Recent activity</h2>
          <p className="mt-1 text-xs text-text-muted">Durable public event rows are loaded explicitly and never recompute campaign decisions.</p>
        </div>
        {pages.eventsError || pages.eventsHasMore ? (
          <Button
            variant="secondary"
            size="sm"
            disabled={pages.eventsLoading}
            onClick={onLoadEvents}
            leftIcon={pages.eventsLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
          >
            {pages.eventsLoading ? 'Loading events' : pages.eventsError ? 'Retry event page' : pages.eventsLoaded ? 'Load more events' : 'Load event page'}
          </Button>
        ) : null}
      </div>
      {pages.eventsError ? <p className="mt-3 text-xs text-status-error" role="alert">{pages.eventsError}</p> : null}
      {events.length ? (
        <ol className="mt-3 max-h-96 divide-y divide-border-subtle overflow-auto border-y border-border-subtle" aria-label="Inspectable campaign events" tabIndex={0}>
          {orderedEvents.map(({ cursor, event }) => (
            <li key={event.event_id} className="grid gap-2 py-2 text-xs md:grid-cols-[100px_minmax(0,1fr)_220px] md:items-start">
              <span className="font-mono text-text-muted">Cursor {cursor}</span>
              <div className="min-w-0">
                <div className="font-semibold text-text-primary">{readable(event.event_type)}</div>
                <div className="mt-1 break-all font-mono text-[10px] text-text-muted">{event.event_id} · sequence {event.sequence} · version {event.aggregate_version}</div>
                {eventSummaryEntries(event).length ? (
                  <dl className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-text-secondary">
                    {eventSummaryEntries(event).map(([label, value]) => <div key={label}><dt className="inline text-text-muted">{label} · </dt><dd className="inline">{value}</dd></div>)}
                  </dl>
                ) : <div className="mt-1 text-[10px] text-text-muted">No public summary fields</div>}
              </div>
              <span className="font-mono text-[10px] text-text-muted md:text-right">{event.created_at}<br />{event.actor_id} · {readable(event.credential_kind)}</span>
            </li>
          ))}
        </ol>
      ) : <p className="mt-3 text-sm text-text-muted">{pages.eventsLoaded ? 'No event rows were returned.' : 'No event detail page is loaded.'}</p>}
    </section>
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
        {campaigns.length === 0 ? (
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

  return (
    <div className="space-y-4">
      <ControlRoomHeader campaigns={campaigns} selectedCampaignId={props.selectedCampaignId} onSelect={props.onSelect} />
      <FreshnessNotice model={model} />
      <BlockerPanel model={model} />
      <CampaignOverview model={model} />
      <CampaignJourney model={model} />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_20rem] xl:grid-cols-[minmax(0,1fr)_22rem]">
        <main className="min-w-0 space-y-4" aria-label="Campaign work and evidence">
          <ActiveWorkPanel model={model} />
          {props.humanOversight}
          {props.campaignRecovery}
          <EvidencePanel model={model} artifacts={props.artifacts} pages={props.pages} onLoadArtifacts={props.onLoadArtifacts} />
          <ActivityPanel events={props.events} pages={props.pages} onLoadEvents={props.onLoadEvents} />
        </main>
        <aside className="min-w-0 space-y-4" aria-label="Campaign authority and context">
          <LaunchAuthority model={model} onStart={props.onStart} onRefresh={props.onRetry} pending={props.startPending} />
          <ControllerPanel model={model} />
          <BudgetPanel model={model} />
          <CollectionPanel model={model} />
        </aside>
      </div>
    </div>
  )
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
  const [startPending, setStartPending] = useState(false)
  const [humanQueue, setHumanQueue] = useState<ParsedHumanWorkQueuePublicV1 | null>(null)
  const [humanQueueError, setHumanQueueError] = useState<string | null>(null)
  const [humanQueueLoading, setHumanQueueLoading] = useState(false)
  const [humanMutationPending, setHumanMutationPending] = useState(false)
  const [humanResponses, setHumanResponses] = useState<Record<string, HumanReviewResponse | undefined>>({})
  const [recoverySnapshot, setRecoverySnapshot] = useState<CampaignRecoveryPublicV1 | null>(null)
  const [recoveryError, setRecoveryError] = useState<string | null>(null)
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
  const transition = useCampaignStore((state) => state.transition)
  const selectedCampaignId = resolveControlRoomCampaignSelection(
    selection.campaignId,
    workspace?.selectedCampaignId,
  )
  const detail = selectedCampaignId ? workspace?.details[selectedCampaignId] : undefined
  const humanSnapshotCursor = detail?.snapshot?.latest_event_cursor ?? null
  const shouldLoadGuidedSetup = Boolean(workspace && !workspace.loading && workspace.campaigns.length === 0)

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
        setRecoveryError(`Recovery authority is unavailable${response.code ? ` (${response.code})` : ''}. The rest of the Control Room remains available.`)
        setRecoveryLoading(false)
        return
      }
      setRecoverySnapshot(response.data)
      setRecoveryError(null)
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
      setSetupError(`Guided setup authority is unavailable${response.code ? ` (${response.code})` : ''}. The setup path remains visible and read-only.`)
      return
    }
    setSetupContext(response.data)
    setSetupConnection('live')
    setSetupError(null)
  }, [workspaceId])

  useEffect(() => wsService.retainCampaignWorkspace(workspaceId), [workspaceId])

  useEffect(() => {
    setupLoadGeneration.current += 1
    setSetupContext(null)
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
    recoveryLoadGeneration.current += 1
    setRecoverySnapshot(null)
    setRecoveryError(null)
    setRecoveryLoading(false)
    setRecoveryConfirmations({ resume: false, repair: false, takeover: false })
    setRecoveryIdempotencyKeys(newRecoveryIdempotencyKeys())
  }, [selectedCampaignId, workspaceId])

  useEffect(() => {
    humanLoadGeneration.current += 1
    setHumanQueue(null)
    setHumanQueueError(null)
    setHumanQueueLoading(false)
    setHumanMutationPending(false)
    setHumanResponses({})
  }, [selectedCampaignId, workspaceId])

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

  const handleStart = async () => {
    if (startPending || !selectedCampaignId || model.kind !== 'snapshot' || !model.authoritative) return
    setStartPending(true)
    try {
      await transition(
        workspaceId,
        selectedCampaignId,
        'start',
        model.snapshot.aggregate_version,
      )
    } finally {
      setStartPending(false)
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
      <div className="mx-auto max-w-[1500px]">
        <ControlRoomContent
          model={model}
          campaigns={workspace?.campaigns || []}
          selectedCampaignId={selectedCampaignId || null}
          events={detail?.pages.events || []}
          artifacts={detail?.pages.artifacts || []}
          pages={detail?.pages || unloadedPages}
          onSelect={handleSelect}
          onRetry={() => {
            if (selectedCampaignId) void refresh(workspaceId, selectedCampaignId)
            else void load(workspaceId, selection.campaignId || undefined)
          }}
          onLoadEvents={() => selectedCampaignId && void loadEventsPage(workspaceId, selectedCampaignId)}
          onLoadArtifacts={() => selectedCampaignId && void loadArtifactsPage(workspaceId, selectedCampaignId)}
          onStart={() => { void handleStart() }}
          startPending={startPending}
          humanOversight={humanOversight}
          campaignRecovery={campaignRecovery}
          guidedSetup={guidedSetup}
        />
      </div>
    </div>
  )
}
