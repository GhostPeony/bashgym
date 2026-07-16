import { memo, useEffect, useMemo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertTriangle,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Square,
} from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { clsx } from 'clsx'
import { useCampaignStore } from '../../../stores/campaignStore'
import { useTerminalStore } from '../../../stores/terminalStore'
import { useWorkspaceStore } from '../../../stores/workspaceStore'
import type {
  CampaignAttempt,
  CampaignComparison,
  CampaignControllerStatus,
  CampaignDetailState,
  CampaignRecord,
  CampaignStatus,
} from '../../../stores/campaignStore'
import { DataNodeShell } from './DataNodeShell'
import { projectCampaignResearch, type CampaignCanvasResearch } from './campaignCanvasModel'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type CampaignNodeType = Node<DataNodeData, 'campaign'>

const ACTIVE_POLL_MS = 3_000
const IDLE_POLL_MS = 12_000

const STATUS_BAR: Record<CampaignStatus, string> = {
  draft: 'bg-background-tertiary',
  validating: 'bg-accent animate-pulse',
  ready: 'bg-accent',
  active: 'bg-accent animate-pulse',
  paused: 'bg-status-warning',
  awaiting_authority: 'bg-status-warning',
  cancelling: 'bg-status-warning animate-pulse',
  completed: 'bg-status-success',
  exhausted: 'bg-status-success',
  failed: 'bg-status-error',
  cancelled: 'bg-background-tertiary',
}

function tone(status?: CampaignStatus): 'neutral' | 'accent' | 'success' | 'warning' | 'error' {
  if (status === 'failed') return 'error'
  if (status === 'completed' || status === 'exhausted') return 'success'
  if (status === 'paused' || status === 'awaiting_authority' || status === 'cancelling') return 'warning'
  if (status === 'active' || status === 'ready' || status === 'validating') return 'accent'
  return 'neutral'
}

function readable(value: string): string {
  return value.replace(/^campaign:/, '').replaceAll('_', ' ').replaceAll(':', ' · ')
}

function shortDigest(value?: string | null): string {
  return value ? `${value.slice(0, 10)}…${value.slice(-6)}` : '—'
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / 1024 ** 3).toFixed(1)} GB`
}

function compactNumber(value: number): string {
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

function budgetSummary(research: CampaignCanvasResearch): string {
  return research.budget.length
    ? research.budget.map((item) => `${readable(item.resource)} ${compactNumber(item.remaining)}`).join(' · ')
    : 'Not recorded'
}

function baselineSummary(research: CampaignCanvasResearch): string {
  const metric = Object.entries(research.baseline.metrics).sort(([left], [right]) => left.localeCompare(right))[0]
  const base = research.baseline.modelRef || 'Model not recorded'
  return metric ? `${base} · ${readable(metric[0])} ${compactNumber(metric[1])}` : base
}

export function CampaignResearchBrief({ research }: { research: CampaignCanvasResearch }) {
  const nextLabel = research.nextAction.kind === 'current'
    ? 'Current action'
    : research.nextAction.kind === 'none'
      ? 'Next action'
      : 'Planned next'
  return (
    <div
      className="rounded-brutal border-brutal border-border-subtle bg-background-secondary"
      role="group"
      aria-label="Campaign research brief"
    >
      <div className="grid grid-cols-2 gap-px bg-border-subtle">
        <div className="min-w-0 bg-background-card px-2 py-1.5">
          <div className="font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">Baseline · {readable(research.baseline.status)}</div>
          <div className="truncate text-[9px] text-text-primary" title={baselineSummary(research)}>{baselineSummary(research)}</div>
        </div>
        <div className="min-w-0 bg-background-card px-2 py-1.5">
          <div className="font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">Budget remaining</div>
          <div className="truncate text-[9px] text-text-primary" title={budgetSummary(research)}>{budgetSummary(research)}</div>
        </div>
      </div>
      <div className="border-t border-border-subtle px-2 py-1.5">
        <div className="font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">Latest hypothesis</div>
        <div className="line-clamp-2 text-[9px] leading-4 text-text-primary">
          {research.latestStudy?.hypothesis || 'No study hypothesis recorded'}
        </div>
        {research.latestStudy ? (
          <div className="truncate font-mono text-[7px] text-text-muted">
            {readable(research.latestStudy.status)} · {readable(research.latestStudy.plannedStage || 'complete')}
          </div>
        ) : null}
      </div>
      <div className="grid grid-cols-2 gap-px border-t border-border-subtle bg-border-subtle">
        <div className="min-w-0 bg-background-card px-2 py-1.5">
          <div className="font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">Latest decision</div>
          <div className="line-clamp-2 text-[9px] leading-4 text-text-primary">{research.latestDecision?.outcome || 'No decision recorded'}</div>
        </div>
        <div className="min-w-0 bg-background-card px-2 py-1.5">
          <div className="font-mono text-[7px] font-bold uppercase tracking-wide text-text-muted">{nextLabel}</div>
          <div className="line-clamp-2 text-[9px] leading-4 text-text-primary">{readable(research.nextAction.label)}</div>
        </div>
      </div>
    </div>
  )
}

interface CampaignEvidencePanelProps {
  campaignId?: string
  campaigns: CampaignRecord[]
  detail?: CampaignDetailState
  controller?: CampaignControllerStatus | null
  chartData: Array<{ step: number; loss: number }>
  latestAttempt?: CampaignAttempt
  latestComparison?: CampaignComparison
  mutating: boolean
  onSelect: (campaignId: string) => void
  onTransition: (action: 'start' | 'pause' | 'resume' | 'cancel') => void
  onRefresh: () => void
}

/** Pure modal body kept separate so the campaign evidence projection is DOM-testable. */
export function CampaignEvidencePanel({
  campaignId,
  campaigns,
  detail,
  controller,
  chartData,
  latestAttempt,
  latestComparison,
  mutating,
  onSelect,
  onTransition,
  onRefresh,
}: CampaignEvidencePanelProps) {
  const research = detail ? projectCampaignResearch(detail) : undefined
  const diagnostics = detail?.ledger?.autoresearch?.diagnostics
  return (
    <>
      <ConfigSection title="Campaign">
        <label className="block font-mono text-[10px] uppercase tracking-wide text-text-muted">
          Workspace campaign
          <select
            aria-label="Workspace campaign"
            value={campaignId || ''}
            onChange={(event) => onSelect(event.target.value)}
            className="nodrag mt-1 w-full rounded-brutal border-brutal border-border bg-background-card px-2 py-1.5 text-xs text-text-primary"
          >
            {campaigns.map((campaign) => (
              <option key={campaign.campaign_id} value={campaign.campaign_id}>{campaign.title}</option>
            ))}
          </select>
        </label>
        {detail ? (
          <>
            <div className="mt-3 flex flex-wrap gap-1.5">
              <ConfigPill tone={tone(detail.campaign.status)}>{readable(detail.campaign.status)}</ConfigPill>
              <ConfigPill tone="neutral">revision {detail.campaign.manifest_revision}</ConfigPill>
              <ConfigPill tone="neutral">version {detail.campaign.version}</ConfigPill>
            </div>
            <ConfigRows>
              <ConfigRow label="Campaign ID" value={detail.campaign.campaign_id} />
              <ConfigRow label="Owner" value={detail.campaign.owner_actor_id} />
              <ConfigRow label="Active study" value={detail.campaign.active_study_id} />
              <ConfigRow label="Active action" value={detail.campaign.active_action_id} />
              <ConfigRow label="Champion" value={detail.campaign.champion_ref || 'Base model retained'} />
              <ConfigRow label="Stop reason" value={detail.campaign.stop_reason} />
            </ConfigRows>
            <div className="mt-3 flex flex-wrap gap-2 nodrag" role="group" aria-label="Campaign controls">
              {detail.campaign.status === 'ready' ? (
                <button className="btn-primary !py-1.5 !text-[10px]" disabled={mutating} onClick={() => onTransition('start')}><Play className="h-3 w-3" />Start</button>
              ) : null}
              {detail.campaign.status === 'active' ? (
                <button className="btn-secondary !py-1.5 !text-[10px]" disabled={mutating} onClick={() => onTransition('pause')}><Pause className="h-3 w-3" />Pause</button>
              ) : null}
              {detail.campaign.status === 'paused' ? (
                <button className="btn-primary !py-1.5 !text-[10px]" disabled={mutating} onClick={() => onTransition('resume')}><Play className="h-3 w-3" />Resume</button>
              ) : null}
              {!['completed', 'exhausted', 'failed', 'cancelled', 'cancelling'].includes(detail.campaign.status) ? (
                <button className="btn-secondary !py-1.5 !text-[10px] !text-status-error" disabled={mutating} onClick={() => onTransition('cancel')}><Square className="h-3 w-3" />Cancel</button>
              ) : null}
              <button className="btn-ghost !py-1.5 !text-[10px]" disabled={detail.loading} onClick={onRefresh}><RefreshCw className={clsx('h-3 w-3', detail.loading && 'animate-spin')} />Refresh</button>
            </div>
          </>
        ) : null}
      </ConfigSection>

      {controller ? (
        <ConfigSection title="Resident Controller">
          <div role="status" aria-label={`Campaign controller ${controller.state}`}>
            <div className="flex flex-wrap gap-1.5">
              <ConfigPill tone={controller.state === 'online' ? 'success' : 'warning'}>
                {readable(controller.state)}
              </ConfigPill>
              {controller.heartbeat_age_seconds != null ? (
                <ConfigPill tone="neutral">heartbeat {compactNumber(controller.heartbeat_age_seconds)}s ago</ConfigPill>
              ) : null}
            </div>
            {controller.guidance ? (
              <div className="mt-2 text-[9px] leading-4 text-text-muted">{controller.guidance}</div>
            ) : null}
          </div>
        </ConfigSection>
      ) : null}

      {research ? <CampaignResearchBrief research={research} /> : null}

      {diagnostics ? (
        <ConfigSection title="AutoResearch Diagnostics">
          <div className="space-y-3" role="group" aria-label="AutoResearch diagnostics">
            <div className="flex flex-wrap gap-1.5">
              <ConfigPill tone={diagnostics.low_signal ? 'warning' : 'success'}>
                {diagnostics.low_signal ? 'low signal detected' : 'signal healthy'}
              </ConfigPill>
              <ConfigPill tone="neutral">
                {diagnostics.checkpoint_comparisons.length} checkpoint comparisons
              </ConfigPill>
              <ConfigPill tone="neutral">{diagnostics.error_slices.length} error slices</ConfigPill>
            </div>

            {diagnostics.signals.length ? (
              <div className="space-y-1" aria-label="Diagnostic signals">
                {diagnostics.signals.slice(0, 8).map((signal) => (
                  <div key={signal.code} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="font-mono text-[9px] font-bold text-text-primary">{readable(signal.code)}</div>
                      <ConfigPill tone={signal.severity === 'critical' ? 'error' : signal.severity === 'warning' ? 'warning' : 'neutral'}>
                        {signal.severity}
                      </ConfigPill>
                    </div>
                    <div className="mt-0.5 text-[9px] leading-4 text-text-muted">{signal.summary}</div>
                  </div>
                ))}
              </div>
            ) : null}

            {diagnostics.checkpoint_comparisons.length ? (
              <div>
                <div className="mb-1 font-mono text-[8px] font-bold uppercase tracking-wide text-text-muted">Checkpoint trajectory</div>
                <div className="space-y-1">
                  {diagnostics.checkpoint_comparisons.slice(0, 8).map((checkpoint) => (
                    <div key={checkpoint.evaluation_result_id} className="grid grid-cols-[1fr_auto_auto] gap-2 rounded-brutal border-brutal border-border-subtle px-2 py-1 font-mono text-[8px] text-text-muted">
                      <span className="truncate">{checkpoint.step != null ? `step ${checkpoint.step}` : checkpoint.role}</span>
                      <span>{compactNumber(checkpoint.metric_value)}</span>
                      <span className={clsx(
                        checkpoint.improvement_from_baseline != null && checkpoint.improvement_from_baseline > 0 && 'text-status-success',
                        checkpoint.improvement_from_baseline != null && checkpoint.improvement_from_baseline < 0 && 'text-status-error',
                      )}>
                        {checkpoint.improvement_from_baseline == null
                          ? 'no baseline delta'
                          : `${checkpoint.improvement_from_baseline >= 0 ? '+' : ''}${compactNumber(checkpoint.improvement_from_baseline)}`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {diagnostics.error_slices.length ? (
              <div>
                <div className="mb-1 font-mono text-[8px] font-bold uppercase tracking-wide text-text-muted">Error slices</div>
                <div className="space-y-1">
                  {diagnostics.error_slices.slice(0, 8).map((slice) => (
                    <div key={slice.slice_path} className="flex items-center justify-between gap-2 rounded-brutal border-brutal border-border-subtle px-2 py-1 font-mono text-[8px]">
                      <span className="truncate text-text-muted">{readable(slice.slice_path)}</span>
                      <span className={clsx(
                        slice.status === 'improved' && 'text-status-success',
                        slice.status === 'regressed' && 'text-status-error',
                        !['improved', 'regressed'].includes(slice.status) && 'text-text-muted',
                      )}>
                        {compactNumber(slice.candidate_value)} · {slice.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {diagnostics.ranked_hypotheses.length ? (
              <div aria-label="Ranked next hypotheses">
                <div className="mb-1 font-mono text-[8px] font-bold uppercase tracking-wide text-text-muted">Ranked next hypotheses · advisory only</div>
                <div className="space-y-1">
                  {diagnostics.ranked_hypotheses.map((hypothesis) => (
                    <div key={hypothesis.hypothesis_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                      <div className="flex items-center gap-1.5 font-mono text-[8px] text-text-muted">
                        <span className="font-bold text-text-primary">#{hypothesis.rank}</span>
                        <ConfigPill tone={hypothesis.action_kind === 'candidate' ? 'accent' : 'neutral'}>{hypothesis.action_kind}</ConfigPill>
                        <span className="truncate">{hypothesis.changed_variable}</span>
                      </div>
                      <div className="mt-1 text-[9px] leading-4 text-text-primary">{hypothesis.hypothesis}</div>
                      <div className="mt-0.5 text-[8px] leading-3 text-text-muted">Falsify: {hypothesis.falsification_criterion}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </ConfigSection>
      ) : null}

      {detail && chartData.length > 1 ? (
        <ConfigSection title={`Loss · ${latestAttempt?.attempt_id || 'latest attempt'}`}>
          <div
            className="h-52 rounded-brutal border-brutal border-border bg-background-secondary p-3"
            role="img"
            aria-label={`Training loss curve, ${chartData.length} points, step ${chartData[0].step} loss ${chartData[0].loss.toFixed(4)} to step ${chartData.at(-1)!.step} loss ${chartData.at(-1)!.loss.toFixed(4)}`}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <XAxis dataKey="step" tick={{ fontSize: 9 }} stroke="var(--text-muted)" />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 9 }} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ background: 'var(--background-card)', border: '2px solid var(--border)' }} />
                <Line type="monotone" dataKey="loss" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </ConfigSection>
      ) : null}

      {detail?.evidence ? (
        <div className="grid gap-4 lg:grid-cols-2" role="group" aria-label="Campaign policy evidence">
          <ConfigSection title="Policy Boundary">
            <ConfigRows>
              <ConfigRow label="Compute profile" value={detail.evidence.compute_profile_id} />
              <ConfigRow label="Approved data" value={detail.evidence.approved_data_scopes.join(', ')} />
              <ConfigRow label="Executors" value={detail.evidence.available_executors.join(', ')} />
              <ConfigRow label="Evidence digest" value={shortDigest(detail.evidence.snapshot_digest)} />
            </ConfigRows>
          </ConfigSection>
          <ConfigSection title="Remaining Budget">
            <ConfigRows>
              {Object.entries(detail.evidence.budget_remaining).map(([unit, remaining]) => (
                <ConfigRow key={unit} label={readable(unit)} value={remaining.toFixed(4)} />
              ))}
            </ConfigRows>
          </ConfigSection>
        </div>
      ) : null}

      {detail?.studies.length ? (
        <ConfigSection title={`Studies (${detail.studies.length})`}>
          <div className="max-h-72 space-y-1 overflow-y-auto pr-1">
            {detail.studies.map((study) => {
              const activeStage = study.stage_plan.items[study.current_stage_index]?.stage || 'complete'
              return (
                <div key={study.study_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <div className="truncate font-mono text-[10px] font-bold text-text-primary">{study.study_id}</div>
                    <ConfigPill tone={study.status.includes('failed') ? 'error' : study.status.includes('passed') || study.status === 'promoted' ? 'success' : 'neutral'}>
                      {readable(study.status)}
                    </ConfigPill>
                  </div>
                  <div className="font-mono text-[8px] text-text-muted">
                    {readable(activeStage)} · stage {Math.min(study.current_stage_index + 1, study.stage_plan.items.length)} / {study.stage_plan.items.length}
                  </div>
                </div>
              )
            })}
          </div>
        </ConfigSection>
      ) : null}

      {latestComparison ? (
        <ConfigSection title="Latest Development Gate">
          <ConfigRows>
            <ConfigRow label="Verdict" value={readable(latestComparison.verdict)} />
            <ConfigRow label="Sample" value={`${latestComparison.sample_count} queries / ${latestComparison.video_count} videos`} />
            <ConfigRow label="Candidate" value={shortDigest(latestComparison.candidate_digest)} />
            <ConfigRow label="Champion" value={shortDigest(latestComparison.champion_digest)} />
            {Object.entries(latestComparison.metrics).slice(0, 10).map(([name, value]) => (
              <ConfigRow key={name} label={readable(name)} value={typeof value === 'number' ? value.toFixed(6) : 'not measured'} />
            ))}
            <ConfigRow label="Blocking reasons" value={latestComparison.blocking_reasons.join('; ') || 'None'} />
            <ConfigRow label="Warnings" value={latestComparison.warnings.join('; ') || 'None'} />
          </ConfigRows>
        </ConfigSection>
      ) : null}

      {detail?.ledger?.linked ? (
        <ConfigSection title="Linked Experiment Ledger">
          <div className="space-y-2" role="group" aria-label="Linked experiment ledger evidence">
            {detail.ledger.projects.map((project) => (
              <div key={project.project.project_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate font-mono text-[10px] font-bold text-text-primary">
                    {project.project.display_name || project.project.project_id}
                  </div>
                  <ConfigPill tone="neutral">{project.project.project_id}</ConfigPill>
                </div>
                <div className="mt-1 grid grid-cols-4 gap-1 font-mono text-[8px] text-text-muted">
                  <span>{project.experiments.length} experiments</span>
                  <span>{project.runs.length} runs</span>
                  <span>{project.evaluations.length} evals</span>
                  <span>{project.decisions.length} decisions</span>
                </div>
                {project.evaluations.slice(0, 3).map((evaluation) => (
                  <div key={evaluation.evaluation_result_id} className="mt-1 border-t border-border-subtle pt-1 font-mono text-[8px] text-text-muted">
                    {evaluation.evaluation_result_id} · {evaluation.evaluation_suite_id} · {readable(evaluation.status)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </ConfigSection>
      ) : null}

      {detail?.ledger?.projects.some((project) => project.decisions.length) ? (
        <ConfigSection title="Decisions">
          <div className="space-y-1" role="group" aria-label="Campaign decisions">
            {detail.ledger.projects.flatMap((project) => project.decisions).slice(0, 12).map((decision) => (
              <div key={decision.decision_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                <div className="font-mono text-[10px] font-bold text-text-primary">{readable(decision.decision_type)}</div>
                <div className="text-[9px] leading-4 text-text-muted">{decision.outcome}</div>
                <div className="font-mono text-[8px] text-text-muted">{decision.decision_id}</div>
              </div>
            ))}
          </div>
        </ConfigSection>
      ) : null}

      {detail ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <ConfigSection title={`Recent Events (${detail.events.length})`}>
            <div className="max-h-72 space-y-1 overflow-y-auto pr-1" role="group" aria-label="Recent campaign events">
              {detail.events.slice(-12).reverse().map((item) => (
                <div key={item.event.event_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                  <div className="font-mono text-[10px] font-bold text-text-primary">{readable(item.event.event_type)}</div>
                  <div className="font-mono text-[8px] text-text-muted">cursor {item.cursor} · {new Date(item.event.created_at).toLocaleString()}</div>
                </div>
              ))}
            </div>
          </ConfigSection>
          <ConfigSection title={`Sealed Evidence (${detail.artifacts.length})`}>
            <div className="max-h-72 space-y-1 overflow-y-auto pr-1" role="group" aria-label="Sealed campaign evidence">
              {detail.artifacts.slice(-12).reverse().map((artifact) => (
                <div key={artifact.artifact_id} className="rounded-brutal border-brutal border-border-subtle px-2 py-1.5">
                  <div className="truncate font-mono text-[10px] font-bold text-text-primary">{readable(artifact.schema_name)}</div>
                  <div className="font-mono text-[8px] text-text-muted">{shortDigest(artifact.sha256)} · {formatBytes(artifact.size_bytes)}</div>
                </div>
              ))}
            </div>
          </ConfigSection>
        </div>
      ) : null}
    </>
  )
}

export const CampaignNode = memo(function CampaignNode({ data, selected }: NodeProps<CampaignNodeType>) {
  const workspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const workspace = useCampaignStore((state) => state.workspaces[workspaceId])
  const load = useCampaignStore((state) => state.load)
  const refresh = useCampaignStore((state) => state.refresh)
  const refreshController = useCampaignStore((state) => state.refreshController)
  const selectCampaign = useCampaignStore((state) => state.select)
  const transition = useCampaignStore((state) => state.transition)
  const updatePanelConfig = useTerminalStore((state) => state.updatePanelConfig)
  const [configOpen, setConfigOpen] = useState(false)
  const [mutating, setMutating] = useState(false)
  const configuredCampaignId = typeof data.adapterConfig?.campaignId === 'string'
    ? data.adapterConfig.campaignId
    : undefined
  const campaignId = configuredCampaignId || workspace?.selectedCampaignId || undefined
  const detail = campaignId ? workspace?.details[campaignId] : undefined

  useEffect(() => {
    void load(workspaceId, configuredCampaignId)
  }, [configuredCampaignId, load, workspaceId])

  useEffect(() => {
    if (!campaignId || configuredCampaignId === campaignId) return
    updatePanelConfig(data.panelId, {
      ...(data.adapterConfig || {}),
      campaignId,
    })
  }, [campaignId, configuredCampaignId, data.adapterConfig, data.panelId, updatePanelConfig])

  useEffect(() => {
    if (!campaignId) return
    const active = ['active', 'validating', 'cancelling'].includes(detail?.campaign.status || '')
    const interval = active ? ACTIVE_POLL_MS : IDLE_POLL_MS
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') {
        void refresh(workspaceId, campaignId)
        void refreshController(workspaceId)
      }
    }, interval)
    return () => window.clearInterval(timer)
  }, [campaignId, detail?.campaign.status, refresh, refreshController, workspaceId])

  const latestAttempt = detail?.attempts.at(-1)
  const loss = useMemo(
    () => latestAttempt ? detail?.lossByAttempt[latestAttempt.attempt_id] || [] : [],
    [detail?.lossByAttempt, latestAttempt],
  )
  const latestComparison = detail?.comparisons.at(-1)
  const ledgerEvaluationCount = detail?.ledger?.projects.reduce(
    (total, project) => total + project.evaluations.length, 0,
  ) || 0
  const ledgerDecisionCount = detail?.ledger?.projects.reduce(
    (total, project) => total + project.decisions.length, 0,
  ) || 0
  const chartData = useMemo(
    () => loss.map((point) => ({ step: point.step, loss: point.value })),
    [loss],
  )
  const research = useMemo(
    () => detail ? projectCampaignResearch(detail) : undefined,
    [detail],
  )

  const handleSelect = async (nextId: string) => {
    updatePanelConfig(data.panelId, { ...(data.adapterConfig || {}), campaignId: nextId })
    await selectCampaign(workspaceId, nextId)
  }

  const runTransition = async (action: 'start' | 'pause' | 'resume' | 'cancel') => {
    if (!detail || mutating) return
    setMutating(true)
    try {
      const reason = action === 'pause'
        ? 'Paused from the BashGym campaign canvas.'
        : action === 'cancel'
          ? 'Cancelled from the BashGym campaign canvas.'
          : undefined
      await transition(workspaceId, detail.campaign.campaign_id, action, detail.campaign.version, reason)
    } finally {
      setMutating(false)
    }
  }

  const buildContext = () => {
    if (!detail) return '## Experiment campaign\n\nNo campaign is selected.'
    const campaign = detail.campaign
    const lines = [
      `## Experiment campaign — ${campaign.title}`,
      '',
      `- campaign: ${campaign.campaign_id}`,
      `- status: ${campaign.status} (version ${campaign.version})`,
      `- objective: ${campaign.objective}`,
      `- manifest revision: ${campaign.manifest_revision}`,
      `- champion: ${campaign.champion_ref || 'base champion retained'}`,
      `- active study: ${campaign.active_study_id || 'none'}`,
      `- active action: ${campaign.active_action_id || 'none'}`,
    ]
    if (research) {
      lines.push('', '### Research brief')
      lines.push(`- baseline status: ${research.baseline.status}`)
      lines.push(`- baseline model: ${research.baseline.modelRef || 'not recorded'}`)
      lines.push(`- baseline run: ${research.baseline.runId || 'not recorded'}`)
      if (research.baseline.evaluationSuiteId) lines.push(`- baseline evaluation suite: ${research.baseline.evaluationSuiteId}`)
      for (const [name, value] of Object.entries(research.baseline.metrics).sort(([left], [right]) => left.localeCompare(right))) {
        lines.push(`- baseline ${name}: ${value}`)
      }
      if (research.latestStudy) {
        lines.push(`- latest study: ${research.latestStudy.studyId} (${research.latestStudy.status})`)
        lines.push(`- latest hypothesis: ${research.latestStudy.hypothesis || 'not recorded'}`)
        lines.push(`- primary variable: ${research.latestStudy.primaryVariable || 'not recorded'}`)
        lines.push(`- falsification criterion: ${research.latestStudy.falsificationCriterion || 'not recorded'}`)
      }
      lines.push(`- budget remaining: ${budgetSummary(research)}`)
      lines.push(`- latest decision: ${research.latestDecision?.outcome || 'not recorded'}`)
      const actionLabel = research.nextAction.kind === 'current'
        ? 'current action'
        : research.nextAction.kind === 'none'
          ? 'next action'
          : 'planned next action'
      lines.push(`- ${actionLabel}: ${research.nextAction.label}`)
    }
    if (detail.evidence) {
      lines.push('', '### Policy and remaining budget')
      lines.push(`- compute profile: ${detail.evidence.compute_profile_id}`)
      lines.push(`- approved data scopes: ${detail.evidence.approved_data_scopes.join(', ')}`)
      for (const [unit, remaining] of Object.entries(detail.evidence.budget_remaining)) {
        lines.push(`- ${unit} remaining: ${remaining}`)
      }
    }
    if (workspace?.controller) {
      lines.push('', '### Resident campaign controller')
      lines.push(`- state: ${workspace.controller.state}`)
      lines.push(`- code: ${workspace.controller.code}`)
      if (workspace.controller.heartbeat_age_seconds != null) {
        lines.push(`- heartbeat age seconds: ${workspace.controller.heartbeat_age_seconds}`)
      }
      if (workspace.controller.guidance) lines.push(`- guidance: ${workspace.controller.guidance}`)
    }
    if (detail.studies.length) {
      lines.push('', '### Studies')
      for (const study of detail.studies.slice(-12)) {
        const stage = study.stage_plan.items[study.current_stage_index]?.stage || 'complete'
        lines.push(`- ${study.study_id}: ${study.status}; cursor ${study.current_stage_index}/${study.stage_plan.items.length}; stage ${stage}`)
      }
    }
    if (latestAttempt) {
      lines.push('', '### Latest attempt')
      lines.push(`- ${latestAttempt.attempt_id}: ${latestAttempt.stage} / ${latestAttempt.status}`)
      lines.push(`- candidate: ${latestAttempt.candidate_digest}`)
      if (loss.length) {
        lines.push(`- loss: ${loss[0].value.toFixed(4)} @${loss[0].step} → ${loss.at(-1)!.value.toFixed(4)} @${loss.at(-1)!.step}`)
      }
    }
    if (latestComparison) {
      lines.push('', '### Latest development comparison')
      lines.push(`- verdict: ${latestComparison.verdict}`)
      lines.push(`- sample: ${latestComparison.sample_count} queries / ${latestComparison.video_count} videos`)
      for (const [name, value] of Object.entries(latestComparison.metrics)) {
        if (typeof value === 'number') lines.push(`- ${name}: ${value.toFixed(6)}`)
      }
      for (const reason of latestComparison.blocking_reasons) lines.push(`- blocker: ${reason}`)
    }
    if (detail.ledger?.linked) {
      lines.push('', '### Linked experiment ledger')
      for (const project of detail.ledger.projects) {
        lines.push(`- project: ${project.project.project_id} (${project.project.display_name})`)
        lines.push(`- experiments: ${project.evidence.experiment_ids.join(', ') || 'none'}`)
        lines.push(`- runs: ${project.evidence.run_ids.join(', ') || 'none'}`)
        for (const evaluation of project.evaluations.slice(0, 12)) {
          lines.push(`- evaluation ${evaluation.evaluation_result_id}: suite ${evaluation.evaluation_suite_id}; run ${evaluation.run_id}; ${evaluation.status}`)
        }
        for (const decision of project.decisions.slice(0, 12)) {
          lines.push(`- decision ${decision.decision_id}: ${decision.outcome}`)
        }
      }
    }
    if (detail.artifacts.length) {
      lines.push('', '### Sealed evidence')
      for (const artifact of detail.artifacts.slice(-12)) {
        lines.push(`- ${artifact.schema_name}: sha256:${artifact.sha256}`)
      }
    }
    lines.push('', 'Protected rows and local artifact paths are intentionally omitted.')
    return lines.join('\n')
  }

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={data.title}
        flowerVariant="campaign"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? buildContext : undefined}
        contextBasename={campaignId ? `campaign_${campaignId}` : 'campaign_context'}
        statusBarClass={detail ? STATUS_BAR[detail.campaign.status] : 'bg-background-tertiary'}
        visualPhase={detail?.campaign.status === 'active' ? 'running' : undefined}
        hue={hueFor('campaign')}
        headerRight={
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              setConfigOpen(true)
            }}
            className="nodrag node-btn node-btn-accent"
            title="Open campaign evidence"
            aria-label="Open campaign evidence"
            aria-haspopup="dialog"
            aria-expanded={configOpen}
          >
            <SlidersHorizontal className="w-3 h-3" />
          </button>
        }
        onFocus={data.onFocus}
        onClose={data.onClose}
      >
        {workspace?.loading && !detail ? (
          <div className="flex justify-center py-5"><Loader2 className="h-4 w-4 animate-spin text-accent" /></div>
        ) : workspace?.error && !detail ? (
          <div className="py-3 text-center font-mono text-[10px] text-status-error">{workspace.error}</div>
        ) : !detail ? (
          <div className="py-4 text-center font-mono text-[10px] text-text-muted">No campaigns in this workspace</div>
        ) : (
          <div className="space-y-2">
            <div className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <div className="truncate font-mono text-[11px] font-bold text-text-primary">{detail.campaign.title}</div>
                <div className="line-clamp-2 text-[10px] leading-4 text-text-muted">{detail.campaign.objective}</div>
              </div>
              <span className={clsx(
                'border-brutal rounded-brutal px-1.5 py-0.5 font-mono text-[8px] font-bold uppercase',
                tone(detail.campaign.status) === 'error' && 'border-status-error/60 bg-status-error/10 text-status-error',
                tone(detail.campaign.status) === 'success' && 'border-status-success/60 bg-status-success/10 text-status-success',
                tone(detail.campaign.status) === 'warning' && 'border-status-warning/60 bg-status-warning/10 text-status-warning',
                tone(detail.campaign.status) === 'accent' && 'border-accent/60 bg-accent/10 text-accent',
                tone(detail.campaign.status) === 'neutral' && 'border-border-subtle text-text-muted',
              )}>{readable(detail.campaign.status)}</span>
            </div>

            <div className="grid grid-cols-3 gap-px overflow-hidden rounded-brutal border-brutal border-border-subtle bg-border-subtle">
              {[
                ['studies', detail.studies.length],
                ['attempts', detail.attempts.length],
                ['evidence', detail.artifacts.length],
              ].map(([label, value]) => (
                <div key={label} className="bg-background-card px-2 py-1 text-center">
                  <div className="font-mono text-[11px] font-bold text-text-primary">{value}</div>
                  <div className="font-mono text-[7px] uppercase tracking-wide text-text-muted">{label}</div>
                </div>
              ))}
            </div>

            {research ? <CampaignResearchBrief research={research} /> : null}

            {workspace?.controller ? (
              <div
                className={clsx(
                  'flex items-center gap-2 font-mono text-[9px]',
                  workspace.controller.state === 'online' ? 'text-status-success' : 'text-status-warning',
                )}
                role="status"
                aria-label={`Campaign controller ${workspace.controller.state}`}
              >
                <span className="font-bold">Controller</span>
                <span>{readable(workspace.controller.state)}</span>
              </div>
            ) : null}

            {chartData.length > 1 ? (
              <div
                className="h-16 rounded-brutal border-brutal border-border-subtle bg-background-secondary p-1"
                role="img"
                aria-label={`Training loss curve, ${chartData.length} points, step ${chartData[0].step} loss ${chartData[0].loss.toFixed(4)} to step ${chartData.at(-1)!.step} loss ${chartData.at(-1)!.loss.toFixed(4)}`}
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <YAxis hide domain={['auto', 'auto']} />
                    <Line type="monotone" dataKey="loss" stroke="var(--accent)" dot={false} strokeWidth={2} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="rounded-brutal border-brutal border-border-subtle px-2 py-2 font-mono text-[9px] text-text-muted">
                {latestAttempt ? `${readable(latestAttempt.stage)} · ${readable(latestAttempt.status)}` : 'Waiting for the first study attempt'}
              </div>
            )}

            {latestComparison ? (
              <div className="flex items-center gap-2 font-mono text-[9px]">
                <span className={clsx(
                  'font-bold uppercase',
                  latestComparison.verdict === 'passed' ? 'text-status-success' : latestComparison.verdict === 'failed' ? 'text-status-error' : 'text-status-warning',
                )}>{readable(latestComparison.verdict)}</span>
                <span className="truncate text-text-muted">{latestComparison.sample_count} queries · {latestComparison.video_count} videos</span>
              </div>
            ) : null}
            {detail.ledger?.linked ? (
              <div className="flex items-center gap-2 font-mono text-[9px] text-text-muted">
                <span className="font-bold text-text-primary">Ledger</span>
                <span>{ledgerEvaluationCount} evals · {ledgerDecisionCount} decisions</span>
              </div>
            ) : null}
            {detail.error ? (
              <div className="flex items-start gap-1 text-[9px] text-status-error"><AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />{detail.error}</div>
            ) : null}
          </div>
        )}
      </DataNodeShell>

      <NodeConfigModal
        isOpen={configOpen}
        onClose={() => setConfigOpen(false)}
        title={detail?.campaign.title || 'Experiment Campaign'}
        description="Durable training, evaluation, and evidence history"
        size="xl"
      >
        <CampaignEvidencePanel
          campaignId={campaignId}
          campaigns={workspace?.campaigns || []}
          detail={detail}
          controller={workspace?.controller}
          chartData={chartData}
          latestAttempt={latestAttempt}
          latestComparison={latestComparison}
          mutating={mutating}
          onSelect={(nextId) => void handleSelect(nextId)}
          onTransition={(action) => void runTransition(action)}
          onRefresh={() => detail && void refresh(workspaceId, detail.campaign.campaign_id)}
        />

      </NodeConfigModal>
    </>
  )
})
