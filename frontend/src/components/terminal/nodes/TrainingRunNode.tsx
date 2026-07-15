import { memo, useEffect, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Loader2, Pause, Play, Save, Server, SlidersHorizontal, Square, Wifi, WifiOff } from 'lucide-react'
import { LineChart, Line, YAxis } from 'recharts'
import { clsx } from 'clsx'
import { useTerminalStore, useTrainingStore, useWorkspaceStore } from '../../../stores'
import type {
  ArtifactRetention,
  HFUploadArtifact,
  TrainingConfig,
  TrainingRun,
  TrainingStrategy,
} from '../../../stores/trainingStore'
import { API_BASE, trainingApi } from '../../../services/api'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import { ConfigPill, ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type TrainingRunNodeType = Node<DataNodeData, 'training'>

const STATUS_BAR: Record<string, string> = {
  planned: 'bg-background-tertiary',
  pending: 'bg-accent',
  queued: 'bg-accent',
  running: 'bg-accent animate-pulse',
  starting: 'bg-accent animate-pulse',
  paused: 'bg-status-warning',
  failed: 'bg-status-error',
  completed: 'bg-status-success'
}

function statusTone(status?: string): 'neutral' | 'accent' | 'success' | 'warning' | 'error' {
  if (status === 'failed') return 'error'
  if (status === 'completed') return 'success'
  if (status === 'paused') return 'warning'
  if (status === 'running' || status === 'starting' || status === 'queued' || status === 'pending') return 'accent'
  return 'neutral'
}

interface TrainingNodeConfig {
  runId?: string
  strategy?: string
  status?: string
  baseModel?: string
  datasetPath?: string
  computeTarget?: string
  originPanelId?: string
  originTerminalId?: string
  originAgent?: string
  correlationId?: string
  summary?: string
  epochs?: number
  batchSize?: number
  learningRate?: number
  gradientAccumulationSteps?: number
  maxSeqLength?: number
  saveSteps?: number
  checkpointLimit?: number
  artifactRetention?: ArtifactRetention
  autoExportGGUF?: boolean
  autoPushHF?: boolean
  hfRepoName?: string
  hfPrivate?: boolean
  hfUploadArtifact?: HFUploadArtifact
  loraRank?: number
  loraAlpha?: number
  loraDropout?: number
  load4Bit?: boolean
  backend?: TrainingBackend
  deviceId?: string
  runtimeJobId?: string
  runtimePid?: number
  runtimeStartedAt?: string
  runtimeDiscovered?: boolean
  progress?: {
    current: number
    total?: number | null
    unit: string
  }
  visual?: {
    phase?: string
    motion?: string
  }
}

type LaunchableTrainingStrategy = Exclude<TrainingStrategy, 'cascade'>
type TrainingBackend = 'local' | 'cloud' | 'remote'

interface TrainingDraft {
  strategy: LaunchableTrainingStrategy
  baseModel: string
  datasetPath: string
  epochs: number
  batchSize: number
  learningRate: number
  gradientAccumulationSteps: number
  maxSeqLength: number
  saveSteps: number
  checkpointLimit: number
  artifactRetention: ArtifactRetention
  autoExportGGUF: boolean
  autoPushHF: boolean
  hfRepoName: string
  hfPrivate: boolean
  hfUploadArtifact: HFUploadArtifact
  loraRank: number
  loraAlpha: number
  loraDropout: number
  load4Bit: boolean
  backend: TrainingBackend
  deviceId: string
}

const LAUNCHABLE_STRATEGIES: LaunchableTrainingStrategy[] = [
  'sft',
  'dpo',
  'grpo',
  'distillation',
  'session_distillation',
]

function launchableStrategy(value?: string): LaunchableTrainingStrategy {
  return LAUNCHABLE_STRATEGIES.includes(value as LaunchableTrainingStrategy)
    ? value as LaunchableTrainingStrategy
    : 'sft'
}

function draftFromTraining(node: TrainingNodeConfig, run: TrainingRun | null): TrainingDraft {
  const config = run?.config
  const backend: TrainingBackend = config?.useRemoteSSH
    ? 'remote'
    : config?.useNemoGym
      ? 'cloud'
      : node.backend || (node.computeTarget?.startsWith('ssh:') ? 'remote' : node.computeTarget === 'cloud' ? 'cloud' : 'local')
  return {
    strategy: launchableStrategy(config?.strategy || node.strategy),
    baseModel: config?.baseModel || node.baseModel || 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    datasetPath: config?.datasetPath || node.datasetPath || '',
    epochs: config?.epochs ?? node.epochs ?? 3,
    batchSize: config?.batchSize ?? node.batchSize ?? 1,
    learningRate: config?.learningRate ?? node.learningRate ?? 2e-5,
    gradientAccumulationSteps: config?.gradientAccumulationSteps ?? node.gradientAccumulationSteps ?? 8,
    maxSeqLength: config?.maxSeqLength ?? node.maxSeqLength ?? 2048,
    saveSteps: config?.saveSteps ?? node.saveSteps ?? 100,
    checkpointLimit: config?.checkpointLimit ?? node.checkpointLimit ?? 1,
    artifactRetention: config?.artifactRetention ?? node.artifactRetention ?? 'adapter_only',
    autoExportGGUF: config?.autoExportGGUF ?? node.autoExportGGUF ?? false,
    autoPushHF: config?.autoPushHF ?? node.autoPushHF ?? false,
    hfRepoName: config?.hfRepoName ?? node.hfRepoName ?? '',
    hfPrivate: config?.hfPrivate ?? node.hfPrivate ?? true,
    hfUploadArtifact: config?.hfUploadArtifact ?? node.hfUploadArtifact ?? 'auto',
    loraRank: config?.loraRank ?? node.loraRank ?? 16,
    loraAlpha: config?.loraAlpha ?? node.loraAlpha ?? 32,
    loraDropout: config?.loraDropout ?? node.loraDropout ?? 0.05,
    load4Bit: config?.load4Bit ?? node.load4Bit ?? true,
    backend,
    deviceId: config?.deviceId || node.deviceId || '',
  }
}

function configFromDraft(draft: TrainingDraft): TrainingConfig {
  return {
    strategy: draft.strategy,
    baseModel: draft.baseModel.trim(),
    datasetPath: draft.datasetPath.trim(),
    epochs: draft.epochs,
    batchSize: draft.batchSize,
    learningRate: draft.learningRate,
    warmupRatio: 0.1,
    gradientAccumulationSteps: draft.gradientAccumulationSteps,
    maxSeqLength: draft.maxSeqLength,
    saveSteps: draft.saveSteps,
    checkpointLimit: draft.checkpointLimit,
    artifactRetention: draft.artifactRetention,
    autoExportGGUF: draft.autoExportGGUF,
    ggufQuantization: 'q4_k_m',
    autoPushHF: draft.autoPushHF,
    hfRepoName: draft.hfRepoName,
    hfPrivate: draft.hfPrivate,
    hfUploadArtifact: draft.hfUploadArtifact,
    loraRank: draft.loraRank,
    loraAlpha: draft.loraAlpha,
    loraDropout: draft.loraDropout,
    load4Bit: draft.load4Bit,
    dpoBeta: 0.1,
    grpoNumGenerations: 4,
    grpoTemperature: 0.7,
    grpoLossType: 'grpo',
    grpoBackend: 'auto',
    grpoUseVllm: false,
    trainingProfile: 'default',
    grpoGroupSize: 4,
    promptsPerRolloutBatch: 8,
    maxToolCallsPerEpisode: 64,
    tokenLevelLoss: false,
    filterZeroStdGroups: false,
    activeSampling: false,
    lmHeadFp32: false,
    interleavedThinking: false,
    sftWarmStartPolicy: 'none',
    dppoBackend: 'auto',
    dppoDivergence: 'binary_tv',
    dppoBinaryTvThreshold: 0.15,
    dppoBinaryKlThreshold: 0.05,
    echoEnabled: false,
    echoAuxLambda: 0.05,
    rwmlEnabled: false,
    rwmlDistanceThreshold: 0.2,
    rwmlEasyPassRateThreshold: 0.8,
    rwmlEasyKeepProbability: 0.1,
    rwmlHistoryWindow: 4,
    rwmlEmbeddingModel: '',
    rwmlKlBeta: 0,
    useLiger: false,
    teacherModel: 'meta-llama/Llama-3.1-70B-Instruct',
    teacherTemperature: 2,
    distillationAlpha: 0.5,
    sessionDistillationAlpha: 0.7,
    sessionDistillationTemperature: 1,
    sessionDistillationMinConfidence: 0.6,
    sessionDistillationMaskPolicy: 'target_span_only',
    sessionDistillationContextMode: 'hint_injected',
    sessionDistillationReader: 'heuristic',
    dataSource: draft.datasetPath.trim() ? 'dataset_path' : 'traces',
    useNemoGym: draft.backend === 'cloud',
    useRemoteSSH: draft.backend === 'remote',
    deviceId: draft.backend === 'remote' ? draft.deviceId.trim() || undefined : undefined,
  }
}

function computeTargetFromDraft(draft: TrainingDraft): string {
  if (draft.backend === 'remote') return draft.deviceId.trim() ? `ssh:${draft.deviceId.trim()}` : 'ssh:remote'
  return draft.backend === 'cloud' ? 'cloud' : 'local'
}

/** Where the run executes: prefer the live payload, fall back to the launch config */
function computeTargetFor(run: TrainingRun | null, config?: TrainingNodeConfig): string | undefined {
  return run?.currentMetrics?.computeTarget ?? config?.computeTarget ?? (
    run?.config?.useRemoteSSH
      ? (run.config.deviceId ? `ssh:${run.config.deviceId}` : 'ssh:remote')
      : run?.config?.useNemoGym
        ? 'cloud'
        : run?.config?.strategy ? 'local' : undefined
  )
}

function fmtDuration(ms: number): string {
  const totalMin = Math.floor(ms / 60000)
  const h = Math.floor(totalMin / 60)
  return h > 0 ? `${h}h ${totalMin % 60}m` : `${totalMin}m`
}

/**
 * Builds the Send-to-terminal markdown payload: a snapshot of run identity,
 * config, telemetry, and health, plus live handles (file paths + API endpoints)
 * the receiving agent can follow for fresh data and run control.
 */
async function buildTrainingContext(runId?: string, nodeConfig?: TrainingNodeConfig): Promise<string> {
  const { currentRun, runs, lossHistory, logs, grpoMetrics } = useTrainingStore.getState()
  const run = runId
    ? (currentRun?.id === runId ? currentRun : runs.find((candidate) => candidate.id === runId) ?? null)
    : currentRun
  const runLossHistory = run?.metricsHistory?.length
    ? run.metricsHistory.map((point) => ({ step: point.step, loss: point.loss, evalLoss: point.evalLoss }))
    : lossHistory
  const stamp = new Date().toISOString()

  if (!run) {
    if (nodeConfig?.status || nodeConfig?.strategy) {
      return [
        `## Planned training work${nodeConfig?.strategy ? ` (${nodeConfig.strategy})` : ''}`,
        '',
        `Canvas state exported from BashGym at ${stamp}.`,
        '',
        `- status: ${nodeConfig.status || 'planned'}`,
        nodeConfig.correlationId ? `- correlation id: ${nodeConfig.correlationId}` : '',
        nodeConfig.originAgent ? `- origin agent: ${nodeConfig.originAgent}` : '',
        nodeConfig.computeTarget ? `- compute target: ${nodeConfig.computeTarget}` : '',
        nodeConfig.baseModel ? `- base model: ${nodeConfig.baseModel}` : '',
        nodeConfig.datasetPath ? `- dataset: ${nodeConfig.datasetPath}` : '',
        nodeConfig.summary ? `- summary: ${nodeConfig.summary}` : '',
        '',
        `- start a run: POST ${API_BASE}/training/start with the same correlation_id`,
        `- refresh workspace context: GET ${API_BASE}/workspace/context?format=json`
      ].filter(Boolean).join('\n')
    }
    return [
      '## Training status',
      '',
      `No active training run (as of ${stamp}).`,
      '',
      `- list past runs: GET ${API_BASE}/training/runs`,
      `- start a run: POST ${API_BASE}/training/start`
    ].join('\n')
  }

  const c = run.config
  const m = run.currentMetrics
  const target = computeTargetFor(run, nodeConfig)
  const lines: string[] = []

  lines.push(`## Training run ${run.id}`)
  lines.push(`Live state exported from the BashGym workspace canvas at ${stamp}.`)
  lines.push('')
  lines.push(`- status: ${run.status}${m?.simulation ? ' (simulation mode — no real GPU training)' : ''}`)
  if (run.error) lines.push(`- error: ${run.error}`)
  lines.push(`- strategy: ${c.strategy ?? 'unknown'}`)
  lines.push(`- base model: ${c.baseModel ?? 'unknown'}`)
  if (target) lines.push(`- compute target: ${target}`)
  if (c.datasetPath) lines.push(`- dataset: ${c.datasetPath}`)
  lines.push(`- started: ${new Date(run.startTime).toISOString()}`)
  if (run.endTime) {
    lines.push(`- ended: ${new Date(run.endTime).toISOString()} (duration ${fmtDuration(run.endTime - run.startTime)})`)
  } else {
    lines.push(`- elapsed: ${fmtDuration(Date.now() - run.startTime)}`)
  }

  lines.push('', '### Config')
  lines.push(`- epochs ${c.epochs}, batch size ${c.batchSize}, lr ${c.learningRate}, max seq ${c.maxSeqLength}`)
  if (c.loraRank != null) {
    lines.push(`- LoRA: rank ${c.loraRank}, alpha ${c.loraAlpha ?? '?'}${c.load4Bit ? ', 4-bit (QLoRA)' : ''}`)
  }
  if (c.strategy === 'dpo' && c.dpoBeta != null) lines.push(`- DPO beta: ${c.dpoBeta}`)
  if (c.strategy === 'grpo') {
    lines.push(`- GRPO: loss ${c.grpoLossType ?? 'grpo'}, generations ${c.grpoNumGenerations ?? '?'}, backend ${c.grpoBackend ?? 'auto'}${c.grpoUseVllm ? ' (vLLM)' : ''}`)
  }
  if ((c.strategy === 'distillation' || c.strategy === 'session_distillation') && c.teacherModel) {
    lines.push(`- teacher: ${c.teacherModel}`)
  }
  if (c.selectedRepos?.length) lines.push(`- repos: ${c.selectedRepos.join(', ')}`)

  if (m) {
    lines.push('', '### Telemetry')
    lines.push(`- step ${m.step}/${m.totalSteps || '?'} (epoch ${m.epoch.toFixed(2)}${m.eta ? `, ETA ${m.eta}` : ''})`)
    lines.push(`- loss ${m.loss.toFixed(4)}${m.evalLoss != null ? `, eval loss ${m.evalLoss.toFixed(4)}` : ''} (grad norm ${m.gradNorm}, lr ${m.learningRate})`)
    if (runLossHistory.length > 1) {
      const first = runLossHistory[0]
      const min = runLossHistory.reduce((a, b) => (b.loss < a.loss ? b : a))
      const last = runLossHistory[runLossHistory.length - 1]
      lines.push(`- loss trend: ${first.loss.toFixed(4)} @step ${first.step} → min ${min.loss.toFixed(4)} @step ${min.step} → ${last.loss.toFixed(4)} @step ${last.step}`)
      lines.push(`- last ${Math.min(10, runLossHistory.length)} losses: ${runLossHistory.slice(-10).map((p) => p.loss.toFixed(4)).join(', ')}`)
    }
    const hw: string[] = []
    if (m.tokensPerSecond != null) hw.push(`${m.tokensPerSecond.toFixed(0)} tok/s`)
    if (m.gpuUtilization != null) hw.push(`GPU ${m.gpuUtilization.toFixed(0)}%`)
    if (m.gpuMemoryGb != null) hw.push(`${m.gpuMemoryGb.toFixed(1)} GB VRAM`)
    if (hw.length) lines.push(`- throughput: ${hw.join(', ')}`)
    const g = grpoMetrics[grpoMetrics.length - 1]
    if (g) {
      const parts: string[] = []
      if (g.reward != null) parts.push(`reward ${g.reward.toFixed(3)}${g.rewardStd != null ? ` ± ${g.rewardStd.toFixed(3)}` : ''}`)
      if (g.kl != null) parts.push(`KL ${g.kl.toFixed(4)}`)
      if (g.fracRewardZeroStd != null) parts.push(`zero-std frac ${g.fracRewardZeroStd.toFixed(2)}`)
      if (g.exitCodeAccuracy != null) parts.push(`exit-code acc ${g.exitCodeAccuracy.toFixed(2)}`)
      if (g.testResultAccuracy != null) parts.push(`test acc ${g.testResultAccuracy.toFixed(2)}`)
      if (parts.length) lines.push(`- GRPO @step ${g.step}: ${parts.join(', ')}`)
    }
  }

  const recent = logs.slice(-5)
  const recentSet = new Set(recent)
  const earlierProblems = logs.filter((l) => l.level !== 'info' && !recentSet.has(l)).slice(-5)
  if (earlierProblems.length) {
    lines.push('', '### Earlier warnings & errors')
    for (const l of earlierProblems) lines.push(`- [${l.level}] ${l.message}`)
  }
  if (recent.length) {
    lines.push('', '### Recent logs')
    for (const l of recent) lines.push(`- ${l.level !== 'info' ? `[${l.level}] ` : ''}${l.message}`)
  }

  try {
    const res = await trainingApi.getRunAnalysis(run.id)
    const analysis = res.ok ? res.data : undefined
    if (analysis?.ok && analysis.verdict) {
      lines.push('', `### Health analysis (verdict: ${analysis.verdict.level})`)
      const findings = analysis.findings ?? []
      for (const f of findings.slice(0, 8)) {
        lines.push(`- [${f.severity}] ${f.code}: ${f.message}${f.next ? ` — next: ${f.next}` : ''}`)
      }
      if (!findings.length) lines.push('- no findings')
    }
  } catch {
    // Analysis unavailable (backend down or metrics.jsonl not written yet) — send the snapshot without it
  }

  lines.push('', '### Live handles')
  if (target?.startsWith('ssh:') || target === 'cloud') {
    lines.push(`- artifacts: on the ${target} host; downloaded artifacts land under ~/.bashgym/models/${run.id}/`)
  } else {
    lines.push(`- artifacts: ~/.bashgym/models/${run.id}/ (training script, metrics.jsonl, checkpoint-*/, final/)`)
  }
  lines.push(`- status: GET ${API_BASE}/training/${run.id}`)
  lines.push(`- metrics history: GET ${API_BASE}/training/runs/${run.id}/metrics`)
  lines.push(`- health analysis: GET ${API_BASE}/training/runs/${run.id}/analysis`)
  if (run.status === 'running' || run.status === 'paused' || run.status === 'starting') {
    lines.push(`- control: POST ${API_BASE}/training/${run.id}/pause | /resume | /stop`)
  }

  return lines.join('\n')
}

export const TrainingRunNode = memo(function TrainingRunNode({ data, selected }: NodeProps<TrainingRunNodeType>) {
  const workspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const [configOpen, setConfigOpen] = useState(false)
  const nodeConfig = (data.adapterConfig || {}) as TrainingNodeConfig
  const currentRun = useTrainingStore((s) => s.currentRun)
  const runs = useTrainingStore((s) => s.runs)
  const lossHistory = useTrainingStore((s) => s.lossHistory)
  const logs = useTrainingStore((s) => s.logs)
  const isConnected = useTrainingStore((s) => s.isConnected)
  const pauseTraining = useTrainingStore((s) => s.pauseTraining)
  const resumeTraining = useTrainingStore((s) => s.resumeTraining)
  const stopTraining = useTrainingStore((s) => s.stopTraining)
  const startTraining = useTrainingStore((s) => s.startTraining)
  const updatePanelConfig = useTerminalStore((s) => s.updatePanelConfig)

  const run = nodeConfig.runId
    ? (currentRun?.id === nodeConfig.runId ? currentRun : runs.find((candidate) => candidate.id === nodeConfig.runId) ?? null)
    : null
  const [draft, setDraft] = useState<TrainingDraft>(() => draftFromTraining(nodeConfig, run))
  const [launching, setLaunching] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const status = run?.status ?? nodeConfig.status
  const m = run?.currentMetrics
  const progress = m && m.totalSteps > 0 ? Math.min(100, (m.step / m.totalSteps) * 100) : 0
  const recentLogs = logs.slice(-2)
  const lossSeries = run?.metricsHistory?.length
    ? run.metricsHistory.map((point) => ({ step: point.step, loss: point.loss, evalLoss: point.evalLoss }))
    : lossHistory

  const computeTarget = computeTargetFor(run, nodeConfig)
  const visualPhase = nodeConfig.visual?.phase || (
    status === 'running' || status === 'starting'
      ? 'running'
      : status === 'pending' ? 'queued' : status
  )
  const canControlRun = Boolean(run && currentRun?.id === run.id)
  const canLaunch = !nodeConfig.runId && !run
  const observedProgress = nodeConfig.progress
  const observedProgressPct = observedProgress?.total
    ? Math.min(100, (observedProgress.current / observedProgress.total) * 100)
    : 0

  useEffect(() => {
    if (!configOpen) return
    setDraft(draftFromTraining((data.adapterConfig || {}) as TrainingNodeConfig, run))
    setConfigError(null)
  }, [configOpen, data.adapterConfig, run])

  const saveDraft = (statusOverride?: string) => {
    const next = {
      ...nodeConfig,
      ...draft,
      status: statusOverride || nodeConfig.status || 'planned',
      computeTarget: computeTargetFromDraft(draft),
    }
    updatePanelConfig(data.panelId, next)
    return next
  }

  const launchDraft = async () => {
    if (!canLaunch || launching) return
    if (!draft.baseModel.trim()) {
      setConfigError('Choose a base model before launching.')
      return
    }
    setLaunching(true)
    setConfigError(null)
    const saved = saveDraft('planned')
    try {
      const runId = await startTraining(configFromDraft(draft), {
        origin: { kind: 'panel', workspace_id: workspaceId, panel_id: data.panelId },
        correlationId: nodeConfig.correlationId,
      })
      const livePanel = useTerminalStore.getState().panels.find((panel) => panel.id === data.panelId)
      updatePanelConfig(data.panelId, {
        ...saved,
        ...(livePanel?.adapterConfig || {}),
        runId,
        status: 'queued',
      })
      setConfigOpen(false)
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : 'Unable to launch training.')
    } finally {
      setLaunching(false)
    }
  }

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      flowerVariant="training"
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={data.hasTerminalConnections
        ? () => buildTrainingContext(nodeConfig.runId, nodeConfig)
        : undefined}
      statusBarClass={status ? STATUS_BAR[status] ?? 'bg-background-tertiary' : 'bg-background-tertiary'}
      hue={hueFor('training')}
      visualPhase={visualPhase}
      headerRight={
        <>
          {isConnected
            ? <Wifi className="w-3 h-3 text-status-success flex-shrink-0" />
            : <WifiOff className="w-3 h-3 text-status-error flex-shrink-0" />}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              setConfigOpen(true)
            }}
            className="nodrag node-btn node-btn-accent"
            title="Configure training node"
          >
            <SlidersHorizontal className="w-3 h-3" />
          </button>
        </>
      }
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      {!run ? (
        nodeConfig.status || nodeConfig.strategy ? (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono">
              <span className="px-1.5 py-0.5 border-brutal border-accent text-accent bg-accent/10 rounded-brutal font-bold uppercase tracking-wider">
                {nodeConfig.status || 'planned'}
              </span>
              <span className="text-text-secondary uppercase">{nodeConfig.strategy || 'sft'}</span>
              {computeTarget && (
                <span
                  className="flex items-center gap-0.5 px-1 py-px border-brutal border-border-subtle rounded-brutal text-text-secondary"
                  title={`Compute target: ${computeTarget}`}
                >
                  <Server className="w-2.5 h-2.5" />
                  {computeTarget}
                </span>
              )}
            </div>
            {nodeConfig.summary && (
              <div className="text-[10px] font-mono text-text-secondary leading-snug">
                {nodeConfig.summary}
              </div>
            )}
            {observedProgress?.total ? (
              <div>
                <div className="mb-0.5 flex justify-between font-mono text-[9px] text-text-muted">
                  <span>{observedProgress.unit}</span>
                  <span>{observedProgress.current}/{observedProgress.total}</span>
                </div>
                <div className="h-1.5 overflow-hidden border-brutal border-border-subtle bg-background-tertiary">
                  <div className="h-full bg-accent" style={{ width: `${observedProgressPct}%` }} />
                </div>
              </div>
            ) : null}
            <div className="grid grid-cols-2 gap-1 text-[9px] font-mono text-text-muted">
              <span className="truncate" title={nodeConfig.baseModel}>model {nodeConfig.baseModel || 'TBD'}</span>
              <span className="truncate" title={nodeConfig.datasetPath}>data {nodeConfig.datasetPath || 'TBD'}</span>
              <span className="truncate" title={nodeConfig.originAgent}>origin {nodeConfig.originAgent || nodeConfig.originTerminalId || 'canvas'}</span>
              <span className="truncate" title={nodeConfig.correlationId}>intent {nodeConfig.correlationId || 'pending'}</span>
            </div>
          </div>
        ) : (
          <div className="text-[10px] font-mono text-text-muted text-center py-3">
            No active training run
            <span className="block mt-1 text-text-secondary">Waiting for run state</span>
          </div>
        )
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span className={clsx(
              'px-1.5 py-0.5 border-brutal rounded-brutal font-bold uppercase tracking-wider',
              run.status === 'running' && 'border-accent text-accent bg-accent/10',
              run.status === 'paused' && 'border-status-warning text-status-warning bg-status-warning/10',
              run.status === 'failed' && 'border-status-error text-status-error bg-status-error/10',
              run.status === 'completed' && 'border-status-success text-status-success bg-status-success/10',
              (run.status === 'starting' || run.status === 'idle') && 'border-border text-text-muted'
            )}>
              {run.status}
            </span>
            <span className="text-text-secondary uppercase">{run.config.strategy ?? nodeConfig.strategy ?? ''}</span>
            {computeTarget && (
              <span
                className="flex items-center gap-0.5 px-1 py-px border-brutal border-border-subtle rounded-brutal text-text-secondary"
                title={`Compute target: ${computeTarget}`}
              >
                <Server className="w-2.5 h-2.5" />
                {computeTarget}
              </span>
            )}
            <span className="flex-1 truncate text-text-muted text-right" title={run.id}>{run.id}</span>
          </div>

          {m && (
            <div>
              <div className="flex justify-between text-[10px] font-mono text-text-muted mb-0.5">
                <span>step {m.step}/{m.totalSteps || '?'}</span>
                <span>epoch {m.epoch.toFixed(2)}</span>
              </div>
              <div className="h-1.5 bg-background-tertiary border-brutal border-border-subtle rounded-brutal overflow-hidden">
                <div className="h-full bg-accent" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {lossSeries.length > 1 && (
            <div>
              <div className="flex justify-between text-[10px] font-mono text-text-muted">
                <span>loss</span>
                <span className="text-text-primary font-semibold">{m ? m.loss.toFixed(4) : '—'}</span>
              </div>
              <LineChart width={280} height={44} data={lossSeries.slice(-80)}>
                <YAxis hide domain={['auto', 'auto']} />
                <Line type="monotone" dataKey="loss" stroke="var(--accent)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </div>
          )}

          {m && (m.gpuUtilization != null || m.tokensPerSecond != null) && (
            <div className="flex gap-3 text-[10px] font-mono text-text-muted">
              {m.gpuUtilization != null && <span>GPU {m.gpuUtilization.toFixed(0)}%</span>}
              {m.gpuMemoryGb != null && <span>{m.gpuMemoryGb.toFixed(1)} GB</span>}
              {m.tokensPerSecond != null && <span>{m.tokensPerSecond.toFixed(0)} tok/s</span>}
            </div>
          )}

          {recentLogs.length > 0 && (
            <div className="border-t border-brutal border-border-subtle pt-1 space-y-0.5">
              {recentLogs.map((l, i) => (
                <div key={i} className="text-[9px] font-mono text-text-muted truncate" title={l.message}>
                  {l.message}
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-1.5">
            {canControlRun && run.status === 'running' && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void pauseTraining() }}
                className="nodrag node-btn node-btn-wide node-btn-warning flex-1 text-status-warning"
              >
                <Pause className="w-3 h-3" /> Pause
              </button>
            )}
            {canControlRun && run.status === 'paused' && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void resumeTraining() }}
                className="nodrag node-btn node-btn-wide node-btn-success flex-1 text-status-success"
              >
                <Play className="w-3 h-3" /> Resume
              </button>
            )}
            {canControlRun && (run.status === 'running' || run.status === 'paused') && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void stopTraining() }}
                className="nodrag node-btn node-btn-wide node-btn-danger flex-1 text-status-error"
              >
                <Square className="w-3 h-3" /> Stop
              </button>
            )}
          </div>
        </div>
      )}
    </DataNodeShell>
    <NodeConfigModal
      isOpen={configOpen}
      onClose={() => setConfigOpen(false)}
      title={`${data.title} Config`}
      description={run ? `Run ${run.id}` : 'Planned training work'}
      size="lg"
      footer={
        <div className="flex w-full items-center justify-between gap-3">
          <span className="min-w-0 font-mono text-[10px] text-status-error">
            {configError}
          </span>
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              type="button"
              className="btn-secondary flex items-center gap-2 text-xs"
              onClick={() => {
                saveDraft()
                setConfigOpen(false)
              }}
            >
              <Save className="h-3.5 w-3.5" />
              Save
            </button>
            {canLaunch ? (
              <button
                type="button"
                className="btn-primary flex items-center gap-2 text-xs"
                onClick={() => void launchDraft()}
                disabled={launching}
              >
                {launching ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                Launch Run
              </button>
            ) : null}
          </div>
        </div>
      }
    >
      <ConfigSection title="Launch Configuration">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="node-field">
            <span className="node-field-label">Method</span>
            <select
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.strategy}
              onChange={(event) => setDraft((current) => ({
                ...current,
                strategy: event.target.value as LaunchableTrainingStrategy,
              }))}
              disabled={!canLaunch}
            >
              <option value="sft">SFT</option>
              <option value="dpo">DPO</option>
              <option value="grpo">GRPO</option>
              <option value="distillation">Knowledge distillation</option>
              <option value="session_distillation">Session distillation</option>
            </select>
          </label>
          <label className="node-field">
            <span className="node-field-label">Compute</span>
            <select
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.backend}
              onChange={(event) => setDraft((current) => ({
                ...current,
                backend: event.target.value as TrainingBackend,
              }))}
              disabled={!canLaunch}
            >
              <option value="local">Local</option>
              <option value="remote">Remote device</option>
              <option value="cloud">Cloud / NeMo Gym</option>
            </select>
          </label>
          <label className="node-field md:col-span-2">
            <span className="node-field-label">Base model</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.baseModel}
              onChange={(event) => setDraft((current) => ({ ...current, baseModel: event.target.value }))}
              placeholder="Hugging Face model id or local path"
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field md:col-span-2">
            <span className="node-field-label">Dataset path</span>
            <input
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.datasetPath}
              onChange={(event) => setDraft((current) => ({ ...current, datasetPath: event.target.value }))}
              placeholder="Leave empty to generate from gold traces"
              disabled={!canLaunch}
            />
          </label>
          {draft.backend === 'remote' ? (
            <label className="node-field md:col-span-2">
              <span className="node-field-label">Remote device id</span>
              <input
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={draft.deviceId}
                onChange={(event) => setDraft((current) => ({ ...current, deviceId: event.target.value }))}
                placeholder="Configured compute device"
                disabled={!canLaunch}
              />
            </label>
          ) : null}
          <label className="node-field">
            <span className="node-field-label">Epochs</span>
            <input
              type="number"
              min={1}
              max={100}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.epochs}
              onChange={(event) => setDraft((current) => ({ ...current, epochs: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Batch size</span>
            <input
              type="number"
              min={1}
              max={64}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.batchSize}
              onChange={(event) => setDraft((current) => ({ ...current, batchSize: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Learning rate</span>
            <input
              type="number"
              min={0}
              step="0.000001"
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.learningRate}
              onChange={(event) => setDraft((current) => ({ ...current, learningRate: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Max sequence</span>
            <input
              type="number"
              min={512}
              max={32768}
              step={256}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.maxSeqLength}
              onChange={(event) => setDraft((current) => ({ ...current, maxSeqLength: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">LoRA rank</span>
            <input
              type="number"
              min={4}
              max={128}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.loraRank}
              onChange={(event) => setDraft((current) => ({ ...current, loraRank: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="node-field">
            <span className="node-field-label">Gradient accumulation</span>
            <input
              type="number"
              min={1}
              max={128}
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.gradientAccumulationSteps}
              onChange={(event) => setDraft((current) => ({ ...current, gradientAccumulationSteps: Number(event.target.value) }))}
              disabled={!canLaunch}
            />
          </label>
          <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2 md:col-span-2">
            <input
              type="checkbox"
              checked={draft.load4Bit}
              onChange={(event) => setDraft((current) => ({ ...current, load4Bit: event.target.checked }))}
              disabled={!canLaunch}
            />
            <span className="font-mono text-[11px] text-text-secondary">Use 4-bit QLoRA loading</span>
          </label>
        </div>
        {!canLaunch ? (
          <p className="font-mono text-[10px] text-text-muted">
            This node is bound to a run. Use the live controls below; add a new Training node for another launch.
          </p>
        ) : null}
      </ConfigSection>

      <ConfigSection title="Storage & Hugging Face">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <label className="node-field md:col-span-2">
            <span className="node-field-label">Post-run artifact policy</span>
            <select
              className="input-brutal min-h-9 font-mono text-[11px]"
              value={draft.artifactRetention}
              onChange={(event) => setDraft((current) => ({
                ...current,
                artifactRetention: event.target.value as ArtifactRetention,
              }))}
              disabled={!canLaunch}
            >
              <option value="adapter_only">Adapter only — recommended</option>
              <option value="adapter_checkpoint">Adapter + resumable checkpoints</option>
              <option value="deployable">Deployable merged model</option>
              <option value="full_run">Full run</option>
            </select>
          </label>
          {(draft.artifactRetention === 'adapter_checkpoint' || draft.artifactRetention === 'full_run') ? (
            <label className="node-field">
              <span className="node-field-label">Checkpoint limit</span>
              <input
                type="number"
                min={1}
                max={20}
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={draft.checkpointLimit}
                onChange={(event) => setDraft((current) => ({ ...current, checkpointLimit: Number(event.target.value) }))}
                disabled={!canLaunch}
              />
            </label>
          ) : null}
          {(draft.artifactRetention === 'deployable' || draft.artifactRetention === 'full_run') ? (
            <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2">
              <input
                type="checkbox"
                checked={draft.autoExportGGUF}
                onChange={(event) => setDraft((current) => ({ ...current, autoExportGGUF: event.target.checked }))}
                disabled={!canLaunch}
              />
              <span className="font-mono text-[11px] text-text-secondary">Also export GGUF</span>
            </label>
          ) : null}
          <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2 md:col-span-2">
            <input
              type="checkbox"
              checked={draft.autoPushHF}
              onChange={(event) => setDraft((current) => ({ ...current, autoPushHF: event.target.checked }))}
              disabled={!canLaunch}
            />
            <span className="font-mono text-[11px] text-text-secondary">Upload to Hugging Face after training</span>
          </label>
          {draft.autoPushHF ? (
            <>
              <label className="node-field">
                <span className="node-field-label">HF repository</span>
                <input
                  className="input-brutal min-h-9 font-mono text-[11px]"
                  value={draft.hfRepoName}
                  onChange={(event) => setDraft((current) => ({ ...current, hfRepoName: event.target.value }))}
                  placeholder="Auto-generated when empty"
                  disabled={!canLaunch}
                />
              </label>
              <label className="node-field">
                <span className="node-field-label">Upload artifact</span>
                <select
                  className="input-brutal min-h-9 font-mono text-[11px]"
                  value={draft.hfUploadArtifact}
                  onChange={(event) => setDraft((current) => ({
                    ...current,
                    hfUploadArtifact: event.target.value as HFUploadArtifact,
                  }))}
                  disabled={!canLaunch}
                >
                  <option value="auto">Auto</option>
                  <option value="adapter">Adapter</option>
                  <option value="merged">Merged model</option>
                </select>
              </label>
              <label className="flex items-center gap-2 border-brutal border-border-subtle bg-background-card px-3 py-2 md:col-span-2">
                <input
                  type="checkbox"
                  checked={draft.hfPrivate}
                  onChange={(event) => setDraft((current) => ({ ...current, hfPrivate: event.target.checked }))}
                  disabled={!canLaunch}
                />
                <span className="font-mono text-[11px] text-text-secondary">
                  Private repository{draft.hfPrivate ? '' : ' — public release requires license and data review'}
                </span>
              </label>
            </>
          ) : null}
          <p className="font-mono text-[10px] text-text-muted md:col-span-2">
            The base model cache is shared. Merged weights and GGUF exports are new model-sized copies.
          </p>
        </div>
      </ConfigSection>

      <ConfigSection title="Run State">
        <div className="flex flex-wrap gap-1.5">
          <ConfigPill tone={statusTone(status)}>{status || 'idle'}</ConfigPill>
          <ConfigPill tone="neutral">{run?.config.strategy || nodeConfig.strategy || 'sft'}</ConfigPill>
          {computeTarget ? <ConfigPill tone="accent">{computeTarget}</ConfigPill> : null}
        </div>
        <ConfigRows>
          <ConfigRow label="Run id" value={run?.id || nodeConfig.runId} />
          <ConfigRow label="Base model" value={run?.config.baseModel || nodeConfig.baseModel} />
          <ConfigRow label="Dataset" value={run?.config.datasetPath || nodeConfig.datasetPath} />
          <ConfigRow label="Origin" value={nodeConfig.originAgent || nodeConfig.originTerminalId || nodeConfig.originPanelId || 'canvas'} />
          <ConfigRow label="Correlation" value={nodeConfig.correlationId} />
          <ConfigRow label="Observed PID" value={nodeConfig.runtimePid} />
          <ConfigRow label="Runtime source" value={nodeConfig.runtimeDiscovered ? 'workspace process observer' : undefined} />
          <ConfigRow
            label="Started"
            value={run ? new Date(run.startTime).toLocaleString() : undefined}
          />
          <ConfigRow
            label="Progress"
            value={m ? `${m.step}/${m.totalSteps || '?'} steps (${progress.toFixed(1)}%)` : undefined}
          />
        </ConfigRows>
      </ConfigSection>

      <ConfigSection title="Training Config">
        <ConfigRows>
          <ConfigRow label="Epochs" value={run?.config.epochs ?? nodeConfig.epochs} />
          <ConfigRow label="Batch size" value={run?.config.batchSize ?? nodeConfig.batchSize} />
          <ConfigRow label="Learning rate" value={run?.config.learningRate ?? nodeConfig.learningRate} />
          <ConfigRow label="Max sequence" value={run?.config.maxSeqLength ?? nodeConfig.maxSeqLength} />
          <ConfigRow label="LoRA rank" value={run?.config.loraRank ?? nodeConfig.loraRank} />
          <ConfigRow label="Summary" value={nodeConfig.summary} />
        </ConfigRows>
      </ConfigSection>

      <ConfigSection title="Live Handles">
        <ConfigRows>
          <ConfigRow label="Status API" value={run ? `${API_BASE}/training/${run.id}` : `${API_BASE}/training/runs`} />
          <ConfigRow label="Metrics API" value={run ? `${API_BASE}/training/runs/${run.id}/metrics` : undefined} />
          <ConfigRow label="Analysis API" value={run ? `${API_BASE}/training/runs/${run.id}/analysis` : undefined} />
          <ConfigRow label="Workspace API" value={`${API_BASE}/workspace/context?format=json`} />
        </ConfigRows>
      </ConfigSection>
    </NodeConfigModal>
    </>
  )
})
