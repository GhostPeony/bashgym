import { memo, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Dumbbell, Pause, Play, Server, SlidersHorizontal, Square, Wifi, WifiOff } from 'lucide-react'
import { LineChart, Line, YAxis } from 'recharts'
import { clsx } from 'clsx'
import { useTrainingStore } from '../../../stores'
import type { TrainingRun } from '../../../stores/trainingStore'
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
  visual?: {
    phase?: string
    motion?: string
  }
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

  const run = nodeConfig.runId
    ? (currentRun?.id === nodeConfig.runId ? currentRun : runs.find((candidate) => candidate.id === nodeConfig.runId) ?? null)
    : currentRun
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

  return (
    <>
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Dumbbell}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={() => buildTrainingContext(nodeConfig.runId, nodeConfig)}
      statusBarClass={status ? STATUS_BAR[status] ?? 'bg-background-tertiary' : 'bg-background-tertiary'}
      hue={hueFor('training')}
      visualPhase={visualPhase}
      motion={nodeConfig.visual?.motion}
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
    >
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
          <ConfigRow label="Epochs" value={run?.config.epochs} />
          <ConfigRow label="Batch size" value={run?.config.batchSize} />
          <ConfigRow label="Learning rate" value={run?.config.learningRate} />
          <ConfigRow label="Max sequence" value={run?.config.maxSeqLength} />
          <ConfigRow label="LoRA rank" value={run?.config.loraRank} />
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
