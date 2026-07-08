import { memo } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import { Dumbbell, Pause, Play, Server, Square, Wifi, WifiOff } from 'lucide-react'
import { LineChart, Line, YAxis } from 'recharts'
import { clsx } from 'clsx'
import { useTrainingStore } from '../../../stores'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import type { DataNodeData } from './types'

export type TrainingRunNodeType = Node<DataNodeData, 'training'>

const STATUS_BAR: Record<string, string> = {
  running: 'bg-accent animate-pulse',
  starting: 'bg-accent animate-pulse',
  paused: 'bg-status-warning',
  failed: 'bg-status-error',
  completed: 'bg-status-success'
}

export const TrainingRunNode = memo(function TrainingRunNode({ data, selected }: NodeProps<TrainingRunNodeType>) {
  const currentRun = useTrainingStore((s) => s.currentRun)
  const lossHistory = useTrainingStore((s) => s.lossHistory)
  const logs = useTrainingStore((s) => s.logs)
  const isConnected = useTrainingStore((s) => s.isConnected)
  const pauseTraining = useTrainingStore((s) => s.pauseTraining)
  const resumeTraining = useTrainingStore((s) => s.resumeTraining)
  const stopTraining = useTrainingStore((s) => s.stopTraining)

  const m = currentRun?.currentMetrics
  const progress = m && m.totalSteps > 0 ? Math.min(100, (m.step / m.totalSteps) * 100) : 0
  const recentLogs = logs.slice(-2)

  // Where the run executes: prefer the live payload, fall back to the launch config
  const computeTarget = m?.computeTarget ?? (
    currentRun?.config?.useRemoteSSH
      ? (currentRun.config.deviceId ? `ssh:${currentRun.config.deviceId}` : 'ssh:remote')
      : currentRun?.config?.useNemoGym
        ? 'cloud'
        : currentRun?.config?.strategy ? 'local' : undefined
  )

  const buildContext = () => {
    if (!currentRun) return '## Training status\n\nNo active training run.'
    return [
      `## Training run ${currentRun.id}`,
      `- status: ${currentRun.status}`,
      `- strategy: ${currentRun.config.strategy ?? 'unknown'}`,
      `- base model: ${currentRun.config.baseModel ?? 'unknown'}`,
      computeTarget ? `- compute target: ${computeTarget}` : '',
      m ? `- step: ${m.step}/${m.totalSteps} (epoch ${m.epoch.toFixed(2)})` : '',
      m ? `- loss: ${m.loss}` : '',
      '- recent logs:',
      ...logs.slice(-5).map((l) => `  - ${l.message}`)
    ].filter(Boolean).join('\n')
  }

  return (
    <DataNodeShell
      panelId={data.panelId}
      title={data.title}
      icon={Dumbbell}
      selected={selected}
      hasConnections={data.hasConnections}
      buildContext={buildContext}
      statusBarClass={currentRun ? STATUS_BAR[currentRun.status] ?? 'bg-background-tertiary' : 'bg-background-tertiary'}
      hue={hueFor('training')}
      headerRight={
        isConnected
          ? <Wifi className="w-3 h-3 text-status-success flex-shrink-0" />
          : <WifiOff className="w-3 h-3 text-status-error flex-shrink-0" />
      }
      onFocus={data.onFocus}
      onClose={data.onClose}
    >
      {!currentRun ? (
        <div className="text-[10px] font-mono text-text-muted text-center py-3">
          No active training run
          <span className="block mt-1 text-text-secondary">Start one from the Training dashboard</span>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center gap-1.5 text-[10px] font-mono">
            <span className={clsx(
              'px-1.5 py-0.5 border-brutal rounded-brutal font-bold uppercase tracking-wider',
              currentRun.status === 'running' && 'border-accent text-accent bg-accent/10',
              currentRun.status === 'paused' && 'border-status-warning text-status-warning bg-status-warning/10',
              currentRun.status === 'failed' && 'border-status-error text-status-error bg-status-error/10',
              currentRun.status === 'completed' && 'border-status-success text-status-success bg-status-success/10',
              (currentRun.status === 'starting' || currentRun.status === 'idle') && 'border-border text-text-muted'
            )}>
              {currentRun.status}
            </span>
            <span className="text-text-secondary uppercase">{currentRun.config.strategy ?? ''}</span>
            {computeTarget && (
              <span
                className="flex items-center gap-0.5 px-1 py-px border-brutal border-border-subtle rounded-brutal text-text-secondary"
                title={`Compute target: ${computeTarget}`}
              >
                <Server className="w-2.5 h-2.5" />
                {computeTarget}
              </span>
            )}
            <span className="flex-1 truncate text-text-muted text-right" title={currentRun.id}>{currentRun.id}</span>
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

          {lossHistory.length > 1 && (
            <div>
              <div className="flex justify-between text-[10px] font-mono text-text-muted">
                <span>loss</span>
                <span className="text-text-primary font-semibold">{m ? m.loss.toFixed(4) : '—'}</span>
              </div>
              <LineChart width={280} height={44} data={lossHistory.slice(-80)}>
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
            {currentRun.status === 'running' && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void pauseTraining() }}
                className="btn-secondary !py-1 !px-2 !text-[10px] flex-1"
              >
                <Pause className="w-3 h-3" /> Pause
              </button>
            )}
            {currentRun.status === 'paused' && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void resumeTraining() }}
                className="btn-secondary !py-1 !px-2 !text-[10px] flex-1"
              >
                <Play className="w-3 h-3" /> Resume
              </button>
            )}
            {(currentRun.status === 'running' || currentRun.status === 'paused') && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); void stopTraining() }}
                className="btn-secondary !py-1 !px-2 !text-[10px] flex-1 hover:!text-status-error"
              >
                <Square className="w-3 h-3" /> Stop
              </button>
            )}
          </div>
        </div>
      )}
    </DataNodeShell>
  )
})
