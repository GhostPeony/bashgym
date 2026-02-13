import { useState, useCallback, useEffect } from 'react'
import {
  Play,
  Pause,
  Square,
  Download,
  Settings,
  RefreshCw,
  Clock,
  Rocket,
  FileStack,
  FolderGit2,
  ArrowRight,
  CheckCircle2,
  BarChart3
} from 'lucide-react'
import { useTrainingStore } from '../../stores'
import { trainingApi, tracesApi, RepoInfo, SystemInfo, ModelRecommendations } from '../../services/api'
import { LossCurve } from './LossCurve'
import { EpochProgress } from './EpochProgress'
import { MetricsGrid } from './MetricsGrid'
import { TrainingConfig } from './TrainingConfig'
import { SystemInfoPanel } from './SystemInfoPanel'
import { TrainingLogs } from './TrainingLogs'
import { clsx } from 'clsx'

export function TrainingDashboard() {
  const { currentRun, lossHistory, startTraining, pauseTraining, resumeTraining, stopTraining, updateMetrics } =
    useTrainingStore()
  const [showConfig, setShowConfig] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [availableRepos, setAvailableRepos] = useState<RepoInfo[]>([])
  const [goldTraceCount, setGoldTraceCount] = useState(0)
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [recommendations, setRecommendations] = useState<ModelRecommendations | null>(null)

  const isRunning = currentRun?.status === 'running'
  const isPaused = currentRun?.status === 'paused'
  const metrics = currentRun?.currentMetrics

  // Fetch repos and trace count on mount
  useEffect(() => {
    tracesApi.listRepos().then((result) => {
      if (result.ok && result.data) {
        setAvailableRepos(result.data)
        const totalTraces = result.data.reduce((sum, r) => sum + (r.trace_count || 0), 0)
        setGoldTraceCount(totalTraces)
      }
    })
  }, [])

  const handleRefreshLossData = useCallback(async () => {
    if (!currentRun || isRefreshing) return

    setIsRefreshing(true)
    try {
      const result = await trainingApi.getStatus(currentRun.id)
      if (result.ok && result.data?.metrics) {
        const apiMetrics = result.data.metrics
        updateMetrics({
          loss: apiMetrics.loss ?? 0,
          learningRate: apiMetrics.learning_rate ?? 0,
          gradNorm: apiMetrics.grad_norm ?? 0,
          epoch: apiMetrics.epoch ?? 0,
          step: apiMetrics.step ?? 0,
          totalSteps: apiMetrics.total_steps ?? 0,
          eta: apiMetrics.eta,
          timestamp: Date.now()
        })
      }
    } finally {
      setIsRefreshing(false)
    }
  }, [currentRun, isRefreshing, updateMetrics])


  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-brand text-2xl text-text-primary">Training Monitor</h1>
            <span className="tag"><span>TRAINING</span></span>
            {metrics?.simulation && (
              <span className="tag"><span>SIMULATION</span></span>
            )}
          </div>
          <p className="text-sm text-text-secondary mt-1">
            {currentRun
              ? `Run: ${currentRun.id} - ${currentRun.config.strategy.toUpperCase()}`
              : 'No active training run'}
            {metrics?.simulation && ' (No GPU/trainer available)'}
          </p>
        </div>

        <div className="flex items-center gap-2">
          {!currentRun ? (
            <button
              onClick={() => setShowConfig(true)}
              className="btn-primary flex items-center gap-2"
              data-tutorial="start-training-button"
            >
              <Play className="w-4 h-4" />
              Start Training
            </button>
          ) : (
            <>
              {isRunning ? (
                <button
                  onClick={pauseTraining}
                  className="btn-icon flex items-center justify-center"
                  title="Pause"
                >
                  <Pause className="w-4 h-4" />
                </button>
              ) : isPaused ? (
                <button
                  onClick={resumeTraining}
                  className="btn-icon flex items-center justify-center"
                  title="Resume"
                >
                  <Play className="w-4 h-4" />
                </button>
              ) : null}
              <button
                onClick={stopTraining}
                className="btn-icon flex items-center justify-center text-status-error"
                title="Stop"
              >
                <Square className="w-4 h-4" />
              </button>
            </>
          )}
          <button
            className="btn-icon flex items-center justify-center"
            title="Settings"
            onClick={() => setShowConfig(true)}
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="section-divider mb-6" />

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-4">
        {/* Loss Curve - Main Chart */}
        <div className="col-span-8 card p-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-brand text-xl text-text-primary">Loss Curve</h2>
            {lossHistory.length > 0 && (
              <button
                className="btn-icon flex items-center justify-center"
                onClick={handleRefreshLossData}
                disabled={isRefreshing}
                title="Refresh data"
              >
                <RefreshCw className={clsx('w-4 h-4', isRefreshing && 'animate-spin')} />
              </button>
            )}
          </div>
          <div className="h-64">
            {lossHistory.length > 0 ? (
              <LossCurve data={lossHistory} />
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-text-muted">
                <BarChart3 className="w-12 h-12 mb-3 opacity-30" />
                <p className="font-mono text-xs uppercase tracking-widest">Start training to see loss curve</p>
              </div>
            )}
          </div>
        </div>

        {/* Epoch Progress */}
        <div className="col-span-4 card p-4">
          <h2 className="font-brand text-xl text-text-primary mb-4">Progress</h2>
          <EpochProgress
            currentEpoch={metrics?.epoch ?? 0}
            totalEpochs={currentRun?.config.epochs ?? 1}
            currentStep={metrics?.step ?? 0}
            totalSteps={metrics?.totalSteps ?? 0}
          />
        </div>

        {/* Metrics Grid */}
        <div className="col-span-12">
          <MetricsGrid
            loss={metrics?.loss}
            learningRate={metrics?.learningRate ?? currentRun?.config.learningRate}
            gradNorm={metrics?.gradNorm}
            eta={metrics?.eta}
            isWaiting={!isRunning && !isPaused}
          />
        </div>

        {/* Training Logs */}
        {currentRun && (
          <div className="col-span-12">
            <TrainingLogs maxHeight={250} defaultExpanded={true} />
          </div>
        )}

        {/* Section divider before config/system panels */}
        {currentRun && (
          <div className="col-span-12 section-divider my-2" />
        )}

        {/* Training Config Card */}
        {currentRun && (
          <div className="col-span-6 card p-4">
            <h2 className="font-brand text-xl text-text-primary mb-4">Configuration</h2>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="card p-4">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Strategy</span>
                <p className="font-brand text-2xl text-text-primary uppercase mt-1">
                  {currentRun.config.strategy}
                </p>
              </div>
              <div className="card p-4">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Base Model</span>
                <p className="font-brand text-2xl text-text-primary mt-1 truncate">{currentRun.config.baseModel}</p>
              </div>
              <div className="card p-4">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Batch Size</span>
                <p className="font-brand text-2xl text-text-primary mt-1">{currentRun.config.batchSize}</p>
              </div>
              <div className="card p-4">
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Learning Rate</span>
                <p className="font-brand text-2xl text-text-primary mt-1">{currentRun.config.learningRate}</p>
              </div>
              {currentRun.config.loraRank && (
                <div className="card p-4">
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">LoRA Rank</span>
                  <p className="font-brand text-2xl text-text-primary mt-1">{currentRun.config.loraRank}</p>
                </div>
              )}
              {currentRun.config.loraAlpha && (
                <div className="card p-4">
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">LoRA Alpha</span>
                  <p className="font-brand text-2xl text-text-primary mt-1">{currentRun.config.loraAlpha}</p>
                </div>
              )}
              {currentRun.config.selectedRepos && currentRun.config.selectedRepos.length > 0 && (
                <div className="col-span-2 card p-4">
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Training Repos</span>
                  <p className="font-brand text-lg text-text-primary mt-1">
                    {currentRun.config.selectedRepos.join(', ')}
                  </p>
                </div>
              )}
              {(!currentRun.config.selectedRepos || currentRun.config.selectedRepos.length === 0) && (
                <div className="col-span-2 card p-4">
                  <span className="font-mono text-xs uppercase tracking-widest text-text-muted">Training Repos</span>
                  <p className="font-brand text-lg text-text-primary mt-1">All repos (Generalist)</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* System Stats */}
        <div className={clsx('card p-4', currentRun ? 'col-span-6' : 'col-span-6')}>
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-brand text-xl text-text-primary">System Resources</h2>
            {currentRun && (
              <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-text-muted">
                <Clock className="w-4 h-4" />
                <span>Runtime: {Math.floor((Date.now() - currentRun.startTime) / 60000)}m</span>
              </div>
            )}
          </div>
          <SystemInfoPanel
            compact={!!currentRun}
            onSystemInfo={setSystemInfo}
            onRecommendations={setRecommendations}
          />
        </div>

        {/* Getting Started Section - Show when no training */}
        {!currentRun && (
          <div className="col-span-6 card p-6">
            <div className="flex items-start gap-6">
              {/* Left - Info */}
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-3">
                  <Rocket className="w-6 h-6 text-primary" />
                  <h2 className="font-brand text-xl text-text-primary">Ready to Train</h2>
                </div>
                <p className="text-text-secondary mb-4">
                  Train a custom model on your coding traces. Choose between a generalist agent
                  that knows all your projects, or a specialist focused on a single codebase.
                </p>

                {/* Steps */}
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className={clsx(
                      'w-8 h-8 border-brutal border-border rounded-brutal flex items-center justify-center font-mono text-xs font-bold',
                      goldTraceCount > 0 ? 'bg-status-success text-white' : 'bg-background-tertiary text-text-muted'
                    )}>
                      {goldTraceCount > 0 ? <CheckCircle2 className="w-4 h-4" /> : '1'}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">Collect Traces</p>
                      <p className="font-mono text-xs text-text-muted">
                        {goldTraceCount > 0
                          ? `${goldTraceCount} gold traces from ${availableRepos.length} repo${availableRepos.length !== 1 ? 's' : ''}`
                          : 'Work with Claude Code to automatically capture traces'
                        }
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 border-brutal border-border rounded-brutal bg-background-tertiary text-text-muted flex items-center justify-center font-mono text-xs font-bold">
                      2
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">Choose Scope</p>
                      <p className="font-mono text-xs text-text-muted">Generalist (all repos), Mixed (select repos), or Specialist (one repo)</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 border-brutal border-border rounded-brutal bg-background-tertiary text-text-muted flex items-center justify-center font-mono text-xs font-bold">
                      3
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">Start Training</p>
                      <p className="font-mono text-xs text-text-muted">Configure strategy (SFT, DPO, GRPO) and hyperparameters</p>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setShowConfig(true)}
                  disabled={goldTraceCount === 0}
                  className={clsx(
                    'mt-6 flex items-center gap-2',
                    goldTraceCount > 0 ? 'btn-primary' : 'btn-secondary opacity-50 cursor-not-allowed'
                  )}
                >
                  <Play className="w-4 h-4" />
                  Configure Training
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>

              {/* Right - Repo Summary */}
              <div className="w-64 card p-4">
                <div className="flex items-center gap-2 mb-3">
                  <FolderGit2 className="w-4 h-4 text-text-muted" />
                  <span className="font-mono text-xs uppercase tracking-widest text-text-primary">Available Data</span>
                </div>

                <div className="border-t border-border mb-3" />

                {availableRepos.length === 0 ? (
                  <div className="text-center py-4">
                    <FileStack className="w-8 h-8 text-text-muted mx-auto mb-2" />
                    <p className="font-mono text-xs uppercase tracking-widest text-text-muted">No traces yet</p>
                    <p className="text-xs text-text-muted mt-1">Start coding to collect data</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {availableRepos.slice(0, 5).map((repo) => (
                      <div key={repo.name} className="flex items-center justify-between py-1 border-b border-border-subtle">
                        <span className="text-sm text-text-secondary truncate">{repo.name}</span>
                        <span className="font-mono text-xs text-text-muted">{repo.trace_count}</span>
                      </div>
                    ))}
                    {availableRepos.length > 5 && (
                      <p className="font-mono text-xs text-text-muted pt-1">+{availableRepos.length - 5} more</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Export Section */}
        {!currentRun && goldTraceCount > 0 && (
          <div className="col-span-12 card p-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="font-brand text-xl text-text-primary">Export Model</h2>
                <p className="font-mono text-xs uppercase tracking-widest text-text-muted mt-1">
                  Export your trained model in various formats
                </p>
              </div>
              <button className="btn-secondary flex items-center gap-2">
                <Download className="w-4 h-4" />
                Export
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Training Config Modal */}
      {showConfig && (
        <TrainingConfig
          onClose={() => setShowConfig(false)}
          onStart={(config) => {
            startTraining(config)
            setShowConfig(false)
          }}
        />
      )}
    </div>
  )
}
