import { useState, useEffect } from 'react'
import {
  Link2,
  Link2Off,
  Settings,
  Server,
  Database,
  Cpu,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Folder,
  Play,
  Square,
  Package,
  Clock,
  AlertCircle,
  History,
  RotateCcw
} from 'lucide-react'
import { integrationApi, IntegrationStatus, IntegrationSettings, ModelVersion, PendingTrace } from '../../services/api'
import { clsx } from 'clsx'

type Tab = 'status' | 'settings' | 'models' | 'traces'

interface StatusCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  status?: 'success' | 'warning' | 'error' | 'neutral'
}

function StatusCard({ title, value, icon, status = 'neutral' }: StatusCardProps) {
  const statusColors = {
    success: 'text-status-success',
    warning: 'text-status-warning',
    error: 'text-status-error',
    neutral: 'text-text-secondary',
  }

  return (
    <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="font-mono text-xs text-text-muted uppercase tracking-widest">{title}</p>
          <p className={clsx('text-2xl font-brand mt-1', statusColors[status])}>{value}</p>
        </div>
        <div className={clsx('p-2 border-2 border-border rounded-brutal bg-background-secondary', statusColors[status])}>
          {icon}
        </div>
      </div>
    </div>
  )
}

export function IntegrationDashboard() {
  const [status, setStatus] = useState<IntegrationStatus | null>(null)
  const [settings, setSettings] = useState<IntegrationSettings | null>(null)
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([])
  const [pendingTraces, setPendingTraces] = useState<PendingTrace[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<Tab>('status')
  const [linking, setLinking] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [watching, setWatching] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const [statusRes, settingsRes, modelsRes, tracesRes] = await Promise.all([
        integrationApi.getStatus(),
        integrationApi.getSettings(),
        integrationApi.listModelVersions(),
        integrationApi.listPendingTraces(),
      ])

      if (statusRes.ok && statusRes.data) setStatus(statusRes.data)
      if (settingsRes.ok && settingsRes.data) setSettings(settingsRes.data)
      if (modelsRes.ok && modelsRes.data) setModelVersions(modelsRes.data)
      if (tracesRes.ok && tracesRes.data) setPendingTraces(tracesRes.data)
    } catch (error) {
      console.error('Failed to fetch integration data:', error)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchData()
    // Poll for updates every 30 seconds
    const interval = setInterval(fetchData, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleLink = async () => {
    setLinking(true)
    const result = await integrationApi.link()
    if (result.ok) {
      await fetchData()
    }
    setLinking(false)
  }

  const handleUnlink = async () => {
    if (!confirm('Unlink bashbros integration? Traces will stop flowing and model sync will be disabled.')) return
    setLinking(true)
    const result = await integrationApi.unlink()
    if (result.ok) {
      await fetchData()
    }
    setLinking(false)
  }

  const handleProcessTraces = async () => {
    setProcessing(true)
    await integrationApi.processTraces()
    // Wait a moment then refresh
    setTimeout(fetchData, 2000)
    setProcessing(false)
  }

  const handleStartWatcher = async () => {
    const result = await integrationApi.startWatcher()
    if (result.ok) {
      setWatching(true)
    }
  }

  const handleStopWatcher = async () => {
    const result = await integrationApi.stopWatcher()
    if (result.ok) {
      setWatching(false)
    }
  }

  const handleUpdateSetting = async (category: string, key: string, value: any) => {
    const updates = { [category]: { [key]: value } }
    const result = await integrationApi.updateSettings(updates)
    if (result.ok && result.data) {
      setSettings(result.data)
    }
  }

  const handleRollback = async (version: string) => {
    if (!confirm(`Rollback to model version ${version}? This will update the sidekick model.`)) return
    const result = await integrationApi.rollbackModel(version)
    if (result.ok) {
      await fetchData()
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-accent" />
      </div>
    )
  }

  const isLinked = status?.linked ?? false

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className={clsx(
              'p-3 border-2 rounded-brutal',
              isLinked ? 'border-status-success bg-background-card' : 'border-border bg-background-secondary'
            )}>
              {isLinked ? (
                <Link2 className="w-6 h-6 text-status-success" />
              ) : (
                <Link2Off className="w-6 h-6 text-text-secondary" />
              )}
            </div>
            <div>
              <h1 className="text-2xl font-brand text-text-primary">Bashbros Integration</h1>
              <p className="text-text-secondary font-mono text-sm">
                {isLinked ? 'Connected - traces flowing, model sync enabled' : 'Not linked - standalone mode'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchData}
              className="btn-icon"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            {isLinked ? (
              <button
                onClick={handleUnlink}
                disabled={linking}
                className="btn-secondary text-status-error border-status-error hover:bg-background-secondary disabled:opacity-50"
              >
                {linking ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unlink'}
              </button>
            ) : (
              <button
                onClick={handleLink}
                disabled={linking}
                className="btn-secondary text-status-success border-status-success hover:bg-background-secondary disabled:opacity-50"
              >
                {linking ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Link Integration'}
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6">
          {(['status', 'settings', 'models', 'traces'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                'px-4 py-2 text-sm font-mono border-2 rounded-brutal capitalize transition-press',
                activeTab === tab
                  ? 'bg-accent text-white border-accent-dark shadow-brutal-sm'
                  : 'text-text-secondary border-border hover:text-text-primary hover:bg-background-secondary hover-press'
              )}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Status Tab */}
        {activeTab === 'status' && (
          <div className="space-y-6">
            {/* Status Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatusCard
                title="Bashbros"
                value={status?.bashbros_connected ? 'Connected' : 'Offline'}
                icon={<Server className="w-5 h-5" />}
                status={status?.bashbros_connected ? 'success' : 'error'}
              />
              <StatusCard
                title="Pending Traces"
                value={status?.pending_traces ?? 0}
                icon={<Database className="w-5 h-5" />}
                status={(status?.pending_traces ?? 0) > 0 ? 'warning' : 'neutral'}
              />
              <StatusCard
                title="Processed"
                value={status?.processed_traces ?? 0}
                icon={<CheckCircle2 className="w-5 h-5" />}
                status="success"
              />
              <StatusCard
                title="Model Version"
                value={status?.current_model_version ?? 'None'}
                icon={<Package className="w-5 h-5" />}
                status={status?.current_model_version ? 'success' : 'neutral'}
              />
            </div>

            {/* Training Status */}
            {status?.training_in_progress && (
              <div className="p-4 border-2 border-status-warning rounded-brutal bg-background-card flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-status-warning" />
                <div>
                  <p className="font-brand text-status-warning">Training in Progress</p>
                  <p className="text-sm text-text-secondary font-mono">A new model is being trained...</p>
                </div>
              </div>
            )}

            {/* Trace Watcher */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-brand text-text-primary">Trace Watcher</h3>
                  <p className="text-sm text-text-secondary font-mono">
                    Automatically process traces from bashbros
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {!watching ? (
                    <button
                      onClick={handleStartWatcher}
                      className="btn-secondary flex items-center gap-2 text-status-success border-status-success text-sm"
                    >
                      <Play className="w-4 h-4" />
                      Start
                    </button>
                  ) : (
                    <button
                      onClick={handleStopWatcher}
                      className="btn-secondary flex items-center gap-2 text-status-error border-status-error text-sm"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  )}
                  <button
                    onClick={handleProcessTraces}
                    disabled={processing}
                    className="btn-secondary flex items-center gap-2 text-accent border-accent text-sm disabled:opacity-50"
                  >
                    {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                    Process Now
                  </button>
                </div>
              </div>
            </div>

            {/* Integration Directory */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <div className="flex items-center gap-2 mb-3">
                <Folder className="w-4 h-4 text-text-secondary" />
                <h3 className="font-brand text-text-primary">Integration Directory</h3>
              </div>
              <div className="terminal-chrome">
                <div className="terminal-header">
                  <div className="terminal-dot-red" />
                  <div className="terminal-dot-yellow" />
                  <div className="terminal-dot-green" />
                </div>
                <code className="block text-sm font-mono text-text-secondary px-3 py-2">
                  ~/.bashgym/integration/
                </code>
              </div>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && settings && (
          <div className="space-y-6">
            {/* Capture Settings */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <h3 className="font-brand text-text-primary mb-4 flex items-center gap-2">
                <Database className="w-4 h-4" />
                Capture Settings
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-2 uppercase tracking-widest">Capture Mode</label>
                  <select
                    value={settings.capture_mode}
                    onChange={(e) => handleUpdateSetting('capture', 'mode', e.target.value)}
                    className="input w-full text-sm"
                  >
                    <option value="everything">Everything - Capture all sessions</option>
                    <option value="successful_only">Successful Only - Only verified traces</option>
                    <option value="sidekick_curated">Sidekick Curated - AI picks teachable moments</option>
                  </select>
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.auto_stream}
                    onChange={(e) => handleUpdateSetting('capture', 'auto_stream', e.target.checked)}
                    className="w-4 h-4 border-brutal rounded-brutal"
                  />
                  <span className="text-text-primary font-mono text-sm">Auto-stream traces</span>
                </label>
              </div>
            </div>

            {/* Training Settings */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <h3 className="font-brand text-text-primary mb-4 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Training Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.auto_training_enabled}
                    onChange={(e) => handleUpdateSetting('training', 'auto_enabled', e.target.checked)}
                    className="w-4 h-4 border-brutal rounded-brutal"
                  />
                  <span className="text-text-primary font-mono text-sm">Enable auto-training</span>
                </label>

                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-2 uppercase tracking-widest">
                    Quality Threshold (gold traces needed)
                  </label>
                  <input
                    type="number"
                    value={settings.quality_threshold}
                    onChange={(e) => handleUpdateSetting('training', 'quality_threshold', parseInt(e.target.value))}
                    min={10}
                    max={500}
                    className="input w-32 text-sm"
                  />
                </div>

                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-2 uppercase tracking-widest">Training Trigger</label>
                  <select
                    value={settings.trigger}
                    onChange={(e) => handleUpdateSetting('training', 'trigger', e.target.value)}
                    className="input w-full text-sm"
                  >
                    <option value="manual">Manual - Only on request</option>
                    <option value="quality_based">Quality Based - When threshold reached</option>
                    <option value="scheduled">Scheduled - Regular intervals</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Model Sync Settings */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <h3 className="font-brand text-text-primary mb-4 flex items-center gap-2">
                <Package className="w-4 h-4" />
                Model Sync Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.auto_export_ollama}
                    onChange={(e) => handleUpdateSetting('model_sync', 'auto_export_ollama', e.target.checked)}
                    className="w-4 h-4 border-brutal rounded-brutal"
                  />
                  <span className="text-text-primary font-mono text-sm">Auto-export to Ollama</span>
                </label>

                <div>
                  <label className="block text-xs font-mono text-text-secondary mb-2 uppercase tracking-widest">Ollama Model Name</label>
                  <input
                    type="text"
                    value={settings.ollama_model_name}
                    onChange={(e) => handleUpdateSetting('model_sync', 'ollama_model_name', e.target.value)}
                    className="input w-64 text-sm font-mono"
                  />
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.notify_on_update}
                    onChange={(e) => handleUpdateSetting('model_sync', 'notify_on_update', e.target.checked)}
                    className="w-4 h-4 border-brutal rounded-brutal"
                  />
                  <span className="text-text-primary font-mono text-sm">Notify on model update</span>
                </label>
              </div>
            </div>

            {/* Security Settings */}
            <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
              <h3 className="font-brand text-text-primary mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Security Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.bashbros_primary}
                    onChange={(e) => handleUpdateSetting('security', 'bashbros_primary', e.target.checked)}
                    className="w-4 h-4 border-brutal rounded-brutal"
                  />
                  <div>
                    <span className="text-text-primary font-mono text-sm">Bashbros as primary security</span>
                    <p className="text-sm text-text-secondary font-mono">
                      Defer security checks to bashbros when linked
                    </p>
                  </div>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="space-y-4">
            {modelVersions.length === 0 ? (
              <div className="text-center py-12 text-text-secondary">
                <Package className="w-12 h-12 mx-auto text-text-muted mb-4" />
                <p className="font-brand text-lg">No model versions available</p>
                <p className="text-sm font-mono">Train your first model to see versions here</p>
              </div>
            ) : (
              modelVersions.map((model) => (
                <div
                  key={model.version}
                  className={clsx(
                    'border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4',
                    model.is_latest && 'border-status-success'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={clsx(
                        'p-2 border-2 rounded-brutal',
                        model.is_latest ? 'border-status-success bg-background-card' : 'border-border bg-background-secondary'
                      )}>
                        <Package className={clsx(
                          'w-5 h-5',
                          model.is_latest ? 'text-status-success' : 'text-text-secondary'
                        )} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-text-primary">{model.version}</span>
                          {model.is_latest && (
                            <span className="tag text-status-success">
                              <span>Latest</span>
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-4 text-sm text-text-secondary font-mono">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3.5 h-3.5" />
                            {new Date(model.created).toLocaleDateString()}
                          </span>
                          <span className="flex items-center gap-1">
                            <Database className="w-3.5 h-3.5" />
                            {model.traces_used} traces
                          </span>
                          <span className="flex items-center gap-1">
                            {model.gguf_available ? (
                              <CheckCircle2 className="w-3.5 h-3.5 text-status-success" />
                            ) : (
                              <XCircle className="w-3.5 h-3.5 text-status-error" />
                            )}
                            GGUF
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-text-secondary font-mono">
                        Quality: {(model.quality_avg * 100).toFixed(0)}%
                      </span>
                      {!model.is_latest && model.gguf_available && (
                        <button
                          onClick={() => handleRollback(model.version)}
                          className="btn-secondary flex items-center gap-1 text-sm"
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                          Rollback
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Traces Tab */}
        {activeTab === 'traces' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-brand text-text-primary">Pending Traces from Bashbros</h3>
              <button
                onClick={handleProcessTraces}
                disabled={processing || pendingTraces.length === 0}
                className="btn-secondary flex items-center gap-2 text-accent border-accent text-sm disabled:opacity-50"
              >
                {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                Process All
              </button>
            </div>

            {pendingTraces.length === 0 ? (
              <div className="text-center py-12 text-text-secondary">
                <Database className="w-12 h-12 mx-auto text-text-muted mb-4" />
                <p className="font-brand text-lg">No pending traces</p>
                <p className="text-sm font-mono">Traces from bashbros will appear here</p>
              </div>
            ) : (
              <div className="space-y-2">
                {pendingTraces.map((trace, index) => (
                  <div
                    key={`${trace.filename}-${index}`}
                    className="card p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="font-mono text-text-primary truncate">{trace.task}</p>
                        <div className="flex items-center gap-3 text-sm text-text-secondary font-mono mt-1">
                          <span>{trace.filename}</span>
                          <span>{trace.steps} steps</span>
                          {trace.verified ? (
                            <span className="text-status-success flex items-center gap-1">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              Verified
                            </span>
                          ) : (
                            <span className="text-text-muted flex items-center gap-1">
                              <AlertCircle className="w-3.5 h-3.5" />
                              Unverified
                            </span>
                          )}
                        </div>
                      </div>
                      <span className="tag">
                        <span>{trace.source}</span>
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
