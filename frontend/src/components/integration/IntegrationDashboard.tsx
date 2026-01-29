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
    <div className="bg-background-secondary rounded-lg border border-border p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-text-muted uppercase tracking-wider">{title}</p>
          <p className={clsx('text-2xl font-bold mt-1', statusColors[status])}>{value}</p>
        </div>
        <div className={clsx('p-2 rounded-lg bg-background-tertiary', statusColors[status])}>
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
        <Loader2 className="w-8 h-8 animate-spin text-accent-primary" />
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
              'p-3 rounded-xl',
              isLinked ? 'bg-status-success/10' : 'bg-background-secondary'
            )}>
              {isLinked ? (
                <Link2 className="w-6 h-6 text-status-success" />
              ) : (
                <Link2Off className="w-6 h-6 text-text-secondary" />
              )}
            </div>
            <div>
              <h1 className="text-2xl font-bold text-text-primary">Bashbros Integration</h1>
              <p className="text-text-secondary">
                {isLinked ? 'Connected - traces flowing, model sync enabled' : 'Not linked - standalone mode'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchData}
              className="p-2 rounded-lg bg-background-secondary hover:bg-background-tertiary text-text-secondary transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            {isLinked ? (
              <button
                onClick={handleUnlink}
                disabled={linking}
                className="px-4 py-2 rounded-lg bg-status-error/10 text-status-error hover:bg-status-error/20 transition-colors disabled:opacity-50"
              >
                {linking ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Unlink'}
              </button>
            ) : (
              <button
                onClick={handleLink}
                disabled={linking}
                className="px-4 py-2 rounded-lg bg-status-success/10 text-status-success hover:bg-status-success/20 transition-colors disabled:opacity-50"
              >
                {linking ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Link Integration'}
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-6 p-1 bg-background-secondary rounded-lg w-fit">
          {(['status', 'settings', 'models', 'traces'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                'px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize',
                activeTab === tab
                  ? 'bg-background-primary text-text-primary shadow-sm'
                  : 'text-text-secondary hover:text-text-primary'
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
              <div className="p-4 bg-status-warning/10 border border-status-warning/30 rounded-lg flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-status-warning" />
                <div>
                  <p className="font-medium text-status-warning">Training in Progress</p>
                  <p className="text-sm text-text-secondary">A new model is being trained...</p>
                </div>
              </div>
            )}

            {/* Trace Watcher */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium text-text-primary">Trace Watcher</h3>
                  <p className="text-sm text-text-secondary">
                    Automatically process traces from bashbros
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {!watching ? (
                    <button
                      onClick={handleStartWatcher}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-status-success/10 text-status-success hover:bg-status-success/20 text-sm"
                    >
                      <Play className="w-4 h-4" />
                      Start
                    </button>
                  ) : (
                    <button
                      onClick={handleStopWatcher}
                      className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-status-error/10 text-status-error hover:bg-status-error/20 text-sm"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  )}
                  <button
                    onClick={handleProcessTraces}
                    disabled={processing}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 text-sm disabled:opacity-50"
                  >
                    {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                    Process Now
                  </button>
                </div>
              </div>
            </div>

            {/* Integration Directory */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <div className="flex items-center gap-2 mb-3">
                <Folder className="w-4 h-4 text-text-secondary" />
                <h3 className="font-medium text-text-primary">Integration Directory</h3>
              </div>
              <code className="text-sm font-mono text-text-secondary bg-background-tertiary px-2 py-1 rounded">
                ~/.bashgym/integration/
              </code>
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && settings && (
          <div className="space-y-6">
            {/* Capture Settings */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <h3 className="font-medium text-text-primary mb-4 flex items-center gap-2">
                <Database className="w-4 h-4" />
                Capture Settings
              </h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Capture Mode</label>
                  <select
                    value={settings.capture_mode}
                    onChange={(e) => handleUpdateSetting('capture', 'mode', e.target.value)}
                    className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
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
                    className="w-4 h-4 rounded border-border"
                  />
                  <span className="text-text-primary">Auto-stream traces</span>
                </label>
              </div>
            </div>

            {/* Training Settings */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <h3 className="font-medium text-text-primary mb-4 flex items-center gap-2">
                <Cpu className="w-4 h-4" />
                Training Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.auto_training_enabled}
                    onChange={(e) => handleUpdateSetting('training', 'auto_enabled', e.target.checked)}
                    className="w-4 h-4 rounded border-border"
                  />
                  <span className="text-text-primary">Enable auto-training</span>
                </label>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">
                    Quality Threshold (gold traces needed)
                  </label>
                  <input
                    type="number"
                    value={settings.quality_threshold}
                    onChange={(e) => handleUpdateSetting('training', 'quality_threshold', parseInt(e.target.value))}
                    min={10}
                    max={500}
                    className="w-32 px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                  />
                </div>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Training Trigger</label>
                  <select
                    value={settings.trigger}
                    onChange={(e) => handleUpdateSetting('training', 'trigger', e.target.value)}
                    className="w-full px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm"
                  >
                    <option value="manual">Manual - Only on request</option>
                    <option value="quality_based">Quality Based - When threshold reached</option>
                    <option value="scheduled">Scheduled - Regular intervals</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Model Sync Settings */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <h3 className="font-medium text-text-primary mb-4 flex items-center gap-2">
                <Package className="w-4 h-4" />
                Model Sync Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.auto_export_ollama}
                    onChange={(e) => handleUpdateSetting('model_sync', 'auto_export_ollama', e.target.checked)}
                    className="w-4 h-4 rounded border-border"
                  />
                  <span className="text-text-primary">Auto-export to Ollama</span>
                </label>

                <div>
                  <label className="block text-sm text-text-secondary mb-2">Ollama Model Name</label>
                  <input
                    type="text"
                    value={settings.ollama_model_name}
                    onChange={(e) => handleUpdateSetting('model_sync', 'ollama_model_name', e.target.value)}
                    className="w-64 px-3 py-2 bg-background-primary border border-border rounded-lg text-text-primary text-sm font-mono"
                  />
                </div>

                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.notify_on_update}
                    onChange={(e) => handleUpdateSetting('model_sync', 'notify_on_update', e.target.checked)}
                    className="w-4 h-4 rounded border-border"
                  />
                  <span className="text-text-primary">Notify on model update</span>
                </label>
              </div>
            </div>

            {/* Security Settings */}
            <div className="bg-background-secondary rounded-lg border border-border p-4">
              <h3 className="font-medium text-text-primary mb-4 flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Security Settings
              </h3>

              <div className="space-y-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.bashbros_primary}
                    onChange={(e) => handleUpdateSetting('security', 'bashbros_primary', e.target.checked)}
                    className="w-4 h-4 rounded border-border"
                  />
                  <div>
                    <span className="text-text-primary">Bashbros as primary security</span>
                    <p className="text-sm text-text-secondary">
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
                <Package className="w-12 h-12 mx-auto opacity-50 mb-4" />
                <p>No model versions available</p>
                <p className="text-sm">Train your first model to see versions here</p>
              </div>
            ) : (
              modelVersions.map((model) => (
                <div
                  key={model.version}
                  className={clsx(
                    'bg-background-secondary rounded-lg border p-4',
                    model.is_latest ? 'border-status-success' : 'border-border'
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={clsx(
                        'p-2 rounded-lg',
                        model.is_latest ? 'bg-status-success/10' : 'bg-background-tertiary'
                      )}>
                        <Package className={clsx(
                          'w-5 h-5',
                          model.is_latest ? 'text-status-success' : 'text-text-secondary'
                        )} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-text-primary">{model.version}</span>
                          {model.is_latest && (
                            <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-status-success/20 text-status-success">
                              Latest
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-4 text-sm text-text-secondary">
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
                      <span className="text-sm text-text-secondary">
                        Quality: {(model.quality_avg * 100).toFixed(0)}%
                      </span>
                      {!model.is_latest && model.gguf_available && (
                        <button
                          onClick={() => handleRollback(model.version)}
                          className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-background-tertiary text-text-secondary hover:text-text-primary text-sm"
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
              <h3 className="font-medium text-text-primary">Pending Traces from Bashbros</h3>
              <button
                onClick={handleProcessTraces}
                disabled={processing || pendingTraces.length === 0}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-accent-primary/10 text-accent-primary hover:bg-accent-primary/20 text-sm disabled:opacity-50"
              >
                {processing ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                Process All
              </button>
            </div>

            {pendingTraces.length === 0 ? (
              <div className="text-center py-12 text-text-secondary">
                <Database className="w-12 h-12 mx-auto opacity-50 mb-4" />
                <p>No pending traces</p>
                <p className="text-sm">Traces from bashbros will appear here</p>
              </div>
            ) : (
              <div className="space-y-2">
                {pendingTraces.map((trace, index) => (
                  <div
                    key={`${trace.filename}-${index}`}
                    className="bg-background-secondary rounded-lg border border-border p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-text-primary truncate">{trace.task}</p>
                        <div className="flex items-center gap-3 text-sm text-text-secondary mt-1">
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
                      <span className="text-xs text-text-muted bg-background-tertiary px-2 py-1 rounded">
                        {trace.source}
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
