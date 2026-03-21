import { useEffect, useState } from 'react'
import { useDeviceStore } from '../../stores/deviceStore'
import type { SSHCandidate, PreflightResult } from '../../services/api'
import { Server, Plus, Search, Trash2, Star, RefreshCw, CheckCircle, XCircle } from 'lucide-react'
import { clsx } from 'clsx'

export function DeviceManager() {
  const { devices, defaultDeviceId, fetchDevices, addDevice, removeDevice, setDefault, runPreflight, discoverFromSSHConfig } = useDeviceStore()

  const [showAddForm, setShowAddForm] = useState(false)
  const [showDiscovery, setShowDiscovery] = useState(false)
  const [candidates, setCandidates] = useState<SSHCandidate[]>([])
  const [testingId, setTestingId] = useState<string | null>(null)
  const [preflightResult, setPreflightResult] = useState<PreflightResult | null>(null)

  // Add form state
  const [formName, setFormName] = useState('')
  const [formHost, setFormHost] = useState('')
  const [formUsername, setFormUsername] = useState('')
  const [formPort, setFormPort] = useState(22)
  const [formKeyPath, setFormKeyPath] = useState('~/.ssh/id_rsa')
  const [formWorkDir, setFormWorkDir] = useState('~/bashgym-training')

  useEffect(() => {
    fetchDevices()
  }, [fetchDevices])

  const resetForm = () => {
    setFormName('')
    setFormHost('')
    setFormUsername('')
    setFormPort(22)
    setFormKeyPath('~/.ssh/id_rsa')
    setFormWorkDir('~/bashgym-training')
  }

  const handleAddDevice = async (e: React.FormEvent) => {
    e.preventDefault()
    await addDevice({
      name: formName,
      host: formHost,
      username: formUsername,
      port: formPort,
      key_path: formKeyPath,
      work_dir: formWorkDir,
    })
    resetForm()
    setShowAddForm(false)
  }

  const handleDiscover = async () => {
    const found = await discoverFromSSHConfig()
    setCandidates(found)
    setShowDiscovery(true)
  }

  const handleAddCandidate = async (candidate: SSHCandidate) => {
    await addDevice({
      name: candidate.ssh_alias,
      host: candidate.host,
      username: candidate.username ?? '',
      port: candidate.port,
      key_path: candidate.key_path,
    })
  }

  const handleTest = async (id: string) => {
    setTestingId(id)
    setPreflightResult(null)
    const result = await runPreflight(id)
    setPreflightResult(result)
    setTestingId(null)
  }

  const isRecentlySeen = (lastSeen?: string) => {
    if (!lastSeen) return false
    const oneHourAgo = Date.now() - 60 * 60 * 1000
    return new Date(lastSeen).getTime() > oneHourAgo
  }

  const formatGpuSummary = (device: typeof devices[number]) => {
    const caps = device.capabilities
    if (!caps) return 'Not scanned yet'
    if (caps.gpus && caps.gpus.length > 0) {
      const gpu = caps.gpus[0]
      const suffix = caps.gpus.length > 1 ? ` +${caps.gpus.length - 1}` : ''
      return `${gpu.name} (${gpu.vram_total_gb.toFixed(0)}GB)${suffix}`
    }
    return caps.os ?? 'No GPU detected'
  }

  const formatDiskFree = (device: typeof devices[number]) => {
    const gb = device.capabilities?.disk_free_gb
    if (gb == null) return null
    return `${gb.toFixed(0)} GB free`
  }

  return (
    <div className="space-y-3">
      {/* Device list */}
      {devices.length > 0 && (
        <div className="space-y-2">
          {devices.map((device) => {
            const isDefault = device.id === defaultDeviceId
            const online = isRecentlySeen(device.last_seen)
            const isTesting = testingId === device.id
            const diskFree = formatDiskFree(device)

            return (
              <div
                key={device.id}
                className={clsx(
                  'card p-3',
                  isDefault && 'border-accent'
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  {/* Left: device info */}
                  <div className="flex items-start gap-3 min-w-0">
                    <Server className="w-4 h-4 text-text-muted mt-0.5 flex-shrink-0" />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm text-text-primary">{device.name}</span>
                        {isDefault && (
                          <span className="font-mono text-xs uppercase tracking-widest text-accent border border-accent px-1 leading-tight">
                            Default
                          </span>
                        )}
                      </div>
                      <p className="font-mono text-xs text-text-muted mt-0.5">
                        {device.username}@{device.host}:{device.port}
                      </p>
                      <p className="font-mono text-xs text-text-secondary mt-0.5">
                        {formatGpuSummary(device)}
                        {diskFree && <span className="text-text-muted ml-2">· {diskFree}</span>}
                      </p>
                    </div>
                  </div>

                  {/* Right: status dot */}
                  <div
                    className={clsx(
                      'w-2.5 h-2.5 rounded-full flex-shrink-0 mt-1',
                      online ? 'bg-green-500' : 'bg-text-muted'
                    )}
                    title={online ? 'Online (seen within 1 hour)' : 'Offline or not yet tested'}
                  />
                </div>

                {/* Actions row */}
                <div className="flex items-center gap-2 mt-3 pt-3 border-t border-border">
                  <button
                    type="button"
                    onClick={() => handleTest(device.id)}
                    disabled={isTesting}
                    className={clsx(
                      'btn-secondary py-1 px-2 text-xs flex items-center gap-1',
                      isTesting && 'opacity-60 cursor-not-allowed'
                    )}
                  >
                    <RefreshCw className={clsx('w-3 h-3', isTesting && 'animate-spin')} />
                    {isTesting ? 'Testing…' : 'Test'}
                  </button>

                  {!isDefault && (
                    <button
                      type="button"
                      onClick={() => setDefault(device.id)}
                      className="btn-secondary py-1 px-2 text-xs flex items-center gap-1"
                    >
                      <Star className="w-3 h-3" />
                      Default
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={() => removeDevice(device.id)}
                    className="btn-secondary py-1 px-2 text-xs flex items-center gap-1 text-red-500 border-red-300 ml-auto"
                  >
                    <Trash2 className="w-3 h-3" />
                    Remove
                  </button>
                </div>

                {/* Preflight result inline — only for this device if it was the last tested */}
                {preflightResult && testingId === null && device.id === devices.find(d => d.id === device.id)?.id && (
                  // Show result only if this is the device we just tested (tracked by store refresh)
                  null
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Preflight result banner */}
      {preflightResult && testingId === null && (
        <div
          className={clsx(
            'card p-3 flex items-start gap-2 border-l-4',
            preflightResult.ok ? 'border-l-green-500' : 'border-l-red-500'
          )}
        >
          {preflightResult.ok ? (
            <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
          ) : (
            <XCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
          )}
          <div className="min-w-0">
            <p className="font-mono text-xs font-bold uppercase tracking-widest text-text-primary">
              {preflightResult.ok ? 'Preflight passed' : 'Preflight failed'}
            </p>
            {preflightResult.ok && (
              <p className="font-mono text-xs text-text-secondary mt-1">
                {preflightResult.python_version && `Python ${preflightResult.python_version}`}
                {preflightResult.gpus && preflightResult.gpus.length > 0 && (
                  <span className="ml-2">· GPU: {preflightResult.gpus.map(g => `${g.name} (${g.vram_total_gb.toFixed(0)}GB)`).join(', ')}</span>
                )}
                {preflightResult.disk_free_gb != null && (
                  <span className="ml-2">· {preflightResult.disk_free_gb.toFixed(0)} GB free</span>
                )}
              </p>
            )}
            {preflightResult.error && (
              <p className="font-mono text-xs text-red-500 mt-1">{preflightResult.error}</p>
            )}
          </div>
        </div>
      )}

      {/* Empty state */}
      {devices.length === 0 && !showAddForm && (
        <div className="card p-6 text-center">
          <Server className="w-8 h-8 text-text-muted mx-auto mb-3" />
          <p className="font-mono text-xs uppercase tracking-widest text-text-muted mb-4">
            No remote training devices configured
          </p>
          <div className="flex items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => setShowAddForm(true)}
              className="btn-secondary py-1.5 px-3 text-xs flex items-center gap-1.5"
            >
              <Plus className="w-3 h-3" />
              Add Device
            </button>
            <button
              type="button"
              onClick={handleDiscover}
              className="btn-secondary py-1.5 px-3 text-xs flex items-center gap-1.5"
            >
              <Search className="w-3 h-3" />
              Discover from SSH Config
            </button>
          </div>
        </div>
      )}

      {/* Add Device form */}
      {showAddForm && (
        <form onSubmit={handleAddDevice} className="card p-4 space-y-3">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
            Add Device
          </p>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block font-mono text-xs text-text-muted mb-1">Name</label>
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                className="input w-full"
                placeholder="DGX Spark"
                required
              />
            </div>
            <div>
              <label className="block font-mono text-xs text-text-muted mb-1">Host</label>
              <input
                type="text"
                value={formHost}
                onChange={(e) => setFormHost(e.target.value)}
                className="input w-full"
                placeholder="192.168.1.100"
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block font-mono text-xs text-text-muted mb-1">Username</label>
              <input
                type="text"
                value={formUsername}
                onChange={(e) => setFormUsername(e.target.value)}
                className="input w-full"
                placeholder="ubuntu"
                required
              />
            </div>
            <div>
              <label className="block font-mono text-xs text-text-muted mb-1">Port</label>
              <input
                type="number"
                value={formPort}
                onChange={(e) => setFormPort(parseInt(e.target.value) || 22)}
                className="input w-full"
                min={1}
                max={65535}
              />
            </div>
          </div>

          <div>
            <label className="block font-mono text-xs text-text-muted mb-1">Key Path</label>
            <input
              type="text"
              value={formKeyPath}
              onChange={(e) => setFormKeyPath(e.target.value)}
              className="input w-full"
              placeholder="~/.ssh/id_rsa"
            />
          </div>

          <div>
            <label className="block font-mono text-xs text-text-muted mb-1">Work Dir</label>
            <input
              type="text"
              value={formWorkDir}
              onChange={(e) => setFormWorkDir(e.target.value)}
              className="input w-full"
              placeholder="~/bashgym-training"
            />
          </div>

          <div className="flex items-center gap-2 pt-1">
            <button type="submit" className="btn-primary py-1.5 px-4 text-xs">
              Add Device
            </button>
            <button
              type="button"
              onClick={() => { setShowAddForm(false); resetForm() }}
              className="btn-secondary py-1.5 px-4 text-xs"
            >
              Cancel
            </button>
          </div>
        </form>
      )}

      {/* Discovery results */}
      {showDiscovery && candidates.length > 0 && (
        <div className="card p-3 space-y-2">
          <p className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-2">
            SSH Config Hosts
          </p>
          {candidates.map((c) => (
            <div
              key={c.ssh_alias}
              className="flex items-center justify-between gap-3 p-2 border border-border"
            >
              <div className="min-w-0">
                <span className="font-mono text-xs font-bold text-text-primary">{c.ssh_alias}</span>
                <span className="font-mono text-xs text-text-muted ml-2">{c.host}</span>
              </div>
              {c.already_added ? (
                <span className="font-mono text-xs uppercase tracking-widest text-text-muted border border-border px-1.5 py-0.5 flex-shrink-0">
                  Already added
                </span>
              ) : (
                <button
                  type="button"
                  onClick={() => handleAddCandidate(c)}
                  className="btn-secondary py-0.5 px-2 text-xs flex items-center gap-1 flex-shrink-0"
                >
                  <Plus className="w-3 h-3" />
                  Add
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {showDiscovery && candidates.length === 0 && (
        <div className="card p-3 text-center">
          <p className="font-mono text-xs text-text-muted">No SSH config hosts found.</p>
        </div>
      )}

      {/* Action bar — shown when devices exist */}
      {devices.length > 0 && (
        <div className="flex items-center gap-2 pt-1">
          <button
            type="button"
            onClick={() => { setShowAddForm(!showAddForm); setShowDiscovery(false) }}
            className="btn-secondary py-1.5 px-3 text-xs flex items-center gap-1.5"
          >
            <Plus className="w-3 h-3" />
            Add Device
          </button>
          <button
            type="button"
            onClick={handleDiscover}
            className="btn-secondary py-1.5 px-3 text-xs flex items-center gap-1.5"
          >
            <Search className="w-3 h-3" />
            Discover from SSH Config
          </button>
        </div>
      )}
    </div>
  )
}
