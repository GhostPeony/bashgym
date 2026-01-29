import { useState, useEffect, useCallback } from 'react'
import {
  ArrowLeft,
  Star,
  BarChart3,
  Download,
  Play,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  Zap,
  GitBranch,
  CheckCircle,
  AlertCircle,
  FileText,
  Settings,
  Activity,
  TrendingUp,
  Loader2,
  Edit3,
  Save,
  X
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, ModelProfile as ModelProfileData } from '../../services/api'

interface ModelProfilePageProps {
  modelId: string
  onBack: () => void
  onCompare: (modelIds: string[]) => void
}

const STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  ready: { bg: 'bg-status-success/20', text: 'text-status-success' },
  needs_eval: { bg: 'bg-status-warning/20', text: 'text-status-warning' },
  training: { bg: 'bg-status-info/20', text: 'text-status-info' },
  archived: { bg: 'bg-text-muted/20', text: 'text-text-muted' },
  regression_detected: { bg: 'bg-status-error/20', text: 'text-status-error' },
}

export function ModelProfilePage({ modelId, onBack, onCompare }: ModelProfilePageProps) {
  const [profile, setProfile] = useState<ModelProfileData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Expanded sections
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['training']))

  // Edit mode
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editTags, setEditTags] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  // Fetch profile
  const fetchProfile = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    const result = await modelsApi.get(modelId)
    if (result.ok && result.data) {
      setProfile(result.data)
      setEditName(result.data.display_name)
      setEditDescription(result.data.description)
      setEditTags(result.data.tags.join(', '))
    } else {
      setError('Failed to load model profile')
    }
    setIsLoading(false)
  }, [modelId])

  useEffect(() => {
    fetchProfile()
  }, [fetchProfile])

  const toggleSection = (section: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(section)) {
        next.delete(section)
      } else {
        next.add(section)
      }
      return next
    })
  }

  const handleStar = async () => {
    if (!profile) return
    await modelsApi.star(modelId, !profile.starred)
    fetchProfile()
  }

  const handleSave = async () => {
    if (!profile) return
    setIsSaving(true)
    const result = await modelsApi.update(modelId, {
      display_name: editName,
      description: editDescription,
      tags: editTags.split(',').map(t => t.trim()).filter(Boolean)
    })
    if (result.ok) {
      setProfile(result.data!)
      setIsEditing(false)
    }
    setIsSaving(false)
  }

  // Action states
  const [isExporting, setIsExporting] = useState(false)
  const [isDeploying, setIsDeploying] = useState(false)
  const [actionMessage, setActionMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const handleEvaluate = async () => {
    const result = await modelsApi.evaluate(modelId)
    if (result.ok) {
      setActionMessage({ type: 'success', text: 'Evaluation queued' })
      setTimeout(() => setActionMessage(null), 3000)
    }
  }

  const handleExport = async (quantization: string) => {
    setIsExporting(true)
    setActionMessage(null)
    try {
      const result = await modelsApi.export(modelId, { format: 'gguf', quantization })
      if (result.ok && result.data) {
        setActionMessage({ type: 'success', text: result.data.message || 'Export started' })
        fetchProfile() // Refresh to show new artifact
      } else {
        setActionMessage({ type: 'error', text: 'Export failed' })
      }
    } finally {
      setIsExporting(false)
      setTimeout(() => setActionMessage(null), 5000)
    }
  }

  const handleDeployToOllama = async () => {
    if (!profile) return
    setIsDeploying(true)
    setActionMessage(null)
    try {
      // Find the best GGUF to deploy
      const gguf = profile.artifacts.gguf_exports[0]
      const quantization = gguf?.quantization || 'q4_k_m'

      const result = await modelsApi.deployToOllama(modelId, { quantization })
      if (result.ok && result.data) {
        setActionMessage({ type: 'success', text: result.data.message })
        fetchProfile()
      } else {
        setActionMessage({ type: 'error', text: 'Deploy failed - check if Ollama is running' })
      }
    } finally {
      setIsDeploying(false)
      setTimeout(() => setActionMessage(null), 5000)
    }
  }

  const handleDownloadArtifact = async (artifactPath: string) => {
    const result = await modelsApi.downloadArtifact(modelId, artifactPath)
    if (result.ok && result.data) {
      // Create a download link
      const link = document.createElement('a')
      link.href = `/api/models/${encodeURIComponent(modelId)}/download?path=${encodeURIComponent(artifactPath)}`
      link.download = result.data.filename
      link.click()
    }
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className="h-full flex flex-col items-center justify-center">
        <AlertCircle className="w-12 h-12 text-status-error mb-4" />
        <h2 className="text-lg font-medium text-text-primary mb-2">Error Loading Model</h2>
        <p className="text-text-muted mb-4">{error || 'Model not found'}</p>
        <button onClick={onBack} className="btn-secondary">
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Models
        </button>
      </div>
    )
  }

  const statusStyle = STATUS_STYLES[profile.status] || STATUS_STYLES.ready

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center gap-4 mb-4">
          <button onClick={onBack} className="p-2 rounded-lg hover:bg-background-tertiary text-text-muted">
            <ArrowLeft className="w-5 h-5" />
          </button>

          <div className="flex-1">
            {isEditing ? (
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                className="input text-xl font-semibold"
              />
            ) : (
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-semibold text-text-primary">{profile.display_name}</h1>
                {profile.starred && <Star className="w-5 h-5 text-yellow-500 fill-yellow-500" />}
                <span className={clsx('px-2 py-0.5 rounded text-xs', statusStyle.bg, statusStyle.text)}>
                  {profile.status.replace('_', ' ')}
                </span>
              </div>
            )}
            <div className="flex items-center gap-2 mt-1 text-sm text-text-muted">
              <span>Based on {profile.base_model}</span>
              <span className="text-text-muted/50">•</span>
              <span className="px-1.5 py-0.5 rounded bg-primary/20 text-primary text-xs">
                {profile.training_strategy.toUpperCase()}
              </span>
              <span className="text-text-muted/50">•</span>
              <span>Trained {new Date(profile.created_at).toLocaleDateString()}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {isEditing ? (
              <>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="btn-primary"
                >
                  {isSaving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                  Save
                </button>
                <button onClick={() => setIsEditing(false)} className="btn-secondary">
                  <X className="w-4 h-4 mr-2" />
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button onClick={() => setIsEditing(true)} className="btn-secondary">
                  <Edit3 className="w-4 h-4 mr-2" />
                  Edit
                </button>
                <button onClick={handleStar} className="btn-secondary">
                  <Star className={clsx('w-4 h-4 mr-2', profile.starred && 'fill-current text-yellow-500')} />
                  {profile.starred ? 'Starred' : 'Star'}
                </button>
                <button onClick={() => onCompare([modelId])} className="btn-secondary">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Compare
                </button>
                <button
                  onClick={() => handleExport('q4_k_m')}
                  disabled={isExporting}
                  className="btn-secondary"
                >
                  {isExporting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
                  Export
                </button>
              </>
            )}
          </div>
        </div>

        {/* Description (editable) */}
        {isEditing ? (
          <div className="mb-4">
            <label className="block text-sm text-text-muted mb-1">Description</label>
            <textarea
              value={editDescription}
              onChange={(e) => setEditDescription(e.target.value)}
              rows={2}
              className="input w-full"
            />
          </div>
        ) : profile.description && (
          <p className="text-text-secondary mb-4">{profile.description}</p>
        )}

        {/* Tags (editable) */}
        {isEditing ? (
          <div className="mb-4">
            <label className="block text-sm text-text-muted mb-1">Tags (comma-separated)</label>
            <input
              type="text"
              value={editTags}
              onChange={(e) => setEditTags(e.target.value)}
              className="input w-full"
              placeholder="production, v3, experimental"
            />
          </div>
        ) : profile.tags.length > 0 && (
          <div className="flex items-center gap-2 mb-4">
            {profile.tags.map(tag => (
              <span key={tag} className="px-2 py-0.5 rounded bg-background-tertiary text-text-muted text-sm">
                {tag}
              </span>
            ))}
          </div>
        )}

        {/* Quick Stats */}
        <div className="grid grid-cols-6 gap-4">
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Custom Eval</div>
            <div className="text-xl font-semibold text-text-primary">
              {profile.custom_eval_pass_rate !== null ? `${profile.custom_eval_pass_rate.toFixed(1)}%` : '—'}
            </div>
          </div>
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Benchmark Avg</div>
            <div className="text-xl font-semibold text-text-primary">
              {profile.benchmark_avg_score !== null ? `${profile.benchmark_avg_score.toFixed(1)}%` : '—'}
            </div>
          </div>
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Model Size</div>
            <div className="text-xl font-semibold text-text-primary">{profile.model_size_display}</div>
            {profile.model_size_params && (
              <div className="text-xs text-text-muted">{profile.model_size_params}</div>
            )}
          </div>
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Latency</div>
            <div className="text-xl font-semibold text-text-primary">
              {profile.inference_latency_ms ? `${profile.inference_latency_ms.toFixed(0)}ms` : '—'}
            </div>
          </div>
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Training Time</div>
            <div className="text-xl font-semibold text-text-primary">{profile.training_duration_display}</div>
          </div>
          <div className="p-3 bg-background-tertiary rounded-lg">
            <div className="text-xs text-text-muted mb-1">Final Loss</div>
            <div className="text-xl font-semibold text-text-primary">
              {profile.final_metrics.final_loss?.toFixed(4) || '—'}
            </div>
          </div>
        </div>

        {/* Action Message */}
        {actionMessage && (
          <div className={clsx(
            'mt-4 p-3 rounded-lg text-sm flex items-center gap-2',
            actionMessage.type === 'success' ? 'bg-status-success/20 text-status-success' : 'bg-status-error/20 text-status-error'
          )}>
            {actionMessage.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
            {actionMessage.text}
          </div>
        )}
      </div>

      {/* Collapsible Sections */}
      <div className="flex-1 overflow-auto p-6 space-y-4">
        {/* Training Details */}
        <CollapsibleSection
          title="Training Details"
          icon={Settings}
          expanded={expandedSections.has('training')}
          onToggle={() => toggleSection('training')}
        >
          <div className="grid grid-cols-2 gap-6">
            {/* Config */}
            <div>
              <h4 className="text-sm font-medium text-text-primary mb-3">Configuration</h4>
              <div className="space-y-2 text-sm">
                {Object.entries(profile.config).slice(0, 10).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-text-muted">{key.replace(/_/g, ' ')}</span>
                    <span className="text-text-primary font-mono">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Loss Curve */}
            <div>
              <h4 className="text-sm font-medium text-text-primary mb-3">Loss Curve</h4>
              {profile.loss_curve.length > 0 ? (
                <div className="h-48 bg-background-tertiary rounded-lg p-4">
                  {/* Simple ASCII-style representation - replace with chart library */}
                  <div className="h-full flex items-end gap-1">
                    {profile.loss_curve.slice(-50).map((point, i) => {
                      const maxLoss = Math.max(...profile.loss_curve.map(p => p.loss))
                      const height = (point.loss / maxLoss) * 100
                      return (
                        <div
                          key={i}
                          className="flex-1 bg-primary/50 rounded-t"
                          style={{ height: `${height}%` }}
                          title={`Step ${point.step}: ${point.loss.toFixed(4)}`}
                        />
                      )
                    })}
                  </div>
                </div>
              ) : (
                <div className="h-48 bg-background-tertiary rounded-lg flex items-center justify-center text-text-muted">
                  No loss curve data
                </div>
              )}
            </div>
          </div>

          {/* Training Metrics */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-text-primary mb-3">Final Metrics</h4>
            <div className="grid grid-cols-4 gap-4">
              {Object.entries(profile.final_metrics).map(([key, value]) => (
                <div key={key} className="p-3 bg-background-tertiary rounded-lg">
                  <div className="text-xs text-text-muted">{key.replace(/_/g, ' ')}</div>
                  <div className="text-lg font-semibold text-text-primary">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CollapsibleSection>

        {/* Benchmark Results */}
        <CollapsibleSection
          title="Benchmark Results"
          icon={BarChart3}
          expanded={expandedSections.has('benchmarks')}
          onToggle={() => toggleSection('benchmarks')}
          badge={Object.keys(profile.benchmarks).length || undefined}
        >
          {Object.keys(profile.benchmarks).length > 0 ? (
            <div className="space-y-4">
              <table className="w-full">
                <thead className="bg-background-tertiary">
                  <tr>
                    <th className="px-4 py-2 text-left text-sm font-medium text-text-muted">Benchmark</th>
                    <th className="px-4 py-2 text-right text-sm font-medium text-text-muted">Score</th>
                    <th className="px-4 py-2 text-right text-sm font-medium text-text-muted">Passed</th>
                    <th className="px-4 py-2 text-right text-sm font-medium text-text-muted">Total</th>
                    <th className="px-4 py-2 text-left text-sm font-medium text-text-muted">Evaluated</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border-subtle">
                  {Object.values(profile.benchmarks).map(bench => (
                    <tr key={bench.benchmark_name}>
                      <td className="px-4 py-3 font-medium text-text-primary">{bench.benchmark_name}</td>
                      <td className="px-4 py-3 text-right text-lg font-semibold text-text-primary">
                        {bench.score.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right text-status-success">{bench.passed}</td>
                      <td className="px-4 py-3 text-right text-text-muted">{bench.total}</td>
                      <td className="px-4 py-3 text-text-muted text-sm">
                        {new Date(bench.evaluated_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8">
              <BarChart3 className="w-12 h-12 text-text-muted mx-auto mb-3 opacity-30" />
              <p className="text-text-muted mb-4">No benchmark results yet</p>
              <button onClick={handleEvaluate} className="btn-primary">
                <Play className="w-4 h-4 mr-2" />
                Run Evaluation
              </button>
            </div>
          )}
        </CollapsibleSection>

        {/* Custom Evaluations */}
        <CollapsibleSection
          title="Custom Evaluations"
          icon={CheckCircle}
          expanded={expandedSections.has('custom_evals')}
          onToggle={() => toggleSection('custom_evals')}
          badge={Object.keys(profile.custom_evals).length || undefined}
        >
          {Object.keys(profile.custom_evals).length > 0 ? (
            <div className="space-y-4">
              {Object.values(profile.custom_evals).map(evalResult => (
                <div key={evalResult.eval_set_id} className="p-4 bg-background-tertiary rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-text-primary">{evalResult.eval_set_id}</span>
                      <span className="px-2 py-0.5 rounded bg-primary/20 text-primary text-xs">
                        {evalResult.eval_type}
                      </span>
                    </div>
                    <span className="text-lg font-semibold text-text-primary">
                      {evalResult.pass_rate.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-text-muted">
                    <span>{evalResult.passed}/{evalResult.total} passed</span>
                    <span>Evaluated {new Date(evalResult.evaluated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <CheckCircle className="w-12 h-12 text-text-muted mx-auto mb-3 opacity-30" />
              <p className="text-text-muted">No custom evaluations yet</p>
            </div>
          )}
        </CollapsibleSection>

        {/* Lineage & Traces */}
        <CollapsibleSection
          title="Lineage & Traces"
          icon={GitBranch}
          expanded={expandedSections.has('lineage')}
          onToggle={() => toggleSection('lineage')}
        >
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-text-primary mb-3">Model Lineage</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Base Model</span>
                  <span className="text-text-primary">{profile.base_model}</span>
                </div>
                {profile.parent_model && (
                  <div className="flex justify-between">
                    <span className="text-text-muted">Parent Model</span>
                    <span className="text-text-primary">{profile.parent_model}</span>
                  </div>
                )}
                {profile.teacher_model && (
                  <div className="flex justify-between">
                    <span className="text-text-muted">Teacher Model</span>
                    <span className="text-text-primary">{profile.teacher_model}</span>
                  </div>
                )}
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-text-primary mb-3">Training Data</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-text-muted">Traces Used</span>
                  <span className="text-text-primary">{profile.training_traces.length}</span>
                </div>
                {profile.training_repos.length > 0 && (
                  <div>
                    <span className="text-text-muted block mb-1">From Repos</span>
                    <div className="flex flex-wrap gap-1">
                      {profile.training_repos.map(repo => (
                        <span key={repo} className="px-2 py-0.5 rounded bg-background-tertiary text-text-muted text-xs">
                          {repo}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </CollapsibleSection>

        {/* Artifacts & Export */}
        <CollapsibleSection
          title="Artifacts & Export"
          icon={FileText}
          expanded={expandedSections.has('artifacts')}
          onToggle={() => toggleSection('artifacts')}
        >
          <div className="space-y-4">
            <div className="text-sm text-text-muted mb-2">
              Model directory: <code className="bg-background-tertiary px-2 py-0.5 rounded">{profile.model_dir}</code>
            </div>

            {/* Checkpoints */}
            {profile.artifacts.checkpoints.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-text-primary mb-2">Checkpoints</h4>
                <div className="flex flex-wrap gap-2">
                  {profile.artifacts.checkpoints.map(cp => (
                    <span key={cp.path} className="px-2 py-1 rounded bg-background-tertiary text-sm text-text-muted">
                      Step {cp.step}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Final & Merged */}
            <div className="grid grid-cols-2 gap-4">
              {profile.artifacts.final_adapter_path && (
                <div className="p-3 bg-background-tertiary rounded-lg">
                  <div className="text-xs text-text-muted mb-1">Final Adapter</div>
                  <div className="text-sm text-text-primary truncate">{profile.artifacts.final_adapter_path}</div>
                </div>
              )}
              {profile.artifacts.merged_path && (
                <div className="p-3 bg-background-tertiary rounded-lg">
                  <div className="text-xs text-text-muted mb-1">Merged Model</div>
                  <div className="text-sm text-text-primary truncate">{profile.artifacts.merged_path}</div>
                </div>
              )}
            </div>

            {/* GGUF Exports */}
            {profile.artifacts.gguf_exports.length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-text-primary mb-2">GGUF Exports</h4>
                <div className="space-y-2">
                  {profile.artifacts.gguf_exports.map(gguf => (
                    <div key={gguf.path} className="flex items-center justify-between p-3 bg-background-tertiary rounded-lg">
                      <div>
                        <span className="text-sm text-text-primary">{gguf.quantization}</span>
                        <span className="text-xs text-text-muted ml-2">
                          {(gguf.size_bytes / (1024 * 1024 * 1024)).toFixed(2)} GB
                        </span>
                      </div>
                      <button
                        onClick={() => handleDownloadArtifact(gguf.path)}
                        className="btn-secondary text-sm"
                      >
                        <Download className="w-3 h-3 mr-1" />
                        Download
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Export Actions */}
            <div className="flex items-center gap-2 pt-4 border-t border-border-subtle">
              <button
                onClick={() => handleExport('q4_k_m')}
                disabled={isExporting}
                className="btn-secondary"
              >
                {isExporting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
                Export GGUF Q4
              </button>
              <button
                onClick={() => handleExport('q8_0')}
                disabled={isExporting}
                className="btn-secondary"
              >
                {isExporting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Download className="w-4 h-4 mr-2" />}
                Export GGUF Q8
              </button>
              <button
                onClick={handleDeployToOllama}
                disabled={isDeploying || profile.artifacts.gguf_exports.length === 0}
                className="btn-primary"
                title={profile.artifacts.gguf_exports.length === 0 ? 'Export a GGUF first' : 'Deploy to Ollama'}
              >
                {isDeploying ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
                Deploy to Ollama
              </button>
            </div>
          </div>
        </CollapsibleSection>
      </div>
    </div>
  )
}

// Collapsible Section Component
interface CollapsibleSectionProps {
  title: string
  icon: React.ElementType
  expanded: boolean
  onToggle: () => void
  badge?: number
  children: React.ReactNode
}

function CollapsibleSection({ title, icon: Icon, expanded, onToggle, badge, children }: CollapsibleSectionProps) {
  return (
    <div className="card-elevated">
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center justify-between hover:bg-background-tertiary transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className="w-5 h-5 text-primary" />
          <span className="font-medium text-text-primary">{title}</span>
          {badge !== undefined && (
            <span className="px-2 py-0.5 rounded-full bg-primary/20 text-primary text-xs">
              {badge}
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-text-muted" />
        ) : (
          <ChevronRight className="w-5 h-5 text-text-muted" />
        )}
      </button>
      {expanded && (
        <div className="p-4 pt-0 border-t border-border-subtle">
          {children}
        </div>
      )}
    </div>
  )
}
