import { useState, useEffect, useRef } from 'react'
import {
  Database,
  Sparkles,
  Play,
  Plus,
  Trash2,
  Settings,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileJson,
  X,
  Layers,
  Cpu,
  RefreshCw,
  Upload,
  Download,
  Wand2,
  BookOpen
} from 'lucide-react'
import { SyntheticGenerator, SyntheticGeneratorState } from './SyntheticGenerator'
import { SeedsPanel } from './SeedsPanel'
import { SettingsPanel } from './SettingsPanel'
import { useTutorialComplete } from '../../hooks'
import { clsx } from 'clsx'
import {
  factoryApi,
  syntheticApi,
  FactoryConfig,
  ColumnConfig,
  SynthesisJob,
  SyntheticJobStatus
} from '../../services/api'
import { TabId, COLUMN_TYPES, RISK_LEVELS, DEFAULT_CONFIG } from './types'

function HowItWorks({ isCollapsed, onToggle }: { isCollapsed: boolean; onToggle: () => void }) {
  return (
    <div className="card-elevated bg-gradient-to-r from-primary/5 to-transparent border-primary/20 mb-6">
      <button
        onClick={onToggle}
        className="w-full p-5 flex items-start gap-4 text-left"
      >
        <div className="p-2 rounded-lg bg-primary/10">
          <Sparkles className="w-6 h-6 text-primary" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-text-primary">
              How Data Creator Works
            </h3>
            <ChevronDown className={clsx(
              'w-4 h-4 text-text-muted transition-transform',
              isCollapsed && '-rotate-90'
            )} />
          </div>
          <p className="text-sm text-text-secondary">
            Powered by NVIDIA NeMo Data Designer
          </p>
        </div>
      </button>

      {!isCollapsed && (
        <div className="px-5 pb-5 space-y-4">
          <p className="text-sm text-text-secondary">
            Data Creator uses NVIDIA NeMo Data Designer to generate high-quality synthetic training data from your gold traces,
            expanding limited real data into larger, diverse training sets.
          </p>

          <div className="flex items-center justify-between text-xs text-text-secondary bg-background-tertiary rounded-lg p-3">
            <span className="font-medium text-text-primary">Gold Traces</span>
            <ChevronRight className="w-4 h-4 text-text-muted" />
            <span>Extract Patterns</span>
            <ChevronRight className="w-4 h-4 text-text-muted" />
            <span>Generate via NIM/Claude</span>
            <ChevronRight className="w-4 h-4 text-text-muted" />
            <span>Validate & Dedupe</span>
            <ChevronRight className="w-4 h-4 text-text-muted" />
            <span className="font-medium text-text-primary">NeMo JSONL</span>
          </div>

          <div className="text-xs text-text-muted">
            <span className="font-medium text-text-secondary">Supported column types: </span>
            LLM, Sampler, Category, Person, DateTime, Expression, UUID, Gaussian, Validator
            <span className="mx-2">&middot;</span>
            <a
              href="https://docs.nvidia.com/nemo/microservices/latest/design-synthetic-data-from-scratch-or-seeds/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline inline-flex items-center gap-1"
            >
              <BookOpen className="w-3 h-3" />
              Learn more
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

export function FactoryDashboard() {
  const [activeTab, setActiveTab] = useState<TabId>('create')
  const [howItWorksCollapsed, setHowItWorksCollapsed] = useState(false)
  const [advancedConfigOpen, setAdvancedConfigOpen] = useState(false)
  const [config, setConfig] = useState<FactoryConfig>(DEFAULT_CONFIG)
  const [jobs, setJobs] = useState<SynthesisJob[]>([])
  const [syntheticJobs, setSyntheticJobs] = useState<SyntheticJobStatus[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [expandedColumn, setExpandedColumn] = useState<string | null>(null)
  const [availableModels, setAvailableModels] = useState<{ id: string; name: string; provider: string }[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { complete: completeTutorialStep } = useTutorialComplete()

  // Synthetic generator state (for header button)
  const [syntheticState, setSyntheticState] = useState<SyntheticGeneratorState | null>(null)

  // Fetch config on mount
  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const [configResult, jobsResult, modelsResult, syntheticJobsResult] = await Promise.all([
          factoryApi.getConfig(),
          factoryApi.listJobs(),
          factoryApi.listModels(),
          syntheticApi.listJobs()
        ])

        if (configResult.ok && configResult.data) {
          setConfig({ ...DEFAULT_CONFIG, ...configResult.data })
        }

        if (jobsResult.ok && jobsResult.data) {
          setJobs(jobsResult.data)
        }

        if (modelsResult.ok && modelsResult.data) {
          setAvailableModels(modelsResult.data)
        }

        if (syntheticJobsResult.ok && syntheticJobsResult.data) {
          setSyntheticJobs(syntheticJobsResult.data)
        }
      } finally {
        setIsLoading(false)
      }
    }
    fetchData()
  }, [])

  const handleSave = async () => {
    setIsSaving(true)
    try {
      await factoryApi.updateConfig(config)
    } finally {
      setIsSaving(false)
    }
  }

  const addColumn = () => {
    const newColumn: ColumnConfig = {
      id: `col_${Date.now()}`,
      name: 'new_column',
      type: 'llm',
      description: '',
      required: true,
      risk_level: 'normal',
      config: { prompt: '' },
      dependencies: [],
      constraints: []
    }
    setConfig({
      ...config,
      columns: [...config.columns, newColumn]
    })
    setExpandedColumn(newColumn.id)
  }

  const removeColumn = (id: string) => {
    setConfig({
      ...config,
      columns: config.columns.filter(c => c.id !== id)
    })
    if (expandedColumn === id) setExpandedColumn(null)
  }

  const updateColumn = (id: string, updates: Partial<ColumnConfig>) => {
    setConfig({
      ...config,
      columns: config.columns.map(c => c.id === id ? { ...c, ...updates } : c)
    })
  }

  const importFromTraces = async () => {
    const result = await factoryApi.importFromGoldTraces()
    if (result.ok && result.data) {
      setConfig({
        ...config,
        seeds: [...config.seeds, ...result.data.seeds]
      })
      completeTutorialStep('import_traces')
    }
  }

  const runSynthesis = async (isPreview: boolean = false) => {
    const result = await factoryApi.runSynthesis({
      preview: isPreview,
      row_count: isPreview ? 50 : config.output.row_count
    })
    if (result.ok && result.data) {
      setJobs([result.data, ...jobs])
      setActiveTab('jobs')
    }
  }

  const exportConfig = () => {
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `factory-config-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleImportClick = () => {
    fileInputRef.current?.click()
  }

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      const imported = JSON.parse(text) as FactoryConfig
      setConfig({ ...DEFAULT_CONFIG, ...imported })
    } catch (err) {
      console.error('Failed to import config:', err)
    }

    // Reset the input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-border-subtle">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-text-primary">Data Creator</h1>
            <p className="text-sm text-text-secondary mt-1">
              Generate high-quality synthetic training data with NVIDIA NeMo
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Hidden file input for import */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImportFile}
              className="hidden"
            />

            {/* Create tab specific controls */}
            {activeTab === 'create' && syntheticState && (
              <button
                onClick={syntheticState.onGenerate}
                disabled={!syntheticState.canGenerate}
                className="btn-primary flex items-center gap-2"
                data-tutorial="generate-button"
              >
                {syntheticState.isGenerating ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Wand2 className="w-4 h-4" />
                )}
                Generate {syntheticState.targetExamples.toLocaleString()} Examples
              </button>
            )}

            {/* Advanced/Settings tab controls */}
            {activeTab !== 'create' && (
              <>
                <button
                  onClick={handleImportClick}
                  className="btn-ghost p-2"
                  title="Import config"
                >
                  <Upload className="w-5 h-5" />
                </button>
                <button
                  onClick={exportConfig}
                  className="btn-ghost p-2"
                  title="Export config"
                >
                  <Download className="w-5 h-5" />
                </button>
                <div className="w-px h-6 bg-border-subtle mx-1" />
                <button
                  onClick={() => runSynthesis(false)}
                  disabled={config.columns.length === 0}
                  className="btn-secondary flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Run Synthesis
                </button>
                <button
                  onClick={handleSave}
                  disabled={isSaving}
                  className="btn-primary flex items-center gap-2"
                >
                  {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                  Save
                </button>
              </>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-6 overflow-x-auto">
          {[
            { id: 'create' as TabId, label: 'Create', icon: Wand2 },
            { id: 'seeds' as TabId, label: 'Seeds', icon: Layers, badge: config.seeds.length > 0 ? config.seeds.length : undefined },
            { id: 'settings' as TabId, label: 'Settings', icon: Settings },
            { id: 'jobs' as TabId, label: 'Jobs', icon: RefreshCw, badge: jobs.filter(j => j.status === 'running').length + syntheticJobs.filter(j => j.status === 'running').length || undefined },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors whitespace-nowrap',
                activeTab === tab.id
                  ? 'bg-primary text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.badge !== undefined && tab.badge > 0 && (
                <span className="px-1.5 py-0.5 text-xs rounded-full bg-white/20">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {/* Create Tab - Main generation interface */}
        {activeTab === 'create' && (
          <div className="p-6 max-w-4xl mx-auto">
            <HowItWorks
              isCollapsed={howItWorksCollapsed}
              onToggle={() => setHowItWorksCollapsed(!howItWorksCollapsed)}
            />

            {/* Main Generator Card */}
            <div className="card-elevated p-6 mb-6">
              <SyntheticGenerator onStateChange={setSyntheticState} />
            </div>

            {/* Advanced Configuration - Collapsible */}
            <div className="card-elevated">
              <button
                onClick={() => setAdvancedConfigOpen(!advancedConfigOpen)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-background-tertiary/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Settings className="w-5 h-5 text-text-muted" />
                  <div>
                    <h3 className="font-medium text-text-primary">Advanced Configuration</h3>
                    <p className="text-sm text-text-muted">Schema columns, model settings, and output format</p>
                  </div>
                </div>
                <ChevronDown className={clsx(
                  'w-5 h-5 text-text-muted transition-transform',
                  !advancedConfigOpen && '-rotate-90'
                )} />
              </button>

              {advancedConfigOpen && (
                <div className="border-t border-border-subtle p-6 space-y-6">
                  {/* Two Column Layout for Model + Output */}
                  <div className="grid grid-cols-2 gap-6">
                    {/* Default Model Config */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-primary" />
                        <h4 className="font-medium text-text-primary text-sm">Default Model</h4>
                      </div>
                      <div className="space-y-3">
                        <div>
                          <label className="block text-xs font-medium text-text-secondary mb-1">Model</label>
                          <select
                            value={config.default_model.model_id}
                            onChange={(e) => setConfig({
                              ...config,
                              default_model: { ...config.default_model, model_id: e.target.value }
                            })}
                            className="input text-sm w-full"
                          >
                            {availableModels.map(m => (
                              <option key={m.id} value={m.id}>{m.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Temperature</label>
                            <input
                              type="number"
                              min="0"
                              max="2"
                              step="0.1"
                              value={config.default_model.temperature}
                              onChange={(e) => setConfig({
                                ...config,
                                default_model: { ...config.default_model, temperature: parseFloat(e.target.value) }
                              })}
                              className="input text-sm w-full"
                            />
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Max Tokens</label>
                            <input
                              type="number"
                              min="1"
                              max="8192"
                              value={config.default_model.max_tokens}
                              onChange={(e) => setConfig({
                                ...config,
                                default_model: { ...config.default_model, max_tokens: parseInt(e.target.value) }
                              })}
                              className="input text-sm w-full"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Output Settings */}
                    <div className="space-y-4">
                      <div className="flex items-center gap-2">
                        <FileJson className="w-4 h-4 text-primary" />
                        <h4 className="font-medium text-text-primary text-sm">Output Settings</h4>
                      </div>
                      <div className="space-y-3">
                        <div>
                          <label className="block text-xs font-medium text-text-secondary mb-1">Task Name</label>
                          <input
                            type="text"
                            value={config.output.task_name}
                            onChange={(e) => setConfig({
                              ...config,
                              output: { ...config.output, task_name: e.target.value }
                            })}
                            placeholder="default_task"
                            className="input text-sm w-full"
                          />
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Format</label>
                            <select
                              value={config.output.format}
                              onChange={(e) => setConfig({
                                ...config,
                                output: { ...config.output, format: e.target.value as 'jsonl' | 'parquet' }
                              })}
                              className="input text-sm w-full"
                            >
                              <option value="jsonl">JSONL</option>
                              <option value="parquet">Parquet</option>
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Train/Val Split</label>
                            <input
                              type="number"
                              min="0.5"
                              max="1"
                              step="0.05"
                              value={config.output.train_val_split}
                              onChange={(e) => setConfig({
                                ...config,
                                output: { ...config.output, train_val_split: parseFloat(e.target.value) }
                              })}
                              className="input text-sm w-full"
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Column Schema - Full Width */}
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <Database className="w-4 h-4 text-primary" />
                        <h4 className="font-medium text-text-primary text-sm">Column Schema</h4>
                        <span className="text-xs text-text-muted">({config.columns.length} columns)</span>
                      </div>
                      <button onClick={addColumn} className="btn-secondary text-xs flex items-center gap-1 py-1 px-2">
                        <Plus className="w-3 h-3" />
                        Add Column
                      </button>
                    </div>

                    <div className="border border-border-subtle rounded-lg divide-y divide-border-subtle">
                {config.columns.map((column) => (
                  <div key={column.id} className="p-4">
                    {/* Column Header */}
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setExpandedColumn(expandedColumn === column.id ? null : column.id)}
                        className="p-1 hover:bg-background-tertiary rounded"
                      >
                        {expandedColumn === column.id ? (
                          <ChevronDown className="w-4 h-4 text-text-muted" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-text-muted" />
                        )}
                      </button>
                      <div className="flex-1 grid grid-cols-4 gap-3">
                        <input
                          type="text"
                          value={column.name}
                          onChange={(e) => updateColumn(column.id, { name: e.target.value })}
                          placeholder="Column name"
                          className="input text-sm"
                        />
                        <select
                          value={column.type}
                          onChange={(e) => updateColumn(column.id, { type: e.target.value as ColumnConfig['type'] })}
                          className="input text-sm"
                        >
                          {COLUMN_TYPES.map(type => (
                            <option key={type.id} value={type.id}>{type.label}</option>
                          ))}
                        </select>
                        <select
                          value={column.risk_level}
                          onChange={(e) => updateColumn(column.id, { risk_level: e.target.value as 'normal' | 'elevated' | 'high' })}
                          className={clsx('input text-sm', RISK_LEVELS.find(r => r.id === column.risk_level)?.color)}
                        >
                          {RISK_LEVELS.map(level => (
                            <option key={level.id} value={level.id}>{level.label}</option>
                          ))}
                        </select>
                        <div className="flex items-center gap-2">
                          <label className="flex items-center gap-1 text-sm text-text-secondary">
                            <input
                              type="checkbox"
                              checked={column.required}
                              onChange={(e) => updateColumn(column.id, { required: e.target.checked })}
                              className="rounded"
                            />
                            Required
                          </label>
                        </div>
                      </div>
                      <button
                        onClick={() => removeColumn(column.id)}
                        className="p-2 text-text-muted hover:text-status-error transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>

                    {/* Expanded Column Config */}
                    {expandedColumn === column.id && (
                      <div className="mt-4 ml-8 p-4 bg-background-tertiary rounded-lg space-y-4">
                        {/* Description */}
                        <div>
                          <label className="block text-xs font-medium text-text-secondary mb-1">Description</label>
                          <input
                            type="text"
                            value={column.description || ''}
                            onChange={(e) => updateColumn(column.id, { description: e.target.value })}
                            placeholder="What this column represents..."
                            className="input text-sm w-full"
                          />
                        </div>

                        {/* Type-specific config */}
                        {column.type === 'llm' && (
                          <div className="space-y-3">
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Prompt Template</label>
                              <textarea
                                value={column.config.prompt || ''}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, prompt: e.target.value } })}
                                placeholder="Generate a {{task_type}} for: {{user_request}}"
                                rows={3}
                                className="input text-sm w-full font-mono"
                              />
                              <p className="text-xs text-text-muted mt-1">Use {'{{column_name}}'} to reference other columns</p>
                            </div>
                            <div className="grid grid-cols-3 gap-3">
                              <div>
                                <label className="block text-xs font-medium text-text-secondary mb-1">Model Override</label>
                                <select
                                  value={column.config.model?.model_id || ''}
                                  onChange={(e) => updateColumn(column.id, {
                                    config: {
                                      ...column.config,
                                      model: e.target.value ? { ...config.default_model, model_id: e.target.value } : undefined
                                    }
                                  })}
                                  className="input text-sm w-full"
                                >
                                  <option value="">Use default</option>
                                  {availableModels.map(m => (
                                    <option key={m.id} value={m.id}>{m.name}</option>
                                  ))}
                                </select>
                              </div>
                              <div>
                                <label className="block text-xs font-medium text-text-secondary mb-1">Temperature</label>
                                <input
                                  type="number"
                                  min="0"
                                  max="2"
                                  step="0.1"
                                  value={column.config.model?.temperature ?? config.default_model.temperature}
                                  onChange={(e) => updateColumn(column.id, {
                                    config: {
                                      ...column.config,
                                      model: {
                                        ...(column.config.model || config.default_model),
                                        temperature: parseFloat(e.target.value)
                                      }
                                    }
                                  })}
                                  className="input text-sm w-full"
                                />
                              </div>
                              <div>
                                <label className="block text-xs font-medium text-text-secondary mb-1">Max Tokens</label>
                                <input
                                  type="number"
                                  min="1"
                                  max="8192"
                                  value={column.config.model?.max_tokens ?? config.default_model.max_tokens}
                                  onChange={(e) => updateColumn(column.id, {
                                    config: {
                                      ...column.config,
                                      model: {
                                        ...(column.config.model || config.default_model),
                                        max_tokens: parseInt(e.target.value)
                                      }
                                    }
                                  })}
                                  className="input text-sm w-full"
                                />
                              </div>
                            </div>
                          </div>
                        )}

                        {column.type === 'category' && (
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Values (comma-separated)</label>
                            <input
                              type="text"
                              value={column.config.values?.join(', ') || ''}
                              onChange={(e) => updateColumn(column.id, {
                                config: { ...column.config, values: e.target.value.split(',').map(v => v.trim()).filter(Boolean) }
                              })}
                              placeholder="option_a, option_b, option_c"
                              className="input text-sm w-full"
                            />
                          </div>
                        )}

                        {column.type === 'expression' && (
                          <div>
                            <label className="block text-xs font-medium text-text-secondary mb-1">Jinja2 Template</label>
                            <textarea
                              value={column.config.template || ''}
                              onChange={(e) => updateColumn(column.id, { config: { ...column.config, template: e.target.value } })}
                              placeholder="{{ user_request | upper }} - {{ task_id }}"
                              rows={2}
                              className="input text-sm w-full font-mono"
                            />
                          </div>
                        )}

                        {column.type === 'gaussian' && (
                          <div className="grid grid-cols-2 gap-3">
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Mean</label>
                              <input
                                type="number"
                                value={column.config.mean ?? 0}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, mean: parseFloat(e.target.value) } })}
                                className="input text-sm w-full"
                              />
                            </div>
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Std Dev</label>
                              <input
                                type="number"
                                value={column.config.std ?? 1}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, std: parseFloat(e.target.value) } })}
                                className="input text-sm w-full"
                              />
                            </div>
                          </div>
                        )}

                        {column.type === 'datetime' && (
                          <div className="grid grid-cols-3 gap-3">
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Format</label>
                              <input
                                type="text"
                                value={column.config.format || '%Y-%m-%d'}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, format: e.target.value } })}
                                placeholder="%Y-%m-%d"
                                className="input text-sm w-full font-mono"
                              />
                            </div>
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Min Date</label>
                              <input
                                type="date"
                                value={column.config.min_date || ''}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, min_date: e.target.value } })}
                                className="input text-sm w-full"
                              />
                            </div>
                            <div>
                              <label className="block text-xs font-medium text-text-secondary mb-1">Max Date</label>
                              <input
                                type="date"
                                value={column.config.max_date || ''}
                                onChange={(e) => updateColumn(column.id, { config: { ...column.config, max_date: e.target.value } })}
                                className="input text-sm w-full"
                              />
                            </div>
                          </div>
                        )}

                        {/* Constraints */}
                        <div>
                          <label className="block text-xs font-medium text-text-secondary mb-2">Constraints</label>
                          <div className="space-y-2">
                            {column.constraints?.map((constraint, idx) => (
                              <div key={idx} className="flex items-center gap-2">
                                <select
                                  value={constraint.type}
                                  onChange={(e) => {
                                    const updated = [...(column.constraints || [])]
                                    updated[idx] = { ...constraint, type: e.target.value }
                                    updateColumn(column.id, { constraints: updated })
                                  }}
                                  className="input text-sm"
                                >
                                  <option value="enum">Enum</option>
                                  <option value="regex">Regex</option>
                                  <option value="min_length">Min Length</option>
                                  <option value="max_length">Max Length</option>
                                </select>
                                <input
                                  type="text"
                                  value={Array.isArray(constraint.value) ? constraint.value.join(', ') : String(constraint.value)}
                                  onChange={(e) => {
                                    const updated = [...(column.constraints || [])]
                                    updated[idx] = {
                                      ...constraint,
                                      value: constraint.type === 'enum'
                                        ? e.target.value.split(',').map(v => v.trim())
                                        : constraint.type.includes('length')
                                          ? parseInt(e.target.value)
                                          : e.target.value
                                    }
                                    updateColumn(column.id, { constraints: updated })
                                  }}
                                  placeholder={constraint.type === 'enum' ? 'value1, value2' : constraint.type === 'regex' ? '^[a-z]+$' : '10'}
                                  className="input text-sm flex-1 font-mono"
                                />
                                <button
                                  onClick={() => {
                                    const updated = column.constraints?.filter((_, i) => i !== idx) || []
                                    updateColumn(column.id, { constraints: updated })
                                  }}
                                  className="p-1 text-text-muted hover:text-status-error"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                            ))}
                            <button
                              onClick={() => {
                                const updated = [...(column.constraints || []), { type: 'regex', value: '', error_message: '' }]
                                updateColumn(column.id, { constraints: updated })
                              }}
                              className="text-xs text-primary hover:underline"
                            >
                              + Add constraint
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                      {config.columns.length === 0 && (
                        <div className="p-6 text-center">
                          <Database className="w-8 h-8 text-text-muted mx-auto mb-2" />
                          <p className="text-sm text-text-muted mb-3">No columns defined yet</p>
                          <button onClick={addColumn} className="btn-secondary text-sm">
                            <Plus className="w-4 h-4 mr-1" />
                            Add First Column
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Column Type Reference - Compact */}
                  <div className="border-t border-border-subtle pt-4 mt-4">
                    <p className="text-xs text-text-muted mb-2">
                      <span className="font-medium text-text-secondary">Column Types: </span>
                      {COLUMN_TYPES.map((type, i) => (
                        <span key={type.id}>
                          {type.label}
                          {i < COLUMN_TYPES.length - 1 && ' \u00b7 '}
                        </span>
                      ))}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Seeds Tab */}
        {activeTab === 'seeds' && (
          <SeedsPanel
            config={config}
            onConfigChange={setConfig}
            onImportFromTraces={importFromTraces}
            onNavigateToCreate={() => setActiveTab('create')}
          />
        )}

        {/* Settings Tab */}
        {activeTab === 'settings' && (
          <SettingsPanel
            config={config}
            onConfigChange={setConfig}
          />
        )}

        {/* Jobs Tab - Unified view of both Factory and Synthetic jobs */}
        {activeTab === 'jobs' && (
          <div className="p-6 space-y-4">
            {jobs.length === 0 && syntheticJobs.length === 0 ? (
              <div className="card-elevated p-12 text-center">
                <RefreshCw className="w-12 h-12 text-text-muted mx-auto mb-4" />
                <h3 className="text-lg font-medium text-text-primary mb-2">No synthesis jobs yet</h3>
                <p className="text-text-muted mb-4">Run synthesis to generate training data</p>
                <button onClick={() => setActiveTab('create')} className="btn-primary">
                  <Wand2 className="w-4 h-4 mr-2" />
                  Go to Create
                </button>
              </div>
            ) : (
              <>
                {/* Synthetic Jobs */}
                {syntheticJobs.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wider">Synthetic Generation Jobs</h3>
                    {syntheticJobs.map(job => (
                      <div key={job.job_id} className="card-elevated p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {job.status === 'running' && <Loader2 className="w-5 h-5 animate-spin text-primary" />}
                            {job.status === 'completed' && <CheckCircle className="w-5 h-5 text-status-success" />}
                            {job.status === 'failed' && <AlertCircle className="w-5 h-5 text-status-error" />}
                            {job.status === 'queued' && <RefreshCw className="w-5 h-5 text-text-muted" />}
                            <div>
                              <div className="flex items-center gap-2">
                                <p className="font-medium text-text-primary">AI-Powered Generation</p>
                                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/20 text-primary">Synthetic</span>
                              </div>
                              <p className="text-sm text-text-muted font-mono">{job.job_id}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className={clsx(
                              'text-sm font-medium capitalize',
                              job.status === 'running' && 'text-primary',
                              job.status === 'completed' && 'text-status-success',
                              job.status === 'failed' && 'text-status-error',
                              job.status === 'queued' && 'text-text-muted'
                            )}>
                              {job.status}
                            </p>
                            {job.progress && job.progress.total > 0 && (
                              <p className="text-sm text-text-muted">
                                {job.progress.current} / {job.progress.total} examples
                              </p>
                            )}
                          </div>
                        </div>
                        {(job.status === 'running' || job.status === 'completed') && job.progress && job.progress.total > 0 && (
                          <div className="mt-3">
                            <div className="h-1.5 bg-background-tertiary rounded-full overflow-hidden">
                              <div
                                className="h-full bg-primary rounded-full transition-all"
                                style={{
                                  width: `${(job.progress.current / job.progress.total) * 100}%`
                                }}
                              />
                            </div>
                          </div>
                        )}
                        {job.output_dir && (
                          <div className="mt-3 pt-3 border-t border-border-subtle">
                            <p className="text-xs text-text-muted">Output: {job.output_dir}</p>
                          </div>
                        )}
                        {job.error && (
                          <div className="mt-3 pt-3 border-t border-border-subtle">
                            <p className="text-xs text-status-error">{job.error}</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Schema-based Factory Jobs */}
                {jobs.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-medium text-text-secondary uppercase tracking-wider">Schema-Based Synthesis Jobs</h3>
                    {jobs.map(job => (
                      <div key={job.id} className="card-elevated p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {job.status === 'running' && <Loader2 className="w-5 h-5 animate-spin text-primary" />}
                            {job.status === 'completed' && <CheckCircle className="w-5 h-5 text-status-success" />}
                            {job.status === 'failed' && <AlertCircle className="w-5 h-5 text-status-error" />}
                            {job.status === 'pending' && <RefreshCw className="w-5 h-5 text-text-muted" />}
                            <div>
                              <div className="flex items-center gap-2">
                                <p className="font-medium text-text-primary">
                                  {job.job_type === 'preview' ? 'Preview' : 'Full Synthesis'}
                                </p>
                                <span className="text-xs px-2 py-0.5 rounded-full bg-background-tertiary text-text-muted">Schema</span>
                              </div>
                              <p className="text-sm text-text-muted">{job.id}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className={clsx(
                              'text-sm font-medium capitalize',
                              job.status === 'running' && 'text-primary',
                              job.status === 'completed' && 'text-status-success',
                              job.status === 'failed' && 'text-status-error',
                              job.status === 'pending' && 'text-text-muted'
                            )}>
                              {job.status}
                            </p>
                            {job.examples_created !== undefined && (
                              <p className="text-sm text-text-muted">
                                {job.examples_created} examples
                                {job.valid_examples !== undefined && ` (${job.valid_examples} valid)`}
                              </p>
                            )}
                          </div>
                        </div>
                        {job.output_path && (
                          <div className="mt-3 pt-3 border-t border-border-subtle">
                            <p className="text-xs text-text-muted">Output: {job.output_path}</p>
                          </div>
                        )}
                        {job.error && (
                          <div className="mt-3 pt-3 border-t border-border-subtle">
                            <p className="text-xs text-status-error">{job.error}</p>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
