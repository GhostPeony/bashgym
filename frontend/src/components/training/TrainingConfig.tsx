import { useState, useEffect } from 'react'
import { X, FolderGit2, Database, Sparkles, Info, Cloud, Monitor, Shield, FileText, Server } from 'lucide-react'
import type { TrainingConfig as TrainingConfigType, DataSource } from '../../stores'
import type { TrainingStrategy } from '../../stores'
import { tracesApi, securityApi, providersApi, cascadeApi, RepoInfo, SecurityDatasetInfo, OllamaModel } from '../../services/api'
import { useTutorialComplete } from '../../hooks'
import { clsx } from 'clsx'
import { DeviceManager } from './DeviceManager'
import { useDeviceStore } from '../../stores/deviceStore'

type TrainingScope = 'all' | 'selected' | 'single'
type TrainingBackend = 'local' | 'remote_ssh' | 'nemo'

interface TrainingConfigProps {
  onClose: () => void
  onStart: (config: TrainingConfigType) => void
}

export function TrainingConfig({ onClose, onStart }: TrainingConfigProps) {
  const [availableRepos, setAvailableRepos] = useState<RepoInfo[]>([])
  const [trainingScope, setTrainingScope] = useState<TrainingScope>('all')
  const [selectedRepos, setSelectedRepos] = useState<string[]>([])
  const [trainingBackend, setTrainingBackend] = useState<TrainingBackend>('local')
  const [dataSource, setDataSource] = useState<DataSource>('traces')
  const [securityDatasets, setSecurityDatasets] = useState<SecurityDatasetInfo[]>([])
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([])
  const [customModelInput, setCustomModelInput] = useState('')
  const [showCustomModel, setShowCustomModel] = useState(false)
  const { complete: completeTutorialStep } = useTutorialComplete()
  const { defaultDeviceId, fetchDevices } = useDeviceStore()

  // Cascade RL state
  const [cascadeDomains, setCascadeDomains] = useState<string[]>([
    'file_operations', 'bash_commands', 'search_and_navigate', 'multi_step_reasoning'
  ])
  const [cascadeStepsPerStage, setCascadeStepsPerStage] = useState(200)
  const [cascadeMode, setCascadeMode] = useState<'simulate' | 'real'>('simulate')

  const [config, setConfig] = useState<TrainingConfigType>({
    strategy: 'sft',
    baseModel: 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    datasetPath: '',  // Empty = auto-generate from gold traces
    epochs: 3,
    batchSize: 1,  // Reduced for 12GB VRAM (uses gradient accumulation)
    learningRate: 2e-5,
    warmupRatio: 0.1,
    gradientAccumulationSteps: 8,
    maxSeqLength: 2048,
    saveSteps: 100,
    // LoRA defaults
    loraRank: 16,
    loraAlpha: 32,
    loraDropout: 0.05,
    load4Bit: true,  // QLoRA enabled by default
    // DPO defaults
    dpoBeta: 0.1,
    // GRPO defaults
    grpoNumGenerations: 4,
    grpoTemperature: 0.7,
    // KD defaults
    teacherModel: 'meta-llama/Llama-3.1-70B-Instruct',
    teacherTemperature: 2.0,
    distillationAlpha: 0.5,
    // Security dataset defaults
    securityDatasetType: '',
    securityDatasetPath: '',
    securityConversionMode: 'direct' as const,
    securityMaxSamples: undefined,
    securityBalanceClasses: true
  })

  // Fetch available repos, security datasets, Ollama models, and devices on mount
  useEffect(() => {
    tracesApi.listRepos().then((result) => {
      if (result.ok && result.data) {
        setAvailableRepos(result.data)
      }
    })
    securityApi.listDatasets().then((result) => {
      if (result.ok && result.data) {
        setSecurityDatasets(result.data)
      }
    })
    providersApi.getOllamaModels().then((result) => {
      if (result.ok && result.data?.models) {
        setOllamaModels(result.data.models)
      }
    })
    fetchDevices()
  }, [fetchDevices])

  // Sync deviceId when defaultDeviceId changes or backend switches to remote
  useEffect(() => {
    if (trainingBackend === 'remote_ssh' && defaultDeviceId) {
      setConfig(prev => ({ ...prev, deviceId: defaultDeviceId }))
    }
  }, [defaultDeviceId, trainingBackend])

  const strategies: { value: TrainingStrategy; label: string; description: string }[] = [
    { value: 'sft', label: 'SFT', description: 'Supervised Fine-Tuning' },
    { value: 'dpo', label: 'DPO', description: 'Direct Preference Optimization' },
    { value: 'grpo', label: 'GRPO', description: 'Group Relative Policy Optimization' },
    { value: 'distillation', label: 'KD', description: 'Knowledge Distillation' },
    { value: 'cascade', label: 'Cascade RL', description: 'Domain-Staged GRPO' },
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Cascade RL uses its own API endpoint
    if (config.strategy === 'cascade') {
      try {
        await cascadeApi.start({
          domains: cascadeDomains,
          baseModel: config.baseModel,
          datasetPath: config.datasetPath || 'data/gold_traces',
          trainStepsPerStage: cascadeStepsPerStage,
          grpoNumGenerations: config.grpoNumGenerations ?? 4,
          grpoTemperature: config.grpoTemperature ?? 0.7,
          learningRate: config.learningRate,
          loraR: config.loraRank ?? 16,
          loraAlpha: config.loraAlpha ?? 32,
          load4Bit: config.load4Bit ?? true,
          useRemoteSsh: config.useRemoteSSH ?? false,
          mode: cascadeMode,
        })
        completeTutorialStep('start_training')
        onClose()
      } catch (err) {
        console.error('Failed to start cascade training:', err)
      }
      return
    }

    // Include selected repos based on training scope
    const reposToUse = trainingScope === 'all' ? [] : selectedRepos
    onStart({ ...config, dataSource, selectedRepos: reposToUse })
    completeTutorialStep('start_training')
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50" style={{ backgroundColor: 'rgba(27, 32, 64, 0.5)' }}>
      <div className="card bg-background-card w-full max-w-lg max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <h2 className="font-brand text-xl text-text-primary">Training Configuration</h2>
          <button
            onClick={onClose}
            className="btn-icon w-8 h-8 flex items-center justify-center"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Data Source */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Data Source
            </label>
            <div className="grid grid-cols-3 gap-2 mb-3">
              <button
                type="button"
                onClick={() => setDataSource('traces')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  dataSource === 'traces'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Database className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Gold Traces</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Default</span>
              </button>
              <button
                type="button"
                onClick={() => setDataSource('dataset_path')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  dataSource === 'dataset_path'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <FileText className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Custom JSONL</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">User-provided</span>
              </button>
              <button
                type="button"
                onClick={() => setDataSource('security_dataset')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  dataSource === 'security_dataset'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Shield className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Security</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Public datasets</span>
              </button>
            </div>

            {/* Security Dataset Config */}
            {dataSource === 'security_dataset' && (
              <div className="card p-3 space-y-3">
                <div>
                  <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Dataset Type</label>
                  <select
                    value={config.securityDatasetType}
                    onChange={(e) => setConfig({ ...config, securityDatasetType: e.target.value })}
                    className="input w-full"
                  >
                    <option value="">Select a dataset...</option>
                    {securityDatasets.map((ds) => (
                      <option key={ds.dataset_type} value={ds.dataset_type}>
                        {ds.name} ({ds.domain})
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Input File Path</label>
                  <input
                    type="text"
                    value={config.securityDatasetPath}
                    onChange={(e) => setConfig({ ...config, securityDatasetPath: e.target.value })}
                    className="input w-full"
                    placeholder="/path/to/dataset.json"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Mode</label>
                    <select
                      value={config.securityConversionMode}
                      onChange={(e) => setConfig({ ...config, securityConversionMode: e.target.value as 'direct' | 'enriched' })}
                      className="input w-full"
                    >
                      <option value="direct">Direct (fast, no API)</option>
                      <option value="enriched">Enriched (LLM reasoning)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Max Samples</label>
                    <input
                      type="number"
                      value={config.securityMaxSamples ?? ''}
                      onChange={(e) => setConfig({ ...config, securityMaxSamples: e.target.value ? parseInt(e.target.value) : undefined })}
                      className="input w-full"
                      placeholder="All"
                      min={10}
                    />
                  </div>
                </div>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.securityBalanceClasses ?? true}
                    onChange={(e) => setConfig({ ...config, securityBalanceClasses: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">Balance malicious/benign classes</span>
                </label>
              </div>
            )}

            {/* Data source info */}
            <div className="card p-3 mt-3 flex items-start gap-2 border-l-4 border-l-accent">
              <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
              <p className="font-mono text-xs text-text-secondary">
                {dataSource === 'traces' && 'Auto-generate training data from your collected gold traces.'}
                {dataSource === 'dataset_path' && 'Provide a pre-formatted JSONL file for training.'}
                {dataSource === 'security_dataset' && 'Ingest a public security dataset (EMBER, PhishTank, etc.) directly into training examples.'}
              </p>
            </div>
          </div>

          {/* Training Scope - only shown for trace-based training */}
          {dataSource === 'traces' && <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Training Data Scope
            </label>
            <div className="grid grid-cols-3 gap-2 mb-3">
              <button
                type="button"
                onClick={() => setTrainingScope('all')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingScope === 'all'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Sparkles className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Generalist</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">All repos</span>
              </button>
              <button
                type="button"
                onClick={() => setTrainingScope('selected')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingScope === 'selected'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Database className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Mixed</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Select repos</span>
              </button>
              <button
                type="button"
                onClick={() => setTrainingScope('single')}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingScope === 'single'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <FolderGit2 className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Specialist</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Single repo</span>
              </button>
            </div>

            {/* Repo Selection */}
            {trainingScope !== 'all' && (
              <div className="card p-3">
                <p className="font-mono text-xs uppercase tracking-widest text-text-muted mb-2">
                  {trainingScope === 'single' ? 'Select one repository:' : 'Select repositories:'}
                </p>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {availableRepos.length === 0 ? (
                    <p className="text-sm text-text-muted py-2">No traces collected yet. Work on some tasks first!</p>
                  ) : (
                    availableRepos.map((repo) => (
                      <label
                        key={repo.name}
                        className={clsx(
                          'flex items-center gap-2 p-2 cursor-pointer hover-press transition-press',
                          selectedRepos.includes(repo.name) && 'bg-accent-light'
                        )}
                      >
                        <input
                          type={trainingScope === 'single' ? 'radio' : 'checkbox'}
                          name="repo"
                          checked={selectedRepos.includes(repo.name)}
                          onChange={() => {
                            if (trainingScope === 'single') {
                              setSelectedRepos([repo.name])
                            } else {
                              setSelectedRepos(prev =>
                                prev.includes(repo.name)
                                  ? prev.filter(r => r !== repo.name)
                                  : [...prev, repo.name]
                              )
                            }
                          }}
                          className="accent-primary"
                        />
                        <FolderGit2 className="w-4 h-4 text-text-muted" />
                        <span className="text-sm text-text-primary flex-1">{repo.name}</span>
                        <span className="font-mono text-xs text-text-muted">{repo.trace_count} traces</span>
                      </label>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Info box */}
            <div className="card p-3 mt-3 flex items-start gap-2 border-l-4 border-l-accent">
              <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
              <p className="font-mono text-xs text-text-secondary">
                {trainingScope === 'all' && 'Train on all collected traces for a versatile generalist agent.'}
                {trainingScope === 'selected' && 'Train on selected repos for focused expertise across projects.'}
                {trainingScope === 'single' && 'Train on a single repo for a highly specialized codebase expert.'}
              </p>
            </div>
          </div>}

          {/* Training Backend */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Training Backend
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('local')
                  setConfig({ ...config, useNemoGym: false, useRemoteSSH: false })
                }}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingBackend === 'local'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Monitor className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Local</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Your GPU (Unsloth)</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('remote_ssh')
                  setConfig({ ...config, useNemoGym: false, useRemoteSSH: true, deviceId: defaultDeviceId || undefined })
                }}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingBackend === 'remote_ssh'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Server className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Remote Device</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">SSH Training</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('nemo')
                  setConfig({ ...config, useNemoGym: true, useRemoteSSH: false })
                }}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingBackend === 'nemo'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Cloud className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">NeMo Cloud</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">NVIDIA Microservices</span>
              </button>
            </div>

            {trainingBackend === 'remote_ssh' && (
              <div className="mt-4">
                <DeviceManager />
              </div>
            )}

            {/* Backend info */}
            <div className="card p-3 mt-3 flex items-start gap-2 border-l-4 border-l-accent">
              <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
              <p className="font-mono text-xs text-text-secondary">
                {trainingBackend === 'local'
                  ? 'Train locally using your GPU with Unsloth. Requires CUDA-capable GPU.'
                  : trainingBackend === 'remote_ssh'
                    ? 'Select or add a remote SSH device for training.'
                    : 'Train using NVIDIA NeMo Microservices for scalable cloud training.'}
              </p>
            </div>
          </div>

          {/* Strategy Selection */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Training Strategy
            </label>
            <div className="grid grid-cols-3 gap-2">
              {strategies.map((s) => (
                <button
                  key={s.value}
                  type="button"
                  onClick={() => setConfig({ ...config, strategy: s.value })}
                  className={clsx(
                    'card px-3 py-2 text-left transition-press',
                    config.strategy === s.value
                      ? 'border-accent bg-accent-light text-accent-dark'
                      : 'text-text-secondary'
                  )}
                >
                  <span className="block font-mono text-xs font-bold uppercase truncate">{s.label}</span>
                  <span className="block font-mono text-[10px] mt-0.5 text-text-muted truncate">{s.description}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Base Model */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Base Model
            </label>
            <select
              value={showCustomModel ? '__custom__' : config.baseModel}
              onChange={(e) => {
                if (e.target.value === '__custom__') {
                  setShowCustomModel(true)
                } else {
                  setShowCustomModel(false)
                  setConfig({ ...config, baseModel: e.target.value })
                }
              }}
              className="input w-full"
            >
              <optgroup label="Qwen 3.5 (Feb 2026)">
                <option value="Qwen/Qwen3.5-0.8B">Qwen3.5-0.8B (dense)</option>
                <option value="Qwen/Qwen3.5-4B">Qwen3.5-4B (dense)</option>
                <option value="Qwen/Qwen3.5-9B">Qwen3.5-9B (dense)</option>
                <option value="Qwen/Qwen3.5-27B">Qwen3.5-27B (dense)</option>
                <option value="Qwen/Qwen3.5-35B-A3B">Qwen3.5-35B-A3B (MoE, ~74GB VRAM)</option>
              </optgroup>
              <optgroup label="Qwen 2.5 Coder">
                <option value="Qwen/Qwen2.5-Coder-1.5B-Instruct">Qwen2.5-Coder-1.5B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-3B-Instruct">Qwen2.5-Coder-3B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-7B-Instruct">Qwen2.5-Coder-7B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-14B-Instruct">Qwen2.5-Coder-14B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-32B-Instruct">Qwen2.5-Coder-32B-Instruct</option>
              </optgroup>
              <optgroup label="Llama">
                <option value="meta-llama/Llama-3.2-1B-Instruct">Llama-3.2-1B-Instruct</option>
                <option value="meta-llama/Llama-3.2-3B-Instruct">Llama-3.2-3B-Instruct</option>
                <option value="meta-llama/Llama-3.1-8B-Instruct">Llama-3.1-8B-Instruct</option>
              </optgroup>
              <optgroup label="DeepSeek">
                <option value="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct">DeepSeek-Coder-V2-Lite</option>
                <option value="deepseek-ai/deepseek-coder-6.7b-instruct">DeepSeek-Coder-6.7B</option>
              </optgroup>
              <optgroup label="Nemotron (NVIDIA)">
                <option value="nvidia/Nemotron-Cascade-2-30B-A3B">Nemotron-Cascade-2-30B-A3B (MoE, DGX Spark)</option>
                <option value="nvidia/Nemotron-3-Nano-4B-Instruct">Nemotron-3-Nano-4B</option>
                <option value="nvidia/Nemotron-Mini-4B-Instruct">Nemotron-Mini-4B</option>
              </optgroup>
              <optgroup label="Other">
                <option value="mistralai/Mistral-7B-Instruct-v0.3">Mistral-7B-Instruct</option>
                <option value="google/gemma-2-9b-it">Gemma-2-9B-IT</option>
              </optgroup>
              <option value="__custom__">Custom model...</option>
            </select>

            {/* Custom model input */}
            {showCustomModel && (
              <div className="flex gap-2 mt-2">
                <input
                  type="text"
                  value={customModelInput}
                  onChange={(e) => {
                    setCustomModelInput(e.target.value)
                    setConfig({ ...config, baseModel: e.target.value })
                  }}
                  className="input flex-1"
                  placeholder="org/model-name (HuggingFace ID or Ollama tag)"
                  autoFocus
                />
              </div>
            )}

            <p className="font-mono text-xs text-text-muted mt-2">
              Select a HuggingFace model to fine-tune, or enter a custom model ID.
              {ollamaModels.length > 0 && ' After training, deploy to Ollama for inference on your DGX Spark.'}
            </p>
          </div>

          {/* Dataset Path - only for traces/custom JSONL modes */}
          {dataSource !== 'security_dataset' && (
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
                Dataset Path
              </label>
              <input
                type="text"
                value={config.datasetPath}
                onChange={(e) => setConfig({ ...config, datasetPath: e.target.value })}
                className="input w-full"
                placeholder={dataSource === 'dataset_path' ? '/path/to/train.jsonl' : 'Leave empty to auto-generate from traces'}
              />
            </div>
          )}

          {/* Hyperparameters Grid */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Hyperparameters
            </label>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Epochs</label>
                <input
                  type="number"
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                  className="input w-full"
                  min={1}
                  max={100}
                />
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Batch Size</label>
                <input
                  type="number"
                  value={config.batchSize}
                  onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                  className="input w-full"
                  min={1}
                  max={64}
                />
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Learning Rate</label>
                <input
                  type="text"
                  value={config.learningRate}
                  onChange={(e) =>
                    setConfig({ ...config, learningRate: parseFloat(e.target.value) })
                  }
                  className="input w-full"
                  placeholder="2e-5"
                />
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Warmup Ratio</label>
                <input
                  type="number"
                  value={config.warmupRatio}
                  onChange={(e) => setConfig({ ...config, warmupRatio: parseFloat(e.target.value) })}
                  className="input w-full"
                  min={0}
                  max={1}
                  step={0.05}
                />
                <p className="font-mono text-xs text-text-muted mt-1">Fraction of total steps for warmup (0-1)</p>
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Gradient Accumulation</label>
                <input
                  type="number"
                  value={config.gradientAccumulationSteps}
                  onChange={(e) => setConfig({ ...config, gradientAccumulationSteps: parseInt(e.target.value) })}
                  className="input w-full"
                  min={1}
                  max={128}
                />
                <p className="font-mono text-xs text-text-muted mt-1">Effective batch = batch_size x this</p>
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Save Steps</label>
                <input
                  type="number"
                  value={config.saveSteps}
                  onChange={(e) => setConfig({ ...config, saveSteps: parseInt(e.target.value) })}
                  className="input w-full"
                  min={10}
                />
                <p className="font-mono text-xs text-text-muted mt-1">Checkpoint every N steps</p>
              </div>
              <div className="col-span-2">
                <label className="block font-mono text-xs text-text-muted mb-2">Max Sequence Length</label>
                <input
                  type="number"
                  value={config.maxSeqLength}
                  onChange={(e) => setConfig({ ...config, maxSeqLength: parseInt(e.target.value) })}
                  className="input w-full"
                  min={512}
                  max={8192}
                  step={256}
                />
              </div>
            </div>
          </div>

          {/* LoRA Config */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              LoRA Configuration
            </label>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">LoRA Rank</label>
                <input
                  type="number"
                  value={config.loraRank}
                  onChange={(e) => setConfig({ ...config, loraRank: parseInt(e.target.value) })}
                  className="input w-full"
                  min={4}
                  max={128}
                />
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">LoRA Alpha</label>
                <input
                  type="number"
                  value={config.loraAlpha}
                  onChange={(e) => setConfig({ ...config, loraAlpha: parseInt(e.target.value) })}
                  className="input w-full"
                  min={4}
                  max={256}
                />
              </div>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">LoRA Dropout</label>
                <input
                  type="number"
                  value={config.loraDropout}
                  onChange={(e) => setConfig({ ...config, loraDropout: parseFloat(e.target.value) })}
                  className="input w-full"
                  min={0}
                  max={0.5}
                  step={0.01}
                />
              </div>
            </div>
            <label className="flex items-center gap-2 mt-3">
              <input
                type="checkbox"
                checked={config.load4Bit ?? true}
                onChange={(e) => setConfig({ ...config, load4Bit: e.target.checked })}
                className="accent-primary"
              />
              <span className="text-sm text-text-secondary">
                Enable 4-bit Quantization (QLoRA) - Reduces VRAM ~50%
              </span>
            </label>
          </div>

          {/* DPO Settings */}
          {config.strategy === 'dpo' && (
            <div className="card p-4 border-l-4 border-l-accent">
              <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
                DPO Settings
              </h3>
              <div>
                <label className="block font-mono text-xs text-text-muted mb-2">Beta (Divergence Penalty)</label>
                <input
                  type="number"
                  value={config.dpoBeta}
                  onChange={(e) => setConfig({ ...config, dpoBeta: parseFloat(e.target.value) })}
                  className="input w-full"
                  min={0.01}
                  max={1}
                  step={0.05}
                />
                <p className="font-mono text-xs text-text-muted mt-1">
                  Controls how much the model can diverge from the reference. Lower = more conservative.
                </p>
              </div>
            </div>
          )}

          {/* GRPO Settings */}
          {config.strategy === 'grpo' && (
            <div className="card p-4 border-l-4 border-l-accent">
              <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
                GRPO Settings
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block font-mono text-xs text-text-muted mb-2">Generations per Prompt</label>
                  <input
                    type="number"
                    value={config.grpoNumGenerations}
                    onChange={(e) => setConfig({ ...config, grpoNumGenerations: parseInt(e.target.value) })}
                    className="input w-full"
                    min={2}
                    max={16}
                  />
                  <p className="font-mono text-xs text-text-muted mt-1">
                    Number of completions sampled per prompt for reward comparison
                  </p>
                </div>
                <div>
                  <label className="block font-mono text-xs text-text-muted mb-2">Sampling Temperature</label>
                  <input
                    type="number"
                    value={config.grpoTemperature}
                    onChange={(e) => setConfig({ ...config, grpoTemperature: parseFloat(e.target.value) })}
                    className="input w-full"
                    min={0.1}
                    max={2}
                    step={0.1}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Knowledge Distillation Config */}
          {config.strategy === 'distillation' && (
            <div className="card p-4 border-l-4 border-l-accent">
              <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-accent" />
                Knowledge Distillation Settings
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block font-mono text-xs text-text-muted mb-2">Teacher Model</label>
                  <select
                    value={config.teacherModel}
                    onChange={(e) => setConfig({ ...config, teacherModel: e.target.value })}
                    className="input w-full"
                  >
                    <optgroup label="Large Language Models">
                      <option value="meta-llama/Llama-3.1-70B-Instruct">Llama-3.1-70B-Instruct</option>
                      <option value="meta-llama/Llama-3.1-405B-Instruct">Llama-3.1-405B-Instruct</option>
                      <option value="Qwen/Qwen2.5-72B-Instruct">Qwen2.5-72B-Instruct</option>
                    </optgroup>
                    <optgroup label="Code Models">
                      <option value="Qwen/Qwen2.5-Coder-32B-Instruct">Qwen2.5-Coder-32B-Instruct</option>
                      <option value="deepseek-ai/DeepSeek-Coder-V2-Instruct">DeepSeek-Coder-V2</option>
                    </optgroup>
                    <optgroup label="Anthropic Claude 4.5 (requires ANTHROPIC_API_KEY)">
                      <option value="claude-opus-4-5-20251101">Claude Opus 4.5 (Best)</option>
                      <option value="claude-sonnet-4-5-20250929">Claude Sonnet 4.5</option>
                      <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 (Fast)</option>
                    </optgroup>
                    <optgroup label="Anthropic Claude 4 (Legacy)">
                      <option value="claude-sonnet-4-20250514">Claude Sonnet 4</option>
                      <option value="claude-opus-4-20250514">Claude Opus 4</option>
                    </optgroup>
                  </select>
                  <p className="font-mono text-xs text-text-muted mt-1">
                    The larger model to distill knowledge from
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block font-mono text-xs text-text-muted mb-2">Temperature</label>
                    <input
                      type="number"
                      value={config.teacherTemperature}
                      onChange={(e) => setConfig({ ...config, teacherTemperature: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0.5}
                      max={10}
                      step={0.5}
                    />
                    <p className="font-mono text-xs text-text-muted mt-1">
                      Softmax temperature (higher = softer distribution)
                    </p>
                  </div>
                  <div>
                    <label className="block font-mono text-xs text-text-muted mb-2">Distillation Alpha</label>
                    <input
                      type="number"
                      value={config.distillationAlpha}
                      onChange={(e) => setConfig({ ...config, distillationAlpha: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="font-mono text-xs text-text-muted mt-1">
                      Weight for soft labels (0 = task loss only, 1 = distillation only)
                    </p>
                  </div>
                </div>

                <div className="card p-3 flex items-start gap-2 border-l-4 border-l-accent">
                  <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
                  <p className="font-mono text-xs text-text-secondary">
                    Knowledge distillation transfers capabilities from a large teacher model to your smaller student model.
                    The student learns to match the teacher's probability distributions, not just the correct answers.
                  </p>
                </div>
              </div>
            </div>
          )}
          {/* Cascade RL Config */}
          {config.strategy === 'cascade' && (
            <div className="card p-4 border-l-4 border-l-accent">
              <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
                Cascade RL Configuration
              </h3>

              {/* Domain selection */}
              <div className="mb-4">
                <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">Training Domains</label>
                <div className="space-y-2">
                  {[
                    { id: 'file_operations', label: 'File Operations', desc: 'Read, write, edit files', reward: 'syntax' },
                    { id: 'bash_commands', label: 'Bash Commands', desc: 'Shell execution', reward: 'execution' },
                    { id: 'search_and_navigate', label: 'Search & Navigate', desc: 'Grep, glob exploration', reward: 'execution' },
                    { id: 'multi_step_reasoning', label: 'Multi-Step Reasoning', desc: 'Planning, multi-tool chains', reward: 'verification' },
                  ].map(domain => (
                    <label key={domain.id} className="flex items-start gap-3 p-2 card cursor-pointer hover-press transition-press">
                      <input
                        type="checkbox"
                        checked={cascadeDomains.includes(domain.id)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setCascadeDomains([...cascadeDomains, domain.id])
                          } else {
                            setCascadeDomains(cascadeDomains.filter(d => d !== domain.id))
                          }
                        }}
                        className="accent-primary mt-1"
                      />
                      <div>
                        <span className="font-mono text-sm">{domain.label}</span>
                        <span className="text-xs text-text-muted ml-2">({domain.reward} reward)</span>
                        <p className="text-xs text-text-secondary">{domain.desc}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Steps per stage */}
              <div className="mb-4">
                <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-1">Steps Per Stage</label>
                <input
                  type="number"
                  value={cascadeStepsPerStage}
                  onChange={e => setCascadeStepsPerStage(Number(e.target.value))}
                  min={10}
                  max={5000}
                  className="input w-full"
                />
                <p className="font-mono text-xs text-text-muted mt-1">
                  GRPO training steps per domain stage (10-5000)
                </p>
              </div>

              {/* Mode toggle */}
              <div className="mb-4">
                <label className="block font-mono text-xs uppercase tracking-widest text-text-muted mb-2">Mode</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setCascadeMode('simulate')}
                    className={clsx(
                      'card px-3 py-1 font-mono text-xs transition-press',
                      cascadeMode === 'simulate'
                        ? 'border-accent bg-accent-light text-accent-dark'
                        : 'text-text-secondary'
                    )}
                  >
                    Simulate
                  </button>
                  <button
                    type="button"
                    onClick={() => setCascadeMode('real')}
                    className={clsx(
                      'card px-3 py-1 font-mono text-xs transition-press',
                      cascadeMode === 'real'
                        ? 'border-accent bg-accent-light text-accent-dark'
                        : 'text-text-secondary'
                    )}
                  >
                    Real Training
                  </button>
                </div>
              </div>

              <div className="card p-3 flex items-start gap-2 border-l-4 border-l-accent">
                <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
                <p className="font-mono text-xs text-text-secondary">
                  Cascade RL trains domain-by-domain using GRPO, with each stage building on the previous checkpoint.
                  After all stages complete, use MOPD distillation to merge domain experts into a unified model.
                </p>
              </div>
            </div>
          )}

          {/* Auto-deploy to Ollama */}
          <div className="mt-4 p-3 card border-l-4 border-l-accent">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.autoDeployOllama ?? false}
                onChange={(e) => setConfig({ ...config, autoDeployOllama: e.target.checked })}
                className="accent-primary"
              />
              <span className="text-sm text-text-secondary">Auto-deploy to Ollama after training</span>
            </div>
            {config.autoDeployOllama && (
              <div className="mt-2">
                <input
                  type="text"
                  value={config.ollamaModelName ?? ''}
                  onChange={(e) => setConfig({ ...config, ollamaModelName: e.target.value })}
                  className="input w-full"
                  placeholder="Model name (auto-generated if empty)"
                />
              </div>
            )}
          </div>

          {/* Auto-push to HuggingFace */}
          <div className="mt-4 p-3 card border-l-4 border-l-accent">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={config.autoPushHF ?? false}
                onChange={(e) => setConfig({ ...config, autoPushHF: e.target.checked })}
                className="accent-primary"
              />
              <span className="text-sm text-text-secondary">Auto-push to HuggingFace Hub after training</span>
            </div>
            {config.autoPushHF && (
              <div className="mt-2 space-y-2">
                <input
                  type="text"
                  value={config.hfRepoName ?? ''}
                  onChange={(e) => setConfig({ ...config, hfRepoName: e.target.value })}
                  className="input w-full"
                  placeholder="Repo name (auto-generated if empty)"
                />
                <div className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.hfPrivate ?? true}
                    onChange={(e) => setConfig({ ...config, hfPrivate: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-xs text-text-muted">Private repository</span>
                </div>
              </div>
            )}
          </div>
        </form>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-border">
          <button type="button" onClick={onClose} className="btn-secondary">
            Cancel
          </button>
          <button onClick={handleSubmit} className="btn-primary">
            Start Training
          </button>
        </div>
      </div>
    </div>
  )
}
