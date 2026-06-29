import { useState, useEffect } from 'react'
import { X, FolderGit2, Database, Sparkles, Info, Cloud, Monitor, Shield, FileText, Server, AlertCircle, Boxes, BookOpen } from 'lucide-react'
import type { TrainingConfig as TrainingConfigType, DataSource, TrainingProfile } from '../../stores'
import type { TrainingStrategy } from '../../stores'
import { tracesApi, securityApi, providersApi, trainingApi, RepoInfo, SecurityDatasetInfo, OllamaModel } from '../../services/api'
import { useTutorialComplete } from '../../hooks'
import { clsx } from 'clsx'
import { DeviceManager } from './DeviceManager'
import { useDeviceStore } from '../../stores/deviceStore'
import { useCascadeStore } from '../../stores/cascadeStore'
import { BaseModelSelect } from '../common/BaseModelSelect'
import { ModelSelect } from '../common/ModelSelect'

type TrainingScope = 'all' | 'selected' | 'single'
type TrainingBackend = 'local' | 'remote_ssh' | 'nemo' | 'managed'

interface TrainingConfigProps {
  onClose: () => void
  onStart: (config: TrainingConfigType) => void
  onOpenGuides?: () => void
}

const TERMINAL_RL_TMAX_DEFAULTS = {
  grpoNumGenerations: 32,
  grpoGroupSize: 32,
  promptsPerRolloutBatch: 8,
  maxToolCallsPerEpisode: 64,
  grpoLossType: 'dapo',
  tokenLevelLoss: true,
  filterZeroStdGroups: true,
  activeSampling: true,
  lmHeadFp32: true,
  interleavedThinking: true,
  sftWarmStartPolicy: 'weak_models_only',
  dppoBackend: 'auto',
  dppoDivergence: 'binary_tv',
  dppoBinaryTvThreshold: 0.15,
  dppoBinaryKlThreshold: 0.05,
} satisfies Partial<TrainingConfigType>

const WORLD_MODEL_DEFAULTS = {
  echoEnabled: false,
  echoAuxLambda: 0.05,
  rwmlEnabled: false,
  rwmlDistanceThreshold: 0.2,
  rwmlEasyPassRateThreshold: 0.8,
  rwmlEasyKeepProbability: 0.1,
  rwmlHistoryWindow: 4,
  rwmlEmbeddingModel: '',
  rwmlKlBeta: 0,
} satisfies Partial<TrainingConfigType>

const RWML_EMBEDDING_MODEL_OPTIONS = [
  { value: 'qwen3-embedding', label: 'Qwen3 Embedding' },
  { value: 'nomic-embed-text', label: 'Nomic Embed Text' },
  { value: 'mxbai-embed-large', label: 'MxBai Embed Large' },
  { value: 'text-embedding-3-large', label: 'OpenAI Text Embedding 3 Large' },
  { value: 'text-embedding-3-small', label: 'OpenAI Text Embedding 3 Small' },
]

const TRAINING_GUIDANCE: Record<TrainingStrategy, {
  fit: string
  start: string
  metrics: string
  caution: string
}> = {
  sft: {
    fit: 'Imitation from gold traces; best for teaching trace format, command style, and project conventions.',
    start: 'Local QLoRA: LR 2e-4 for adapters, 1-3 epochs, LoRA r=16 or 32, alpha=r or 2r, max length 2k-8k.',
    metrics: 'Watch train/eval loss, grad norm, token count, and whether tool output or final actions are truncated.',
    caution: 'If loss falls but held-out pass@k is flat, improve trace quality or masking before adding RL.',
  },
  dpo: {
    fit: 'Preference learning from chosen/rejected answers for the same prompt.',
    start: 'Use beta=0.1, adapter LR around 1e-5, and enough max length to keep both preference completions intact.',
    metrics: 'Watch chosen reward, rejected reward, reward margin, preference accuracy, and chosen/rejected logprobs.',
    caution: 'Accuracy stuck near 0.5 usually means the preference pairs are weak or mislabeled.',
  },
  grpo: {
    fit: 'Verifier-guided RL: sample multiple attempts, score them, and learn from differences inside each group.',
    start: 'Terminal RL: DAPO, group size 8-32, active sampling, zero-std filtering, token-level loss, FP32 LM head.',
    metrics: 'Watch reward, reward_std, frac_reward_zero_std, KL, pass@1/pass@k, timeouts, tamper rate, and tool-call count.',
    caution: 'Use RL when the base model sometimes succeeds. If every group is all pass or all fail, fix data or task difficulty first.',
  },
  distillation: {
    fit: 'Teacher-guided transfer from a larger model into a smaller student.',
    start: 'Use this for hard tasks where pass@k is zero or when a stronger teacher can create cleaner reasoning traces.',
    metrics: 'Watch student loss, held-out pass@k, teacher agreement, and whether distilled traces preserve safe tool use.',
    caution: 'Distillation can copy teacher mistakes; keep verifier and holdout checks in the loop.',
  },
  cascade: {
    fit: 'Domain-staged GRPO that trains easier terminal skills before harder multi-step tasks.',
    start: 'Begin in simulate mode, then run short real stages with verifier-backed domains and stable GRPO settings.',
    metrics: 'Watch per-domain pass@k, forgetting between stages, verifier errors, and final unified release-gate evidence.',
    caution: 'A weak early stage can poison later stages; gate each domain before merging.',
  },
}

function FieldLabel({ children, hint }: { children: string; hint: string }) {
  return (
    <label className="flex items-center gap-1.5 font-mono text-xs text-text-muted mb-2">
      <span>{children}</span>
      <span title={hint} className="inline-flex cursor-help">
        <Info
          className="w-3.5 h-3.5 text-accent flex-shrink-0"
          aria-label={`${children} guidance`}
        />
      </span>
    </label>
  )
}

function GuidanceNote({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex items-start gap-2 border-l-4 border-l-accent border-brutal border-border rounded-brutal bg-background-card p-3">
      <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
      <div>
        <p className="font-mono text-[10px] uppercase tracking-widest text-text-primary">{title}</p>
        <p className="font-mono text-xs text-text-secondary mt-1">{body}</p>
      </div>
    </div>
  )
}

function SectionKicker({ title, body }: { title: string; body: string }) {
  return (
    <div className="border-b border-border pb-3">
      <p className="font-mono text-[10px] uppercase tracking-widest text-accent mb-1">{title}</p>
      <p className="text-sm leading-relaxed text-text-secondary">{body}</p>
    </div>
  )
}

export function TrainingConfig({ onClose, onStart, onOpenGuides }: TrainingConfigProps) {
  const [availableRepos, setAvailableRepos] = useState<RepoInfo[]>([])
  const [trainingScope, setTrainingScope] = useState<TrainingScope>('all')
  const [selectedRepos, setSelectedRepos] = useState<string[]>([])
  const [trainingBackend, setTrainingBackend] = useState<TrainingBackend>('local')
  const [managedPlatform, setManagedPlatform] = useState<'together' | 'openai' | 'fireworks'>('together')
  const [managedAccountId, setManagedAccountId] = useState('')
  const [managedMsg, setManagedMsg] = useState<string | null>(null)
  const [dataSource, setDataSource] = useState<DataSource>('traces')
  const [securityDatasets, setSecurityDatasets] = useState<SecurityDatasetInfo[]>([])
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([])
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState(false)
  const { complete: completeTutorialStep } = useTutorialComplete()
  const { defaultDeviceId, fetchDevices } = useDeviceStore()

  // Cascade RL state
  const [cascadeDomains, setCascadeDomains] = useState<string[]>([
    'file_operations', 'bash_commands', 'search_and_navigate', 'multi_step_reasoning'
  ])
  const [cascadeStepsPerStage, setCascadeStepsPerStage] = useState(200)
  const [cascadeMode, setCascadeMode] = useState<'simulate' | 'real'>('simulate')
  const cascadeError = useCascadeStore((s) => s.error)
  const startCascade = useCascadeStore((s) => s.startCascade)
  const setPreflightError = useCascadeStore((s) => s.setPreflightError)

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
    ...WORLD_MODEL_DEFAULTS,
    useLiger: false,
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
  const strategyGuide = TRAINING_GUIDANCE[config.strategy]
  const effectiveBatch = config.batchSize * config.gradientAccumulationSteps
  const grpoGroupSize = config.grpoGroupSize ?? config.grpoNumGenerations ?? 4
  const grpoAttemptsPerBatch = (config.promptsPerRolloutBatch ?? 8) * grpoGroupSize

  const applyTrainingProfile = (profile: TrainingProfile) => {
    if (profile === 'terminal_rl_tmax_like') {
      setConfig((prev) => ({
        ...prev,
        trainingProfile: profile,
        ...TERMINAL_RL_TMAX_DEFAULTS,
      }))
      return
    }

    setConfig((prev) => ({
      ...prev,
      trainingProfile: 'default',
      grpoGroupSize: prev.grpoNumGenerations ?? 4,
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
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Cascade RL uses its own API endpoint + store-driven error surfacing
    if (config.strategy === 'cascade') {
      const result = await startCascade({
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
      if (result.ok) {
        completeTutorialStep('start_training')
        onClose()
      }
      // If !ok, cascadeStore.error is set; banner renders inline below.
      return
    }

    // Managed fine-tune (Together/OpenAI) submits a hosted job instead of a local run
    if (trainingBackend === 'managed') {
      setManagedMsg('Submitting…')
      const result = await trainingApi.managedSubmit({
        platform: managedPlatform,
        base_model: config.baseModel,
        dataset_path: config.datasetPath || 'data/gold_traces/train.jsonl',
        n_epochs: config.epochs,
        account_id: managedPlatform === 'fireworks' ? managedAccountId || undefined : undefined,
      })
      if (result.ok && result.data?.job_id) {
        setManagedMsg(`Submitted ${result.data.backend} job ${result.data.job_id} — ${result.data.status}`)
      } else {
        setManagedMsg(result.data?.error || result.error || 'Submit failed')
      }
      completeTutorialStep('start_training')
      return
    }

    // Include selected repos based on training scope
    const reposToUse = trainingScope === 'all' ? [] : selectedRepos
    onStart({ ...config, dataSource, selectedRepos: reposToUse })
    completeTutorialStep('start_training')
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 p-4" style={{ backgroundColor: 'rgba(27, 32, 64, 0.5)' }}>
      <div className="card bg-background-card w-[min(96vw,1180px)] max-h-[92vh] overflow-hidden">
        {/* Header */}
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between px-5 md:px-6 py-4 border-b border-border">
          <div>
            <h2 className="font-brand text-xl text-text-primary">Training Configuration</h2>
            <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted mt-1">
              Setup, objective, and launch controls
            </p>
          </div>
          <div className="flex items-center gap-2">
            {onOpenGuides && (
              <button
                type="button"
                onClick={onOpenGuides}
                className="btn-secondary flex items-center gap-2 text-xs"
                title="Open training guides"
              >
                <BookOpen className="w-4 h-4" />
                Guides
              </button>
            )}
            <button
              onClick={onClose}
              className="btn-icon w-8 h-8 flex items-center justify-center"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="overflow-y-auto max-h-[calc(92vh-148px)] p-5 md:p-6">
          <div className="grid grid-cols-1 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.55fr)] gap-5 xl:gap-6">
            <div className="space-y-5 xl:self-start">
              <SectionKicker
                title="Run setup"
                body="Choose the data source, compute surface, and training objective before tuning individual parameters."
              />
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
            <div className="grid grid-cols-2 gap-2">
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
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('managed')
                  setConfig({ ...config, useNemoGym: false, useRemoteSSH: false })
                }}
                className={clsx(
                  'card p-3 text-center transition-press',
                  trainingBackend === 'managed'
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'text-text-secondary'
                )}
              >
                <Boxes className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-mono text-xs font-bold uppercase">Managed</span>
                <span className="block font-mono text-xs mt-1 text-text-muted">Together / OpenAI</span>
              </button>
            </div>

            {trainingBackend === 'remote_ssh' && (
              <div className="mt-4">
                <DeviceManager />
              </div>
            )}

            {trainingBackend === 'managed' && (
              <div className="mt-4 card p-3 border-l-4 border-l-accent space-y-2">
                <label className="block font-mono text-xs text-text-muted mb-1">Platform</label>
                <select
                  value={managedPlatform}
                  onChange={(e) => setManagedPlatform(e.target.value as 'together' | 'openai' | 'fireworks')}
                  className="input w-full"
                >
                  <option value="together">Together</option>
                  <option value="openai">OpenAI</option>
                  <option value="fireworks">Fireworks</option>
                </select>
                {managedPlatform === 'fireworks' && (
                  <div>
                    <label className="block font-mono text-xs text-text-muted mb-1">Fireworks account ID</label>
                    <input
                      value={managedAccountId}
                      onChange={(e) => setManagedAccountId(e.target.value)}
                      placeholder="your-account-id"
                      className="input w-full"
                    />
                  </div>
                )}
                <p className="font-mono text-xs text-text-muted">
                  Submits a hosted fine-tune job (no local GPU). Connect the platform in
                  Settings → Models first so the API key is reused.
                </p>
                {managedMsg && (
                  <p className="font-mono text-xs text-text-secondary">{managedMsg}</p>
                )}
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
                    : trainingBackend === 'managed'
                      ? 'Submit a hosted fine-tune job to Together or OpenAI — no local GPU.'
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

          {/* Training setup guide */}
          <div className="border-brutal border-border rounded-brutal p-4 bg-background-secondary">
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-4 h-4 text-accent" />
              <h3 className="font-mono text-xs uppercase tracking-widest text-text-secondary">
                Setup Guide
              </h3>
            </div>
            <div className="space-y-3">
              <p className="font-mono text-xs text-text-secondary">{strategyGuide.fit}</p>
              <div className="grid grid-cols-1 lg:grid-cols-3 xl:grid-cols-1 gap-3">
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                    Start Here
                  </p>
                  <p className="font-mono text-xs text-text-secondary">{strategyGuide.start}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                    Watch
                  </p>
                  <p className="font-mono text-xs text-text-secondary">{strategyGuide.metrics}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted mb-1">
                    Check First
                  </p>
                  <p className="font-mono text-xs text-text-secondary">{strategyGuide.caution}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-2 gap-2 pt-2 border-t border-border">
                <div>
                  <p
                    className="font-mono text-[10px] uppercase tracking-widest text-text-muted"
                    title="Optimizer batch = per-device batch size multiplied by gradient accumulation steps."
                  >
                    Effective Batch
                  </p>
                  <p className="font-mono text-sm text-text-primary">{effectiveBatch}</p>
                </div>
                <div>
                  <p
                    className="font-mono text-[10px] uppercase tracking-widest text-text-muted"
                    title="Keep enough context for prompt, tool calls, observations, and final answer. Truncation can hide failures."
                  >
                    Max Length
                  </p>
                  <p className="font-mono text-sm text-text-primary">{config.maxSeqLength}</p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">LoRA</p>
                  <p className="font-mono text-sm text-text-primary">
                    r{config.loraRank}/a{config.loraAlpha}
                  </p>
                </div>
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">RL Group</p>
                  <p className="font-mono text-sm text-text-primary">{grpoGroupSize}</p>
                </div>
              </div>
              {config.strategy === 'sft' && (
                <GuidanceNote
                  title="SFT gate"
                  body="Before moving to RL, confirm chat-template rendering, assistant-only loss masking, and held-out pass@k. Falling loss alone is not enough."
                />
              )}
              {config.strategy === 'dpo' && (
                <GuidanceNote
                  title="DPO gate"
                  body="Use pairs with the same prompt and a real chosen/rejected contrast. If preference accuracy stays near 0.5, repair the pair quality first."
                />
              )}
              {config.strategy === 'grpo' && (
                <GuidanceNote
                  title="GRPO contrast"
                  body={`This setup samples ${grpoAttemptsPerBatch} attempts per rollout batch. Watch reward_std and frac_reward_zero_std before increasing steps.`}
                />
              )}
              {config.strategy === 'cascade' && (
                <GuidanceNote
                  title="Stage gate"
                  body="Gate each domain with pass@k, verifier integrity, and forgetting checks before the next stage inherits that checkpoint."
                />
              )}
            </div>
          </div>
            </div>

            <div className="space-y-5 xl:border-l xl:border-border xl:pl-6">
              <SectionKicker
                title="Training details"
                body="Tune the model, optimizer, adapters, advanced objectives, and post-training export targets."
              />

          {/* Base Model */}
          <div>
            <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-3">
              Base Model
            </label>
            <BaseModelSelect
              value={config.baseModel}
              onChange={(baseModel) => setConfig({ ...config, baseModel })}
              className="input w-full"
            />

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
            <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-3 gap-4">
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
                <FieldLabel hint="Per-step device batch. Multiply by gradient accumulation for the optimizer batch shown in the setup guide.">
                  Batch Size
                </FieldLabel>
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
                <FieldLabel hint="Use conservative values for full fine-tunes and DPO. QLoRA SFT adapters can tolerate higher rates than preference or RL runs.">
                  Learning Rate
                </FieldLabel>
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
                <FieldLabel hint="Warmup reduces early gradient spikes. Short smoke runs can use less; long unstable runs usually benefit from some warmup.">
                  Warmup Ratio
                </FieldLabel>
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
                <FieldLabel hint="Raises effective batch without increasing per-step VRAM. Effective batch = batch size times accumulation steps.">
                  Gradient Accumulation
                </FieldLabel>
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
              <div className="md:col-span-2 2xl:col-span-3">
                <FieldLabel hint="Set high enough to keep prompts, tool calls, observations, verifier output, and final answer. Truncated completions can break RL signals.">
                  Max Sequence Length
                </FieldLabel>
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
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <FieldLabel hint="Adapter capacity. r16-r32 is a practical starting range for coding traces; raise only when underfitting is clear.">
                  LoRA Rank
                </FieldLabel>
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
                <FieldLabel hint="Adapter scaling. Common starters are alpha = rank or alpha = 2 x rank.">
                  LoRA Alpha
                </FieldLabel>
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
                <FieldLabel hint="Regularizes adapters. Keep low for strong gold traces; increase slightly when overfitting appears.">
                  LoRA Dropout
                </FieldLabel>
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
                <FieldLabel hint="Controls reference-policy pull. Lower values are more conservative; start around 0.1 and watch reward margin plus held-out pass@k.">
                  Beta (Divergence Penalty)
                </FieldLabel>
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
              <div className="mb-4">
                <FieldLabel hint="The TMax terminal profile applies safer defaults for long verifier-backed rollouts: DAPO, larger groups, active sampling, zero-std filtering, and FP32 LM head.">
                  Training Profile
                </FieldLabel>
                <select
                  value={config.trainingProfile ?? 'default'}
                  onChange={(e) => applyTrainingProfile(e.target.value as TrainingProfile)}
                  className="input w-full"
                >
                  <option value="default">Default GRPO</option>
                  <option value="terminal_rl_tmax_like">TMax Terminal RL</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <FieldLabel hint="Completions sampled per prompt. Larger groups improve comparison quality only when the verifier produces reward variation.">
                    Group Size
                  </FieldLabel>
                  <input
                    type="number"
                    value={config.grpoGroupSize ?? config.grpoNumGenerations}
                    onChange={(e) => {
                      const groupSize = parseInt(e.target.value)
                      setConfig({
                        ...config,
                        grpoGroupSize: groupSize,
                        grpoNumGenerations: groupSize,
                      })
                    }}
                    className="input w-full"
                    min={2}
                    max={64}
                  />
                  <p className="font-mono text-xs text-text-muted mt-1">
                    Number of completions sampled per prompt for reward comparison
                  </p>
                </div>
                <div>
                  <FieldLabel hint="Controls rollout diversity. Too low can collapse reward variance; too high can increase invalid tool calls and timeouts.">
                    Sampling Temperature
                  </FieldLabel>
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
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <FieldLabel hint="DAPO and Dr. GRPO reduce length-bias issues for long rollouts. Use the default only when sequence lengths are short and stable.">
                    Loss Variant
                  </FieldLabel>
                  <select
                    value={config.grpoLossType ?? 'grpo'}
                    onChange={(e) => setConfig({ ...config, grpoLossType: e.target.value })}
                    className="input w-full"
                  >
                    <option value="grpo">GRPO (default)</option>
                    <option value="gspo">GSPO — sequence-level (Qwen)</option>
                    <option value="dr_grpo">Dr. GRPO</option>
                    <option value="dapo">DAPO</option>
                    <option value="bnpo">BNPO</option>
                  </select>
                  <p className="font-mono text-xs text-text-muted mt-1">
                    GSPO is more stable for long sequences and MoE models
                  </p>
                </div>
                <div>
                  <FieldLabel hint="Auto detects the available stack. Use TRL + vLLM, verl, or SkyRL only when the environment is installed and smoke-tested.">
                    Compute Backend
                  </FieldLabel>
                  <select
                    value={config.grpoBackend ?? 'auto'}
                    onChange={(e) => setConfig({ ...config, grpoBackend: e.target.value })}
                    className="input w-full"
                  >
                    <option value="auto">Auto (detect)</option>
                    <option value="unsloth">Unsloth</option>
                    <option value="plain">Plain transformers</option>
                    <option value="trl_vllm">TRL + vLLM</option>
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <FieldLabel hint="DPPO backends need train-policy logprob replay support. Auto falls back to GRPO when no real DPPO stack is available.">
                    DPPO Backend
                  </FieldLabel>
                  <select
                    value={config.dppoBackend ?? 'auto'}
                    onChange={(e) => setConfig({ ...config, dppoBackend: e.target.value })}
                    className="input w-full"
                  >
                    <option value="auto">Auto detect</option>
                    <option value="verl">verl</option>
                    <option value="skyrl">SkyRL</option>
                    <option value="tmax_open_instruct">TMax open-instruct</option>
                    <option value="grpo_fallback">GRPO fallback</option>
                  </select>
                </div>
                <div>
                  <FieldLabel hint="Binary-TV is easier to reason about for pass/fail terminal rewards; Binary-KL gives a logprob-style bound.">
                    DPPO Divergence
                  </FieldLabel>
                  <select
                    value={config.dppoDivergence ?? 'binary_tv'}
                    onChange={(e) => setConfig({ ...config, dppoDivergence: e.target.value })}
                    className="input w-full"
                  >
                    <option value="binary_tv">Binary-TV</option>
                    <option value="binary_kl">Binary-KL</option>
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <FieldLabel hint="Masks or flags updates when binary total variation gets too large. Tighten if the model drifts despite rewards.">
                    Binary-TV Threshold
                  </FieldLabel>
                  <input
                    type="number"
                    value={config.dppoBinaryTvThreshold ?? 0.15}
                    onChange={(e) => setConfig({ ...config, dppoBinaryTvThreshold: parseFloat(e.target.value) })}
                    className="input w-full"
                    min={0}
                    max={1}
                    step={0.01}
                  />
                </div>
                <div>
                  <FieldLabel hint="Masks or flags updates when binary KL exceeds the bound. Keep small for conservative policy improvement.">
                    Binary-KL Threshold
                  </FieldLabel>
                  <input
                    type="number"
                    value={config.dppoBinaryKlThreshold ?? 0.05}
                    onChange={(e) => setConfig({ ...config, dppoBinaryKlThreshold: parseFloat(e.target.value) })}
                    className="input w-full"
                    min={0}
                    step={0.01}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <FieldLabel hint="Number of prompts rolled out per batch. Attempts per batch = prompt groups x group size.">
                    Prompt Groups per Batch
                  </FieldLabel>
                  <input
                    type="number"
                    value={config.promptsPerRolloutBatch ?? 8}
                    onChange={(e) => setConfig({ ...config, promptsPerRolloutBatch: parseInt(e.target.value) })}
                    className="input w-full"
                    min={1}
                    max={128}
                  />
                </div>
                <div>
                  <FieldLabel hint="Caps each terminal episode. Too low truncates valid solutions; too high can hide inefficient or looping policies.">
                    Max Tool Calls
                  </FieldLabel>
                  <input
                    type="number"
                    value={config.maxToolCallsPerEpisode ?? 64}
                    onChange={(e) => setConfig({ ...config, maxToolCallsPerEpisode: parseInt(e.target.value) })}
                    className="input w-full"
                    min={1}
                    max={512}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.tokenLevelLoss ?? false}
                    onChange={(e) => setConfig({ ...config, tokenLevelLoss: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">Token-level loss</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.filterZeroStdGroups ?? false}
                    onChange={(e) => setConfig({ ...config, filterZeroStdGroups: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">Filter zero-std groups</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.activeSampling ?? false}
                    onChange={(e) => setConfig({ ...config, activeSampling: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">Active sampling</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.lmHeadFp32 ?? false}
                    onChange={(e) => setConfig({ ...config, lmHeadFp32: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">FP32 LM head</span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={config.interleavedThinking ?? false}
                    onChange={(e) => setConfig({ ...config, interleavedThinking: e.target.checked })}
                    className="accent-primary"
                  />
                  <span className="text-sm text-text-secondary">Interleaved thinking</span>
                </label>
                <div>
                  <FieldLabel hint="Warm-start weak policies before RL. Skip when held-out SFT behavior is already strong enough to produce verifier contrast.">
                    SFT Warm Start
                  </FieldLabel>
                  <select
                    value={config.sftWarmStartPolicy ?? 'none'}
                    onChange={(e) => setConfig({ ...config, sftWarmStartPolicy: e.target.value })}
                    className="input w-full"
                  >
                    <option value="none">None</option>
                    <option value="weak_models_only">Weak models only</option>
                    <option value="always">Always</option>
                  </select>
                </div>
              </div>
              <label className="flex items-center gap-2 mt-3">
                <input
                  type="checkbox"
                  checked={config.grpoUseVllm ?? false}
                  onChange={(e) => setConfig({ ...config, grpoUseVllm: e.target.checked })}
                  className="accent-primary"
                />
                <span className="text-sm text-text-secondary">
                  Use vLLM-backed generation (faster rollouts; requires vLLM in the env)
                </span>
              </label>
              <div className="mt-5 pt-4 border-t border-border">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div>
                    <h4 className="font-mono text-xs uppercase tracking-widest text-text-secondary">
                      World Model Objectives
                    </h4>
                    <p className="font-mono text-xs text-text-muted mt-1">
                      ECHO predicts terminal observations; RWML rewards accurate next-state predictions in embedding space.
                    </p>
                  </div>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-text-muted whitespace-nowrap">
                    JEPA-style
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <label className="flex items-start gap-2 text-sm text-text-secondary">
                    <input
                      type="checkbox"
                      checked={config.echoEnabled ?? false}
                      onChange={(e) => setConfig({ ...config, echoEnabled: e.target.checked })}
                      className="accent-primary mt-1"
                    />
                    <span>
                      <span className="block">ECHO observation loss</span>
                      <span className="block font-mono text-xs text-text-muted">
                        Start at lambda 0.05; watch environment-prediction loss and pass@k.
                      </span>
                    </span>
                  </label>
                  <label className="flex items-start gap-2 text-sm text-text-secondary">
                    <input
                      type="checkbox"
                      checked={config.rwmlEnabled ?? false}
                      onChange={(e) => setConfig({ ...config, rwmlEnabled: e.target.checked })}
                      className="accent-primary mt-1"
                    />
                    <span>
                      <span className="block">RWML latent reward</span>
                      <span className="block font-mono text-xs text-text-muted">
                        Keep hard transitions; sub-sample easy terminal dynamics examples.
                      </span>
                    </span>
                  </label>
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <FieldLabel hint="Auxiliary observation-prediction weight. Start small, around 0.05, and confirm pass@k does not regress while ECHO loss improves.">
                      ECHO Lambda
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.echoAuxLambda ?? 0.05}
                      onChange={(e) => setConfig({ ...config, echoAuxLambda: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      step={0.01}
                    />
                  </div>
                  <div>
                    <FieldLabel hint="RWML compares predicted and observed next states in embedding space. Keep this stable across runs so distances are comparable.">
                      Embedding Model
                    </FieldLabel>
                    <select
                      value={
                        showCustomEmbeddingModel ||
                        ((config.rwmlEmbeddingModel ?? '') !== '' &&
                          !RWML_EMBEDDING_MODEL_OPTIONS.some((model) => model.value === config.rwmlEmbeddingModel))
                          ? '__custom__'
                          : config.rwmlEmbeddingModel ?? ''
                      }
                      onChange={(e) => {
                        if (e.target.value === '__custom__') {
                          setShowCustomEmbeddingModel(true)
                        } else {
                          setShowCustomEmbeddingModel(false)
                          setConfig({ ...config, rwmlEmbeddingModel: e.target.value })
                        }
                      }}
                      className="input w-full"
                    >
                      <option value="">Backend default</option>
                      {RWML_EMBEDDING_MODEL_OPTIONS.map((model) => (
                        <option key={model.value} value={model.value}>
                          {model.label}
                        </option>
                      ))}
                      <option value="__custom__">Custom embedding model...</option>
                    </select>
                    {(showCustomEmbeddingModel ||
                      ((config.rwmlEmbeddingModel ?? '') !== '' &&
                        !RWML_EMBEDDING_MODEL_OPTIONS.some((model) => model.value === config.rwmlEmbeddingModel))) && (
                      <input
                        type="text"
                        value={config.rwmlEmbeddingModel ?? ''}
                        onChange={(e) => setConfig({ ...config, rwmlEmbeddingModel: e.target.value })}
                        className="input w-full mt-2"
                        placeholder="provider/model-or-local-tag"
                        spellCheck={false}
                        autoComplete="off"
                      />
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                  <div>
                    <FieldLabel hint="Embedding-distance threshold for a good next-state prediction. Tighten only after checking the distribution on held-out replays.">
                      Distance
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.rwmlDistanceThreshold ?? 0.2}
                      onChange={(e) => setConfig({ ...config, rwmlDistanceThreshold: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0.01}
                      max={2}
                      step={0.01}
                    />
                  </div>
                  <div>
                    <FieldLabel hint="Pass-rate threshold for marking terminal dynamics as easy. Easy examples should be downsampled, not deleted blindly.">
                      Easy Pass
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.rwmlEasyPassRateThreshold ?? 0.8}
                      onChange={(e) => setConfig({ ...config, rwmlEasyPassRateThreshold: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      max={1}
                      step={0.05}
                    />
                  </div>
                  <div>
                    <FieldLabel hint="Probability of keeping easy transitions. Lower values focus RWML on hard state changes without losing calibration coverage.">
                      Easy Keep
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.rwmlEasyKeepProbability ?? 0.1}
                      onChange={(e) => setConfig({ ...config, rwmlEasyKeepProbability: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      max={1}
                      step={0.05}
                    />
                  </div>
                  <div>
                    <FieldLabel hint="How many prior terminal states inform the latent comparison. More history helps multi-step dynamics but increases payload size.">
                      History
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.rwmlHistoryWindow ?? 4}
                      onChange={(e) => setConfig({ ...config, rwmlHistoryWindow: parseInt(e.target.value) })}
                      className="input w-full"
                      min={0}
                      max={64}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div>
                    <FieldLabel hint="Optional divergence penalty on latent reward learning. Keep at 0 until RWML reward is correlated with held-out behavior.">
                      RWML KL Beta
                    </FieldLabel>
                    <input
                      type="number"
                      value={config.rwmlKlBeta ?? 0}
                      onChange={(e) => setConfig({ ...config, rwmlKlBeta: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      step={0.01}
                    />
                  </div>
                  <div className="font-mono text-xs text-text-muted self-end">
                    Track ECHO loss, RWML reward/pass rate, embedding-distance distribution, hard-transition coverage,
                    and held-out terminal pass@1/pass@k.
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Liger fused-CE — plain backend (SFT/DPO) large-vocab OOM fix */}
          <label className="flex items-center gap-2 mt-1">
            <input
              type="checkbox"
              checked={config.useLiger ?? false}
              onChange={(e) => setConfig({ ...config, useLiger: e.target.checked })}
              className="accent-primary"
            />
            <span className="text-sm text-text-secondary">
              Liger fused cross-entropy (plain backend) — avoids 262k-vocab OOM for Gemma; requires liger-kernel
            </span>
          </label>

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
                  <ModelSelect
                    value={config.teacherModel ?? ''}
                    onChange={(teacherModel) => setConfig({ ...config, teacherModel })}
                    placeholder="Select a teacher model..."
                    className="input w-full"
                  />
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

              {cascadeError && (
                <div className="card p-3 mb-4 flex items-start gap-2 border-l-4 border-l-status-error bg-status-error/10">
                  <AlertCircle className="w-4 h-4 text-status-error flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-mono text-xs font-bold text-status-error uppercase tracking-widest mb-1">
                      Cascade Preflight Failed
                    </p>
                    <p className="font-mono text-xs text-text-primary whitespace-pre-wrap break-words">
                      {cascadeError}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setPreflightError(null)}
                    className="btn-icon w-6 h-6 flex items-center justify-center text-status-error"
                    aria-label="Dismiss"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}

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
            </div>
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
