import { useState, useEffect } from 'react'
import { X, FolderGit2, Database, Sparkles, Info, Cloud, Monitor } from 'lucide-react'
import type { TrainingConfig as TrainingConfigType, TrainingStrategy } from '../../stores'
import { tracesApi, RepoInfo } from '../../services/api'
import { useTutorialComplete } from '../../hooks'
import { clsx } from 'clsx'

type TrainingScope = 'all' | 'selected' | 'single'
type TrainingBackend = 'local' | 'nemo'

interface TrainingConfigProps {
  onClose: () => void
  onStart: (config: TrainingConfigType) => void
}

export function TrainingConfig({ onClose, onStart }: TrainingConfigProps) {
  const [availableRepos, setAvailableRepos] = useState<RepoInfo[]>([])
  const [trainingScope, setTrainingScope] = useState<TrainingScope>('all')
  const [selectedRepos, setSelectedRepos] = useState<string[]>([])
  const [trainingBackend, setTrainingBackend] = useState<TrainingBackend>('local')
  const { complete: completeTutorialStep } = useTutorialComplete()

  const [config, setConfig] = useState<TrainingConfigType>({
    strategy: 'sft',
    baseModel: 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    datasetPath: '',  // Empty = auto-generate from gold traces
    epochs: 3,
    batchSize: 1,  // Reduced for 12GB VRAM (uses gradient accumulation)
    learningRate: 2e-5,
    loraRank: 16,
    loraAlpha: 32,
    use4BitQuantization: true,  // QLoRA enabled by default
    warmupSteps: 100,
    maxSeqLength: 2048,
    // KD defaults
    teacherModel: 'meta-llama/Llama-3.1-70B-Instruct',
    temperature: 2.0,
    kdAlpha: 0.5
  })

  // Fetch available repos on mount
  useEffect(() => {
    tracesApi.listRepos().then((result) => {
      if (result.ok && result.data) {
        setAvailableRepos(result.data)
      }
    })
  }, [])

  const strategies: { value: TrainingStrategy; label: string; description: string }[] = [
    { value: 'sft', label: 'SFT', description: 'Supervised Fine-Tuning' },
    { value: 'dpo', label: 'DPO', description: 'Direct Preference Optimization' },
    { value: 'grpo', label: 'GRPO', description: 'Group Relative Policy Optimization' },
    { value: 'kd', label: 'KD', description: 'Knowledge Distillation' }
  ]

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Include selected repos based on training scope
    const reposToUse = trainingScope === 'all' ? [] : selectedRepos
    onStart({ ...config, selectedRepos: reposToUse })
    completeTutorialStep('start_training')
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background-secondary rounded-2xl shadow-elevated w-full max-w-lg max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border-subtle">
          <h2 className="text-lg font-semibold text-text-primary">Training Configuration</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-background-tertiary text-text-muted"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {/* Training Scope - NEW */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Training Data Scope
            </label>
            <div className="grid grid-cols-3 gap-2 mb-3">
              <button
                type="button"
                onClick={() => setTrainingScope('all')}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  trainingScope === 'all'
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border-color hover:border-primary/50 text-text-secondary'
                )}
              >
                <Sparkles className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-semibold text-sm">Generalist</span>
                <span className="block text-xs mt-1 text-text-muted">All repos</span>
              </button>
              <button
                type="button"
                onClick={() => setTrainingScope('selected')}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  trainingScope === 'selected'
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border-color hover:border-primary/50 text-text-secondary'
                )}
              >
                <Database className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-semibold text-sm">Mixed</span>
                <span className="block text-xs mt-1 text-text-muted">Select repos</span>
              </button>
              <button
                type="button"
                onClick={() => setTrainingScope('single')}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  trainingScope === 'single'
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border-color hover:border-primary/50 text-text-secondary'
                )}
              >
                <FolderGit2 className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-semibold text-sm">Specialist</span>
                <span className="block text-xs mt-1 text-text-muted">Single repo</span>
              </button>
            </div>

            {/* Repo Selection */}
            {trainingScope !== 'all' && (
              <div className="border border-border-color rounded-lg p-3 bg-background-tertiary">
                <p className="text-xs text-text-muted mb-2">
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
                          'flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-background-secondary transition-colors',
                          selectedRepos.includes(repo.name) && 'bg-primary/10'
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
                        <span className="text-xs text-text-muted">{repo.trace_count} traces</span>
                      </label>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Info box */}
            <div className="flex items-start gap-2 mt-3 p-2 rounded-lg bg-status-info/10 border border-status-info/20">
              <Info className="w-4 h-4 text-status-info flex-shrink-0 mt-0.5" />
              <p className="text-xs text-text-secondary">
                {trainingScope === 'all' && 'Train on all collected traces for a versatile generalist agent.'}
                {trainingScope === 'selected' && 'Train on selected repos for focused expertise across projects.'}
                {trainingScope === 'single' && 'Train on a single repo for a highly specialized codebase expert.'}
              </p>
            </div>
          </div>

          {/* Training Backend */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Training Backend
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('local')
                  setConfig({ ...config, useNemoGym: false })
                }}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  trainingBackend === 'local'
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border-color hover:border-primary/50 text-text-secondary'
                )}
              >
                <Monitor className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-semibold text-sm">Local</span>
                <span className="block text-xs mt-1 text-text-muted">Your GPU (Unsloth)</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setTrainingBackend('nemo')
                  setConfig({ ...config, useNemoGym: true })
                }}
                className={clsx(
                  'p-3 rounded-lg border text-center transition-all',
                  trainingBackend === 'nemo'
                    ? 'border-primary bg-primary/10 text-primary'
                    : 'border-border-color hover:border-primary/50 text-text-secondary'
                )}
              >
                <Cloud className="w-5 h-5 mx-auto mb-1" />
                <span className="block font-semibold text-sm">NeMo Cloud</span>
                <span className="block text-xs mt-1 text-text-muted">NVIDIA Microservices</span>
              </button>
            </div>

            {/* Backend info */}
            <div className="flex items-start gap-2 mt-3 p-2 rounded-lg bg-status-info/10 border border-status-info/20">
              <Info className="w-4 h-4 text-status-info flex-shrink-0 mt-0.5" />
              <p className="text-xs text-text-secondary">
                {trainingBackend === 'local'
                  ? 'Train locally using your GPU with Unsloth for fast LoRA fine-tuning. Requires CUDA-capable GPU.'
                  : 'Train using NVIDIA NeMo Microservices for scalable cloud training. Requires NVIDIA_API_KEY.'}
              </p>
            </div>
          </div>

          {/* Strategy Selection */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Training Strategy
            </label>
            <div className="grid grid-cols-3 gap-2">
              {strategies.map((s) => (
                <button
                  key={s.value}
                  type="button"
                  onClick={() => setConfig({ ...config, strategy: s.value })}
                  className={clsx(
                    'p-3 rounded-lg border text-center transition-all',
                    config.strategy === s.value
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-border-color hover:border-primary/50 text-text-secondary'
                  )}
                >
                  <span className="block font-semibold">{s.label}</span>
                  <span className="block text-xs mt-1 text-text-muted">{s.description}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Base Model */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Base Model
            </label>
            <select
              value={config.baseModel}
              onChange={(e) => setConfig({ ...config, baseModel: e.target.value })}
              className="input w-full"
            >
              <optgroup label="Qwen (Recommended)">
                <option value="Qwen/Qwen2.5-Coder-1.5B-Instruct">Qwen2.5-Coder-1.5B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-3B-Instruct">Qwen2.5-Coder-3B-Instruct</option>
                <option value="Qwen/Qwen2.5-Coder-7B-Instruct">Qwen2.5-Coder-7B-Instruct</option>
              </optgroup>
              <optgroup label="Llama">
                <option value="meta-llama/Llama-3.2-1B-Instruct">Llama-3.2-1B-Instruct</option>
                <option value="meta-llama/Llama-3.2-3B-Instruct">Llama-3.2-3B-Instruct</option>
              </optgroup>
              <optgroup label="Other">
                <option value="mistralai/Mistral-7B-Instruct-v0.3">Mistral-7B-Instruct</option>
              </optgroup>
            </select>
          </div>

          {/* Dataset Path */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Dataset Path
            </label>
            <input
              type="text"
              value={config.datasetPath}
              onChange={(e) => setConfig({ ...config, datasetPath: e.target.value })}
              className="input w-full"
              placeholder="Leave empty to auto-generate from traces"
            />
          </div>

          {/* Hyperparameters Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">Epochs</label>
              <input
                type="number"
                value={config.epochs}
                onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                className="input w-full"
                min={1}
                max={10}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">
                Batch Size
              </label>
              <input
                type="number"
                value={config.batchSize}
                onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) })}
                className="input w-full"
                min={1}
                max={32}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-primary mb-2">
                Learning Rate
              </label>
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
              <label className="block text-sm font-medium text-text-primary mb-2">
                Warmup Steps
              </label>
              <input
                type="number"
                value={config.warmupSteps}
                onChange={(e) => setConfig({ ...config, warmupSteps: parseInt(e.target.value) })}
                className="input w-full"
                min={0}
              />
            </div>
          </div>

          {/* LoRA Config */}
          <div>
            <h3 className="text-sm font-medium text-text-primary mb-3">LoRA Configuration</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-text-secondary mb-2">LoRA Rank</label>
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
                <label className="block text-sm text-text-secondary mb-2">LoRA Alpha</label>
                <input
                  type="number"
                  value={config.loraAlpha}
                  onChange={(e) => setConfig({ ...config, loraAlpha: parseInt(e.target.value) })}
                  className="input w-full"
                  min={4}
                  max={256}
                />
              </div>
            </div>
            <label className="flex items-center gap-2 mt-3">
              <input
                type="checkbox"
                checked={config.use4BitQuantization ?? true}
                onChange={(e) => setConfig({ ...config, use4BitQuantization: e.target.checked })}
                className="accent-primary"
              />
              <span className="text-sm text-text-secondary">
                Enable 4-bit Quantization (QLoRA) - Reduces VRAM ~50%
              </span>
            </label>
          </div>

          {/* Max Sequence Length */}
          <div>
            <label className="block text-sm font-medium text-text-primary mb-2">
              Max Sequence Length
            </label>
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

          {/* Knowledge Distillation Config */}
          {config.strategy === 'kd' && (
            <div className="border border-primary/20 rounded-lg p-4 bg-primary/5">
              <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" />
                Knowledge Distillation Settings
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm text-text-secondary mb-2">Teacher Model</label>
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
                    <optgroup label="OpenAI (requires OPENAI_API_KEY)">
                      <option value="gpt-4o">GPT-4o</option>
                      <option value="gpt-4-turbo">GPT-4 Turbo</option>
                    </optgroup>
                  </select>
                  <p className="text-xs text-text-muted mt-1">
                    The larger model to distill knowledge from
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-text-secondary mb-2">Temperature</label>
                    <input
                      type="number"
                      value={config.temperature}
                      onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0.5}
                      max={10}
                      step={0.5}
                    />
                    <p className="text-xs text-text-muted mt-1">
                      Softmax temperature (higher = softer distribution)
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm text-text-secondary mb-2">KD Alpha</label>
                    <input
                      type="number"
                      value={config.kdAlpha}
                      onChange={(e) => setConfig({ ...config, kdAlpha: parseFloat(e.target.value) })}
                      className="input w-full"
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="text-xs text-text-muted mt-1">
                      Weight for distillation loss (0 = task only, 1 = KD only)
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-2 p-2 rounded-lg bg-status-info/10 border border-status-info/20">
                  <Info className="w-4 h-4 text-status-info flex-shrink-0 mt-0.5" />
                  <p className="text-xs text-text-secondary">
                    Knowledge distillation transfers capabilities from a large teacher model to your smaller student model.
                    The student learns to match the teacher's probability distributions, not just the correct answers.
                  </p>
                </div>
              </div>
            </div>
          )}
        </form>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-border-subtle">
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
