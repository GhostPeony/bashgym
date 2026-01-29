/**
 * Bash Gym API Client
 * Handles communication with the Python backend
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api'

// Types matching backend schemas
export interface TaskRequest {
  prompt: string
  task_id?: string
  repository_url?: string
  timeout?: number
}

export interface TaskResponse {
  task_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  message?: string
  created_at?: string
  completed_at?: string
  duration_seconds?: number
  result?: Record<string, any>
  trace_path?: string
}

export interface TrainingRequest {
  strategy?: 'sft' | 'dpo' | 'grpo'
  dataset_path?: string
  base_model?: string
  num_epochs?: number
  batch_size?: number
  learning_rate?: number
  use_lora?: boolean
  lora_rank?: number
  lora_alpha?: number
  warmup_steps?: number
  max_seq_length?: number
  auto_export_gguf?: boolean
  gguf_quantization?: string
  use_nemo_gym?: boolean  // Use NVIDIA NeMo cloud training
}

export interface TrainingResponse {
  run_id: string
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed'
  strategy: string
  message?: string
  started_at?: string
  completed_at?: string
  metrics?: Record<string, number>
  output_path?: string
}

export interface ModelInfo {
  model_id: string
  path: string
  created_at: string
  base_model?: string
  strategy?: string
  has_gguf: boolean
  gguf_path?: string
}

export interface ExportRequest {
  format: 'gguf' | 'safetensors' | 'lora'
  quantization?: string
}

export interface ExportResponse {
  model_id: string
  format: string
  status: string
  message?: string
  output_path?: string
}

export interface TraceQuality {
  success_rate: number
  verification_score: number
  complexity_score: number
  length_score: number
  tool_diversity: number
  efficiency_score: number
  total_score: number
}

export interface RepoInfo {
  name: string
  path?: string
  git_remote?: string
  git_branch?: string
  is_git_repo: boolean
  trace_count?: number  // Used in repo list response
}

// Quality tier based on NVIDIA NeMo recommendations
export type TraceQualityTier = 'gold' | 'silver' | 'bronze' | 'rejected'

export interface TraceInfo {
  trace_id: string
  task_id: string
  task_description: string
  status: 'gold' | 'silver' | 'bronze' | 'failed' | 'pending'
  quality_tier?: TraceQualityTier
  steps_count: number
  quality: TraceQuality
  repo?: RepoInfo
  repos_count: number
  created_at?: string
  promoted_at?: string
}

export interface RouterStats {
  total_requests: number
  teacher_requests: number
  student_requests: number
  teacher_success_rate: number
  student_success_rate: number
  avg_teacher_latency: number
  avg_student_latency: number
  current_student_rate: number
}

export interface SystemStats {
  gold_traces_count: number
  silver_traces_count: number
  bronze_traces_count: number
  failed_traces_count: number
  pending_traces_count: number
  models_count: number
  base_model: string
  auto_export_gguf: boolean
  active_tasks: number
  active_training_runs: number
}

export interface HealthCheck {
  status: string
  timestamp: string
  version: string
}

interface ApiResponse<T> {
  ok: boolean
  data?: T
  error?: string
}

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  try {
    // Use Electron IPC proxy if available (avoids CORS in dev)
    if (window.bashgym?.api) {
      const result = await window.bashgym.api.fetch(`${API_BASE}${endpoint}`, options)
      return result as ApiResponse<T>
    }

    // Direct fetch fallback
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers
      }
    })

    const text = await response.text()
    try {
      const data = JSON.parse(text)
      return { ok: response.ok, data, error: response.ok ? undefined : data?.detail || text }
    } catch {
      return { ok: false, error: text || `HTTP ${response.status}` }
    }
  } catch (error) {
    return { ok: false, error: String(error) }
  }
}

// System API
export const systemApi = {
  health: () =>
    request<HealthCheck>('/health'),

  stats: () =>
    request<SystemStats>('/stats')
}

// Task API
export const taskApi = {
  submit: (task: TaskRequest) =>
    request<TaskResponse>('/tasks', {
      method: 'POST',
      body: JSON.stringify(task)
    }),

  getStatus: (taskId: string) =>
    request<TaskResponse>(`/tasks/${taskId}`),

  list: (status?: string, limit?: number) =>
    request<TaskResponse[]>(`/tasks${status ? `?status=${status}` : ''}${limit ? `${status ? '&' : '?'}limit=${limit}` : ''}`)
}

// Training API
export const trainingApi = {
  start: (config: TrainingRequest) =>
    request<TrainingResponse>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config)
    }),

  getStatus: (runId: string) =>
    request<TrainingResponse>(`/training/${runId}`),

  pause: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/pause`, { method: 'POST' }),

  resume: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/resume`, { method: 'POST' }),

  stop: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/stop`, { method: 'POST' }),

  list: (status?: string, limit?: number) =>
    request<TrainingResponse[]>(`/training${status ? `?status=${status}` : ''}${limit ? `${status ? '&' : '?'}limit=${limit}` : ''}`)
}

// Legacy Models API (see modelsApi below for full implementation)

// Evaluation types
export interface EvaluationRequest {
  model_id: string
  benchmarks: string[]
  num_samples?: number
}

export interface BenchmarkResult {
  score: number
  passed: number
  total: number
  duration_seconds: number
}

export interface EvaluationResponse {
  job_id: string
  model_id: string
  benchmarks: string[]
  status: 'pending' | 'running' | 'completed' | 'failed'
  results?: Record<string, BenchmarkResult>
  error?: string
  created_at?: string
}

// Evaluation API
export const evaluatorApi = {
  run: (req: EvaluationRequest) =>
    request<EvaluationResponse>('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  getStatus: (jobId: string) =>
    request<EvaluationResponse>(`/evaluation/${jobId}`),

  list: () =>
    request<EvaluationResponse[]>('/evaluation')
}

// Traces API
export const tracesApi = {
  list: (options?: { status?: 'gold' | 'silver' | 'bronze' | 'failed' | 'pending', repo?: string, limit?: number }) => {
    const params = new URLSearchParams()
    if (options?.status) params.set('status', options.status)
    if (options?.repo) params.set('repo', options.repo)
    if (options?.limit) params.set('limit', String(options.limit))
    const queryString = params.toString()
    return request<TraceInfo[]>(`/traces${queryString ? `?${queryString}` : ''}`)
  },

  listRepos: () =>
    request<RepoInfo[]>('/traces/repos'),

  stats: () =>
    request<{
      timeline: { time: string; gold: number; failed: number; pending: number }[]
      totals: { gold: number; failed: number; pending: number; total: number }
    }>('/traces/stats'),

  getGold: (limit?: number) =>
    request<TraceInfo[]>(`/traces/gold${limit ? `?limit=${limit}` : ''}`),

  get: (traceId: string) =>
    request<TraceInfo>(`/traces/${traceId}`),

  promote: (traceId: string) =>
    request<{ success: boolean; message: string }>(`/traces/${traceId}/promote`, { method: 'POST' }),

  demote: (traceId: string) =>
    request<{ success: boolean; message: string }>(`/traces/${traceId}/demote`, { method: 'POST' }),

  // Sync traces from ~/.bashgym/ to project data/ directory
  sync: () =>
    request<{ synced: { pending: number; gold: number; silver: number; bronze: number; failed: number }; project_dir: string }>('/traces/sync', { method: 'POST' }),

  // Auto-classify pending traces into quality tiers based on NVIDIA NeMo thresholds
  autoClassify: (options?: {
    gold_success_rate?: number;
    gold_quality_score?: number;
    silver_success_rate?: number;
    silver_quality_score?: number;
    bronze_success_rate?: number;
    bronze_quality_score?: number;
    dry_run?: boolean;
    auto_promote?: boolean;
  }) => {
    const params = new URLSearchParams()
    if (options?.gold_success_rate !== undefined) params.set('gold_success_rate', String(options.gold_success_rate))
    if (options?.gold_quality_score !== undefined) params.set('gold_quality_score', String(options.gold_quality_score))
    if (options?.silver_success_rate !== undefined) params.set('silver_success_rate', String(options.silver_success_rate))
    if (options?.silver_quality_score !== undefined) params.set('silver_quality_score', String(options.silver_quality_score))
    if (options?.bronze_success_rate !== undefined) params.set('bronze_success_rate', String(options.bronze_success_rate))
    if (options?.bronze_quality_score !== undefined) params.set('bronze_quality_score', String(options.bronze_quality_score))
    if (options?.dry_run !== undefined) params.set('dry_run', String(options.dry_run))
    if (options?.auto_promote !== undefined) params.set('auto_promote', String(options.auto_promote))
    const queryString = params.toString()
    return request<{
      classifications: {
        gold: string[];
        silver: string[];
        bronze: string[];
        rejected: string[];
        failed: string[];
        pending: string[];
      };
      detailed: {
        gold: Array<{ id: string; success_rate: number; quality_score: number; steps: number }>;
        silver: Array<{ id: string; success_rate: number; quality_score: number; steps: number }>;
        bronze: Array<{ id: string; success_rate: number; quality_score: number; steps: number }>;
        rejected: Array<{ id: string; success_rate: number; quality_score: number; steps: number }>;
      };
      thresholds: {
        gold: { success_rate: number; quality_score: number };
        silver: { success_rate: number; quality_score: number };
        bronze: { success_rate: number; quality_score: number };
      };
      dry_run: boolean;
      auto_promote: boolean;
      summary: {
        gold: number;
        silver: number;
        bronze: number;
        rejected: number;
        failed: number;
        pending: number;
        total_processed: number;
      };
      dpo_pairs: Array<{
        chosen: string;
        chosen_success_rate: number;
        rejected: string;
        rejected_success_rate: number;
        quality_gap: number;
      }>;
      dpo_pairs_count: number;
      training_recommendations: {
        sft_eligible: number;
        dpo_chosen_pool: number;
        dpo_rejected_pool: number;
        note: string;
      };
    }>(`/traces/auto-classify${queryString ? `?${queryString}` : ''}`, { method: 'POST' })
  },

  // Generate training examples from a trace session
  generateExamples: (traceId: string, options?: { min_success_rate?: number }) =>
    request<{
      trace_id: string;
      examples: Array<{
        example_id: string;
        user_prompt: string;
        assistant_response: string;
        step_count: number;
        success_rate: number;
        confidence: number;
      }>;
      total_steps: number;
      examples_generated: number;
    }>(`/traces/${traceId}/generate-examples`, {
      method: 'POST',
      body: JSON.stringify(options || {})
    })
}

// Router API
export type RoutingStrategy =
  | 'teacher_only'
  | 'student_only'
  | 'confidence_based'
  | 'task_complexity'
  | 'progressive'
  | 'random_sample'

export const routerApi = {
  getStats: () =>
    request<RouterStats>('/router/stats'),

  setStrategy: (strategy: RoutingStrategy) =>
    request<{ success: boolean; strategy: string }>(`/router/strategy?strategy=${strategy}`, {
      method: 'POST'
    }),

  setStudentRate: (rate: number) =>
    request<{ success: boolean; rate: number }>(`/router/student-rate?rate=${rate}`, {
      method: 'POST'
    })
}

// Verification API
export interface VerificationResult {
  passed: boolean
  tests_run: number
  tests_passed: number
  message?: string
}

export const verifyApi = {
  run: (taskId: string) =>
    request<VerificationResult>(`/verify/${taskId}`, { method: 'POST' })
}

// Synthesis API
export interface SynthesisResult {
  success: boolean
  examples_created: number
  output_path?: string
  message?: string
}

export const synthesisApi = {
  run: () =>
    request<SynthesisResult>('/synthesize', { method: 'POST' })
}

// Hooks API - Multi-tool support
export interface ToolStatus {
  name: string
  installed: boolean
  hooks_installed: boolean
  hooks_path: string | null
  adapter_type: string
}

export interface HooksStatus {
  // Legacy fields for backward compatibility
  hooks_dir?: string
  post_tool_use_installed?: boolean
  session_end_installed?: boolean
  all_installed?: boolean
  platform?: string
  // New multi-tool format
  tools?: ToolStatus[]
  summary?: {
    installed_count: number
    configured_count: number
    installed_names: string[]
    configured_names: string[]
  }
}

export interface HooksInstallResult {
  success: boolean
  tools?: Record<string, { success: boolean; message: string }>
  errors?: string[]
  // Legacy fields
  hooks_dir?: string
  installed?: string[]
  message?: string
}

export interface HooksInstallRequest {
  tools?: string[]
}

export const hooksApi = {
  getStatus: () =>
    request<HooksStatus>('/hooks/status'),

  install: (params?: HooksInstallRequest) =>
    request<HooksInstallResult>('/hooks/install', {
      method: 'POST',
      body: JSON.stringify(params || {})
    })
}

// System Info API (Hardware detection)
export interface GpuInfo {
  vendor: string
  model: string
  vram: number
  vram_used?: number
  driver?: string
  temperature?: number
  utilization?: number
}

export interface SystemInfo {
  gpus: GpuInfo[]
  total_ram: number
  available_ram: number
  platform: string
  arch: string
  cuda_available: boolean
  cuda_version?: string
  python_available: boolean
  python_version?: string
}

export interface ModelRecommendations {
  max_vram_gb: number
  cuda_available: boolean
  recommended_models: string[]
  recommended_quantization: string
  recommended_batch_size: number
  warning?: string
}

export const systemInfoApi = {
  getInfo: (refresh?: boolean) =>
    request<SystemInfo>(`/system/info${refresh ? '?refresh=true' : ''}`),

  getGpus: () =>
    request<GpuInfo[]>('/system/gpus'),

  getRecommendations: () =>
    request<ModelRecommendations>('/system/recommendations')
}

// Model Providers API
export interface ProviderStatus {
  type: string
  name: string
  available: boolean
  endpoint?: string
  model_count?: number
  error?: string
}

export interface LocalModel {
  id: string
  name: string
  provider: string
  size_gb?: number
  parameter_size?: string
  is_code_model: boolean
  is_local: boolean
  supports_training: boolean
  supports_inference: boolean
  context_length?: number
  description?: string
}

export interface OllamaModel {
  name: string
  size: number
  size_gb: number
  modified_at: string
  family: string
  parameter_size: string
  quantization: string
  is_code_model: boolean
  provider: string
}

export interface ProvidersResponse {
  providers: ProviderStatus[]
  summary: {
    available: number
    total: number
  }
}

export interface ModelsResponse {
  local: LocalModel[]
  training: LocalModel[]
  teacher: LocalModel[]
  inference: LocalModel[]
}

export interface OllamaModelsResponse {
  available: boolean
  error?: string
  models: OllamaModel[]
}

export const providersApi = {
  getProviders: () =>
    request<ProvidersResponse>('/providers'),

  getModels: (params?: { include_local?: boolean; include_cloud?: boolean; code_only?: boolean }) => {
    const searchParams = new URLSearchParams()
    if (params?.include_local !== undefined) searchParams.set('include_local', String(params.include_local))
    if (params?.include_cloud !== undefined) searchParams.set('include_cloud', String(params.include_cloud))
    if (params?.code_only !== undefined) searchParams.set('code_only', String(params.code_only))
    const query = searchParams.toString()
    return request<ModelsResponse>(`/providers/models${query ? `?${query}` : ''}`)
  },

  getOllamaModels: () =>
    request<OllamaModelsResponse>('/providers/ollama/models'),

  pullOllamaModel: (modelName: string) =>
    request<{ status: string; model: string; message: string }>('/providers/ollama/pull', {
      method: 'POST',
      body: JSON.stringify({ model_name: modelName })
    }),

  deleteOllamaModel: (modelName: string) =>
    request<{ status: string; model: string }>(`/providers/ollama/models/${encodeURIComponent(modelName)}`, {
      method: 'DELETE'
    })
}

// Factory API - Data Designer, Privacy, Prompt Optimization

// Seed example for training data generation
export interface SeedExample {
  id: string
  data: Record<string, string>  // Column name -> value mapping
  source: 'manual' | 'imported' | 'gold_trace'
  created_at?: string
  trace_id?: string  // If from gold trace
  tags?: string[]
  quality?: number
}

// Model configuration for LLM columns
export interface ModelConfig {
  model_id: string
  temperature: number
  max_tokens: number
  system_prompt?: string
}

// Column dependency rule
export interface ColumnDependency {
  depends_on: string  // Column ID
  condition: 'equals' | 'not_equals' | 'contains' | 'exists'
  value?: string
  required_when_true: boolean
}

// Validation constraint
export interface ColumnConstraint {
  type: 'enum' | 'regex' | 'json_schema' | 'min_length' | 'max_length'
  value: string | string[] | number
  error_message?: string
}

// Enhanced column configuration
export interface ColumnConfig {
  id: string
  name: string
  type: 'llm' | 'sampler' | 'category' | 'person' | 'datetime' | 'expression' | 'uuid' | 'gaussian' | 'validator'
  description?: string
  required: boolean
  risk_level: 'normal' | 'elevated' | 'high'  // Safety marking
  config: {
    // LLM column
    prompt?: string
    model?: ModelConfig
    // Category/Sampler
    values?: string[]
    weights?: number[]  // For weighted sampling
    // Sampler distributions
    distribution?: 'uniform' | 'gaussian' | 'poisson' | 'bernoulli'
    mean?: number
    std?: number
    // DateTime
    format?: string
    min_date?: string
    max_date?: string
    // Expression (Jinja2)
    template?: string
    // Validator
    validation_code?: string  // Python validation code
  }
  dependencies?: ColumnDependency[]
  constraints?: ColumnConstraint[]
}

export interface PrivacyConfig {
  enabled: boolean
  epsilon: number
  pii_types: string[]
  replacement_strategy: 'synthetic' | 'mask' | 'hash'
}

export interface PromptOptConfig {
  enabled: boolean
  intensity: 'light' | 'medium' | 'heavy'
  max_demos: number
  metric_threshold: number  // 0-1, stop when reached
  target_metric: 'accuracy' | 'f1' | 'custom'
}

// Output configuration for synthesis
export interface OutputConfig {
  row_count: number
  format: 'jsonl' | 'csv' | 'parquet'
  task_name: string  // For GRPO integration
  include_task_name: boolean
  train_val_split: number  // 0-1, portion for training
  include_negative_examples: boolean
  negative_example_ratio: number  // 0-1
}

// Safety configuration
export interface SafetyConfig {
  enabled: boolean
  block_dangerous_commands: boolean
  require_confirmation_for_high_risk: boolean
  max_risk_level: 'normal' | 'elevated' | 'high'
  blocked_patterns: string[]  // Regex patterns to block
}

export interface FactoryConfig {
  columns: ColumnConfig[]
  seeds: SeedExample[]
  privacy: PrivacyConfig
  prompt_optimization: PromptOptConfig
  output: OutputConfig
  safety: SafetyConfig
  default_model: ModelConfig
}

// Preview result for inspection
export interface PreviewRow {
  id: string
  data: Record<string, string>
  validation_errors: string[]
  risk_flags: string[]
}

export interface PreviewResult {
  rows: PreviewRow[]
  total_generated: number
  valid_count: number
  invalid_count: number
  validation_summary: Record<string, number>  // Error type -> count
  column_coverage: Record<string, number>  // Column -> non-empty percentage
}

export interface SynthesisJob {
  id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  job_type: 'preview' | 'full'
  created_at?: string
  completed_at?: string
  examples_created?: number
  valid_examples?: number
  output_path?: string
  error?: string
  config_snapshot?: Partial<FactoryConfig>
}

export const factoryApi = {
  getConfig: () =>
    request<FactoryConfig>('/factory/config'),

  updateConfig: (config: FactoryConfig) =>
    request<{ success: boolean; message: string }>('/factory/config', {
      method: 'PUT',
      body: JSON.stringify(config)
    }),

  // Seeds management
  getSeeds: () =>
    request<SeedExample[]>('/factory/seeds'),

  addSeed: (seed: Omit<SeedExample, 'id' | 'created_at'>) =>
    request<SeedExample>('/factory/seeds', {
      method: 'POST',
      body: JSON.stringify(seed)
    }),

  deleteSeed: (seedId: string) =>
    request<{ success: boolean }>(`/factory/seeds/${seedId}`, { method: 'DELETE' }),

  importSeeds: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return request<{ imported: number; seeds: SeedExample[] }>('/factory/seeds/import', {
      method: 'POST',
      body: formData,
      headers: {}  // Let browser set Content-Type for FormData
    })
  },

  importFromGoldTraces: (traceIds?: string[]) =>
    request<{ imported: number; seeds: SeedExample[] }>('/factory/seeds/from-traces', {
      method: 'POST',
      body: JSON.stringify({ trace_ids: traceIds })
    }),

  // Preview mode
  generatePreview: (rowCount?: number) =>
    request<PreviewResult>('/factory/preview', {
      method: 'POST',
      body: JSON.stringify({ row_count: rowCount || 50 })
    }),

  // Synthesis jobs
  listJobs: () =>
    request<SynthesisJob[]>('/factory/jobs'),

  runSynthesis: (options?: { preview?: boolean; row_count?: number }) =>
    request<SynthesisJob>('/factory/synthesize', {
      method: 'POST',
      body: JSON.stringify(options || {})
    }),

  getJob: (jobId: string) =>
    request<SynthesisJob>(`/factory/jobs/${jobId}`),

  cancelJob: (jobId: string) =>
    request<{ success: boolean }>(`/factory/jobs/${jobId}/cancel`, { method: 'POST' }),

  // Validation
  validateColumn: (columnId: string, sampleSize?: number) =>
    request<{ valid: boolean; errors: string[]; sample: PreviewRow[] }>(
      `/factory/columns/${columnId}/validate`,
      { method: 'POST', body: JSON.stringify({ sample_size: sampleSize || 10 }) }
    ),

  // Available models
  listModels: () =>
    request<{ id: string; name: string; provider: string }[]>('/factory/models')
}

// Observability API - Guardrails & Profiler

export interface GuardrailEvent {
  timestamp: string
  check_type: string
  location: string
  action_taken: string
  confidence: number
  model_source?: string
  original_content: string
  modified_content?: string
  details: Record<string, any>
}

export interface GuardrailStats {
  total_events: number
  by_action: Record<string, number>
  by_type: Record<string, number>
  by_source: Record<string, number>
  block_rate: number
}

export interface TraceSummary {
  trace_id: string
  name: string
  duration_ms: number
  total_spans: number
  llm_calls: Record<string, any>
  tool_calls: Record<string, any>
  bottlenecks?: Array<Record<string, any>>
}

export interface ObservabilityMetrics {
  profiler: {
    total_traces?: number
    avg_duration_ms?: number
    total_tokens?: number
    avg_tokens_per_trace?: number
  }
  guardrails: GuardrailStats
}

export interface ObservabilitySettings {
  guardrails: {
    enabled: boolean
    pii_filtering: boolean
    injection_detection: boolean
    code_safety: boolean
  }
  profiler: {
    enabled: boolean
    profile_tokens: boolean
    profile_latency: boolean
  }
}

export const observabilityApi = {
  // Traces (profiler)
  listTraces: (limit = 50, offset = 0) =>
    request<{ traces: TraceSummary[]; total: number }>(
      `/observability/traces?limit=${limit}&offset=${offset}`
    ),

  getTrace: (traceId: string) =>
    request<TraceSummary>(`/observability/traces/${traceId}`),

  // Guardrail events
  listGuardrailEvents: (options?: {
    action?: string
    check_type?: string
    model_source?: string
    limit?: number
  }) => {
    const params = new URLSearchParams()
    if (options?.action) params.set('action', options.action)
    if (options?.check_type) params.set('check_type', options.check_type)
    if (options?.model_source) params.set('model_source', options.model_source)
    if (options?.limit) params.set('limit', String(options.limit))
    const query = params.toString()
    return request<{ events: GuardrailEvent[]; total: number }>(
      `/observability/guardrails/events${query ? `?${query}` : ''}`
    )
  },

  getGuardrailStats: () =>
    request<GuardrailStats>('/observability/guardrails/stats'),

  getDpoNegatives: (limit = 100) =>
    request<{ negatives: Array<Record<string, any>>; total: number }>(
      `/observability/guardrails/dpo-negatives?limit=${limit}`
    ),

  // Aggregated metrics
  getMetrics: () =>
    request<ObservabilityMetrics>('/observability/metrics'),

  // Settings
  getSettings: () =>
    request<ObservabilitySettings>('/observability/settings'),

  updateGuardrailSettings: (settings: Partial<{
    enabled: boolean
    pii_filtering: boolean
    injection_detection: boolean
    code_safety: boolean
  }>) =>
    request<{ status: string; updated: Record<string, any> }>(
      '/observability/settings/guardrails',
      { method: 'POST', body: JSON.stringify(settings) }
    ),

  updateProfilerSettings: (settings: Partial<{ enabled: boolean }>) =>
    request<{ status: string; updated: Record<string, any> }>(
      '/observability/settings/profiler',
      { method: 'POST', body: JSON.stringify(settings) }
    )
}

// ============================================================================
// Models API - Model Registry and Profiles
// ============================================================================

export interface ModelSummary {
  model_id: string
  display_name: string
  description: string
  tags: string[]
  starred: boolean
  base_model: string
  training_strategy: string
  status: string
  created_at: string
  custom_eval_pass_rate: number | null
  benchmark_avg_score: number | null
  model_size_display: string
  inference_latency_ms: number | null
  training_duration_display: string
}

export interface BenchmarkResult {
  benchmark_name: string
  score: number
  passed: number
  total: number
  metrics: Record<string, number>
  evaluated_at: string
}

export interface CustomEvalResult {
  eval_set_id: string
  eval_type: string
  passed: number
  total: number
  pass_rate: number
  failures: Array<Record<string, any>>
  evaluated_at: string
}

export interface ModelArtifacts {
  checkpoints: Array<{ path: string; step: number; epoch?: number; loss?: number }>
  final_adapter_path: string | null
  merged_path: string | null
  gguf_exports: Array<{ path: string; quantization: string; size_bytes: number; created_at: string }>
}

export interface EvaluationRecord {
  evaluated_at: string
  benchmarks: Record<string, number>
  custom_eval_score: number | null
  overall_score: number | null
}

export interface ModelProfile {
  // Identity
  model_id: string
  run_id: string
  display_name: string
  description: string
  tags: string[]
  starred: boolean
  created_at: string

  // Lineage
  base_model: string
  training_strategy: string
  teacher_model: string | null
  training_traces: string[]
  parent_model: string | null
  training_repos: string[]

  // Training
  config: Record<string, any>
  started_at: string | null
  completed_at: string | null
  duration_seconds: number
  training_duration_display: string
  loss_curve: Array<{ step: number; loss: number; val_loss?: number; learning_rate?: number; epoch?: number }>
  final_metrics: Record<string, number>

  // Artifacts
  artifacts: ModelArtifacts
  model_dir: string

  // Evaluations
  benchmarks: Record<string, BenchmarkResult>
  custom_evals: Record<string, CustomEvalResult>
  evaluation_history: EvaluationRecord[]

  // Operational
  model_size_bytes: number
  model_size_display: string
  model_size_params: string | null
  inference_latency_ms: number | null
  status: string
  deployed_to: string | null

  // Computed
  custom_eval_pass_rate: number | null
  benchmark_avg_score: number | null
}

export interface LeaderboardEntry {
  rank: number
  model_id: string
  display_name: string
  value: number
  base_model: string
  strategy: string
}

export interface TrendDataPoint {
  timestamp: string
  model_id: string
  display_name: string
  value: number
}

export const modelsApi = {
  // List models with filtering
  list: (options?: {
    strategy?: string
    base_model?: string
    status?: string
    tags?: string
    starred?: boolean
    sort_by?: string
    sort_order?: 'asc' | 'desc'
    limit?: number
    offset?: number
  }) => {
    const params = new URLSearchParams()
    if (options?.strategy) params.set('strategy', options.strategy)
    if (options?.base_model) params.set('base_model', options.base_model)
    if (options?.status) params.set('status', options.status)
    if (options?.tags) params.set('tags', options.tags)
    if (options?.starred) params.set('starred', 'true')
    if (options?.sort_by) params.set('sort_by', options.sort_by)
    if (options?.sort_order) params.set('sort_order', options.sort_order)
    if (options?.limit) params.set('limit', String(options.limit))
    if (options?.offset) params.set('offset', String(options.offset))
    const query = params.toString()
    return request<{ models: ModelSummary[]; total: number }>(
      `/models${query ? `?${query}` : ''}`
    )
  },

  // Get full model profile
  get: (modelId: string) =>
    request<ModelProfile>(`/models/${encodeURIComponent(modelId)}`),

  // Update model metadata
  update: (modelId: string, updates: {
    display_name?: string
    description?: string
    tags?: string[]
    starred?: boolean
  }) =>
    request<ModelProfile>(`/models/${encodeURIComponent(modelId)}`, {
      method: 'POST',
      body: JSON.stringify(updates)
    }),

  // Delete/archive model
  delete: (modelId: string, archive = true) =>
    request<{ status: string; archived: boolean }>(
      `/models/${encodeURIComponent(modelId)}?archive=${archive}`,
      { method: 'DELETE' }
    ),

  // Star/unstar model
  star: (modelId: string, starred = true) =>
    request<{ status: string; starred: boolean }>(
      `/models/${encodeURIComponent(modelId)}/star?starred=${starred}`,
      { method: 'POST' }
    ),

  // Trigger evaluation
  evaluate: (modelId: string) =>
    request<{ status: string; model_id: string; message: string }>(
      `/models/${encodeURIComponent(modelId)}/evaluate`,
      { method: 'POST' }
    ),

  // Get artifacts
  getArtifacts: (modelId: string) =>
    request<{ model_id: string; model_dir: string; artifacts: ModelArtifacts }>(
      `/models/${encodeURIComponent(modelId)}/artifacts`
    ),

  // Rescan model directory
  rescan: (modelId: string) =>
    request<ModelProfile>(
      `/models/${encodeURIComponent(modelId)}/rescan`,
      { method: 'POST' }
    ),

  // Compare models
  compare: (modelIds: string[], metrics?: string[]) =>
    request<{ models: Record<string, Record<string, any>> }>(
      '/models/compare',
      {
        method: 'POST',
        body: JSON.stringify({ model_ids: modelIds, metrics })
      }
    ),

  // Get leaderboard
  leaderboard: (metric = 'custom_eval_pass_rate', limit = 10) =>
    request<{ metric: string; entries: LeaderboardEntry[] }>(
      `/models/leaderboard?metric=${metric}&limit=${limit}`
    ),

  // Get trends
  trends: (metric = 'benchmark_avg_score', days = 30) =>
    request<{ metric: string; data: TrendDataPoint[] }>(
      `/models/trends?metric=${metric}&days=${days}`
    ),

  // Run eval set on model
  runEval: (modelId: string, evalSetId: string, options?: { maxTokens?: number; temperature?: number }) =>
    request<{
      model_id: string
      eval_set_id: string
      status: string
      passed: number
      failed: number
      total: number
      pass_rate: number
      results: Array<{
        case_id: string
        case_name: string
        passed: boolean
        output?: string
        error?: string
      }>
    }>(`/models/${encodeURIComponent(modelId)}/run-eval`, {
      method: 'POST',
      body: JSON.stringify({
        eval_set_id: evalSetId,
        max_tokens: options?.maxTokens ?? 4096,
        temperature: options?.temperature ?? 0.0
      })
    }),

  // Export model to GGUF or other formats
  export: (modelId: string, options?: { format?: 'gguf' | 'safetensors' | 'lora'; quantization?: string }) =>
    request<ExportResponse>(`/models/${encodeURIComponent(modelId)}/export`, {
      method: 'POST',
      body: JSON.stringify({
        format: options?.format ?? 'gguf',
        quantization: options?.quantization ?? 'q4_k_m'
      })
    }),

  // Deploy model to Ollama
  deployToOllama: (modelId: string, options?: { modelName?: string; quantization?: string }) =>
    request<{ status: string; model_name: string; message: string }>(
      `/models/${encodeURIComponent(modelId)}/deploy-ollama`,
      {
        method: 'POST',
        body: JSON.stringify({
          model_name: options?.modelName,
          quantization: options?.quantization ?? 'q4_k_m'
        })
      }
    ),

  // Download artifact (returns blob URL for download)
  downloadArtifact: (modelId: string, artifactPath: string) =>
    request<{ download_url: string; filename: string }>(
      `/models/${encodeURIComponent(modelId)}/download?path=${encodeURIComponent(artifactPath)}`
    ),
}

// Custom Eval Set types
export interface EvalSetSummary {
  eval_set_id: string
  name: string
  description: string
  num_cases: number
  generation_mode: string
  source_traces: string[]
  created_at: string
}

export interface EvalCase {
  case_id: string
  name: string
  description: string
  system_prompt?: string
  user_prompt: string
  expected_behavior: string
  verification: {
    method: string
    expected_output?: string
    test_command?: string
    check_files?: string[]
    llm_criteria?: string
  }
  source_trace_id?: string
}

export interface EvalSetFull {
  eval_set_id: string
  name: string
  description: string
  cases: EvalCase[]
  generation_mode: string
  source_traces: string[]
  created_at: string
}

// Custom Eval Sets API
export const evalSetsApi = {
  // List all eval sets
  list: () =>
    request<EvalSetSummary[]>('/models/eval-sets'),

  // Get specific eval set
  get: (evalSetId: string) =>
    request<EvalSetFull>(`/models/eval-sets/${encodeURIComponent(evalSetId)}`),

  // Generate eval set from traces
  generate: (options: {
    name: string
    description?: string
    traceIds?: string[]
    mode?: 'replay' | 'variation' | 'both'
    maxCases?: number
    includeFailedTraces?: boolean
  }) =>
    request<{
      status: string
      eval_set_id: string
      name: string
      num_cases: number
      message: string
    }>('/models/eval-sets/generate', {
      method: 'POST',
      body: JSON.stringify({
        name: options.name,
        description: options.description,
        trace_ids: options.traceIds,
        mode: options.mode ?? 'both',
        max_cases: options.maxCases ?? 50,
        include_failed_traces: options.includeFailedTraces ?? false
      })
    }),

  // Delete eval set
  delete: (evalSetId: string) =>
    request<{ status: string; eval_set_id: string }>(
      `/models/eval-sets/${encodeURIComponent(evalSetId)}`,
      { method: 'DELETE' }
    )
}

// =============================================================================
// HuggingFace Integration
// =============================================================================

export interface HFStatus {
  enabled: boolean
  pro_enabled: boolean
  username: string
  namespace: string
  token_configured: boolean
  token_source: 'env' | 'stored' | ''
}

export interface HFConfigureResponse {
  success: boolean
  username?: string
  pro_enabled?: boolean
}

export interface HFJob {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  hardware: string
  created_at: string
  logs_url?: string
  error_message?: string
}

export interface HFJobSubmitRequest {
  dataset_repo: string
  output_repo: string
  hardware?: string
  base_model?: string
  num_epochs?: number
  learning_rate?: number
}

export interface HFSpace {
  space_name: string
  url: string
  status: 'building' | 'running' | 'stopped' | 'error'
}

export interface HFSpaceCreateRequest {
  model_repo: string
  space_name: string
  private?: boolean
  gpu_duration?: number
}

export interface HFDataset {
  repo_id: string
  url: string
  train_count: number
  val_count: number
}

export interface HFDatasetUploadRequest {
  local_path: string
  repo_name: string
  private?: boolean
  metadata?: Record<string, unknown>
}

export interface HFInferenceGenerateRequest {
  model: string
  prompt: string
  max_tokens?: number
  temperature?: number
}

export interface HFInferenceGenerateResponse {
  text: string
  finish_reason: string
  model: string
  provider?: string
}

export interface HFInferenceEmbedRequest {
  model: string
  texts: string[]
}

export interface HFInferenceEmbedResponse {
  embeddings: number[][]
  model: string
  dimension: number
}

// HuggingFace API
export const hfApi = {
  // Status
  getStatus: () =>
    request<HFStatus>('/hf/status'),

  // Token Configuration
  configureToken: (token: string) =>
    request<HFConfigureResponse>('/hf/configure', {
      method: 'POST',
      body: JSON.stringify({ token })
    }),

  removeToken: () =>
    request<{ success: boolean; deleted: boolean }>('/hf/configure', {
      method: 'DELETE'
    }),

  // Jobs
  listJobs: () =>
    request<HFJob[]>('/hf/jobs'),

  submitJob: (req: HFJobSubmitRequest) =>
    request<HFJob>('/hf/jobs', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  getJob: (jobId: string) =>
    request<HFJob>(`/hf/jobs/${encodeURIComponent(jobId)}`),

  getJobLogs: (jobId: string) =>
    request<{ logs: string }>(`/hf/jobs/${encodeURIComponent(jobId)}/logs`),

  cancelJob: (jobId: string) =>
    request<{ status: string; job_id: string }>(
      `/hf/jobs/${encodeURIComponent(jobId)}`,
      { method: 'DELETE' }
    ),

  // Spaces
  listSpaces: () =>
    request<HFSpace[]>('/hf/spaces'),

  createSpace: (req: HFSpaceCreateRequest) =>
    request<HFSpace>('/hf/spaces', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  getSpaceStatus: (spaceName: string) =>
    request<HFSpace>(`/hf/spaces/${encodeURIComponent(spaceName)}/status`),

  deleteSpace: (spaceName: string) =>
    request<{ status: string; space_name: string }>(
      `/hf/spaces/${encodeURIComponent(spaceName)}`,
      { method: 'DELETE' }
    ),

  // Datasets
  listDatasets: (prefix?: string) =>
    request<string[]>(`/hf/datasets${prefix ? `?prefix=${encodeURIComponent(prefix)}` : ''}`),

  uploadDataset: (req: HFDatasetUploadRequest) =>
    request<HFDataset>('/hf/datasets', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  deleteDataset: (repoName: string) =>
    request<{ status: string; repo_name: string }>(
      `/hf/datasets/${encodeURIComponent(repoName)}`,
      { method: 'DELETE' }
    ),

  // Inference
  generate: (req: HFInferenceGenerateRequest) =>
    request<HFInferenceGenerateResponse>('/hf/inference/generate', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  embed: (req: HFInferenceEmbedRequest) =>
    request<HFInferenceEmbedResponse>('/hf/inference/embed', {
      method: 'POST',
      body: JSON.stringify(req)
    })
}

// =============================================================================
// Synthetic Data Generation API
// =============================================================================

export interface SyntheticGenerateRequest {
  strategy: 'trace_seeded' | 'augmented' | 'schema_driven'
  repo_filter: 'single' | 'multi' | 'all'
  selected_repos: string[]
  preset: 'quick_test' | 'balanced' | 'production' | 'custom'
  target_examples?: number
  multiplier?: number
  provider: 'nim' | 'anthropic'
  merge_mode: 'synthetic_only' | 'mixed' | 'synthetic_weighted'
}

export interface SyntheticJobStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: { current: number; total: number }
  config?: Record<string, any>
  output_dir?: string
  error?: string
}

export interface SyntheticPreset {
  label: string
  description: string
  target_examples: number | null
}

export const syntheticApi = {
  // Start a new synthetic generation job
  generate: (req: SyntheticGenerateRequest) =>
    request<{ job_id: string; status: string }>('/factory/synthetic/generate', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  // Get job status
  getJobStatus: (jobId: string) =>
    request<SyntheticJobStatus>(`/factory/synthetic/jobs/${jobId}`),

  // List all jobs
  listJobs: () =>
    request<SyntheticJobStatus[]>('/factory/synthetic/jobs'),

  // Get available presets
  getPresets: () =>
    request<Record<string, SyntheticPreset>>('/factory/synthetic/presets')
}

// =============================================================================
// Integration API (Bashbros)
// =============================================================================

export interface IntegrationStatus {
  enabled: boolean
  linked: boolean
  linked_at: string | null
  bashbros_connected: boolean
  bashgym_connected: boolean
  pending_traces: number
  processed_traces: number
  current_model_version: string | null
  training_in_progress: boolean
}

export interface IntegrationSettings {
  version: string
  updated_at: string | null
  updated_by: string | null
  enabled: boolean
  linked_at: string | null
  capture_mode: 'everything' | 'successful_only' | 'sidekick_curated'
  auto_stream: boolean
  auto_training_enabled: boolean
  quality_threshold: number
  trigger: 'manual' | 'quality_based' | 'scheduled'
  bashbros_primary: boolean
  policy_path: string | null
  auto_export_ollama: boolean
  ollama_model_name: string
  notify_on_update: boolean
}

export interface ModelVersion {
  version: string
  created: string
  traces_used: number
  quality_avg: number
  is_latest: boolean
  gguf_available: boolean
}

export interface PendingTrace {
  filename: string
  task: string
  source: string
  verified: boolean
  steps: number
}

export interface ExportModelRequest {
  run_id: string
  quantization?: string
  traces_used?: number
  quality_avg?: number
}

export interface ExportModelResponse {
  success: boolean
  version?: string
  gguf_path?: string
  ollama_registered: boolean
  error?: string
}

export const integrationApi = {
  // Get integration status
  getStatus: () =>
    request<IntegrationStatus>('/integration/status'),

  // Get integration settings
  getSettings: () =>
    request<IntegrationSettings>('/integration/settings'),

  // Update integration settings
  updateSettings: (updates: Record<string, Record<string, any>>) =>
    request<IntegrationSettings>('/integration/settings', {
      method: 'PUT',
      body: JSON.stringify(updates)
    }),

  // Link integration
  link: () =>
    request<{ success: boolean; linked: boolean; linked_at: string | null }>('/integration/link', {
      method: 'POST'
    }),

  // Unlink integration
  unlink: () =>
    request<{ success: boolean; linked: boolean }>('/integration/unlink', {
      method: 'POST'
    }),

  // List pending traces
  listPendingTraces: () =>
    request<PendingTrace[]>('/integration/traces/pending'),

  // Process pending traces
  processTraces: () =>
    request<{ status: string; pending_count: number }>('/integration/traces/process', {
      method: 'POST'
    }),

  // List model versions
  listModelVersions: () =>
    request<ModelVersion[]>('/integration/models/versions'),

  // Export model to GGUF
  exportModel: (req: ExportModelRequest) =>
    request<ExportModelResponse>('/integration/models/export', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  // Rollback to previous model version
  rollbackModel: (version: string) =>
    request<ExportModelResponse>('/integration/models/rollback', {
      method: 'POST',
      body: JSON.stringify({ version })
    }),

  // Start trace watcher
  startWatcher: () =>
    request<{ status: string; message: string }>('/integration/watcher/start', {
      method: 'POST'
    }),

  // Stop trace watcher
  stopWatcher: () =>
    request<{ status: string; message: string }>('/integration/watcher/stop', {
      method: 'POST'
    }),

  // Get security policy
  getSecurityPolicy: () =>
    request<{ available: boolean; path: string | null; content: string | null; bashbros_primary: boolean }>(
      '/integration/security/policy'
    ),

  // Get integration directory info
  getDirectory: () =>
    request<{ base_path: string; directories: Record<string, string>; exists: boolean }>(
      '/integration/directory'
    )
}

// Achievements API
export interface AchievementStatusResponse {
  id: string
  name: string
  description: string
  category: string
  rarity: string
  icon: string
  points: number
  earned: boolean
  earned_at: string | null
  progress: number
}

export interface LifetimeStatsResponse {
  traces: {
    total: number
    gold: number
    silver: number
    bronze: number
    failed: number
    pending: number
    highest_quality: number
    avg_quality: number
    total_steps: number
    most_used_tool: string
    unique_repos: number
  }
  training: {
    runs_completed: number
    runs_by_strategy: Record<string, number>
    lowest_loss: number | null
    models_finetuned: number
    models_exported: number
    total_examples_generated: number
  }
  factory: {
    jobs_completed: number
    total_generated: number
    total_valid: number
  }
  router: {
    total_routed: number
    student_success_rate: number
    teacher_success_rate: number
  }
  first_activity: string
  days_active: number
  achievement_points: number
}

export interface AchievementsListResponse {
  achievements: AchievementStatusResponse[]
  earned_count: number
  total_count: number
  total_points: number
}

export interface RecentAchievementsResponse {
  recent: AchievementStatusResponse[]
  earned_count: number
  total_count: number
  total_points: number
}

export interface RefreshAchievementsResponse {
  newly_earned: AchievementStatusResponse[]
  earned_count: number
  total_count: number
  total_points: number
}

export const achievementsApi = {
  getStats: () =>
    request<LifetimeStatsResponse>('/achievements/stats'),

  getAll: () =>
    request<AchievementsListResponse>('/achievements'),

  getRecent: () =>
    request<RecentAchievementsResponse>('/achievements/recent'),

  refresh: () =>
    request<RefreshAchievementsResponse>('/achievements/refresh', { method: 'POST' })
}
