/**
 * Bash Gym API Client
 * Handles communication with the Python backend
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8002/api'
const BACKEND_START_COMMAND = 'bashgym serve --host 127.0.0.1 --port 8002'

function normalizeApiError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error || 'Unknown API error')
  const lower = raw.toLowerCase()
  const looksOffline =
    lower.includes('failed to fetch') ||
    lower.includes('fetch failed') ||
    lower.includes('networkerror') ||
    lower.includes('err_connection_refused') ||
    lower.includes('load failed')

  if (!looksOffline) {
    return raw
  }

  return [
    `Backend API is not reachable at ${API_BASE}.`,
    `Start it with: ${BACKEND_START_COMMAND}`,
    'If the backend is already running on another port, set VITE_API_URL to that /api URL.',
    `Original error: ${raw}`,
  ].join(' ')
}

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
  strategy?: 'sft' | 'dpo' | 'grpo' | 'distillation' | 'cascade'
  dataset_path?: string
  base_model?: string
  model_type?: string
  num_epochs?: number
  batch_size?: number
  learning_rate?: number
  warmup_ratio?: number
  gradient_accumulation_steps?: number
  max_seq_length?: number
  save_steps?: number
  // LoRA
  use_lora?: boolean
  lora_rank?: number
  lora_alpha?: number
  lora_dropout?: number
  load_in_4bit?: boolean
  // Strategy-specific
  dpo_beta?: number
  grpo_num_generations?: number
  grpo_temperature?: number
  grpo_loss_type?: string
  grpo_backend?: string
  grpo_use_vllm?: boolean
  training_profile?: string
  grpo_group_size?: number
  prompts_per_rollout_batch?: number
  max_tool_calls_per_episode?: number
  token_level_loss?: boolean
  filter_zero_std_groups?: boolean
  active_sampling?: boolean
  lm_head_fp32?: boolean
  interleaved_thinking?: boolean
  sft_warm_start_policy?: string
  dppo_backend?: string
  dppo_divergence?: string
  dppo_binary_tv_threshold?: number
  dppo_binary_kl_threshold?: number
  echo_enabled?: boolean
  echo_aux_lambda?: number
  rwml_enabled?: boolean
  rwml_distance_threshold?: number
  rwml_easy_pass_rate_threshold?: number
  rwml_easy_keep_probability?: number
  rwml_history_window?: number
  rwml_embedding_model?: string
  rwml_kl_beta?: number
  sft_backend?: string
  dpo_backend?: string
  use_liger?: boolean
  // Knowledge Distillation
  teacher_model?: string
  teacher_temperature?: number
  distillation_alpha?: number
  // Export
  auto_export_gguf?: boolean
  gguf_quantization?: string
  auto_deploy_ollama?: boolean
  ollama_model_name?: string
  // HuggingFace
  auto_push_hf?: boolean
  hf_repo_name?: string
  hf_private?: boolean
  // Backend
  use_nemo_gym?: boolean
  use_remote_ssh?: boolean
  device_id?: string
  selected_repos?: string[]
  // Data source
  data_source?: 'traces' | 'dataset_path' | 'security_dataset'
  security_dataset_type?: string
  security_dataset_path?: string
  security_conversion_mode?: string
  security_max_samples?: number
  security_balance_classes?: boolean
}

export interface TrainingResponse {
  run_id: string
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed'
  strategy: string
  message?: string
  started_at?: string
  completed_at?: string
  metrics?: TrainingStatusMetrics
  output_path?: string
}

// Metrics dict emitted by the trainer progress callbacks (see bashgym/gym/trainer.py
// and bashgym/api/routes.py). Mostly numeric, but `eta` is a preformatted string
// (e.g. "4m 30s") and `simulation` is a boolean flag.
export interface TrainingStatusMetrics {
  epoch?: number
  total_epochs?: number
  step?: number
  total_steps?: number
  loss?: number
  learning_rate?: number
  grad_norm?: number
  eval_loss?: number
  samples_processed?: number
  final_loss?: number
  eta?: string
  simulation?: boolean
  [key: string]: number | string | boolean | undefined
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
  cognitive_quality: number
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
  tool_breakdown?: Record<string, number>
  source_tool?: string
}

export interface TraceDetailInfo extends TraceInfo {
  duration_seconds?: number
  step_outcomes?: (boolean | null)[]
  cognitive_summary?: {
    planning_phases: number
    reflections: number
    thinking_steps: number
    cognitive_coverage: number
  }
  raw_metrics?: {
    total_steps: number
    successful_steps: number
    failed_steps: number
    unique_tools: number
    unique_commands: number
    cognitive_steps: number
  }
}

export interface ToolAnalyticsStat {
  tool: string
  calls: number
  sessions: number
  success_rate: number
  total_tokens: number
}

export interface TraceAnalytics {
  tool_stats: ToolAnalyticsStat[]
  quality_distribution: Record<string, number>
  totals: {
    sessions: number
    steps: number
    tokens: number
  }
  training_readiness: {
    sft_ready: number
    dpo_pairs_possible: number
    total_trainable: number
  }
  source_breakdown: Array<{ source: string; traces: number; steps: number; tokens: number }>
  cost_total_usd: number
  avg_quality_score: number
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

// Provider Health
export interface ProviderHealth {
  available: boolean
  latency_ms: number
  error: string | null
  models_loaded: string[]
  last_checked: string
  gpu_memory_used_mb: number | null
  gpu_memory_total_mb: number | null
}

export interface ProvidersHealthResponse {
  providers: Record<string, ProviderHealth>
  model_map: Record<string, string>
}

export interface RouterConfigResponse {
  strategy: string
  student_rate: number
  teacher_model: { name: string; type: string } | null
  student_model: { name: string; type: string } | null
  providers: {
    providers: Record<string, any>
    model_map: Record<string, string>
    total_models: number
  }
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

// Device management types
export interface DeviceCapabilities {
  python_version?: string
  cuda_version?: string
  gpus?: Array<{ name: string; vram_total_gb: number; vram_free_gb: number }>
  disk_free_gb?: number
  hostname?: string
  os?: string
}

export interface Device {
  id: string
  name: string
  host: string
  port: number
  username: string
  key_path: string
  work_dir: string
  is_default: boolean
  added_at: string
  last_seen?: string
  capabilities?: DeviceCapabilities
}

export interface NewDevice {
  name: string
  host: string
  port?: number
  username: string
  key_path?: string
  work_dir?: string
}

export interface SSHCandidate {
  ssh_alias: string
  host: string
  username?: string
  port: number
  key_path?: string
  already_added: boolean
  existing_device_id?: string
}

export interface DiscoverResult {
  candidates: SSHCandidate[]
  ssh_config_path: string
}

export interface PreflightResult {
  ok: boolean
  python_version?: string
  disk_free_gb?: number
  hostname?: string
  os_info?: string
  cuda_version?: string
  gpus?: Array<{ name: string; vram_total_gb: number; vram_free_gb: number }>
  error?: string
  device?: Device
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
      const apiResult = result as ApiResponse<T>
      if (!apiResult.ok && apiResult.error) {
        return { ...apiResult, error: normalizeApiError(apiResult.error) }
      }
      return apiResult
    }

    // Direct fetch fallback
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': localStorage.getItem('bashgym_api_key') || '',
        'X-Requested-With': 'XMLHttpRequest',
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
    return { ok: false, error: normalizeApiError(error) }
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
export interface TrainingRunSummary {
  run_id: string
  modified: number
  has_metrics: boolean
  has_final: boolean
}

export interface RunMetricPoint {
  step: number
  loss: number
  epoch?: number | null
  learning_rate?: number | null
  ts?: number
}

export interface DatasetInspectMessage {
  role: string | null
  content: string | null
  tool_calls?: unknown[] | null
  truncated?: boolean
}

export interface DatasetInspectExample {
  index: number
  messages: DatasetInspectMessage[]
  warnings: string[]
}

export interface DatasetInspectReport {
  total: number
  offset: number
  limit: number
  examples: DatasetInspectExample[]
  with_warnings_in_slice: number
}

export const trainingApi = {
  start: (config: TrainingRequest) =>
    request<TrainingResponse>('/training/start', {
      method: 'POST',
      body: JSON.stringify(config)
    }),

  managedSubmit: (body: { platform: string; base_model: string; dataset_path: string; n_epochs?: number; learning_rate?: number; suffix?: string; api_key?: string; account_id?: string }) =>
    request<{ job_id: string; backend: string; status: string; output_model?: string; error?: string }>('/training/managed/submit', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  managedPoll: (platform: string, jobId: string) =>
    request<{ job_id: string; backend: string; status: string; output_model?: string; error?: string }>(`/training/managed/${platform}/${jobId}`),

  getStatus: (runId: string) =>
    request<TrainingResponse>(`/training/${runId}`),

  pause: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/pause`, { method: 'POST' }),

  resume: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/resume`, { method: 'POST' }),

  stop: (runId: string) =>
    request<{ success: boolean; message: string }>(`/training/${runId}/stop`, { method: 'POST' }),

  list: (status?: string, limit?: number) =>
    request<TrainingResponse[]>(`/training${status ? `?status=${status}` : ''}${limit ? `${status ? '&' : '?'}limit=${limit}` : ''}`),

  // Export training examples to JSONL on the server
  exportExamples: (options?: { trace_ids?: string[]; include_gold_only?: boolean; train_split?: number }) =>
    request<{
      success: boolean
      train_path?: string
      val_path?: string
      train_count: number
      val_count: number
      message?: string
    }>('/training/export', {
      method: 'POST',
      body: JSON.stringify(options || {}),
    }),

  // Persisted run history (metrics.jsonl written next to checkpoints)
  listRuns: () =>
    request<{ runs: TrainingRunSummary[] }>('/training/runs'),

  getRunMetrics: (runId: string) =>
    request<{ run_id: string; metrics: RunMetricPoint[] }>(
      `/training/runs/${encodeURIComponent(runId)}/metrics`
    ),

  // Inspect exported training examples with chat-template validation
  inspectDataset: (offset = 0, limit = 10, path = '') =>
    request<DatasetInspectReport>(
      `/training/dataset/inspect?offset=${offset}&limit=${limit}${path ? `&path=${encodeURIComponent(path)}` : ''}`
    ),

  // Download exported JSONL file as browser download
  downloadExport: async (split: 'train' | 'val' = 'train') => {
    try {
      const response = await fetch(`${API_BASE}/training/export/download?split=${split}`, {
        credentials: 'include',
      })
      if (!response.ok) {
        const text = await response.text()
        return { ok: false as const, error: text }
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${split}.jsonl`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      return { ok: true as const, data: { downloaded: true } }
    } catch (e) {
      return { ok: false as const, error: String(e) }
    }
  },

  // Fetch last N lines of a run's training.log from disk
  getLog: (runId: string, opts?: { tail?: number; grep?: string }) => {
    const params = new URLSearchParams()
    if (opts?.tail !== undefined) params.set('tail', String(opts.tail))
    if (opts?.grep) params.set('grep', opts.grep)
    const qs = params.toString()
    return request<{
      run_id: string
      path: string
      total_lines: number
      truncated: boolean
      lines: string[]
    }>(`/training/${encodeURIComponent(runId)}/log${qs ? `?${qs}` : ''}`)
  },

  // List saved checkpoints on disk
  listCheckpoints: () =>
    request<Array<{
      id: string
      run_id: string
      kind: 'final' | 'merged' | 'intermediate'
      path: string
      size_mb: number
      created_at: string
      base_model: string | null
    }>>('/training/checkpoints'),

  // Delete a checkpoint directory
  deleteCheckpoint: (id: string) =>
    request<{ deleted: boolean; id: string }>(
      `/training/checkpoints/${encodeURIComponent(id)}`,
      { method: 'DELETE' }
    ),
}

// =============================================================================
// Data Designer API
// =============================================================================

export interface DesignerPipelineInfo {
  name: string
  description: string
  columns: string[]
}

export interface DesignerPreviewRequest {
  pipeline: string
  num_records?: number
  provider?: string
  provider_endpoint?: string
  text_model?: string
  code_model?: string
  judge_model?: string
}

export interface DesignerCreateRequest {
  pipeline: string
  num_records?: number
  seed_source?: string
  seed_type?: string
  seed_format?: string
  provider?: string
  provider_endpoint?: string
  text_model?: string
  code_model?: string
  judge_model?: string
  mcp_backend?: string
  output_dir?: string
  export_nemo?: boolean
  keep_only_passing?: boolean
  train_val_split?: number
}

export interface DesignerModel {
  id: string
  name: string
  provider: string
  is_code_model?: boolean
  is_local?: boolean
  parameter_size?: string | null
  description?: string | null
}

export interface DesignerJobStatus {
  job_id: string
  status: string
  pipeline: string
  num_records: number
  progress?: { current: number; total: number }
  output_dir?: string
  export_result?: Record<string, unknown>
  error?: string
}

export const designerApi = {
  listPipelines: () =>
    request<{ pipelines: DesignerPipelineInfo[]; available: boolean }>(
      '/factory/designer/pipelines'
    ),

  preview: (req: DesignerPreviewRequest) =>
    request<{ records: Record<string, unknown>[]; columns: string[]; count: number }>(
      '/factory/designer/preview',
      {
        method: 'POST',
        body: JSON.stringify({ num_records: 5, provider: 'nvidia', ...req }),
      }
    ),

  create: (req: DesignerCreateRequest) =>
    request<DesignerJobStatus>('/factory/designer/create', {
      method: 'POST',
      body: JSON.stringify({ num_records: 100, provider: 'nvidia', ...req }),
    }),

  getJob: (jobId: string) =>
    request<DesignerJobStatus>(`/factory/designer/jobs/${encodeURIComponent(jobId)}`),

  listModels: (codeOnly = false) =>
    request<{ models: DesignerModel[]; provider_models: string[]; available: boolean }>(
      `/factory/designer/models?code_only=${codeOnly}`
    ),
}

// =============================================================================
// Executable Environment API
// =============================================================================

export interface EnvironmentAxis {
  name: string
  value: string
  source?: string
  weight?: number | null
  metadata?: Record<string, unknown>
}

export interface EnvironmentVerifier {
  kind: string
  command: string
  path?: string | null
  reward_type?: string
  success_threshold?: number
  timeout_sec?: number
  metadata?: Record<string, unknown>
}

export interface EnvironmentBuildSpec {
  context_dir?: string
  dockerfile?: string | null
  base_image?: string | null
  compose_file?: string | null
  setup_commands?: string[]
  network_disabled?: boolean
  metadata?: Record<string, unknown>
}

export interface EnvironmentRolloutSpec {
  harness?: string
  max_steps?: number
  max_tool_calls?: number
  timeout_sec?: number
  bash_timeout_sec?: number
  max_prompt_tokens?: number
  max_response_tokens?: number
  metadata?: Record<string, unknown>
}

export interface TerminalEnvironmentSpec {
  id: string
  instruction: string
  source: string
  domain: string
  skills: string[]
  axes: EnvironmentAxis[]
  fixtures: Array<Record<string, unknown>>
  verifier: EnvironmentVerifier
  build: EnvironmentBuildSpec
  rollout: EnvironmentRolloutSpec
  files: Record<string, string>
  source_uri?: string | null
  license?: string | null
  metadata?: Record<string, unknown>
}

export interface EnvironmentMixReport {
  total: number
  domain_distribution: Record<string, number>
  skill_distribution: Record<string, number>
  axis_balance: Record<string, number>
  verifier_distribution: Record<string, number>
  mean_pass_rates?: Record<string, number>
}

export interface EnvironmentImportError {
  index: number
  id?: string
  error?: string
  validation_errors?: string[]
}

export interface EnvironmentNormalizeResponse {
  environments: TerminalEnvironmentSpec[]
  report: EnvironmentMixReport
  errors: EnvironmentImportError[]
}

export interface EnvironmentPipelineInfo {
  name: string
  available: boolean
  description: string
  outputs: string[]
}

export interface EnvironmentPipelinesResponse {
  available: boolean
  data_designer_available: boolean
  pipelines: EnvironmentPipelineInfo[]
  registered_pipelines: string[]
  external_sources: Record<string, string>
}

export interface EnvironmentDecontaminateResponse {
  environments: TerminalEnvironmentSpec[]
  report: { kept: number; dropped: number; drop_reasons: Record<string, number> }
  mix_report: EnvironmentMixReport
}

export interface EnvironmentMaterializeResponse {
  build: {
    env_id: string
    path: string
    files_written: string[]
  }
}

export const environmentApi = {
  pipelines: () =>
    request<EnvironmentPipelinesResponse>('/environments/pipelines'),

  normalize: (req: {
    records: Record<string, unknown>[]
    source?: string
    source_uri?: string
    preserve_raw?: boolean
  }) =>
    request<EnvironmentNormalizeResponse>('/environments/normalize', {
      method: 'POST',
      body: JSON.stringify({ source: 'external', preserve_raw: true, ...req }),
    }),

  importJsonl: (req: { path: string; source?: string; preserve_raw?: boolean }) =>
    request<EnvironmentNormalizeResponse>('/environments/import-jsonl', {
      method: 'POST',
      body: JSON.stringify({ source: 'tmax', preserve_raw: true, ...req }),
    }),

  decontaminate: (req: {
    environments: TerminalEnvironmentSpec[]
    benchmark_texts: string[]
    big_n?: number
    small_n?: number
    jaccard_threshold?: number
  }) =>
    request<EnvironmentDecontaminateResponse>('/environments/decontaminate', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  materialize: (req: {
    environment: TerminalEnvironmentSpec
    output_dir: string
    overwrite?: boolean
  }) =>
    request<EnvironmentMaterializeResponse>('/environments/materialize', {
      method: 'POST',
      body: JSON.stringify({ overwrite: false, ...req }),
    }),
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

// Advanced eval: held-out trace gate + benchmark wiring (bashgym/eval modules)
export interface EvalEndpointSpec {
  provider?: string
  base_url?: string
  model?: string
  api_key?: string
}

export interface HeldoutEvalRequest {
  model_id: string
  dataset_path: string
  candidate: EvalEndpointSpec
  base: EvalEndpointSpec
  metric?: string
  limit?: number
  min_trace_delta?: number
  max_forgetting_drop?: number
  require_ci_excludes_zero?: boolean
  forgetting_drops?: Record<string, number>
  environment_evidence?: HeldoutEnvironmentEvidence
  n_resamples?: number
  seed?: number
}

export interface HeldoutEnvironmentEvidence {
  passk?: EnvironmentPassKReport | Record<string, unknown> | null
  holdout_gate?: EnvironmentHoldoutGateResult | EnvironmentHoldoutGateResponse | Record<string, unknown> | null
  holdout_comparison?:
    | EnvironmentHoldoutComparisonResult
    | EnvironmentHoldoutComparisonResponse
    | Record<string, unknown>
    | null
  spurious_reward_control?:
    | EnvironmentSpuriousRewardControlResult
    | EnvironmentSpuriousRewardControlResponse
    | Record<string, unknown>
    | null
  external_benchmarks?: ExternalBenchmarkIngestResponse | ExternalBenchmarkReport | Record<string, unknown> | null
  world_model_quality?: Record<string, unknown> | null
  external_benchmark_min_scores?: Record<string, number> | null
  external_benchmarks_required?: boolean
  required?: boolean
}

export interface HeldoutBootstrap {
  mean: number
  ci_low: number
  ci_high: number
  significant: boolean
  better: boolean
  n_clusters: number
}

export interface HeldoutReleaseGate {
  schema_version: string
  ship: boolean
  trace_ship: boolean
  environment_ship: boolean
  external_benchmark_ship?: boolean
  world_model_quality_present?: boolean
  world_model_quality_diagnostic_only?: boolean
  world_model_quality_signal?: string
  world_model_quality_findings?: string[]
  world_model_quality?: {
    present: boolean
    diagnostic_only: boolean
    signal: string
    metrics: Record<string, number>
    findings: string[]
    coverage?: Record<string, unknown> | null
  }
  trace_reasons: string[]
  environment_reasons: string[]
  external_benchmark_reasons?: string[]
  environment_required: boolean
  environment_sections: string[]
  blocking_environment_sections: string[]
  external_benchmark_sections?: string[]
  blocking_external_benchmark_sections?: string[]
  world_model_quality_sections?: string[]
}

export interface HeldoutReport {
  n: number
  n_clusters: number
  metric: string
  base_pass_rate: number
  candidate_pass_rate: number
  trace_delta: number
  bootstrap: HeldoutBootstrap
  forgetting_drops: Record<string, number>
  ship: boolean
  reasons: string[]
  release_gate?: HeldoutReleaseGate
  environment_evidence?: HeldoutEnvironmentEvidence
}

export interface HeldoutJobResponse {
  job_id: string
  model_id: string
  metric: string
  status: 'running' | 'completed' | 'failed'
  report?: HeldoutReport
  error?: string
  created_at?: string
}

export interface VerdictResponse {
  model_id: string
  display_name: string
  latest_heldout_eval: HeldoutReport | null
  n_heldout_evals: number
  latest_environment_holdout_eval: EnvironmentHoldoutGateResult | Record<string, unknown> | null
  n_environment_holdout_evals: number
}

export interface BenchmarkIngestRequest {
  model_id: string
  base_results: Record<string, unknown>
  candidate_results: Record<string, unknown>
  max_forgetting_drop?: number
}

export interface NormalizedBenchmarkResult {
  name: string
  score: number
  passed: number
  total: number
  metrics: Record<string, number>
  error?: string | null
}

export interface ExternalBenchmarkReport {
  scores: Record<string, number>
  failures: string[]
  results: NormalizedBenchmarkResult[]
}

export interface ExternalBenchmarkIngestRequest {
  model_id: string
  results: unknown
  benchmark_name?: string | null
  source?: string | null
  record_to_registry?: boolean
}

export interface ExternalBenchmarkIngestResponse {
  model_id: string
  report: ExternalBenchmarkReport
  recorded: string[]
}

export interface EnvironmentAttemptPayload {
  environment_id: string
  attempt_index: number
  passed: boolean
  reward?: number | null
  verifier_status?: string | null
  timeout?: boolean
  tool_calls?: number | null
  tokens?: number | null
  action_tokens?: number | null
  observation_tokens?: number | null
  metadata?: Record<string, unknown>
}

export interface EnvironmentCommandAttemptPayload {
  environment_id: string
  attempt_index: number
  commands: string[]
  metadata?: Record<string, unknown>
}

export interface EnvironmentPassKRequest {
  model_id?: string | null
  environments: TerminalEnvironmentSpec[]
  attempts: EnvironmentAttemptPayload[]
  k_values?: number[]
  record_to_registry?: boolean
}

export interface EnvironmentPassKReport {
  k_values: number[]
  n_environments: number
  n_attempts: number
  pass_at_k: Record<string, number>
  mean_success_rate: number
  per_environment: Record<string, Record<string, number>>
  attempt_summary: {
    timeout_rate: number
    mean_tool_calls?: number | null
    mean_tokens?: number | null
    mean_action_tokens?: number | null
    mean_observation_tokens?: number | null
    verifier_status_distribution: Record<string, number>
  }
  warnings: string[]
}

export interface EnvironmentPassKResponse {
  model_id?: string | null
  report: EnvironmentPassKReport
  recorded: string[]
}

export interface EnvironmentHoldoutGateRequest {
  model_id?: string | null
  environments: TerminalEnvironmentSpec[]
  attempts: EnvironmentAttemptPayload[]
  split_by?: string
  holdout_fraction?: number
  seed?: number
  k_values?: number[]
  min_pass_at_1?: number
  max_timeout_rate?: number
  max_tamper_rate?: number
  require_no_contamination?: boolean
  record_to_registry?: boolean
}

export interface EnvironmentHoldoutGateResult {
  schema_version: string
  split: {
    schema_version: string
    split_by: string
    fraction: number
    seed: number
    n_train: number
    n_holdout: number
    train_ids: string[]
    holdout_ids: string[]
    train_hashes: string[]
    holdout_hashes: string[]
    train_group_keys: string[]
    holdout_group_keys: string[]
  }
  contamination: string[]
  report: EnvironmentPassKReport
  gate: {
    ship: boolean
    reasons: string[]
    thresholds: {
      min_pass_at_1: number
      max_timeout_rate: number
      max_tamper_rate: number
      require_no_contamination: boolean
    }
    observed: {
      pass_at_1: number
      timeout_rate: number
      tamper_rate: number
    }
  }
}

export interface EnvironmentHoldoutGateResponse {
  model_id?: string | null
  result: EnvironmentHoldoutGateResult
  recorded: string[]
}

export interface EnvironmentHoldoutComparisonRequest {
  environments: TerminalEnvironmentSpec[]
  base_attempts: EnvironmentAttemptPayload[]
  candidate_attempts: EnvironmentAttemptPayload[]
  split_by?: string
  cluster_by?: string
  holdout_fraction?: number
  seed?: number
  k_values?: number[]
  compare_k?: number
  min_delta?: number
  min_candidate_pass_at_1?: number
  require_ci_excludes_zero?: boolean
  max_candidate_timeout_rate?: number
  max_candidate_tamper_rate?: number
  require_no_contamination?: boolean
  n_resamples?: number
}

export interface EnvironmentHoldoutComparisonResult {
  schema_version: string
  split: EnvironmentHoldoutGateResult['split']
  cluster_by: string
  compare_metric: string
  contamination: string[]
  base_report: EnvironmentPassKReport
  candidate_report: EnvironmentPassKReport
  per_environment: Record<string, {
    cluster: string
    base: number
    candidate: number
    delta: number
  }>
  bootstrap: {
    mean: number
    ci_low: number
    ci_high: number
    significant: boolean
    better: boolean
    n: number
    n_clusters: number
  }
  gate: {
    ship: boolean
    reasons: string[]
    thresholds: {
      min_delta: number
      min_candidate_pass_at_1: number
      require_ci_excludes_zero: boolean
      max_candidate_timeout_rate: number
      max_candidate_tamper_rate: number
      require_no_contamination: boolean
    }
    observed: {
      delta: number
      ci_low: number
      ci_high: number
      candidate_pass_at_1: number
      candidate_timeout_rate: number
      candidate_tamper_rate: number
    }
  }
}

export interface EnvironmentHoldoutComparisonResponse {
  result: EnvironmentHoldoutComparisonResult
}

export interface EnvironmentSpuriousRewardControlRequest {
  environments: TerminalEnvironmentSpec[]
  attempts: EnvironmentAttemptPayload[]
  control_attempts?: EnvironmentAttemptPayload[] | null
  split_by?: string
  holdout_fraction?: number
  seed?: number
  k_values?: number[]
  n_trials?: number
  random_pass_probability?: number
  min_observed_pass_at_1?: number
  max_control_pass_at_1?: number
  min_lift_over_control?: number
  require_no_contamination?: boolean
}

export interface EnvironmentSpuriousRewardControlResult {
  schema_version: string
  split: EnvironmentHoldoutGateResult['split']
  contamination: string[]
  observed_report: EnvironmentPassKReport
  control: {
    mode: string
    n_trials: number
    random_pass_probability?: number | null
    report?: EnvironmentPassKReport | null
    pass_at_k_summary: Record<string, {
      mean: number
      p05: number
      p50: number
      p95: number
      min: number
      max: number
    }>
  }
  gate: {
    ship: boolean
    reasons: string[]
    thresholds: {
      min_observed_pass_at_1: number
      max_control_pass_at_1: number
      min_lift_over_control: number
      require_no_contamination: boolean
    }
    observed: {
      observed_pass_at_1: number
      control_pass_at_1: number
      control_stat: string
      lift_over_control: number
    }
  }
}

export interface EnvironmentSpuriousRewardControlResponse {
  result: EnvironmentSpuriousRewardControlResult
}

export interface EnvironmentRolloutObservation {
  command: string
  cwd: string
  exit_code: number
  stdout: string
  stderr: string
  duration_sec: number
  timeout: boolean
  blocked: boolean
}

export interface EnvironmentRolloutResultPayload {
  attempt: EnvironmentAttemptPayload
  workspace: string
  observations: EnvironmentRolloutObservation[]
  verifier_observation?: EnvironmentRolloutObservation | null
}

export interface EnvironmentRolloutSamplingReport {
  sampling_enabled: boolean
  active_sampling_refills: number
  zero_std_groups_dropped: number
  all_zero_groups_dropped: number
  all_one_groups_dropped: number
  effective_prompt_groups: number
  requested_prompt_groups: number
  candidate_prompt_groups: number
  maintained_batch: boolean
  selected_environment_ids: string[]
  dropped_environment_ids: string[]
}

export interface EnvironmentDppoReadinessReport {
  attempts: number
  attempts_with_behavior_logprobs: number
  behavior_logprob_tokens: number
  missing_behavior_logprob_attempts: number
  attempts_with_train_logprobs: number
  train_logprob_tokens: number
  missing_train_logprob_attempts: number
  rollout_logprobs_ready: boolean
  optimizer_logprobs_ready: boolean
  needs_train_logprob_replay: boolean
}

export interface EnvironmentDppoReplaySummary {
  schema_version: string
  records: number
  environments: number
  environment_ids: string[]
  behavior_logprobs_ready_records: number
  train_logprobs_ready_records: number
  train_logprob_replay_required_records: number
  world_model_records: number
  world_model?: {
    records: number
    records_missing_world_model: number
    rwml_transitions: number
    rwml_mean_transitions_per_record: number
    rwml_mean_prior_pairs: number
    rwml_max_prior_pairs: number
    echo_segments: number
    echo_action_chars: number
    echo_observation_chars: number
    echo_observation_char_fraction: number
  }
  missing_behavior_logprob_records?: number
  mismatched_train_logprob_records?: number
  dppo?: {
    n_tokens: number
    masked_updates: number
    masked_fraction: number
    mean_binary_tv: number
    max_binary_tv: number
    mean_binary_kl: number
    max_binary_kl: number
    mean_abs_logprob_diff: number
    max_abs_logprob_diff: number
    mean_abs_policy_mismatch: number
    max_abs_policy_mismatch: number
    collapse_warning: boolean
    divergence: string
    threshold: number
  }
  path: string
  input_path?: string
  batch_id?: string | null
}

export interface DPPOTrainLogprobsSpec {
  environment_id: string
  attempt_index: number
  token_logprobs: number[]
  tokens?: string[] | null
  model?: string | null
  base_url?: string | null
}

export interface DPPOReplayEnrichRequest {
  input_path: string
  output_path: string
  train_logprobs: DPPOTrainLogprobsSpec[]
  divergence?: 'binary_tv' | 'binary_kl'
  threshold?: number | null
}

export interface DPPOSmokeLaunchPlanRequest {
  replay_path: string
  output_dir: string
  base_model: string
  backend?: 'auto' | 'verl' | 'skyrl' | 'tmax_open_instruct' | 'grpo_fallback'
  max_steps?: number
  n_gpus_per_node?: number
  write_script?: boolean
  command_template?: string | null
  echo_enabled?: boolean
  echo_aux_lambda?: number
  rwml_enabled?: boolean
  rwml_distance_threshold?: number
  rwml_easy_pass_rate_threshold?: number
  rwml_easy_keep_probability?: number
  rwml_history_window?: number
  rwml_embedding_model?: string
  rwml_kl_beta?: number
}

export interface DPPOSmokeLaunchPlan {
  backend: string
  requested_backend: string
  available: boolean
  fallback_to_grpo: boolean
  runnable: boolean
  reason: string
  command: string[]
  cwd?: string | null
  env: Record<string, string>
  warnings: string[]
  script_path?: string | null
  replay_path: string
  output_dir: string
  base_model: string
  max_steps: number
  n_gpus_per_node: number
  world_model?: {
    echo_enabled?: boolean
    echo_aux_lambda?: number
    rwml_enabled?: boolean
    rwml_distance_threshold?: number
    rwml_easy_pass_rate_threshold?: number
    rwml_easy_keep_probability?: number
    rwml_history_window?: number
    rwml_embedding_model?: string
    rwml_kl_beta?: number
  }
}

export interface EnvironmentRolloutPassKRequest {
  model_id?: string | null
  environments: TerminalEnvironmentSpec[]
  command_attempts: EnvironmentCommandAttemptPayload[]
  k_values?: number[]
  workspace_root?: string | null
  keep_workspace?: boolean
  allow_dangerous_commands?: boolean
  stop_on_error?: boolean
  record_to_registry?: boolean
}

export interface EnvironmentRolloutPassKResponse {
  model_id?: string | null
  report: EnvironmentPassKReport
  attempts: EnvironmentAttemptPayload[]
  rollouts: EnvironmentRolloutResultPayload[]
  recorded: string[]
  sampling_report?: EnvironmentRolloutSamplingReport | null
  dppo_report?: EnvironmentDppoReadinessReport | null
  dppo_replay?: EnvironmentDppoReplaySummary | null
}

export interface EnvironmentCanarySuiteRequest {
  categories?: string[]
  workspace_root?: string | null
  keep_workspace?: boolean
}

export interface EnvironmentCanaryResult {
  canary_id: string
  category: string
  name: string
  guarded: boolean
  expected_status: string
  verifier_status: string
  passed: boolean
  tamper_detected: boolean
  workspace?: string | null
}

export interface EnvironmentCanarySummary {
  total: number
  guarded: number
  failed: number
  guard_rate: number
  categories: Record<string, number>
  results: EnvironmentCanaryResult[]
}

export interface EnvironmentCanarySpec {
  id: string
  name: string
  category: string
  description: string
  environment: TerminalEnvironmentSpec
  attack_commands: string[]
  expected_status: string
}

export interface EnvironmentCanarySuiteResponse {
  summary: EnvironmentCanarySummary
  canaries: EnvironmentCanarySpec[]
  rollouts: EnvironmentRolloutResultPayload[]
}

export interface EnvironmentModelRolloutPassKRequest {
  model_id?: string | null
  endpoint: EvalEndpointSpec
  environments: TerminalEnvironmentSpec[]
  attempts_per_environment?: number
  k_values?: number[]
  workspace_root?: string | null
  keep_workspace?: boolean
  allow_dangerous_commands?: boolean
  stop_on_error?: boolean
  max_tool_calls?: number | null
  max_observation_chars?: number
  temperature?: number
  max_tokens?: number
  request_timeout?: number
  use_tool_calling?: boolean
  capture_logprobs?: boolean
  top_logprobs?: number | null
  filter_zero_std_groups?: boolean
  active_sampling?: boolean
  target_prompt_groups?: number | null
  dppo_replay_output_path?: string | null
  include_world_model_replay?: boolean
  rwml_history_window?: number
  record_to_registry?: boolean
}

export const evalAdvancedApi = {
  runHeldout: (req: HeldoutEvalRequest) =>
    request<HeldoutJobResponse>('/eval/heldout', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  heldoutStatus: (jobId: string) =>
    request<HeldoutJobResponse>(`/eval/heldout/${jobId}`),

  heldoutList: (limit = 20) =>
    request<HeldoutJobResponse[]>(`/eval/heldout?limit=${limit}`),

  verdict: (modelId: string) =>
    request<VerdictResponse>(`/eval/verdict/${encodeURIComponent(modelId)}`),

  benchmarkCommands: (params: { base_url: string; model: string; tasks?: string; include?: string }) => {
    const qs = new URLSearchParams({ base_url: params.base_url, model: params.model })
    if (params.tasks) qs.set('tasks', params.tasks)
    if (params.include) qs.set('include', params.include)
    return request<{ commands: Record<string, string[]> }>(`/eval/benchmark-commands?${qs.toString()}`)
  },

  ingestBenchmarks: (req: BenchmarkIngestRequest) =>
    request<{
      model_id: string
      forgetting: { drops: Record<string, number>; regressed: Record<string, number>; worst: [string, number] | null }
      recorded: string[]
      max_forgetting_drop: number
      forgetting_ok: boolean
      worst: [string, number] | null
    }>('/eval/benchmarks/ingest', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  ingestExternalBenchmarks: (req: ExternalBenchmarkIngestRequest) =>
    request<ExternalBenchmarkIngestResponse>('/eval/benchmarks/external-ingest', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  environmentPassk: (req: EnvironmentPassKRequest) =>
    request<EnvironmentPassKResponse>('/eval/environments/passk', {
      method: 'POST',
      body: JSON.stringify({ record_to_registry: false, ...req })
    }),

  environmentHoldoutGate: (req: EnvironmentHoldoutGateRequest) =>
    request<EnvironmentHoldoutGateResponse>('/eval/environments/holdout-gate', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  environmentHoldoutComparison: (req: EnvironmentHoldoutComparisonRequest) =>
    request<EnvironmentHoldoutComparisonResponse>('/eval/environments/holdout-comparison', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  environmentSpuriousRewardControl: (req: EnvironmentSpuriousRewardControlRequest) =>
    request<EnvironmentSpuriousRewardControlResponse>('/eval/environments/spurious-reward-control', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  environmentLocalRolloutPassk: (req: EnvironmentRolloutPassKRequest) =>
    request<EnvironmentRolloutPassKResponse>('/eval/environments/local-rollout-passk', {
      method: 'POST',
      body: JSON.stringify({ record_to_registry: false, keep_workspace: true, ...req })
    }),

  environmentModelRolloutPassk: (req: EnvironmentModelRolloutPassKRequest) =>
    request<EnvironmentRolloutPassKResponse>('/eval/environments/model-rollout-passk', {
      method: 'POST',
      body: JSON.stringify({ record_to_registry: false, keep_workspace: true, ...req })
    }),

  environmentRewardHackingCanaries: (req: EnvironmentCanarySuiteRequest = {}) =>
    request<EnvironmentCanarySuiteResponse>('/eval/environments/reward-hacking-canaries', {
      method: 'POST',
      body: JSON.stringify({ keep_workspace: true, ...req })
    }),

  enrichDppoReplay: (req: DPPOReplayEnrichRequest) =>
    request<{ dppo_replay: EnvironmentDppoReplaySummary }>('/eval/environments/dppo-replay/enrich', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  planDppoSmoke: (req: DPPOSmokeLaunchPlanRequest) =>
    request<{ plan: DPPOSmokeLaunchPlan }>('/eval/environments/dppo-replay/smoke-plan', {
      method: 'POST',
      body: JSON.stringify(req)
    })
}

// Traces API
export const tracesApi = {
  list: (options?: { status?: 'gold' | 'silver' | 'bronze' | 'failed' | 'pending', repo?: string, source_tool?: string, limit?: number, offset?: number }) => {
    const params = new URLSearchParams()
    if (options?.status) params.set('status', options.status)
    if (options?.repo) params.set('repo', options.repo)
    if (options?.source_tool) params.set('source_tool', options.source_tool)
    if (options?.limit) params.set('limit', String(options.limit))
    if (options?.offset) params.set('offset', String(options.offset))
    const queryString = params.toString()
    return request<{ traces: TraceInfo[]; total: number; offset: number; limit: number; counts: { gold: number; silver: number; bronze: number; failed: number; pending: number } }>(`/traces${queryString ? `?${queryString}` : ''}`)
  },

  listRepos: () =>
    request<RepoInfo[]>('/traces/repos'),

  stats: (options?: { range?: string }) => {
    const params = new URLSearchParams()
    if (options?.range) params.set('range', options.range)
    const qs = params.toString()
    return request<{
      timeline: { time: string; gold: number; failed: number; pending: number }[]
      totals: { gold: number; failed: number; pending: number; total: number }
    }>(`/traces/stats${qs ? `?${qs}` : ''}`)
  },

  getGold: (limit?: number) =>
    request<TraceInfo[]>(`/traces/gold${limit ? `?limit=${limit}` : ''}`),

  get: (traceId: string) =>
    request<TraceDetailInfo>(`/traces/${traceId}`),

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

  // Trigger import of new Claude Code sessions from ~/.claude/projects/
  triggerImport: () =>
    request<{ imported: number; total: number; skipped: number; errors: number; new_trace_ids: string[] }>(
      '/traces/import',
      { method: 'POST' }
    ),

  // Import traces from a specific source tool
  importBySource: (source: string, options?: { days?: number; limit?: number; force?: boolean }) =>
    request<{
      source: string
      imported: number
      skipped: number
      errors: number
      total: number
      new_trace_ids: string[]
    }>(`/traces/import/${source}`, {
      method: 'POST',
      body: JSON.stringify(options || {}),
    }),

  // Upload and import trace files from external AI tools (ChatGPT, MCP)
  uploadAndImport: async (file: File, source: 'chatgpt' | 'mcp', force = false): Promise<ApiResponse<{
    source: string
    imported_count: number
    skipped_count: number
    failed_count: number
    total_steps: number
    errors: string[]
  }>> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('source', source)
    formData.append('force', String(force))

    try {
      const response = await fetch(`${API_BASE}/traces/upload/import`, {
        method: 'POST',
        body: formData,
        credentials: 'include',
      })
      const text = await response.text()
      try {
        const data = JSON.parse(text)
        return { ok: response.ok, data }
      } catch {
        return { ok: false, error: text }
      }
    } catch (e) {
      return { ok: false, error: String(e) }
    }
  },

  // Import traces from all detected source tools
  importAll: (options?: { days?: number; limit?: number }) =>
    request<{
      results: Array<{
        source: string
        imported: number
        skipped: number
        errors: number
        total: number
      }>
      total_imported: number
    }>('/traces/import/all', {
      method: 'POST',
      body: JSON.stringify(options || {}),
    }),

  // Get count of traces created since a given ISO timestamp
  importSince: (since: string) =>
    request<{ count: number; traces: string[] }>(
      `/traces/import-since?since=${encodeURIComponent(since)}`
    ),

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
    }),

  getAnalytics: () =>
    request<TraceAnalytics>('/traces/analytics')
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
    }),

  getConfig: () =>
    request<RouterConfigResponse>('/router/config'),

  setStudentProvider: (providerType: string, modelName: string) =>
    request<{ success: boolean; provider: string; model: string; is_local: boolean }>(
      `/router/student-provider?provider_type=${encodeURIComponent(providerType)}&model_name=${encodeURIComponent(modelName)}`,
      { method: 'POST' }
    ),
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
  // Largest model (billions of params) that fits per regime, from the estimator.
  regime_capacities?: Record<string, number>
  // Budget is unified memory (RAM-backed, e.g. DGX Spark)
  unified_memory?: boolean
}

export interface DiscoveredModelFit {
  id: string
  params_billions: number | null
  can_infer: boolean | null
  can_qlora: boolean | null
  can_lora: boolean | null
  can_full: boolean | null
}

export interface DiscoveredModel {
  id: string
  downloads: number
  likes: number
  tags: string[]
  pipeline_tag?: string | null
  params_billions: number | null
  hf_url: string
  fit?: DiscoveredModelFit | null
}

export const systemInfoApi = {
  getInfo: (refresh?: boolean) =>
    request<SystemInfo>(`/system/info${refresh ? '?refresh=true' : ''}`),

  getGpus: () =>
    request<GpuInfo[]>('/system/gpus'),

  // Pass a registered SSH device_id to target that machine's discovered budget
  // (e.g. a unified-memory DGX Spark) instead of the local GPU.
  getRecommendations: (deviceId?: string) =>
    request<ModelRecommendations>(
      `/system/recommendations${deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : ''}`
    ),

  // Live-discover open base models from HuggingFace for the fine-tuning directory.
  discoverModels: (deviceId?: string) =>
    request<{ models: DiscoveredModel[]; budget_vram_gb?: number; error?: string }>(
      `/models/discover${deviceId ? `?device_id=${encodeURIComponent(deviceId)}` : ''}`
    )
}

export const sshApi = {
  preflight: () =>
    request<{ ok: boolean; python_version?: string; disk_free_gb?: number; error?: string; host?: string; username?: string }>('/ssh/preflight'),
}

export const deviceApi = {
  list: () => request<Device[]>('/devices'),
  add: (device: NewDevice) => request<Device>('/devices', { method: 'POST', body: JSON.stringify(device), headers: { 'Content-Type': 'application/json' } }),
  update: (id: string, device: Partial<NewDevice>) => request<Device>(`/devices/${id}`, { method: 'PUT', body: JSON.stringify(device), headers: { 'Content-Type': 'application/json' } }),
  remove: (id: string) => request<{ ok: boolean }>(`/devices/${id}`, { method: 'DELETE' }),
  preflight: (id: string) => request<PreflightResult>(`/devices/${id}/preflight`, { method: 'POST' }),
  setDefault: (id: string) => request<Device>(`/devices/${id}/set-default`, { method: 'POST' }),
  discover: () => request<DiscoverResult>('/devices/discover', { method: 'POST' }),
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

  connect: (body: { platform?: string; base_url?: string; api_key?: string; default_model?: string; name?: string }) =>
    request<{ ok: boolean; provider_type?: string; available?: boolean; models?: string[]; error?: string }>('/providers/connect', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

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
    }),

  getHealth: () =>
    request<ProvidersHealthResponse>('/providers/health'),

  warmupOllamaModel: (modelName: string) =>
    request<{ success: boolean; model: string }>(
      `/providers/ollama/warmup?model_name=${encodeURIComponent(modelName)}`,
      { method: 'POST' }
    ),
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

export interface ToolStat {
  tool: string
  calls: number
  avg_duration_ms: number
  success_rate: number
  total_tokens: number
}

export interface TraceSummary {
  trace_id: string
  name: string
  duration_ms: number
  total_spans: number
  status?: 'success' | 'error' | 'in_progress'
  llm_calls: Record<string, any>
  tool_calls: Record<string, any>
  bottlenecks?: Array<Record<string, any>>
}

export interface TraceSpan {
  span_id: string
  name: string
  kind: string
  duration_ms: number
  status: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  latency_ms: number
  attributes: Record<string, any>
}

export interface TraceDetail extends TraceSummary {
  spans: TraceSpan[]
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
    request<TraceDetail>(`/observability/traces/${traceId}`),

  getToolStats: () =>
    request<ToolStat[]>('/observability/tool-stats'),

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
  hf_repo_id?: string | null
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
  hf_repo_id?: string | null

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
  metrics?: Record<string, number>
}

export interface HFJobSubmitRequest {
  dataset_repo: string
  output_repo: string
  hardware?: string
  base_model?: string
  num_epochs?: number
  learning_rate?: number
  strategy?: 'sft' | 'dpo' | 'distillation'
  batch_size?: number
  lora_r?: number
  lora_alpha?: number
  max_seq_length?: number
}

export interface HFHardwareTier {
  id: string
  gpu: string | null
  vram_gb: number
  cost_per_hour: number
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

export interface HFModelPushRequest {
  model_id: string
  repo_name?: string
  private?: boolean
  push_gguf?: boolean
  generate_card?: boolean
}

export interface HFModelPushResponse {
  repo_id: string
  url: string
  gguf_repo_id?: string
  gguf_url?: string
  card_generated: boolean
}

export interface HFMyModel {
  id: string
  url: string
  downloads: number
  likes: number
  private: boolean
  last_modified: string
  pipeline_tag?: string
  tags: string[]
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

  getHardware: () =>
    request<HFHardwareTier[]>('/hf/jobs/hardware'),

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
    }),

  pushModel: (req: HFModelPushRequest) =>
    request<HFModelPushResponse>('/hf/models/push', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  listMyModels: () =>
    request<HFMyModel[]>('/hf/models/mine'),

  deleteHFModel: (repoId: string) =>
    request<{ status: string }>(`/hf/models/${encodeURIComponent(repoId)}`, { method: 'DELETE' }),

  // Storage Buckets (huggingface_hub v1.10+)
  listBuckets: (namespace?: string) =>
    request<any[]>(`/hf/buckets${namespace ? `?namespace=${encodeURIComponent(namespace)}` : ''}`),
  createBucket: (body: { bucket_id: string; private?: boolean }) =>
    request<{ bucket_id: string; url: string }>('/hf/buckets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  listBucketTree: (bucketId: string, prefix?: string) =>
    request<any[]>(`/hf/buckets/${encodeURIComponent(bucketId)}/tree${prefix ? `?prefix=${encodeURIComponent(prefix)}` : ''}`),
  getBucketInfo: (bucketId: string) =>
    request<any>(`/hf/buckets/${encodeURIComponent(bucketId)}/info`),
  syncBucket: (body: { source: string; dest: string; delete?: boolean; dry_run?: boolean }) =>
    request<any>('/hf/buckets/sync', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  deleteBucket: (bucketId: string) =>
    request<{ status: string }>(`/hf/buckets/${encodeURIComponent(bucketId)}`, { method: 'DELETE' }),
  copyFiles: (body: { source: string; destination: string }) =>
    request<{ source: string; destination: string; status: string }>('/hf/copy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  // Agent Traces
  uploadTraces: (body: { trace_dir: string; repo_id: string; private?: boolean }) =>
    request<{ repo_id: string; url: string; num_traces: number; total_size_bytes: number }>('/hf/traces/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  listTraceDatasets: (prefix?: string) =>
    request<any[]>(`/hf/traces/datasets${prefix ? `?prefix=${encodeURIComponent(prefix)}` : ''}`),
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

// Decision-DPO mining (trace data quality)
export interface DataQualityDefaults {
  generate_decision_dpo: boolean
  require_successful_verification: boolean
  min_trace_steps: number
  max_trace_steps: number
}

export interface DecisionDpoRequest {
  gold_dir?: string
  failed_dir?: string
  generate_decision_dpo?: boolean
  require_successful_verification?: boolean
  min_trace_steps?: number
  max_trace_steps?: number
}

export interface DecisionDpoJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  n_training_examples?: number
  n_dpo_pairs?: number
  dpo_output_path?: string
  error?: string
}

export const dataQualityApi = {
  defaults: () =>
    request<DataQualityDefaults>('/factory/data-quality/defaults'),

  mineDecisionDpo: (req: DecisionDpoRequest) =>
    request<DecisionDpoJob>('/factory/decision-dpo/generate', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  decisionDpoStatus: (jobId: string) =>
    request<DecisionDpoJob>(`/factory/decision-dpo/jobs/${jobId}`)
}

// =============================================================================
// Cascade RL API
// =============================================================================

export const cascadeApi = {
  start: (config: {
    domains: string[]
    baseModel: string
    datasetPath: string
    trainStepsPerStage: number
    grpoNumGenerations: number
    grpoTemperature: number
    learningRate: number
    loraR: number
    loraAlpha: number
    load4Bit: boolean
    useRemoteSsh: boolean
    mode: string
  }) =>
    request<{ status: string; domains: string[]; stages: number }>('/cascade/start', {
      method: 'POST',
      body: JSON.stringify({
        domains: config.domains,
        base_model: config.baseModel,
        dataset_path: config.datasetPath,
        train_steps_per_stage: config.trainStepsPerStage,
        grpo_num_generations: config.grpoNumGenerations,
        grpo_temperature: config.grpoTemperature,
        learning_rate: config.learningRate,
        lora_r: config.loraR,
        lora_alpha: config.loraAlpha,
        load_in_4bit: config.load4Bit,
        use_remote_ssh: config.useRemoteSsh,
        mode: config.mode,
      }),
    }),
  stop: () => request<{ status: string }>('/cascade/stop', { method: 'POST' }),
  getStatus: () => request<Record<string, unknown>>('/cascade/status'),
  distill: (config: {
    studentModel?: string
    distillationAlpha?: number
    temperature?: number
    trainSteps?: number
  }) =>
    request<{ status: string }>('/cascade/distill', {
      method: 'POST',
      body: JSON.stringify({
        student_model: config.studentModel || '',
        distillation_alpha: config.distillationAlpha || 0.5,
        temperature: config.temperature || 2.0,
        train_steps: config.trainSteps || 500,
      }),
    }),
  getDistillStatus: () => request<Record<string, unknown>>('/cascade/distill/status'),
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

// =============================================================================
// Security Dataset API
// =============================================================================

export interface SecurityDatasetInfo {
  dataset_type: string
  name: string
  domain: string
  description: string
  input_formats: string[]
  example_sources: string[]
}

export interface SecurityIngestRequest {
  dataset_type: string
  input_path: string
  mode?: 'direct' | 'enriched'
  max_samples?: number
  balance_classes?: boolean
  benign_ratio?: number
  output_dir?: string
  train_split?: number
  enrichment_provider?: string
  enrichment_model?: string
}

export interface SecurityIngestResponse {
  job_id: string
  status: string
  message: string
}

export interface SecurityJobStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  dataset_type: string
  mode: string
  created_at: string
  completed_at?: string
  total_samples_read: number
  examples_generated: number
  train_path?: string
  val_path?: string
  train_count: number
  val_count: number
  error?: string
}

export const securityApi = {
  listDatasets: () =>
    request<SecurityDatasetInfo[]>('/security/datasets'),

  startIngestion: (req: SecurityIngestRequest) =>
    request<SecurityIngestResponse>('/security/ingest', {
      method: 'POST',
      body: JSON.stringify(req)
    }),

  getJobStatus: (jobId: string) =>
    request<SecurityJobStatus>(`/security/jobs/${jobId}`),

  listJobs: () =>
    request<SecurityJobStatus[]>('/security/jobs')
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

// Agent Chat API — Session types
export interface AgentSessionMeta {
  session_id: string
  name: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface AgentSessionMessage {
  id: string
  role: string
  content: string
  timestamp: number
  context_used: string[]
}

export interface PendingAction {
  type: string  // "shell_command"
  command: string
  reason: string
  token: string
}

export interface AgentChatResponse {
  response: string
  context_used: string[]
  pending_action?: PendingAction | null
}

export const agentApi = {
  chat: (message: string, history?: Array<{ role: string; content: string }>) =>
    request<AgentChatResponse>('/agent/chat', {
      method: 'POST',
      body: JSON.stringify({ message, history })
    }),

  confirmAction: (token: string, approved: boolean, sessionId?: string) =>
    request<AgentChatResponse>('/agent/confirm-action', {
      method: 'POST',
      body: JSON.stringify({ token, approved, session_id: sessionId ?? null })
    }),

  listSessions: () =>
    request<AgentSessionMeta[]>('/agent/sessions'),

  loadSession: (sessionId: string) =>
    request<AgentSessionMessage[]>(`/agent/sessions/${sessionId}`),

  saveSession: (sessionId: string, name: string, messages: AgentSessionMessage[]) =>
    request('/agent/sessions', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, name, messages })
    }),

  deleteSession: (sessionId: string) =>
    request(`/agent/sessions/${sessionId}`, { method: 'DELETE' }),
}

// Orchestrator API
// Orchestrator API response types (see bashgym/api/orchestrator_routes.py)
export type OrchestratorJobStatus =
  | 'decomposing'
  | 'awaiting_approval'
  | 'executing'
  | 'dispatched'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface OrchestratorTaskDTO {
  id: string
  title?: string
  description?: string
  priority?: 'CRITICAL' | 'HIGH' | 'NORMAL' | 'LOW'
  status?:
    | 'pending'
    | 'assigned'
    | 'running'
    | 'completed'
    | 'failed'
    | 'blocked'
    | 'cancelled'
    | 'retrying'
    | 'dispatched'
  dependencies?: string[]
  files_touched?: string[]
  estimated_turns?: number
  budget_usd?: number
  retry_count?: number
  worker_prompt?: string
  worker_id?: string
  result?: {
    cost_usd?: number
    duration_seconds?: number
    error?: string
    success?: boolean
  }
}

export interface OrchestratorStatusResponse {
  job_id: string
  status: OrchestratorJobStatus
  error?: string | null
  title?: string
  dag?: {
    tasks?: OrchestratorTaskDTO[]
    stats?: Record<string, number>
  }
  stats?: Record<string, number>
  total_cost?: number
  total_time?: number
  completed?: number
  failed?: number
  budget?: {
    limit_usd: number
    spent_usd: number
    remaining_usd: number
    exceeded: boolean
  }
  synthesis?: {
    merge_successes?: number
    merge_failures?: number
  }
}

export interface OrchestratorJobSummary {
  job_id: string
  status: OrchestratorJobStatus
  title?: string | null
  task_count?: number
  stats?: Record<string, number>
}

export interface OrchestratorProvider {
  provider: string
  default_model: string
  env_key: string
  base_url?: string
}

export const orchestratorApi = {
  submitSpec: (spec: {
    title: string
    description: string
    constraints?: string[]
    acceptance_criteria?: string[]
    repository?: string
    base_branch?: string
    max_budget_usd?: number
    max_workers?: number
    llm_config?: {
      provider?: string
      model?: string
      temperature?: number
    }
  }) =>
    request<{ job_id: string; status: OrchestratorJobStatus; provider: string; model: string }>(
      '/orchestrate/submit',
      { method: 'POST', body: JSON.stringify(spec) }
    ),

  approveJob: (jobId: string) =>
    request<{ job_id: string; status: OrchestratorJobStatus }>(
      `/orchestrate/${jobId}/approve`,
      { method: 'POST', body: '{}' }
    ),

  getStatus: (jobId: string) =>
    request<OrchestratorStatusResponse>(`/orchestrate/${jobId}/status`),

  retryTask: (jobId: string, taskId: string, modifiedPrompt?: string) =>
    request<{ job_id: string; task_id: string; status: string }>(
      `/orchestrate/${jobId}/task/${taskId}/retry`,
      {
        method: 'POST',
        body: JSON.stringify({ modified_prompt: modifiedPrompt })
      }
    ),

  cancelJob: (jobId: string) =>
    request<{ status: string; job_id: string }>(`/orchestrate/${jobId}`, { method: 'DELETE' }),

  listProviders: () => request<{ providers: OrchestratorProvider[] }>('/orchestrate/providers'),

  listJobs: () => request<{ jobs: OrchestratorJobSummary[] }>('/orchestrate/jobs'),
}

// Pipeline API
export interface PipelineConfig {
  watch_enabled: boolean
  watch_debounce_seconds: number
  classify_enabled: boolean
  classify_gold_min_success_rate: number
  classify_gold_min_steps: number
  classify_fail_max_success_rate: number
  generate_enabled: boolean
  generate_gold_threshold: number
  train_enabled: boolean
  train_examples_threshold: number
  cascade_enabled: boolean
  cascade_gold_threshold: number
  cascade_base_model: string
  cascade_mode: 'simulate' | 'real'
  cascade_train_steps_per_stage: number
  cascade_min_domain_examples: number
  cascade_use_remote_ssh: boolean
  cascade_repo_domains_enabled: boolean
}

export interface PipelineStatus {
  watcher_running: boolean
  config: PipelineConfig
  gold_count: number
  pending_count: number
  failed_count: number
  cascade_auto_trigger?: Record<string, unknown> | null
}

export const pipelineApi = {
  getConfig: () =>
    request<PipelineConfig>('/pipeline/config'),

  updateConfig: (updates: Partial<PipelineConfig>) =>
    request<PipelineConfig>('/pipeline/config', {
      method: 'PUT',
      body: JSON.stringify(updates),
    }),

  getStatus: () =>
    request<PipelineStatus>('/pipeline/status'),

  triggerStage: (stage: 'import' | 'classify' | 'cascade') =>
    request<{ imported?: number; classified?: number; total?: number }>(
      `/pipeline/trigger/${stage}`,
      { method: 'POST' }
    ),
}

// Settings API types
export interface EnvKeyStatus {
  key: string
  display_name: string
  masked_value: string
  is_set: boolean
  source: string
}

export interface EnvKeysResponse {
  keys: EnvKeyStatus[]
}

export interface EnvTestResponse {
  key: string
  valid: boolean
  message: string
  status_code?: number
}

export const settingsApi = {
  getEnvKeys: () =>
    request<EnvKeysResponse>('/settings/env'),

  updateEnvKeys: (values: Record<string, string>) =>
    request<{ success: boolean; updated: string[] }>('/settings/env', {
      method: 'PUT',
      body: JSON.stringify({ values }),
    }),

  testEnvKey: (key: string, value?: string) =>
    request<EnvTestResponse>('/settings/env/test', {
      method: 'POST',
      body: JSON.stringify({ key, value }),
    }),
}

// AutoResearch API
export interface AutoResearchStartConfig {
  searchParams: string[]
  maxExperiments: number
  trainSteps: number
  mutationRate: number
  mutationScale: number
  mode: 'simulate' | 'real'
  baseModel: string
  learningRate: number
  loraRank: number
  loraAlpha: number
  loraDropout: number
  warmupRatio: number
  gradientAccumulationSteps: number
  batchSize: number
  maxSeqLength: number
  load4Bit: boolean
}

export interface EnvironmentRecipeProposalConfig {
  environments?: TerminalEnvironmentSpec[]
  environmentPath?: string
  source?: string
  maxExperiments?: number
  sampleSize?: number
  passAt1Target?: number
  mutationRate?: number
  mutationScale?: number
  seed?: number
  outputPath?: string
}

export interface EnvironmentRecipeExperiment {
  experiment_id: number
  config_snapshot: Record<string, unknown>
  metric_value: number
  improved: boolean
  duration_seconds: number
  timestamp: string
}

export interface EnvironmentRecipeProposal {
  schema_version: string
  source_count: number
  selected_count: number
  selected_environment_ids: string[]
  recipe: Record<string, unknown>
  metric: number
  mix_report: EnvironmentMixReport
  selected_environments: Array<{
    id: string
    domain: string
    skills: string[]
    verifier_kind: string
    fixture_kinds: string[]
    'pass@1'?: number | null
  }>
}

export interface EnvironmentRecipeProposalResponse {
  status: string
  source_count: number
  best_metric: number | null
  proposal: EnvironmentRecipeProposal
  experiments: EnvironmentRecipeExperiment[]
  validation_errors: EnvironmentImportError[]
  output_path?: string | null
}

// Research index — Firecrawl-grounded news feed + AutoResearch advice
export interface ResearchNewsItem {
  kind: 'github' | 'paper'
  title: string
  url: string
  source: string
  summary: string
  score: number
}

export interface ResearchAdvice {
  configured: boolean
  context: Record<string, unknown>
  techniques: { title: string; url: string; abstract: string; score: number }[]
  issues: { repository: string; url: string; snippet: string; title: string }[]
  prior: { suggested: Record<string, unknown>; notes: string[] }
}

export const researchApi = {
  news: (k = 20) =>
    request<{ configured: boolean; count: number; items: ResearchNewsItem[] }>(
      `/research/news?k=${k}`
    ),

  advise: (body: { base_model?: string; strategy?: string; family?: string; domain?: string }) =>
    request<ResearchAdvice>('/research/advise', {
      method: 'POST',
      body: JSON.stringify(body)
    })
}

export const autoresearchApi = {
  start: (config: AutoResearchStartConfig) =>
    request<{ status: string; message: string }>('/autoresearch/start', {
      method: 'POST',
      body: JSON.stringify({
        search_params: config.searchParams,
        max_experiments: config.maxExperiments,
        train_steps: config.trainSteps,
        mutation_rate: config.mutationRate,
        mutation_scale: config.mutationScale,
        mode: config.mode || 'simulate',
        base_config: {
          base_model: config.baseModel,
          learning_rate: config.learningRate,
          lora_r: config.loraRank,
          lora_alpha: config.loraAlpha,
          lora_dropout: config.loraDropout,
          warmup_ratio: config.warmupRatio,
          gradient_accumulation_steps: config.gradientAccumulationSteps,
          batch_size: config.batchSize,
          max_seq_length: config.maxSeqLength,
          load_in_4bit: config.load4Bit,
        }
      })
    }),

  stop: () =>
    request<{ status: string }>('/autoresearch/stop', { method: 'POST' }),

  pause: () =>
    request<{ status: string }>('/autoresearch/pause', { method: 'POST' }),

  resume: () =>
    request<{ status: string }>('/autoresearch/resume', { method: 'POST' }),

  status: () =>
    request<any>('/autoresearch/status'),

  proposeEnvironmentRecipe: (config: EnvironmentRecipeProposalConfig) =>
    request<EnvironmentRecipeProposalResponse>('/autoresearch/environment-recipe/propose', {
      method: 'POST',
      body: JSON.stringify({
        environments: config.environments || [],
        environment_path: config.environmentPath || null,
        source: config.source || 'tmax',
        max_experiments: config.maxExperiments ?? 8,
        sample_size: config.sampleSize ?? 64,
        pass_at_1_target: config.passAt1Target ?? 0.35,
        mutation_rate: config.mutationRate ?? 0.35,
        mutation_scale: config.mutationScale ?? 0.2,
        seed: config.seed ?? 0,
        output_path: config.outputPath || null,
      }),
    }),

  // Trace research
  startTraceResearch: (config: { searchParams: string[]; maxExperiments: number; mutationRate: number; mutationScale: number }) =>
    request<{ status: string; message: string }>('/autoresearch/trace-research/start', {
      method: 'POST',
      body: JSON.stringify({
        search_params: config.searchParams,
        max_experiments: config.maxExperiments,
        mutation_rate: config.mutationRate,
        mutation_scale: config.mutationScale,
      })
    }),

  stopTraceResearch: () =>
    request<{ status: string }>('/autoresearch/trace-research/stop', { method: 'POST' }),

  pauseTraceResearch: () =>
    request<{ status: string }>('/autoresearch/trace-research/pause', { method: 'POST' }),

  resumeTraceResearch: () =>
    request<{ status: string }>('/autoresearch/trace-research/resume', { method: 'POST' }),

  traceResearchStatus: () =>
    request<any>('/autoresearch/trace-research/status'),

  // Schema research
  schemaResearch: {
    start: (config: { baseTemplate: string; maxExperiments: number; mutationRate: number; mutationScale: number; mode: string }) =>
      request<{ status: string; template: string }>('/autoresearch/schema-research/start', {
        method: 'POST',
        body: JSON.stringify({
          base_template: config.baseTemplate,
          max_experiments: config.maxExperiments,
          mutation_rate: config.mutationRate,
          mutation_scale: config.mutationScale,
          mode: config.mode,
        }),
      }),
    stop: () => request<{ status: string }>('/autoresearch/schema-research/stop', { method: 'POST' }),
    pause: () => request<{ status: string }>('/autoresearch/schema-research/pause', { method: 'POST' }),
    resume: () => request<{ status: string }>('/autoresearch/schema-research/resume', { method: 'POST' }),
    getStatus: () => request<any>('/autoresearch/schema-research/status'),
    getQuality: () => request<any>('/autoresearch/schema-research/quality'),
  },
}
