import { create } from 'zustand'
import { trainingApi } from '../services/api'
import { useActivityStore } from './activityStore'
import { useCanvasOrchestratorStore } from './canvasOrchestratorStore'
import { useTerminalStore } from './terminalStore'
import {
  createTrainingCorrelationId,
  inferTrainingComputeTarget,
  resolveTrainingOrigin,
  trainingQueuedPayloadFromResponse,
  type TrainingOrigin,
} from './trainingCanvasLifecycle'

export type TrainingStrategy =
  | 'sft'
  | 'dpo'
  | 'grpo'
  | 'distillation'
  | 'session_distillation'
  | 'cascade'
export type TrainingStatus = 'idle' | 'starting' | 'running' | 'paused' | 'completed' | 'failed'
export type TrainingProfile = 'default' | 'terminal_rl_tmax_like'
export type ArtifactRetention = 'adapter_only' | 'adapter_checkpoint' | 'deployable' | 'full_run'
export type HFUploadArtifact = 'auto' | 'adapter' | 'merged'

export interface TrainingMetrics {
  loss: number
  learningRate: number
  gradNorm: number
  epoch: number
  step: number
  totalSteps: number
  eta?: string
  simulation?: boolean  // True when running in simulation mode (no GPU/trainer)
  timestamp: number
  // Richer per-step metrics the trainer emits (forwarded from training:progress).
  evalLoss?: number
  samplesProcessed?: number
  tokensPerSecond?: number
  gpuMemoryGb?: number
  gpuUtilization?: number
  /** Where the run executes, e.g. 'local', 'ssh:<device>', 'cloud' */
  computeTarget?: string
  sessionDistillationLoss?: number
  sessionDistillationKl?: number
  sessionDistillationCe?: number
  sessionDistillationMaskedTokens?: number
}

export type DataSource = 'traces' | 'dataset_path' | 'security_dataset'

export interface TrainingConfig {
  strategy: TrainingStrategy
  baseModel: string
  datasetPath: string
  epochs: number
  batchSize: number
  learningRate: number
  warmupRatio: number
  gradientAccumulationSteps: number
  maxSeqLength: number
  saveSteps: number
  checkpointLimit?: number
  artifactRetention?: ArtifactRetention
  // LoRA settings
  loraRank?: number
  loraAlpha?: number
  loraDropout?: number
  load4Bit?: boolean  // QLoRA: load model in 4-bit quantization
  // Strategy-specific
  dpoBeta?: number
  grpoNumGenerations?: number
  grpoTemperature?: number
  grpoLossType?: string  // grpo | gspo | dr_grpo | dapo | bnpo
  grpoBackend?: string   // auto | unsloth | plain | trl_vllm
  grpoUseVllm?: boolean
  trainingProfile?: TrainingProfile
  grpoGroupSize?: number
  promptsPerRolloutBatch?: number
  maxToolCallsPerEpisode?: number
  tokenLevelLoss?: boolean
  filterZeroStdGroups?: boolean
  activeSampling?: boolean
  lmHeadFp32?: boolean
  interleavedThinking?: boolean
  sftWarmStartPolicy?: string
  dppoBackend?: string
  dppoDivergence?: string
  dppoBinaryTvThreshold?: number
  dppoBinaryKlThreshold?: number
  echoEnabled?: boolean
  echoAuxLambda?: number
  rwmlEnabled?: boolean
  rwmlDistanceThreshold?: number
  rwmlEasyPassRateThreshold?: number
  rwmlEasyKeepProbability?: number
  rwmlHistoryWindow?: number
  rwmlEmbeddingModel?: string
  rwmlKlBeta?: number
  useLiger?: boolean     // plain backend: Liger fused-linear-CE (262k-vocab OOM fix)
  // Knowledge Distillation
  teacherModel?: string
  teacherTemperature?: number
  distillationAlpha?: number
  // Session Distillation
  sessionDistillationAlpha?: number
  sessionDistillationTemperature?: number
  sessionDistillationMinConfidence?: number
  sessionDistillationMaskPolicy?: string
  sessionDistillationContextMode?: string
  sessionDistillationReader?: string
  // Repo selection
  selectedRepos?: string[]  // Repos to include in training (empty = all)
  // Training backend
  useNemoCustomizer?: boolean
  /** @deprecated Compatibility alias for hosted NeMo Customizer. */
  useNemoGym?: boolean
  useRemoteSSH?: boolean
  deviceId?: string
  // Data source selection
  dataSource?: DataSource
  securityDatasetType?: string
  securityDatasetPath?: string
  securityConversionMode?: 'direct' | 'enriched'
  securityMaxSamples?: number
  securityBalanceClasses?: boolean
  // Auto-deploy
  autoDeployOllama?: boolean
  ollamaModelName?: string
  // HuggingFace
  autoPushHF?: boolean
  hfRepoName?: string
  hfPrivate?: boolean
  hfUploadArtifact?: HFUploadArtifact
  autoExportGGUF?: boolean
  ggufQuantization?: string
}

export interface TrainingRun {
  id: string
  config: TrainingConfig
  status: TrainingStatus
  startTime: number
  endTime?: number
  currentMetrics?: TrainingMetrics
  metricsHistory: TrainingMetrics[]
  error?: string
  reconnected?: boolean  // True when run was reconnected after backend restart
}

export interface TrainingLog {
  timestamp: number
  message: string
  level: 'info' | 'warning' | 'error'
}

export interface TrainingLaunchOptions {
  origin?: TrainingOrigin
  correlationId?: string
}

export interface GrpoMetric {
  step: number
  loss?: number
  reward?: number
  rewardStd?: number
  fracRewardZeroStd?: number
  kl?: number
  echoLoss?: number
  rwmlPassRate?: number
  embeddingDistanceMean?: number
  embeddingDistanceP95?: number
  exitCodeAccuracy?: number
  testResultAccuracy?: number
  gradNorm?: number
  learningRate?: number
  activeSamplingRefills?: number
  zeroStdGroupsDropped?: number
  effectivePromptGroups?: number
  requestedPromptGroups?: number
  candidatePromptGroups?: number
  grpoGroupSize?: number
  timestamp: number
}

interface TrainingState {
  // Current training
  currentRun: TrainingRun | null
  runs: TrainingRun[]

  // Metrics streaming
  lossHistory: Array<{ step: number; loss: number; evalLoss?: number }>
  isConnected: boolean

  // Training logs
  logs: TrainingLog[]

  // Parsed GRPO per-step metrics (extracted from training:log TRL stats dicts)
  grpoMetrics: GrpoMetric[]

  // Cross-component overrides applied on next TrainingConfig open
  baseModelOverride: string | null
  datasetPathOverride: string | null

  // Actions
  startTraining: (config: TrainingConfig, options?: TrainingLaunchOptions) => Promise<string>
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  stopTraining: () => Promise<void>

  updateMetrics: (metrics: TrainingMetrics) => void
  setStatus: (status: TrainingStatus) => void
  setConnected: (connected: boolean) => void
  addLog: (log: TrainingLog) => void
  clearLogs: () => void
  clearGrpoMetrics: () => void
  setBaseModelOverride: (path: string | null) => void
  setDatasetPathOverride: (path: string | null) => void
  hydrateFromReconnect: (runId: string, metrics: TrainingMetrics) => void

  getRun: (id: string) => TrainingRun | undefined
}

// TRL/backend stats dicts look like: {'loss': 0.12, 'grad_norm': 0.4,
//   'reward': 0.5, 'kl': 0.02, 'echo_loss': 1.1, 'rwml_pass_rate': 0.7,
//   'step': 42, ...}. We do a fast substring gate then regex-extract fields.
const STATS_SIGNATURE_KEYS = [
  "'loss':",
  "'reward':",
  "'kl':",
  "'echo_loss':",
  "'rwml_pass_rate':",
]
const FIELD_RE = /'([a-z_]+)':\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)/g

function parseStatsLine(line: string): GrpoMetric | null {
  if (!STATS_SIGNATURE_KEYS.some((k) => line.includes(k))) return null
  const parsed: Record<string, number> = {}
  let match: RegExpExecArray | null
  FIELD_RE.lastIndex = 0
  while ((match = FIELD_RE.exec(line)) !== null) {
    const key = match[1]
    const raw = match[2]
    if (raw === 'nan' || raw === 'inf' || raw === '-inf') continue
    const num = Number(raw)
    if (Number.isFinite(num)) parsed[key] = num
  }
  if (
    !('loss' in parsed) &&
    !('reward' in parsed) &&
    !('echo_loss' in parsed) &&
    !('rwml_pass_rate' in parsed)
  ) return null
  const step = parsed.step ?? 0
  return {
    step,
    loss: parsed.loss,
    reward: parsed.reward,
    rewardStd: parsed.reward_std,
    fracRewardZeroStd: parsed.frac_reward_zero_std,
    kl: parsed.kl,
    echoLoss: parsed.echo_loss,
    rwmlPassRate: parsed.rwml_pass_rate,
    embeddingDistanceMean: parsed.embedding_distance_mean ?? parsed.rwml_embedding_distance_mean,
    embeddingDistanceP95: parsed.embedding_distance_p95 ?? parsed.rwml_embedding_distance_p95,
    exitCodeAccuracy: parsed.exit_code_accuracy,
    testResultAccuracy: parsed.test_result_accuracy,
    gradNorm: parsed.grad_norm,
    learningRate: parsed.learning_rate,
    activeSamplingRefills: parsed.active_sampling_refills,
    zeroStdGroupsDropped: parsed.zero_std_groups_dropped,
    effectivePromptGroups: parsed.effective_prompt_groups,
    requestedPromptGroups: parsed.requested_prompt_groups,
    candidatePromptGroups: parsed.candidate_prompt_groups,
    grpoGroupSize: parsed.grpo_group_size,
    timestamp: Date.now(),
  }
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  currentRun: null,
  runs: [],
  lossHistory: [],
  isConnected: false,
  logs: [],
  grpoMetrics: [],
  baseModelOverride: null,
  datasetPathOverride: null,

  startTraining: async (config: TrainingConfig, options) => {
    const origin = options?.origin ?? resolveTrainingOrigin(useTerminalStore.getState())
    const correlationId = options?.correlationId || createTrainingCorrelationId()
    const computeTarget = inferTrainingComputeTarget(config)

    // Set initial state
    const tempRunId = `run-${Date.now()}`
    const tempRun: TrainingRun = {
      id: tempRunId,
      config,
      status: 'starting',
      startTime: Date.now(),
      metricsHistory: []
    }

    set({
      currentRun: tempRun,
      runs: [...get().runs, tempRun],
      lossHistory: [],
      logs: [],  // Clear logs when starting new training
      grpoMetrics: []
    })

    try {
      // Call the actual API to start training
      const response = await trainingApi.start({
        strategy: config.strategy,
        base_model: config.baseModel,
        dataset_path: config.datasetPath,
        num_epochs: config.epochs,
        batch_size: config.batchSize,
        learning_rate: config.learningRate,
        warmup_ratio: config.warmupRatio,
        gradient_accumulation_steps: config.gradientAccumulationSteps,
        max_seq_length: config.maxSeqLength,
        save_steps: config.saveSteps,
        checkpoint_limit: config.checkpointLimit,
        artifact_retention: config.artifactRetention,
        // LoRA
        lora_rank: config.loraRank,
        lora_alpha: config.loraAlpha,
        lora_dropout: config.loraDropout,
        load_in_4bit: config.load4Bit,
        // Strategy-specific
        dpo_beta: config.dpoBeta,
        grpo_num_generations: config.grpoNumGenerations,
        grpo_temperature: config.grpoTemperature,
        grpo_loss_type: config.grpoLossType,
        grpo_backend: config.grpoBackend,
        grpo_use_vllm: config.grpoUseVllm,
        training_profile: config.trainingProfile,
        grpo_group_size: config.grpoGroupSize,
        prompts_per_rollout_batch: config.promptsPerRolloutBatch,
        max_tool_calls_per_episode: config.maxToolCallsPerEpisode,
        token_level_loss: config.tokenLevelLoss,
        filter_zero_std_groups: config.filterZeroStdGroups,
        active_sampling: config.activeSampling,
        lm_head_fp32: config.lmHeadFp32,
        interleaved_thinking: config.interleavedThinking,
        sft_warm_start_policy: config.sftWarmStartPolicy,
        dppo_backend: config.dppoBackend,
        dppo_divergence: config.dppoDivergence,
        dppo_binary_tv_threshold: config.dppoBinaryTvThreshold,
        dppo_binary_kl_threshold: config.dppoBinaryKlThreshold,
        echo_enabled: config.echoEnabled,
        echo_aux_lambda: config.echoAuxLambda,
        rwml_enabled: config.rwmlEnabled,
        rwml_distance_threshold: config.rwmlDistanceThreshold,
        rwml_easy_pass_rate_threshold: config.rwmlEasyPassRateThreshold,
        rwml_easy_keep_probability: config.rwmlEasyKeepProbability,
        rwml_history_window: config.rwmlHistoryWindow,
        rwml_embedding_model: config.rwmlEmbeddingModel,
        rwml_kl_beta: config.rwmlKlBeta,
        use_liger: config.useLiger,
        // Knowledge Distillation
        teacher_model: config.teacherModel,
        teacher_temperature: config.teacherTemperature,
        distillation_alpha: config.distillationAlpha,
        session_distillation_alpha: config.sessionDistillationAlpha,
        session_distillation_temperature: config.sessionDistillationTemperature,
        session_distillation_min_confidence: config.sessionDistillationMinConfidence,
        session_distillation_mask_policy: config.sessionDistillationMaskPolicy,
        session_distillation_context_mode: config.sessionDistillationContextMode,
        session_distillation_reader: config.sessionDistillationReader,
        // Backend & repos
        selected_repos: config.selectedRepos,
        use_nemo_customizer: config.useNemoCustomizer ?? config.useNemoGym,
        use_remote_ssh: config.useRemoteSSH,
        device_id: config.deviceId,
        compute_target: computeTarget,
        origin,
        correlation_id: correlationId,
        // Data source
        data_source: config.dataSource,
        security_dataset_type: config.securityDatasetType,
        security_dataset_path: config.securityDatasetPath,
        security_conversion_mode: config.securityConversionMode,
        security_max_samples: config.securityMaxSamples,
        security_balance_classes: config.securityBalanceClasses,
        // Auto-deploy
        auto_deploy_ollama: config.autoDeployOllama,
        ollama_model_name: config.ollamaModelName,
        // HuggingFace
        auto_push_hf: config.autoPushHF,
        hf_repo_name: config.hfRepoName,
        hf_private: config.hfPrivate,
        hf_upload_artifact: config.hfUploadArtifact,
        auto_export_gguf: config.autoExportGGUF,
        gguf_quantization: config.ggufQuantization,
      })

      if (response.ok && response.data) {
        const apiRunId = response.data.run_id
        const status = response.data.status === 'pending' ? 'starting' : response.data.status
        // Update with actual run ID from backend
        set((state) => ({
          currentRun: state.currentRun ? { ...state.currentRun, id: apiRunId, status } : null,
          runs: state.runs.map(r => r.id === tempRunId ? { ...r, id: apiRunId, status } : r)
        }))

        const queuedPayload = trainingQueuedPayloadFromResponse(response.data, {
          strategy: config.strategy,
          baseModel: config.baseModel,
          datasetPath: config.datasetPath,
          origin,
          correlationId,
          computeTarget,
        })

        // The POST itself is authoritative. WebSocket delivery can now miss a
        // queued event without leaving the canvas or Activity node unaware.
        useActivityStore.getState().addEvent(
          'training:queued',
          queuedPayload as unknown as Record<string, unknown>,
        )
        try {
          useCanvasOrchestratorStore.getState().handleTrainingQueued(queuedPayload)
        } catch (canvasError) {
          console.warn('[training] unable to materialize canvas run node', canvasError)
        }
        return apiRunId
      } else {
        // API call failed
        set((state) => ({
          currentRun: state.currentRun ? { ...state.currentRun, status: 'failed', error: response.error || 'Failed to start training' } : null
        }))
        throw new Error(response.error || 'Failed to start training')
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Unknown error'
      set((state) => ({
        currentRun: state.currentRun ? { ...state.currentRun, status: 'failed', error: errorMsg } : null
      }))
      throw error
    }
  },

  pauseTraining: async () => {
    const currentRun = get().currentRun
    if (!currentRun) return

    try {
      await trainingApi.pause(currentRun.id)
      set((state) => ({
        currentRun: state.currentRun ? { ...state.currentRun, status: 'paused' } : null
      }))
    } catch (error) {
      console.error('Failed to pause training:', error)
    }
  },

  resumeTraining: async () => {
    const currentRun = get().currentRun
    if (!currentRun) return

    try {
      await trainingApi.resume(currentRun.id)
      set((state) => ({
        currentRun: state.currentRun ? { ...state.currentRun, status: 'running' } : null
      }))
    } catch (error) {
      console.error('Failed to resume training:', error)
    }
  },

  stopTraining: async () => {
    const currentRun = get().currentRun
    if (!currentRun) return

    try {
      await trainingApi.stop(currentRun.id)
      set((state) => {
        if (!state.currentRun) return state

        const completedRun: TrainingRun = {
          ...state.currentRun,
          status: 'completed',
          endTime: Date.now()
        }

        return {
          currentRun: null,
          runs: state.runs.map((r) =>
            r.id === completedRun.id ? completedRun : r
          )
        }
      })
    } catch (error) {
      console.error('Failed to stop training:', error)
    }
  },

  updateMetrics: (metrics: TrainingMetrics) => {
    set((state) => {
      if (!state.currentRun) return state

      const newLossHistory = [
        ...state.lossHistory,
        { step: metrics.step, loss: metrics.loss, evalLoss: metrics.evalLoss }
      ].slice(-500) // Keep last 500 points

      return {
        currentRun: {
          ...state.currentRun,
          status: 'running',
          currentMetrics: metrics,
          metricsHistory: [...state.currentRun.metricsHistory, metrics]
        },
        lossHistory: newLossHistory
      }
    })
  },

  setStatus: (status: TrainingStatus) => {
    set((state) => {
      if (!state.currentRun) return state
      return {
        currentRun: { ...state.currentRun, status }
      }
    })
  },

  setConnected: (connected: boolean) => {
    set({ isConnected: connected })
  },

  hydrateFromReconnect: (runId: string, metrics: TrainingMetrics) => {
    const reconnectedRun: TrainingRun = {
      id: runId,
      config: {} as TrainingConfig,  // Will be populated from API
      status: 'running',
      startTime: Date.now(),
      metricsHistory: [metrics],
      currentMetrics: metrics,
      reconnected: true,
    }

    set((state) => ({
      currentRun: reconnectedRun,
      runs: [...state.runs, reconnectedRun],
      lossHistory: metrics.loss != null ? [{ step: metrics.step, loss: metrics.loss }] : [],
    }))

    // Fetch full run details from API to populate config
    trainingApi.getStatus(runId).then((response) => {
      if (response.ok && response.data) {
        set((state) => ({
          currentRun: state.currentRun?.id === runId
            ? { ...state.currentRun, startTime: new Date(response.data!.started_at || Date.now()).getTime() }
            : state.currentRun,
        }))
      }
    }).catch(() => {})  // Best-effort hydration
  },

  getRun: (id: string) => {
    return get().runs.find((r) => r.id === id)
  },

  addLog: (log: TrainingLog) => {
    const parsed = parseStatsLine(log.message)
    set((state) => ({
      logs: [...state.logs, log].slice(-1000),
      grpoMetrics: parsed
        ? [...state.grpoMetrics, parsed].slice(-300)
        : state.grpoMetrics,
    }))
  },

  clearLogs: () => {
    set({ logs: [] })
  },

  clearGrpoMetrics: () => {
    set({ grpoMetrics: [] })
  },

  setBaseModelOverride: (path: string | null) => {
    set({ baseModelOverride: path })
  },

  setDatasetPathOverride: (path: string | null) => {
    set({ datasetPathOverride: path })
  },
}))
