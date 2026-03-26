import { create } from 'zustand'
import { trainingApi } from '../services/api'

export type TrainingStrategy = 'sft' | 'dpo' | 'grpo' | 'distillation' | 'cascade'
export type TrainingStatus = 'idle' | 'starting' | 'running' | 'paused' | 'completed' | 'failed'

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
  // LoRA settings
  loraRank?: number
  loraAlpha?: number
  loraDropout?: number
  load4Bit?: boolean  // QLoRA: load model in 4-bit quantization
  // Strategy-specific
  dpoBeta?: number
  grpoNumGenerations?: number
  grpoTemperature?: number
  // Knowledge Distillation
  teacherModel?: string
  teacherTemperature?: number
  distillationAlpha?: number
  // Repo selection
  selectedRepos?: string[]  // Repos to include in training (empty = all)
  // Training backend
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

interface TrainingState {
  // Current training
  currentRun: TrainingRun | null
  runs: TrainingRun[]

  // Metrics streaming
  lossHistory: Array<{ step: number; loss: number }>
  isConnected: boolean

  // Training logs
  logs: TrainingLog[]

  // Actions
  startTraining: (config: TrainingConfig) => Promise<string>
  pauseTraining: () => Promise<void>
  resumeTraining: () => Promise<void>
  stopTraining: () => Promise<void>

  updateMetrics: (metrics: TrainingMetrics) => void
  setStatus: (status: TrainingStatus) => void
  setConnected: (connected: boolean) => void
  addLog: (log: TrainingLog) => void
  clearLogs: () => void
  hydrateFromReconnect: (runId: string, metrics: TrainingMetrics) => void

  getRun: (id: string) => TrainingRun | undefined
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  currentRun: null,
  runs: [],
  lossHistory: [],
  isConnected: false,
  logs: [],

  startTraining: async (config: TrainingConfig) => {
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
      logs: []  // Clear logs when starting new training
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
        // LoRA
        lora_rank: config.loraRank,
        lora_alpha: config.loraAlpha,
        lora_dropout: config.loraDropout,
        load_in_4bit: config.load4Bit,
        // Strategy-specific
        dpo_beta: config.dpoBeta,
        grpo_num_generations: config.grpoNumGenerations,
        grpo_temperature: config.grpoTemperature,
        // Knowledge Distillation
        teacher_model: config.teacherModel,
        teacher_temperature: config.teacherTemperature,
        distillation_alpha: config.distillationAlpha,
        // Backend & repos
        selected_repos: config.selectedRepos,
        use_nemo_gym: config.useNemoGym,
        use_remote_ssh: config.useRemoteSSH,
        device_id: config.deviceId,
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
      })

      if (response.ok && response.data) {
        const apiRunId = response.data.run_id
        // Update with actual run ID from backend
        set((state) => ({
          currentRun: state.currentRun ? { ...state.currentRun, id: apiRunId, status: 'running' } : null,
          runs: state.runs.map(r => r.id === tempRunId ? { ...r, id: apiRunId, status: 'running' } : r)
        }))
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
        { step: metrics.step, loss: metrics.loss }
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
    set((state) => ({
      logs: [...state.logs, log].slice(-1000) // Keep last 1000 logs
    }))
  },

  clearLogs: () => {
    set({ logs: [] })
  }
}))
