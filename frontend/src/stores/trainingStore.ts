import { create } from 'zustand'
import { trainingApi } from '../services/api'

export type TrainingStrategy = 'sft' | 'dpo' | 'grpo' | 'kd'
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
  loraRank?: number
  loraAlpha?: number
  use4BitQuantization?: boolean  // Enables QLoRA (LoRA + 4-bit quantization)
  warmupSteps: number
  maxSeqLength: number
  // Repo selection
  selectedRepos?: string[]  // Repos to include in training (empty = all)
  // Training backend
  useNemoGym?: boolean  // Use NVIDIA NeMo cloud training instead of local
  // Knowledge Distillation specific
  teacherModel?: string
  temperature?: number
  kdAlpha?: number  // Weight for distillation loss vs task loss
  // Data source selection
  dataSource?: DataSource
  securityDatasetType?: string
  securityDatasetPath?: string
  securityConversionMode?: 'direct' | 'enriched'
  securityMaxSamples?: number
  securityBalanceClasses?: boolean
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
        epochs: config.epochs,
        batch_size: config.batchSize,
        learning_rate: config.learningRate,
        lora_rank: config.loraRank,
        lora_alpha: config.loraAlpha,
        warmup_steps: config.warmupSteps,
        max_seq_length: config.maxSeqLength,
        selected_repos: config.selectedRepos,
        use_nemo_gym: config.useNemoGym,
        data_source: config.dataSource,
        security_dataset_type: config.securityDatasetType,
        security_dataset_path: config.securityDatasetPath,
        security_conversion_mode: config.securityConversionMode,
        security_max_samples: config.securityMaxSamples,
        security_balance_classes: config.securityBalanceClasses
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
