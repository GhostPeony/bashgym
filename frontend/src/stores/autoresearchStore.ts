import { create } from 'zustand'
import { autoresearchApi } from '../services/api'

export interface ExperimentResult {
  experimentId: number
  totalExperiments: number
  configSnapshot: Record<string, number | boolean>
  metricValue: number
  bestMetric: number
  improved: boolean
  durationSeconds: number
}

export interface TraceExperimentResult {
  experimentId: number
  totalExperiments: number
  configSnapshot: Record<string, number | boolean | string>
  examplesGenerated: number
  uniqueRepos: number
  avgExampleLength: number
  metricValue: number
  bestMetric: number
  improved: boolean
  durationSeconds: number
}

export type AutoResearchStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed'
export type AutoResearchMode = 'hyperparam' | 'trace'

export interface AutoResearchStartConfig {
  searchParams: string[]
  maxExperiments: number
  trainSteps: number
  mutationRate: number
  mutationScale: number
  // Base training config to start from
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

interface AutoResearchState {
  // Hyperparam search
  status: AutoResearchStatus
  experiments: ExperimentResult[]
  bestMetric: number | null
  bestConfig: Record<string, number | boolean> | null
  totalExperiments: number
  currentExperiment: number

  // Trace research
  traceStatus: AutoResearchStatus
  traceExperiments: TraceExperimentResult[]
  traceBestMetric: number | null
  traceBestPipeline: Record<string, any> | null
  traceTotalExperiments: number
  traceCurrentExperiment: number

  // Active mode
  activeMode: AutoResearchMode

  // Hyperparam actions
  addExperiment: (result: ExperimentResult) => void
  setStatus: (status: AutoResearchStatus) => void
  reset: () => void
  start: (config: AutoResearchStartConfig) => Promise<void>
  stop: () => Promise<void>
  pause: () => Promise<void>
  resume: () => Promise<void>

  // Trace research actions
  addTraceExperiment: (result: TraceExperimentResult) => void
  setTraceStatus: (status: AutoResearchStatus) => void
  resetTrace: () => void
  startTraceResearch: (config: { searchParams: string[]; maxExperiments: number; mutationRate: number; mutationScale: number }) => Promise<void>
  stopTraceResearch: () => Promise<void>
  pauseTraceResearch: () => Promise<void>
  resumeTraceResearch: () => Promise<void>

  // Mode
  setActiveMode: (mode: AutoResearchMode) => void
}

export const useAutoResearchStore = create<AutoResearchState>((set, get) => ({
  // Hyperparam state
  status: 'idle',
  experiments: [],
  bestMetric: null,
  bestConfig: null,
  totalExperiments: 0,
  currentExperiment: 0,

  // Trace research state
  traceStatus: 'idle',
  traceExperiments: [],
  traceBestMetric: null,
  traceBestPipeline: null,
  traceTotalExperiments: 0,
  traceCurrentExperiment: 0,

  activeMode: 'hyperparam',

  // ─── Hyperparam actions ────────────────────────────────────────────

  addExperiment: (result: ExperimentResult) => {
    set((state) => ({
      experiments: [...state.experiments, result],
      currentExperiment: result.experimentId,
      totalExperiments: result.totalExperiments,
      bestMetric: result.bestMetric,
      bestConfig: result.improved ? result.configSnapshot : state.bestConfig,
    }))
  },

  setStatus: (status: AutoResearchStatus) => {
    set({ status })
  },

  reset: () => {
    set({
      status: 'idle',
      experiments: [],
      bestMetric: null,
      bestConfig: null,
      totalExperiments: 0,
      currentExperiment: 0,
    })
  },

  start: async (config: AutoResearchStartConfig) => {
    set({
      status: 'running',
      experiments: [],
      bestMetric: null,
      bestConfig: null,
      totalExperiments: config.maxExperiments,
      currentExperiment: 0,
    })
    try {
      const response = await autoresearchApi.start(config)
      if (!response.ok) {
        set({ status: 'failed' })
      }
    } catch (error) {
      set({ status: 'failed' })
    }
  },

  stop: async () => {
    try {
      await autoresearchApi.stop()
      set({ status: 'idle' })
    } catch {/* */}
  },

  pause: async () => {
    try {
      await autoresearchApi.pause()
      set({ status: 'paused' })
    } catch {/* */}
  },

  resume: async () => {
    try {
      await autoresearchApi.resume()
      set({ status: 'running' })
    } catch {/* */}
  },

  // ─── Trace research actions ────────────────────────────────────────

  addTraceExperiment: (result: TraceExperimentResult) => {
    set((state) => ({
      traceExperiments: [...state.traceExperiments, result],
      traceCurrentExperiment: result.experimentId,
      traceTotalExperiments: result.totalExperiments,
      traceBestMetric: result.bestMetric,
      traceBestPipeline: result.improved ? result.configSnapshot : state.traceBestPipeline,
    }))
  },

  setTraceStatus: (status: AutoResearchStatus) => {
    set({ traceStatus: status })
  },

  resetTrace: () => {
    set({
      traceStatus: 'idle',
      traceExperiments: [],
      traceBestMetric: null,
      traceBestPipeline: null,
      traceTotalExperiments: 0,
      traceCurrentExperiment: 0,
    })
  },

  startTraceResearch: async (config) => {
    set({
      traceStatus: 'running',
      traceExperiments: [],
      traceBestMetric: null,
      traceBestPipeline: null,
      traceTotalExperiments: config.maxExperiments,
      traceCurrentExperiment: 0,
    })
    try {
      const response = await autoresearchApi.startTraceResearch(config)
      if (!response.ok) {
        set({ traceStatus: 'failed' })
      }
    } catch (error) {
      set({ traceStatus: 'failed' })
    }
  },

  stopTraceResearch: async () => {
    try {
      await autoresearchApi.stopTraceResearch()
      set({ traceStatus: 'idle' })
    } catch {/* */}
  },

  pauseTraceResearch: async () => {
    try {
      await autoresearchApi.pauseTraceResearch()
      set({ traceStatus: 'paused' })
    } catch {/* */}
  },

  resumeTraceResearch: async () => {
    try {
      await autoresearchApi.resumeTraceResearch()
      set({ traceStatus: 'running' })
    } catch {/* */}
  },

  setActiveMode: (mode: AutoResearchMode) => {
    set({ activeMode: mode })
  },
}))
