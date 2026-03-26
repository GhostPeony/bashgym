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

export interface SchemaExperimentResult {
  experimentId: number
  totalExperiments: number
  configSnapshot: Record<string, number | boolean | string>
  metricValue: number
  bestMetric: number
  improved: boolean
  durationSeconds: number
}

export type AutoResearchStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed'
export type AutoResearchMode = 'hyperparam' | 'trace' | 'schema'

export type ExperimentMode = 'simulate' | 'real'

export interface AutoResearchStartConfig {
  searchParams: string[]
  maxExperiments: number
  trainSteps: number
  mutationRate: number
  mutationScale: number
  mode: ExperimentMode
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

  // Schema research
  schemaStatus: AutoResearchStatus
  schemaExperiments: SchemaExperimentResult[]
  schemaBestMetric: number | null
  schemaBestGenome: Record<string, any> | null
  schemaTotalExperiments: number
  schemaCurrentExperiment: number
  schemaTemplate: string

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

  // Schema research actions
  addSchemaExperiment: (result: SchemaExperimentResult) => void
  setSchemaStatus: (status: AutoResearchStatus) => void
  resetSchema: () => void
  startSchemaResearch: (config: { baseTemplate: string; maxExperiments: number; mutationRate: number; mutationScale: number; mode: string }) => Promise<void>
  stopSchemaResearch: () => Promise<void>
  pauseSchemaResearch: () => Promise<void>
  resumeSchemaResearch: () => Promise<void>
  setSchemaTemplate: (template: string) => void

  // Mode
  setActiveMode: (mode: AutoResearchMode) => void
}

export const useAutoResearchStore = create<AutoResearchState>((set, _get) => ({
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

  // Schema research state
  schemaStatus: 'idle',
  schemaExperiments: [],
  schemaBestMetric: null,
  schemaBestGenome: null,
  schemaTotalExperiments: 0,
  schemaCurrentExperiment: 0,
  schemaTemplate: 'coding_agent_sft',

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
    } catch (_error) {
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
    } catch (_error) {
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

  // ─── Schema research actions ──────────────────────────────────────

  addSchemaExperiment: (result: SchemaExperimentResult) => {
    set((state) => ({
      schemaExperiments: [...state.schemaExperiments, result],
      schemaCurrentExperiment: result.experimentId,
      schemaTotalExperiments: result.totalExperiments,
      schemaBestMetric: result.bestMetric,
      schemaBestGenome: result.improved ? result.configSnapshot : state.schemaBestGenome,
    }))
  },

  setSchemaStatus: (status: AutoResearchStatus) => {
    set({ schemaStatus: status })
  },

  resetSchema: () => {
    set({
      schemaStatus: 'idle',
      schemaExperiments: [],
      schemaBestMetric: null,
      schemaBestGenome: null,
      schemaTotalExperiments: 0,
      schemaCurrentExperiment: 0,
    })
  },

  startSchemaResearch: async (config) => {
    set({
      schemaStatus: 'running',
      schemaExperiments: [],
      schemaBestMetric: null,
      schemaBestGenome: null,
      schemaTotalExperiments: config.maxExperiments,
      schemaCurrentExperiment: 0,
    })
    try {
      const response = await autoresearchApi.schemaResearch.start(config)
      if (!response.ok) {
        set({ schemaStatus: 'failed' })
      }
    } catch (_error) {
      set({ schemaStatus: 'failed' })
    }
  },

  stopSchemaResearch: async () => {
    try {
      await autoresearchApi.schemaResearch.stop()
      set({ schemaStatus: 'idle' })
    } catch {/* */}
  },

  pauseSchemaResearch: async () => {
    try {
      await autoresearchApi.schemaResearch.pause()
      set({ schemaStatus: 'paused' })
    } catch {/* */}
  },

  resumeSchemaResearch: async () => {
    try {
      await autoresearchApi.schemaResearch.resume()
      set({ schemaStatus: 'running' })
    } catch {/* */}
  },

  setSchemaTemplate: (template: string) => {
    set({ schemaTemplate: template })
  },

  setActiveMode: (mode: AutoResearchMode) => {
    set({ activeMode: mode })
  },
}))
