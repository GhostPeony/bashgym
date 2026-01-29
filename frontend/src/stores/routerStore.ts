import { create } from 'zustand'

export type RoutingStrategy =
  | 'teacher_only'
  | 'student_only'
  | 'confidence_based'
  | 'round_robin'
  | 'progressive_handoff'

export interface RoutingStats {
  totalRequests: number
  teacherRequests: number
  studentRequests: number
  teacherSuccessRate: number
  studentSuccessRate: number
  avgTeacherLatency: number
  avgStudentLatency: number
  currentStudentRate: number
}

export interface RoutingDecision {
  id: string
  timestamp: number
  model: 'teacher' | 'student'
  confidence: number
  latency: number
  success: boolean
  task?: string
}

interface RouterState {
  // Configuration
  strategy: RoutingStrategy
  studentRate: number
  confidenceThreshold: number

  // Stats
  stats: RoutingStats
  recentDecisions: RoutingDecision[]

  // Actions
  setStrategy: (strategy: RoutingStrategy) => void
  setStudentRate: (rate: number) => void
  setConfidenceThreshold: (threshold: number) => void
  updateStats: (stats: Partial<RoutingStats>) => void
  addDecision: (decision: RoutingDecision) => void
}

export const useRouterStore = create<RouterState>((set, get) => ({
  strategy: 'progressive_handoff',
  studentRate: 10,
  confidenceThreshold: 0.7,

  stats: {
    totalRequests: 0,
    teacherRequests: 0,
    studentRequests: 0,
    teacherSuccessRate: 0,
    studentSuccessRate: 0,
    avgTeacherLatency: 0,
    avgStudentLatency: 0,
    currentStudentRate: 10
  },

  recentDecisions: [],

  setStrategy: (strategy) => set({ strategy }),

  setStudentRate: (rate) => set({ studentRate: rate }),

  setConfidenceThreshold: (threshold) => set({ confidenceThreshold: threshold }),

  updateStats: (stats) =>
    set((state) => ({
      stats: { ...state.stats, ...stats }
    })),

  addDecision: (decision) =>
    set((state) => ({
      recentDecisions: [decision, ...state.recentDecisions].slice(0, 100)
    }))
}))
