import { create } from 'zustand'
import {
  achievementsApi,
  type LifetimeStatsResponse,
  type AchievementStatusResponse,
} from '../services/api'

interface AchievementState {
  stats: LifetimeStatsResponse | null
  achievements: AchievementStatusResponse[]
  recentUnlocks: AchievementStatusResponse[]
  earnedCount: number
  totalCount: number
  totalPoints: number
  loading: boolean

  fetchStats: () => Promise<void>
  fetchAchievements: () => Promise<void>
  fetchRecent: () => Promise<void>
  refresh: () => Promise<AchievementStatusResponse[]>
}

export const useAchievementStore = create<AchievementState>((set, get) => ({
  stats: null,
  achievements: [],
  recentUnlocks: [],
  earnedCount: 0,
  totalCount: 0,
  totalPoints: 0,
  loading: false,

  fetchStats: async () => {
    const result = await achievementsApi.getStats()
    if (result.ok && result.data) {
      set({ stats: result.data })
    }
  },

  fetchAchievements: async () => {
    set({ loading: true })
    const result = await achievementsApi.getAll()
    if (result.ok && result.data) {
      set({
        achievements: result.data.achievements,
        earnedCount: result.data.earned_count,
        totalCount: result.data.total_count,
        totalPoints: result.data.total_points,
      })
    }
    set({ loading: false })
  },

  fetchRecent: async () => {
    const result = await achievementsApi.getRecent()
    if (result.ok && result.data) {
      set({
        recentUnlocks: result.data.recent,
        earnedCount: result.data.earned_count,
        totalCount: result.data.total_count,
        totalPoints: result.data.total_points,
      })
    }
  },

  refresh: async () => {
    const result = await achievementsApi.refresh()
    if (result.ok && result.data) {
      set({
        earnedCount: result.data.earned_count,
        totalCount: result.data.total_count,
        totalPoints: result.data.total_points,
      })
      // Re-fetch full lists after refresh
      get().fetchAchievements()
      get().fetchRecent()
      return result.data.newly_earned
    }
    return []
  },
}))
