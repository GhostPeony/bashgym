import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface AccentPreset {
  name: string
  hue: number
}

export const ACCENT_PRESETS: AccentPreset[] = [
  { name: 'Wisteria', hue: 258 },
  { name: 'Rose', hue: 350 },
  { name: 'Moss', hue: 140 },
  { name: 'Marigold', hue: 35 },
  { name: 'Lavender', hue: 270 },
  { name: 'Teal Leaf', hue: 175 },
]

const DEFAULT_HUE = 258

function applyAccentHue(hue: number) {
  document.documentElement.style.setProperty('--accent-hue', String(hue))
}

interface AccentState {
  accentHue: number
  setAccentHue: (hue: number) => void
  randomizeHue: () => void
  resetHue: () => void
}

export const useAccentStore = create<AccentState>()(
  persist(
    (set) => ({
      accentHue: DEFAULT_HUE,

      setAccentHue: (hue: number) => {
        const clamped = Math.round(((hue % 360) + 360) % 360)
        set({ accentHue: clamped })
        applyAccentHue(clamped)
      },

      randomizeHue: () => {
        const hue = Math.floor(Math.random() * 360)
        set({ accentHue: hue })
        applyAccentHue(hue)
      },

      resetHue: () => {
        set({ accentHue: DEFAULT_HUE })
        applyAccentHue(DEFAULT_HUE)
      },
    }),
    {
      name: 'bashgym-accent-hue',
      onRehydrateStorage: () => (state) => {
        if (state) {
          applyAccentHue(state.accentHue)
        }
      }
    }
  )
)
