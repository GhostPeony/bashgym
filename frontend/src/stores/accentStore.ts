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

export interface TerminalFgPreset {
  name: string
  hue: number // -1 = default (black/cream)
}

export const TERMINAL_FG_PRESETS: TerminalFgPreset[] = [
  { name: 'Default', hue: -1 },
  { name: 'Rose', hue: 350 },
  { name: 'Moss', hue: 140 },
  { name: 'Sky', hue: 210 },
  { name: 'Amber', hue: 30 },
  { name: 'Violet', hue: 270 },
]

const DEFAULT_HUE = 258
const DEFAULT_TERMINAL_FG_HUE = -1

function applyAccentHue(hue: number) {
  document.documentElement.style.setProperty('--accent-hue', String(hue))
}

interface AccentState {
  accentHue: number
  terminalFgHue: number // -1 = default (black in light, cream in dark)
  setAccentHue: (hue: number) => void
  setTerminalFgHue: (hue: number) => void
  randomizeHue: () => void
  resetHue: () => void
  resetTerminalFgHue: () => void
}

export const useAccentStore = create<AccentState>()(
  persist(
    (set) => ({
      accentHue: DEFAULT_HUE,
      terminalFgHue: DEFAULT_TERMINAL_FG_HUE,

      setAccentHue: (hue: number) => {
        const clamped = Math.round(((hue % 360) + 360) % 360)
        set({ accentHue: clamped })
        applyAccentHue(clamped)
      },

      setTerminalFgHue: (hue: number) => {
        if (hue < 0) {
          set({ terminalFgHue: -1 })
        } else {
          set({ terminalFgHue: Math.round(((hue % 360) + 360) % 360) })
        }
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

      resetTerminalFgHue: () => {
        set({ terminalFgHue: DEFAULT_TERMINAL_FG_HUE })
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
