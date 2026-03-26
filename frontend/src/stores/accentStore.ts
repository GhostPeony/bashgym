import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface AccentPreset {
  name: string
  hue: number
}

export const ACCENT_PRESETS: AccentPreset[] = [
  { name: 'Wisteria', hue: 258 },
  { name: 'Rose', hue: 350 },
  { name: 'Coral', hue: 8 },
  { name: 'Marigold', hue: 35 },
  { name: 'Lime', hue: 88 },
  { name: 'Moss', hue: 140 },
  { name: 'Teal Leaf', hue: 175 },
  { name: 'Sky', hue: 210 },
  { name: 'Cobalt', hue: 228 },
  { name: 'Lavender', hue: 270 },
  { name: 'Orchid', hue: 308 },
]

export interface TerminalFgPreset {
  name: string
  hue: number
}

export const DEFAULT_TERMINAL_FG_HUE = -1
export const BLACK_TERMINAL_FG_HUE = -2
export const WHITE_TERMINAL_FG_HUE = -3

export const TERMINAL_FG_PRESETS: TerminalFgPreset[] = [
  { name: 'Default', hue: DEFAULT_TERMINAL_FG_HUE },
  { name: 'Black', hue: BLACK_TERMINAL_FG_HUE },
  { name: 'White', hue: WHITE_TERMINAL_FG_HUE },
  { name: 'Rose', hue: 350 },
  { name: 'Moss', hue: 140 },
  { name: 'Sky', hue: 210 },
  { name: 'Amber', hue: 30 },
  { name: 'Violet', hue: 270 },
  { name: 'Aqua', hue: 185 },
]

const DEFAULT_HUE = 258

function applyAccentHue(hue: number) {
  document.documentElement.style.setProperty('--accent-hue', String(hue))
}

export function isTerminalFgHue(value: number): boolean {
  return value >= 0
}

export function getTerminalFgColor(theme: 'light' | 'dark', value: number): string {
  if (value === BLACK_TERMINAL_FG_HUE) {
    return '#000000'
  }

  if (value === WHITE_TERMINAL_FG_HUE) {
    return '#FFFFFF'
  }

  if (!isTerminalFgHue(value)) {
    return theme === 'dark' ? '#F0EDE8' : '#000000'
  }

  return theme === 'dark'
    ? `hsl(${value}, 30%, 82%)`
    : `hsl(${value}, 45%, 25%)`
}

export function getTerminalFgLabel(value: number): string {
  if (value === DEFAULT_TERMINAL_FG_HUE) return 'default'
  if (value === BLACK_TERMINAL_FG_HUE) return 'black'
  if (value === WHITE_TERMINAL_FG_HUE) return 'white'
  return `${Math.round(((value % 360) + 360) % 360)}deg`
}

interface AccentState {
  accentHue: number
  terminalFgHue: number
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
        if (
          hue === DEFAULT_TERMINAL_FG_HUE ||
          hue === BLACK_TERMINAL_FG_HUE ||
          hue === WHITE_TERMINAL_FG_HUE
        ) {
          set({ terminalFgHue: hue })
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
