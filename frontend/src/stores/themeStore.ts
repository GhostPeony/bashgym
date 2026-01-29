import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type Theme = 'light' | 'dark' | 'system'
type ResolvedTheme = 'light' | 'dark'

function getSystemTheme(): ResolvedTheme {
  if (typeof window !== 'undefined' && window.matchMedia) {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  return 'dark'
}

function applyTheme(resolvedTheme: ResolvedTheme) {
  if (resolvedTheme === 'dark') {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
}

interface ThemeState {
  theme: Theme
  resolvedTheme: ResolvedTheme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'dark',
      resolvedTheme: 'dark',

      setTheme: (theme) => {
        const resolvedTheme = theme === 'system' ? getSystemTheme() : theme
        set({ theme, resolvedTheme })
        applyTheme(resolvedTheme)
        // Sync with Electron if available
        window.bashgym?.theme.set(resolvedTheme)
      },

      toggleTheme: () => {
        const current = get().theme
        const newTheme: Theme = current === 'dark' ? 'light' : current === 'light' ? 'system' : 'dark'
        get().setTheme(newTheme)
      }
    }),
    {
      name: 'bash-gym-theme',
      onRehydrateStorage: () => (state) => {
        if (state) {
          const resolvedTheme = state.theme === 'system' ? getSystemTheme() : state.theme
          state.resolvedTheme = resolvedTheme
          applyTheme(resolvedTheme)
        }
      }
    }
  )
)

// Listen for system theme changes
if (typeof window !== 'undefined' && window.matchMedia) {
  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
  mediaQuery.addEventListener('change', (e) => {
    const state = useThemeStore.getState()
    if (state.theme === 'system') {
      const newResolvedTheme: ResolvedTheme = e.matches ? 'dark' : 'light'
      useThemeStore.setState({ resolvedTheme: newResolvedTheme })
      applyTheme(newResolvedTheme)
      window.bashgym?.theme.set(newResolvedTheme)
    }
  })
}
