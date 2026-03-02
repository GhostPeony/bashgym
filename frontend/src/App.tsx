import { useEffect, lazy, Suspense } from 'react'
import { MainLayout } from './components/layout/MainLayout'
import { SettingsModal } from './components/common'
import { OnboardingModal } from './components/onboarding/OnboardingModal'
import { useThemeStore, useAccentStore, useUIStore, useAuthStore } from './stores'
import { useTutorialStore } from './stores/tutorialStore'
import { useGlobalHotkeys } from './hooks'
import { wsService } from './services'
import { isWeb } from './utils/platform'

// Tree-shaken in Electron builds (isWeb is a compile-time constant)
const LoginPage = isWeb ? lazy(() => import('./components/auth/LoginPage').then(m => ({ default: m.LoginPage }))) : null

function App() {
  const { theme } = useThemeStore()
  const { accentHue } = useAccentStore()
  const { setOnboardingOpen } = useUIStore()
  const { hasSeenIntro } = useTutorialStore()
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore()

  // Apply theme on mount
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  // Apply accent hue on mount
  useEffect(() => {
    document.documentElement.style.setProperty('--accent-hue', String(accentHue))
  }, [accentHue])

  // Check auth on mount (web mode only)
  useEffect(() => {
    if (isWeb) {
      checkAuth()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Connect WebSocket on mount — delay until authenticated in web mode
  useEffect(() => {
    if (isWeb && !isAuthenticated) return

    const timer = setTimeout(() => {
      console.log('App: Initiating WebSocket connection...')
      wsService.connect()
    }, 100)
    return () => {
      clearTimeout(timer)
      wsService.disconnect()
    }
  }, [isAuthenticated])

  // Show onboarding on first visit
  useEffect(() => {
    if (!hasSeenIntro) {
      setOnboardingOpen(true)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Global keyboard shortcuts
  useGlobalHotkeys()

  // Web mode: show login page if not authenticated
  if (isWeb) {
    if (isLoading) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-background">
          <div className="text-text-secondary font-mono text-sm uppercase tracking-wider animate-pulse">
            Loading...
          </div>
        </div>
      )
    }
    if (!isAuthenticated && LoginPage) {
      return (
        <Suspense fallback={null}>
          <LoginPage />
        </Suspense>
      )
    }
  }

  return (
    <>
      <MainLayout />
      <SettingsModal />
      <OnboardingModal />
    </>
  )
}

export default App
