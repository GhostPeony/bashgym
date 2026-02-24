import { useEffect } from 'react'
import { MainLayout } from './components/layout/MainLayout'
import { SettingsModal } from './components/common'
import { OnboardingModal } from './components/onboarding/OnboardingModal'
import { useThemeStore, useAccentStore, useUIStore } from './stores'
import { useTutorialStore } from './stores/tutorialStore'
import { useGlobalHotkeys } from './hooks'
import { wsService } from './services'

function App() {
  const { theme } = useThemeStore()
  const { accentHue } = useAccentStore()
  const { setOnboardingOpen } = useUIStore()
  const { hasSeenIntro } = useTutorialStore()

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

  // Connect WebSocket on mount (with small delay to avoid race conditions)
  useEffect(() => {
    const timer = setTimeout(() => {
      console.log('App: Initiating WebSocket connection...')
      wsService.connect()
    }, 100)
    return () => {
      clearTimeout(timer)
      wsService.disconnect()
    }
  }, [])

  // Show onboarding on first visit
  useEffect(() => {
    if (!hasSeenIntro) {
      setOnboardingOpen(true)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Global keyboard shortcuts
  useGlobalHotkeys()

  return (
    <>
      <MainLayout />
      <SettingsModal />
      <OnboardingModal />
    </>
  )
}

export default App
