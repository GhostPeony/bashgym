import { useEffect } from 'react'
import { MainLayout } from './components/layout/MainLayout'
import { SettingsModal } from './components/common'
import { useThemeStore, useAccentStore } from './stores'
import { useGlobalHotkeys } from './hooks'
import { wsService } from './services'

function App() {
  const { theme } = useThemeStore()
  const { accentHue } = useAccentStore()

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

  // Global keyboard shortcuts
  useGlobalHotkeys()

  return (
    <>
      <MainLayout />
      <SettingsModal />
    </>
  )
}

export default App
