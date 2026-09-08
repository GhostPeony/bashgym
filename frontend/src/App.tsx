import { useEffect, lazy, Suspense } from 'react'
import { useThemeStore } from './stores/themeStore'
import { useAccentStore } from './stores/accentStore'
import { useAuthStore } from './stores/authStore'
import { isWeb } from './utils/platform'

// Tree-shaken in Electron builds (isWeb is a compile-time constant)
const LoginPage = isWeb
  ? lazy(() => import('./components/auth/LoginPage').then((m) => ({ default: m.LoginPage })))
  : null
const ResearchStudio = isWeb
  ? lazy(() =>
      import('./components/studio/ResearchStudio').then((m) => ({ default: m.ResearchStudio }))
    )
  : null
const DesktopShell = isWeb
  ? null
  : lazy(() =>
      import('./components/layout/DesktopShell').then((m) => ({ default: m.DesktopShell }))
    )

// Browser sessions pair with the local research service before loading data.
const requireWebAuth = isWeb

function App() {
  const { theme } = useThemeStore()
  const { accentHue } = useAccentStore()
  const { isAuthenticated, isLoading, checkAuth } = useAuthStore()

  // Apply theme on mount
  useEffect(() => {
    if (theme === 'dark' && !isWeb) {
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
    if (requireWebAuth && !isAuthenticated) return

    let disposed = false
    let disconnect: (() => void) | undefined
    const timer = setTimeout(() => {
      void import('./services/websocket').then(({ wsService }) => {
        if (disposed) return
        wsService.connect()
        disconnect = () => wsService.disconnect()
      })
    }, 100)
    return () => {
      disposed = true
      clearTimeout(timer)
      disconnect?.()
    }
  }, [isAuthenticated])

  // Browser mode requires a verified session in both development and production.
  if (requireWebAuth) {
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
      {ResearchStudio ? (
        <Suspense fallback={null}>
          <ResearchStudio />
        </Suspense>
      ) : DesktopShell ? (
        <Suspense fallback={null}>
          <DesktopShell />
        </Suspense>
      ) : null}
    </>
  )
}

export default App
