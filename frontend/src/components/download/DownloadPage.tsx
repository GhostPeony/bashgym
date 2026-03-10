import { useState, useEffect } from 'react'
import { Download, Monitor, Apple, Terminal, ExternalLink, CheckCircle, ArrowRight } from 'lucide-react'
import { clsx } from 'clsx'

type Platform = 'windows' | 'mac' | 'linux' | 'unknown'

function detectPlatform(): Platform {
  const ua = navigator.userAgent.toLowerCase()
  if (ua.includes('win')) return 'windows'
  if (ua.includes('mac')) return 'mac'
  if (ua.includes('linux')) return 'linux'
  return 'unknown'
}

const PLATFORMS = {
  windows: {
    label: 'Windows',
    icon: Monitor,
    ext: '.exe',
    target: 'nsis',
    requirements: 'Windows 10+ (64-bit)',
  },
  mac: {
    label: 'macOS',
    icon: Apple,
    ext: '.dmg',
    target: 'dmg',
    requirements: 'macOS 12+ (Apple Silicon & Intel)',
  },
  linux: {
    label: 'Linux',
    icon: Terminal,
    ext: '.AppImage',
    target: 'AppImage',
    requirements: 'Ubuntu 20.04+ or equivalent',
  },
} as const

const FEATURES = [
  'Full terminal workspace with multi-panel PTY sessions',
  'Orchestrator for parallel Claude Code workers with git worktrees',
  'Agent chat panel with direct Claude CLI integration',
  'File browser with native filesystem access',
  'Browser pane for live preview',
  'One-click hook installation for trace capture',
  'Integration nodes (Neon, Vercel, custom adapters)',
]

interface ReleaseInfo {
  tag: string
  assets: Array<{ name: string; url: string; size: number }>
}

export function DownloadPage() {
  const [platform, setPlatform] = useState<Platform>('unknown')
  const [release, setRelease] = useState<ReleaseInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setPlatform(detectPlatform())
  }, [])

  // Try to fetch latest release info from GitHub (cached in sessionStorage for 15 min)
  useEffect(() => {
    const CACHE_KEY = 'bashgym_release'
    const CACHE_TTL = 15 * 60 * 1000 // 15 minutes

    const fetchRelease = async () => {
      // Check cache first
      try {
        const cached = sessionStorage.getItem(CACHE_KEY)
        if (cached) {
          const { data, ts } = JSON.parse(cached)
          if (Date.now() - ts < CACHE_TTL) {
            setRelease(data)
            setLoading(false)
            return
          }
        }
      } catch { /* ignore corrupted cache */ }

      try {
        const res = await fetch('https://api.github.com/repos/ghost-peony/bashgym/releases/latest')
        if (res.ok) {
          const data = await res.json()
          const releaseData = {
            tag: data.tag_name,
            assets: (data.assets || []).map((a: any) => ({
              name: a.name,
              url: a.browser_download_url,
              size: a.size,
            })),
          }
          setRelease(releaseData)
          sessionStorage.setItem(CACHE_KEY, JSON.stringify({ data: releaseData, ts: Date.now() }))
        }
      } catch {
        // No releases available yet — that's fine
      } finally {
        setLoading(false)
      }
    }
    fetchRelease()
  }, [])

  const getDownloadUrl = (plat: 'windows' | 'mac' | 'linux') => {
    if (!release) return null
    const ext = PLATFORMS[plat].ext
    const asset = release.assets.find(a => a.name.endsWith(ext))
    return asset?.url ?? null
  }

  const formatSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024)
    return `${mb.toFixed(0)} MB`
  }

  const primaryPlatform = platform !== 'unknown' ? platform : 'windows'
  const otherPlatforms = (['windows', 'mac', 'linux'] as const).filter(p => p !== primaryPlatform)

  return (
    <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">
      {/* Hero */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <img src="/ghost-icon.png" alt="BashGym" className="w-16 h-16 object-cover" />
        </div>
        <h1 className="font-brand text-4xl font-bold text-text-primary">
          <span className="text-accent">/</span>BashGym Desktop
        </h1>
        <p className="text-lg text-text-secondary max-w-xl mx-auto">
          The full development gym experience. Terminal workspace, orchestrator, agent chat,
          and native integrations — all in one app.
        </p>
      </div>

      {/* Primary Download */}
      <div className="border-brutal border-border rounded-brutal bg-background-card shadow-brutal p-8 text-center space-y-4">
        {loading ? (
          <div className="py-8 text-text-muted font-mono text-sm">Checking for releases...</div>
        ) : release ? (
          <>
            <p className="font-mono text-xs text-text-muted uppercase tracking-widest">
              Latest Release: {release.tag}
            </p>
            {(() => {
              const url = getDownloadUrl(primaryPlatform)
              const info = PLATFORMS[primaryPlatform]
              const Icon = info.icon
              return url ? (
                <a
                  href={url}
                  className="inline-flex items-center gap-3 px-8 py-4 bg-accent text-white font-mono font-semibold text-lg border-brutal border-border rounded-brutal shadow-brutal hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none transition-all"
                >
                  <Download className="w-6 h-6" />
                  Download for {info.label}
                  <span className="text-sm opacity-75">({info.ext})</span>
                </a>
              ) : (
                <div className="py-4 text-text-muted font-mono text-sm">
                  No {info.label} build available yet
                </div>
              )
            })()}
            <p className="text-xs text-text-muted">{PLATFORMS[primaryPlatform].requirements}</p>

            {/* Other platforms */}
            <div className="flex items-center justify-center gap-6 pt-4 border-t border-accent/15">
              {otherPlatforms.map(plat => {
                const url = getDownloadUrl(plat)
                const info = PLATFORMS[plat]
                const Icon = info.icon
                return url ? (
                  <a
                    key={plat}
                    href={url}
                    className="flex items-center gap-2 text-sm text-accent-dark hover:text-accent font-mono transition-colors"
                  >
                    <Icon className="w-4 h-4" />
                    {info.label} ({info.ext})
                  </a>
                ) : (
                  <span key={plat} className="flex items-center gap-2 text-sm text-text-muted font-mono">
                    <Icon className="w-4 h-4" />
                    {info.label} — coming soon
                  </span>
                )
              })}
            </div>
          </>
        ) : (
          <div className="space-y-4 py-4">
            <p className="text-text-secondary">
              No releases available yet. Desktop builds are coming soon.
            </p>
            <p className="text-sm text-text-muted">
              In the meantime, you can build from source:
            </p>
            <div className="terminal-chrome max-w-md mx-auto">
              <pre className="p-4 text-left text-xs font-mono text-text-secondary">
{`git clone https://github.com/ghost-peony/bashgym
cd bashgym/frontend
npm install
npm run build`}
              </pre>
            </div>
          </div>
        )}
      </div>

      {/* Desktop-only Features */}
      <div className="space-y-4">
        <h2 className="font-brand text-2xl font-semibold text-text-primary">
          Why Desktop?
        </h2>
        <p className="text-text-secondary text-sm">
          The web version gives you traces, training, and analytics. The desktop app
          adds everything that needs local access:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {FEATURES.map((feature, i) => (
            <div
              key={i}
              className="flex items-start gap-2 p-3 border-brutal border-border rounded-brutal bg-background-card shadow-brutal-sm"
            >
              <CheckCircle className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
              <span className="text-sm text-text-secondary">{feature}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Connect existing desktop */}
      <div className="border-brutal border-accent rounded-brutal bg-background-card p-6 space-y-3">
        <h3 className="font-brand text-lg font-semibold text-text-primary">
          Already have the desktop app?
        </h3>
        <p className="text-sm text-text-secondary">
          Connect your local Claude Code traces to this web instance for remote
          monitoring and training.
        </p>
        <div className="terminal-chrome">
          <pre className="p-3 text-xs font-mono text-text-secondary">
{`npx bashgym connect ${window.location.origin} --token YOUR_API_TOKEN`}
          </pre>
        </div>
        <p className="text-xs text-text-muted">
          This configures your local Claude Code hooks to send traces to this server.
          Run it on the machine where you use Claude Code.
        </p>
      </div>
    </div>
  )
}
