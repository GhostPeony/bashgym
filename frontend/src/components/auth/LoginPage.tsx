import { Github } from 'lucide-react'

export function LoginPage() {
  const handleLogin = () => {
    window.location.href = '/api/auth/github'
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm border-2 border-border bg-background-card relative"
        style={{ boxShadow: '4px 4px 0 var(--border)' }}>
        {/* Header */}
        <div className="p-8 pb-4 text-center">
          <img
            src="/ghost-icon.png"
            alt="BashGym"
            className="w-16 h-16 mx-auto mb-4 object-cover"
          />
          <h1 className="font-brand text-2xl mb-1">
            <span className="text-accent">/</span>
            <span className="text-text-primary">BashGym</span>
          </h1>
          <p className="text-text-secondary text-sm font-mono uppercase tracking-wider">
            Agent Training Gym
          </p>
        </div>

        {/* Divider */}
        <div className="border-t border-border mx-6" />

        {/* Login Action */}
        <div className="p-8 pt-6">
          <button
            onClick={handleLogin}
            className="btn-primary w-full flex items-center justify-center gap-3 py-3 font-mono text-sm uppercase tracking-wider"
          >
            <Github className="w-5 h-5" />
            Sign in with GitHub
          </button>

          <p className="text-text-muted text-xs text-center mt-4 font-mono">
            Authentication via GitHub OAuth
          </p>
        </div>
      </div>
    </div>
  )
}
