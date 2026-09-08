import { useState, type FormEvent } from 'react'
import { ArrowRight, Flower2, KeyRound } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import '../../styles/studio.css'

export function LoginPage() {
  const [code, setCode] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const pair = useAuthStore((state) => state.pair)
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!code.trim() || pending) return
    setPending(true)
    setError(null)
    try {
      await pair(code.trim())
      setCode('')
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Pairing failed. Please retry.')
    } finally {
      setPending(false)
    }
  }
  return (
    <main className="research-studio studio-pairing">
      <div className="studio-pairing-intro">
        <span className="studio-wordmark">
          <Flower2 aria-hidden="true" /> BashGym
        </span>
        <p className="studio-eyebrow">A quiet research studio</p>
        <h1>
          Make room for
          <br />
          the next idea.
        </h1>
        <p>One hypothesis. One considered change. Evidence you can come back to.</p>
        <ol className="studio-loop" aria-label="Research loop">
          {['Evaluate', 'Experiment', 'Compare', 'Decide'].map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
      </div>
      <form className="studio-pairing-card" onSubmit={submit}>
        <span className="studio-icon-tile">
          <KeyRound aria-hidden="true" />
        </span>
        <h2>Connect your studio</h2>
        <p>Enter the local pairing code shown by your research service to open your projects.</p>
        <label htmlFor="pairing-code">Pairing code</label>
        <input
          id="pairing-code"
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={code}
          onChange={(event) => setCode(event.target.value)}
          required
          aria-describedby={error ? 'pairing-error' : 'pairing-help'}
          disabled={pending}
        />
        <p id="pairing-help" className="studio-caption">
          Your session stays in a secure browser cookie.
        </p>
        <p className="studio-caption">
          Run <code>bashgym init</code> in your terminal to get a pairing code.
        </p>
        {error && (
          <p id="pairing-error" className="studio-notice" role="alert">
            {error}
          </p>
        )}
        <button className="studio-primary" disabled={pending || !code.trim()} type="submit">
          {pending ? 'Connecting…' : 'Open studio'}
          <ArrowRight size={17} aria-hidden="true" />
        </button>
        <details className="studio-alternate-auth">
          <summary>Other sign-in options</summary>
          <a href="/api/auth/github">Sign in with GitHub</a>
        </details>
      </form>
    </main>
  )
}
