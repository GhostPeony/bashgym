import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/**
 * Top-level error boundary. Without one, any render-time throw unmounts the
 * whole React tree and the user sees a blank screen with no explanation. This
 * catches the error and shows it instead. Styled with inline styles so it
 * renders even if the theme/CSS layer is what failed.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('App crashed (caught by ErrorBoundary):', error, info.componentStack)
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0a0a0a',
            color: '#e5e5e5',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            padding: '2rem',
          }}
        >
          <div style={{ maxWidth: 640 }}>
            <h1 style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>
              Something broke while rendering the app.
            </h1>
            <p style={{ color: '#a3a3a3', marginBottom: '1rem', lineHeight: 1.5 }}>
              The error below was caught by the top-level boundary so the whole screen
              didn&apos;t go blank. Check the console for the full stack.
            </p>
            <pre
              style={{
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                background: '#1a1a1a',
                padding: '1rem',
                borderRadius: 8,
                fontSize: '0.8rem',
                color: '#fca5a5',
              }}
            >
              {this.state.error.message}
            </pre>
            <button
              type="button"
              onClick={() => window.location.reload()}
              style={{
                marginTop: '1rem',
                padding: '0.5rem 1rem',
                background: '#262626',
                color: '#e5e5e5',
                border: '1px solid #404040',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Reload
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
