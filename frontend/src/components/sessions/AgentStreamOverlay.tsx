import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot,
  ChevronDown,
  GripHorizontal,
  Palette,
  SendHorizontal,
  Terminal,
  X
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  getTerminalFgColor,
  useAccentStore,
  useTerminalStore,
  useThemeStore,
  useUIStore
} from '../../stores'
import {
  sendHermesStreamPrompt,
  useAgentStreamStore,
  type HermesStreamMessage,
  type HermesStreamSnapshot
} from '../../stores/agentStreamStore'
import {
  clampStreamGeometry,
  normalizeTerminalFeed,
  type StreamGeometry
} from './agentStreamPresentation'

const GEOMETRY_KEY = 'bashgym_agent_stream_geometry_v3'

type StreamSource =
  | {
      id: string
      kind: 'terminal'
      label: string
      subtitle: string
      terminalId: string
      lines: string[]
      busy: boolean
    }
  | {
      id: string
      kind: 'hermes'
      label: string
      subtitle: string
      panelId: string
      snapshot: HermesStreamSnapshot
      busy: boolean
    }

interface PointerInteraction {
  kind: 'move' | 'resize'
  startX: number
  startY: number
  geometry: StreamGeometry
}

function defaultGeometry(sidebarOpen: boolean): StreamGeometry {
  return {
    x: sidebarOpen ? 268 : 12,
    y: 112,
    width: 380,
    height: 470
  }
}

function loadGeometry(sidebarOpen: boolean): StreamGeometry {
  try {
    const stored = localStorage.getItem(GEOMETRY_KEY)
    if (stored) {
      const value = JSON.parse(stored) as Partial<StreamGeometry>
      if ([value.x, value.y, value.width, value.height].every(Number.isFinite)) {
        return value as StreamGeometry
      }
    }
  } catch {
    // Use the anchored default.
  }
  return defaultGeometry(sidebarOpen)
}

function saveGeometry(geometry: StreamGeometry): void {
  try {
    localStorage.setItem(GEOMETRY_KEY, JSON.stringify(geometry))
  } catch {
    // Geometry persistence is optional.
  }
}

function hermesTranscript(messages: HermesStreamMessage[]): HermesStreamMessage[] {
  return messages.filter((message) => message.content.trim())
}

/** Ambient, draggable transcript for terminal agents and Hermes chats. */
export function AgentStreamOverlay() {
  const isOpen = useUIStore((state) => state.isAgentStreamOpen)
  const setOpen = useUIStore((state) => state.setAgentStreamOpen)
  const setSettingsOpen = useUIStore((state) => state.setSettingsOpen)
  const isSidebarOpen = useUIStore((state) => state.isSidebarOpen)
  const panels = useTerminalStore((state) => state.panels)
  const sessions = useTerminalStore((state) => state.sessions)
  const hermesStreams = useAgentStreamStore((state) => state.hermesStreams)
  const hermesVersion = useAgentStreamStore((state) => state.version)
  const accentHue = useAccentStore((state) => state.accentHue)
  const terminalFgHue = useAccentStore((state) => state.terminalFgHue)
  const resolvedTheme = useThemeStore((state) => state.resolvedTheme)
  const [selectedId, setSelectedId] = useState('')
  const [prompt, setPrompt] = useState('')
  const [feedback, setFeedback] = useState<string | null>(null)
  const [geometry, setGeometry] = useState(() => loadGeometry(isSidebarOpen))
  const transcriptRef = useRef<HTMLDivElement>(null)
  const interactionRef = useRef<PointerInteraction | null>(null)

  const sources = useMemo<StreamSource[]>(() => {
    void hermesVersion
    const terminalSources: StreamSource[] = panels.flatMap((panel) => {
      if (panel.type !== 'terminal' || !panel.terminalId) return []
      const session = sessions.get(panel.terminalId)
      if (!session) return []
      return [
        {
          id: `terminal:${session.id}`,
          kind: 'terminal' as const,
          label: panel.title,
          subtitle: session.agentKind
            ? `${session.agentKind} · ${session.status.replace('_', ' ')}`
            : 'shell',
          terminalId: session.id,
          lines: normalizeTerminalFeed(session.lastOutput ?? []),
          busy: session.status === 'running' || session.status === 'tool_calling'
        }
      ]
    })
    const hermesSources: StreamSource[] = Array.from(hermesStreams.values()).map((snapshot) => ({
      id: `hermes:${snapshot.panelId}`,
      kind: 'hermes' as const,
      label: snapshot.label,
      subtitle: snapshot.sending ? snapshot.activity || 'responding' : snapshot.endpointId,
      panelId: snapshot.panelId,
      snapshot,
      busy: snapshot.sending
    }))
    return [...terminalSources, ...hermesSources]
  }, [hermesStreams, hermesVersion, panels, sessions])

  useEffect(() => {
    if (sources.length === 0) {
      setSelectedId('')
      return
    }
    if (!sources.some((source) => source.id === selectedId)) setSelectedId(sources[0].id)
  }, [selectedId, sources])

  const selected = sources.find((source) => source.id === selectedId) ?? sources[0]
  const terminalLines = useMemo(
    () => (selected?.kind === 'terminal' ? selected.lines : []),
    [selected]
  )
  const hermesMessages = useMemo(
    () => (selected?.kind === 'hermes' ? hermesTranscript(selected.snapshot.messages) : []),
    [selected]
  )

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: 'smooth' })
  }, [hermesMessages, terminalLines])

  const clampToViewport = useCallback(
    (next: StreamGeometry) =>
      clampStreamGeometry(next, {
        width: window.innerWidth,
        height: window.innerHeight
      }),
    []
  )

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      const interaction = interactionRef.current
      if (!interaction) return
      const deltaX = event.clientX - interaction.startX
      const deltaY = event.clientY - interaction.startY
      const next =
        interaction.kind === 'move'
          ? {
              ...interaction.geometry,
              x: interaction.geometry.x + deltaX,
              y: interaction.geometry.y + deltaY
            }
          : {
              ...interaction.geometry,
              width: interaction.geometry.width + deltaX,
              height: interaction.geometry.height + deltaY
            }
      setGeometry(clampToViewport(next))
    }
    const handlePointerUp = () => {
      if (!interactionRef.current) return
      interactionRef.current = null
      setGeometry((current) => {
        saveGeometry(current)
        return current
      })
    }
    const handleViewportResize = () => setGeometry((current) => clampToViewport(current))
    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerup', handlePointerUp)
    window.addEventListener('resize', handleViewportResize)
    return () => {
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerup', handlePointerUp)
      window.removeEventListener('resize', handleViewportResize)
    }
  }, [clampToViewport])

  const beginInteraction = (kind: PointerInteraction['kind'], event: React.PointerEvent) => {
    event.preventDefault()
    interactionRef.current = {
      kind,
      startX: event.clientX,
      startY: event.clientY,
      geometry
    }
  }

  const submitPrompt = () => {
    const text = prompt.trim()
    if (!text || !selected) return
    if (selected.kind === 'terminal') {
      window.bashgym?.terminal.write(selected.terminalId, `${text}\r`)
      setFeedback('Sent to terminal')
    } else if (sendHermesStreamPrompt(selected.panelId, text)) {
      setFeedback('Sent to Hermes')
    } else {
      setFeedback('Open the canvas to reconnect this Hermes feed')
      return
    }
    setPrompt('')
  }

  const feedTextColor = getTerminalFgColor(resolvedTheme, terminalFgHue)
  const glassBackground =
    resolvedTheme === 'dark'
      ? `hsla(${accentHue}, 24%, 9%, 0.78)`
      : `hsla(${accentHue}, 35%, 97%, 0.72)`
  const composerBackground =
    resolvedTheme === 'dark' ? `hsla(${accentHue}, 18%, 18%, 0.72)` : 'rgba(255,255,255,0.68)'

  if (!isOpen) return null

  return (
    <aside
      className="fixed z-40 flex overflow-hidden rounded-2xl backdrop-blur-2xl"
      style={{
        left: geometry.x,
        top: geometry.y,
        width: geometry.width,
        height: geometry.height,
        background: glassBackground,
        boxShadow: `0 18px 60px hsla(${accentHue}, 42%, 18%, 0.24), inset 0 1px rgba(255,255,255,0.16)`
      }}
      aria-label="Live agent feed"
    >
      <div className="flex min-h-0 w-full flex-col">
        <header className="flex flex-none items-center gap-2 px-3 pb-1 pt-2.5">
          <button
            type="button"
            onPointerDown={(event) => beginInteraction('move', event)}
            onDoubleClick={() => {
              const reset = clampToViewport(defaultGeometry(isSidebarOpen))
              setGeometry(reset)
              saveGeometry(reset)
            }}
            className="flex h-7 w-7 cursor-grab items-center justify-center rounded-lg text-text-muted hover:bg-white/10 hover:text-text-primary active:cursor-grabbing"
            title="Drag to move · double-click to reset position"
            aria-label="Move agent feed"
          >
            <GripHorizontal className="h-4 w-4" />
          </button>
          <span
            className={clsx('status-dot flex-shrink-0', selected?.busy ? 'status-success' : '')}
          />
          {selected?.kind === 'hermes' ? (
            <Bot className="h-4 w-4 flex-shrink-0 text-accent" />
          ) : (
            <Terminal className="h-4 w-4 flex-shrink-0 text-accent" />
          )}
          <label className="relative min-w-0 flex-1">
            <span className="sr-only">Viewing agent feed</span>
            <select
              value={selected?.id ?? ''}
              onChange={(event) => {
                setSelectedId(event.target.value)
                setFeedback(null)
              }}
              className="w-full cursor-pointer appearance-none truncate bg-transparent pr-4 font-mono text-[11px] font-semibold text-text-primary outline-none"
              title="Choose terminal or Hermes chat"
              aria-label="Viewing agent feed"
            >
              {sources.length === 0 ? <option value="">No active agents</option> : null}
              {sources.map((source) => (
                <option key={source.id} value={source.id}>
                  {source.kind === 'hermes' ? 'Hermes' : 'Terminal'} · {source.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-0 top-0.5 h-3 w-3 text-text-muted" />
            <span className="block truncate font-mono text-[9px] uppercase tracking-wider text-text-muted">
              {selected?.subtitle ?? 'Start a terminal or add Hermes'}
            </span>
          </label>
          <button
            type="button"
            onClick={() => setSettingsOpen(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-text-muted hover:bg-white/10 hover:text-accent"
            title="Appearance — follows Accent and Terminal Text Color"
            aria-label="Open appearance settings"
          >
            <Palette className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-text-muted hover:bg-white/10 hover:text-text-primary"
            title="Hide agent feed"
            aria-label="Hide agent feed"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </header>

        <div
          ref={transcriptRef}
          className="min-h-0 flex-1 overflow-y-auto px-4 py-3 [mask-image:linear-gradient(to_bottom,transparent_0,black_12%,black_100%)]"
          aria-live="polite"
        >
          {selected?.kind === 'hermes' ? (
            <div className="space-y-4">
              {hermesMessages.length > 0 ? (
                hermesMessages.map((message, index) => (
                  <div
                    key={`${message.role}-${index}`}
                    className={clsx(
                      'max-w-[92%]',
                      message.role === 'user' ? 'ml-auto text-right' : null
                    )}
                  >
                    <div
                      className={clsx(
                        'mb-1 font-mono text-[9px] font-semibold uppercase tracking-[0.14em]',
                        message.role === 'user' ? 'text-text-muted' : 'text-accent'
                      )}
                    >
                      {message.role === 'user' ? 'You' : selected.label}
                    </div>
                    <div className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-text-primary">
                      {message.content}
                    </div>
                  </div>
                ))
              ) : (
                <p className="pt-8 text-center font-mono text-[11px] text-text-muted">
                  Waiting for this Hermes conversation…
                </p>
              )}
              {selected.snapshot.sending ? (
                <div className="font-mono text-[10px] text-accent">
                  {selected.snapshot.activity || 'Hermes is responding…'}
                </div>
              ) : null}
              {selected.snapshot.error ? (
                <div className="font-mono text-[10px] text-status-error">
                  {selected.snapshot.error}
                </div>
              ) : null}
            </div>
          ) : (
            <div
              className="space-y-1.5 font-mono text-[11px] leading-relaxed"
              style={{ color: feedTextColor }}
            >
              {terminalLines.length > 0 ? (
                terminalLines.map((line, index) => {
                  const recency = (index + 1) / terminalLines.length
                  return (
                    <div
                      key={`${index}-${line}`}
                      className="whitespace-pre-wrap break-words"
                      style={{ opacity: 0.3 + recency * 0.7 }}
                    >
                      {line}
                    </div>
                  )
                })
              ) : (
                <p className="pt-8 text-center text-text-muted">Waiting for terminal output…</p>
              )}
            </div>
          )}
        </div>

        <footer className="flex-none px-3 pb-3 pt-1">
          <div
            className="flex items-end gap-2 rounded-xl px-3 py-2 shadow-sm backdrop-blur-xl transition-shadow focus-within:shadow-md"
            style={{
              background: composerBackground,
              boxShadow: `inset 0 0 0 1px hsla(${accentHue}, 30%, 55%, 0.18)`
            }}
          >
            <textarea
              value={prompt}
              onChange={(event) => {
                setPrompt(event.target.value)
                setFeedback(null)
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault()
                  submitPrompt()
                }
              }}
              rows={1}
              className="max-h-24 min-h-8 min-w-0 flex-1 resize-none bg-transparent py-1 text-[13px] leading-relaxed text-text-primary outline-none placeholder:text-text-muted"
              placeholder={selected ? `Message ${selected.label}…` : 'No agent selected'}
              disabled={!selected}
              aria-label={selected ? `Message ${selected.label}` : 'No agent selected'}
            />
            <button
              type="button"
              onClick={submitPrompt}
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-full text-white shadow-md transition-transform hover:scale-105 disabled:cursor-not-allowed disabled:opacity-35"
              style={{ background: `hsl(${accentHue}, 38%, 54%)` }}
              disabled={!selected || !prompt.trim()}
              title={selected ? `Send to ${selected.label}` : 'No agent selected'}
              aria-label={selected ? `Send message to ${selected.label}` : 'Send message'}
            >
              <SendHorizontal className="h-4 w-4" />
            </button>
          </div>
          {feedback ? (
            <div className="px-2 pt-1.5 font-mono text-[9px] text-text-muted">{feedback}</div>
          ) : null}
        </footer>
      </div>

      <button
        type="button"
        onPointerDown={(event) => beginInteraction('resize', event)}
        className="absolute bottom-1 right-1 h-5 w-5 cursor-nwse-resize opacity-40 hover:opacity-80"
        title="Drag to resize agent feed"
        aria-label="Resize agent feed"
      >
        <span className="absolute bottom-1 right-1 h-2.5 w-2.5 border-b-2 border-r-2 border-current" />
      </button>
    </aside>
  )
}
