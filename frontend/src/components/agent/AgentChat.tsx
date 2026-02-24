import { useState, useRef, useEffect, useCallback } from 'react'
import { X, Send, MessageSquare, Loader2, AlertCircle, Plus, History, Trash2, Tag, ChevronRight, ArrowLeft, Terminal, CheckCircle, XCircle } from 'lucide-react'
import { useUIStore, useAgentStore } from '../../stores'
import { agentApi } from '../../services/api'
import { clsx } from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type View = 'chat' | 'history'

function relativeTime(dateStr: string): string {
  const now = Date.now()
  const then = new Date(dateStr).getTime()
  const diff = now - then
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 7) return `${days}d ago`
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export function AgentChat() {
  const { isAgentChatOpen, setAgentChatOpen } = useUIStore()
  const store = useAgentStore()
  const {
    sessions, activeSessionId, messages, isLoading, error, pendingAction,
    createSession, switchSession, deleteSession,
    addUserMessage, addAssistantMessage, setLoading, setError, setPendingAction,
    saveActiveSession, loadSessions, initializeDefaultSession,
  } = store

  const [input, setInput] = useState('')
  const [view, setView] = useState<View>('chat')
  const [contextSources, setContextSources] = useState<string[]>([])
  const [showContext, setShowContext] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeSession = sessions.find(s => s.sessionId === activeSessionId)

  // Initialize on mount
  useEffect(() => {
    initializeDefaultSession()
    loadSessions()
  }, [])

  // Load active session messages from backend on mount
  useEffect(() => {
    if (activeSessionId && messages.length === 0) {
      agentApi.loadSession(activeSessionId).then(result => {
        if (result.ok && result.data && result.data.length > 0) {
          for (const msg of result.data) {
            if (msg.role === 'user') {
              store.addUserMessage(msg.content)
            } else if (msg.role === 'assistant') {
              store.addAssistantMessage(msg.content, msg.context_used ?? [])
            }
          }
        }
      }).catch(() => {})
    }
  }, [activeSessionId])

  // Track latest context sources from assistant messages
  useEffect(() => {
    const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant')
    if (lastAssistant?.contextUsed?.length) {
      setContextSources(lastAssistant.contextUsed)
    }
  }, [messages])

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  // Scroll to bottom instantly on reopen
  useEffect(() => {
    if (isAgentChatOpen) {
      const timer = setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'instant' as ScrollBehavior })
      }, 50)
      return () => clearTimeout(timer)
    }
  }, [isAgentChatOpen])

  // Focus input when panel opens
  useEffect(() => {
    if (isAgentChatOpen && view === 'chat') {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isAgentChatOpen, view])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isLoading) return

    setError(null)
    setPendingAction(null)
    addUserMessage(text)
    setInput('')
    setLoading(true)

    try {
      const history = messages
        .map(m => ({ role: m.role, content: m.content }))
      history.push({ role: 'user', content: text })

      const result = await agentApi.chat(text, history)

      if (result.ok && result.data) {
        if (result.data.pending_action) {
          setPendingAction(result.data.pending_action)
        } else if (result.data.response) {
          addAssistantMessage(result.data.response, result.data.context_used ?? [])
          saveActiveSession()
        }
      } else {
        setError(result.error || 'Failed to get response')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleConfirmAction = async (approved: boolean) => {
    if (!pendingAction) return
    const token = pendingAction.token
    setPendingAction(null)
    setLoading(true)
    try {
      const result = await agentApi.confirmAction(token, approved, activeSessionId ?? undefined)
      if (result.ok && result.data) {
        addAssistantMessage(result.data.response, result.data.context_used ?? [])
        saveActiveSession()
      } else {
        setError(result.error || 'Action confirmation failed')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewSession = () => {
    createSession()
    setView('chat')
  }

  const handleSwitchSession = async (sessionId: string) => {
    await switchSession(sessionId)
    setView('chat')
  }

  const handleDeleteSession = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    await deleteSession(sessionId)
  }

  if (!isAgentChatOpen) return null

  const sortedSessions = [...sessions].sort((a, b) =>
    new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  )

  return (
    <div className="fixed bottom-20 right-6 z-50 w-[420px] max-w-[calc(100vw-3rem)] h-[600px] max-h-[calc(100vh-8rem)] flex flex-col
      bg-background-card dark:bg-[#141418]
      border-brutal border-border rounded-brutal shadow-brutal
      dark:border-[rgba(255,255,255,0.12)] dark:shadow-[0_0_40px_hsla(var(--accent-hue),20%,50%,0.15)]
      overflow-hidden"
    >
      {/* Header — accent fill, hard accent-dark bottom border for depth */}
      <div className="flex items-center justify-between px-4 py-3 bg-accent border-b-[3px] border-accent-dark">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {view === 'history' ? (
            <button
              onClick={() => setView('chat')}
              className="w-8 h-8 flex items-center justify-center text-white/60 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
          ) : (
            <div className="w-8 h-8 border-[2px] border-white/25 rounded-brutal bg-white/15 flex items-center justify-center shrink-0">
              <MessageSquare className="w-4 h-4 text-white" />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <h2 className="font-brand text-base text-white leading-tight">
              {view === 'history' ? 'Sessions' : 'Peony'}
            </h2>
            <p className="font-mono text-[9px] uppercase tracking-[0.15em] text-white/50 truncate mt-0.5">
              {view === 'history'
                ? `${sessions.length} conversation${sessions.length !== 1 ? 's' : ''}`
                : (activeSession?.name ?? 'Botanical Assistant')
              }
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {view === 'chat' && (
            <button
              onClick={() => setView('history')}
              className="w-8 h-8 flex items-center justify-center text-white/50 hover:text-white transition-colors"
              title="Session history"
            >
              <History className="w-4 h-4" />
            </button>
          )}
          {view === 'history' && (
            <button
              onClick={handleNewSession}
              className="w-8 h-8 flex items-center justify-center text-white/50 hover:text-white transition-colors"
              title="New conversation"
            >
              <Plus className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={() => setAgentChatOpen(false)}
            className="w-8 h-8 flex items-center justify-center text-white/50 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Context sources bar */}
      {view === 'chat' && contextSources.length > 0 && (
        <button
          onClick={() => setShowContext(p => !p)}
          className="flex items-center gap-2 px-4 py-2 border-b border-border/50 dark:border-[rgba(255,255,255,0.06)] bg-accent/[0.04] dark:bg-accent/[0.03] text-left w-full hover:bg-accent/[0.07] transition-colors"
        >
          <ChevronRight className={clsx(
            'w-3 h-3 text-text-muted transition-transform duration-150 shrink-0',
            showContext && 'rotate-90'
          )} />
          <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-text-muted">
            {showContext ? 'Context Sources' : `${contextSources.length} context source${contextSources.length !== 1 ? 's' : ''}`}
          </span>
        </button>
      )}
      {view === 'chat' && showContext && contextSources.length > 0 && (
        <div className="flex gap-1.5 flex-wrap px-4 py-2 border-b border-border/50 dark:border-[rgba(255,255,255,0.06)] bg-accent/[0.04] dark:bg-accent/[0.03]">
          {contextSources.map(ctx => (
            <span key={ctx} className="tag text-[9px]">
              {ctx}
            </span>
          ))}
        </div>
      )}

      {/* ---- Chat View ---- */}
      {view === 'chat' && (
        <>
          <div className="flex-1 overflow-auto px-4 py-4 space-y-4">
            {/* Welcome — render-only, not stored */}
            {messages.length === 0 && (
              <div className="flex justify-start">
                <div className="max-w-[88%] px-4 py-3 text-sm leading-relaxed bg-background-primary dark:bg-[rgba(255,255,255,0.04)] text-text-primary border-brutal border-border dark:border-[rgba(255,255,255,0.08)] rounded-brutal">
                  <div className="prose-brutal">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {`## Peony

I can help you with your training pipeline — from traces to fine-tuned models.

- **Plan training runs** — configure SFT, DPO, or GRPO strategies
- **Check system status** — GPU utilization, running jobs, model health
- **Generate data** — synthesize examples from your gold traces
- **Inspect traces** — quality scores, session breakdowns, export stats

> Ask me anything about your traces, models, or training pipeline.`}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            )}

            {messages.map(msg => (
              <div
                key={msg.id}
                className={clsx(
                  'flex',
                  msg.role === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                <div className="max-w-[88%]">
                  <div
                    className={clsx(
                      'px-4 py-3 text-sm leading-relaxed',
                      msg.role === 'user'
                        ? 'bg-accent text-white border-brutal border-accent-dark rounded-brutal shadow-brutal-sm'
                        : 'bg-background-primary dark:bg-[rgba(255,255,255,0.04)] text-text-primary border-brutal border-border dark:border-[rgba(255,255,255,0.08)] rounded-brutal'
                    )}
                  >
                    <div className={clsx(
                      'prose-brutal',
                      msg.role === 'user' && 'prose-brutal-inverted'
                    )}>
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                  {/* Context tags on assistant messages */}
                  {msg.role === 'assistant' && msg.contextUsed && msg.contextUsed.length > 0 && (
                    <div className="flex gap-1.5 mt-1.5 flex-wrap">
                      {msg.contextUsed.map(ctx => (
                        <span key={ctx} className="inline-flex items-center font-mono text-[9px] uppercase tracking-wider text-accent/60 dark:text-accent/50">
                          <Tag className="w-2.5 h-2.5 mr-1" />{ctx}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-background-primary dark:bg-[rgba(255,255,255,0.04)] border-brutal border-border dark:border-[rgba(255,255,255,0.08)] rounded-brutal px-4 py-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-accent animate-spin" />
                  <span className="font-mono text-xs text-accent/70 uppercase tracking-widest">Thinking</span>
                </div>
              </div>
            )}

            {/* Shell command confirmation gate */}
            {pendingAction && (
              <div className="flex justify-start">
                <div className="max-w-[88%] border-brutal border-status-warning/60 rounded-brutal bg-status-warning/5 p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <Terminal className="w-3.5 h-3.5 text-status-warning shrink-0" />
                    <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-status-warning font-bold">
                      Shell Command
                    </span>
                  </div>
                  <div className="font-mono text-xs text-text-primary bg-background-primary dark:bg-[rgba(0,0,0,0.3)] border border-border/50 rounded px-3 py-2 mb-2 break-all">
                    {pendingAction.command}
                  </div>
                  {pendingAction.reason && (
                    <p className="text-xs text-text-muted mb-3">{pendingAction.reason}</p>
                  )}
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleConfirmAction(true)}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal border-status-success/40 bg-status-success/10 text-status-success rounded-brutal hover:bg-status-success/20 transition-colors"
                    >
                      <CheckCircle className="w-3 h-3" />
                      Run it
                    </button>
                    <button
                      onClick={() => handleConfirmAction(false)}
                      className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold uppercase tracking-wide border-brutal border-status-error/40 bg-status-error/10 text-status-error rounded-brutal hover:bg-status-error/20 transition-colors"
                    >
                      <XCircle className="w-3 h-3" />
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="flex justify-start">
                <div className="bg-status-error/10 border-brutal border-status-error/30 rounded-brutal px-4 py-3 flex items-center gap-2 max-w-[88%]">
                  <AlertCircle className="w-4 h-4 text-status-error shrink-0" />
                  <span className="text-sm text-status-error">{error}</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="px-4 pb-4 pt-2 border-t-[3px] border-border dark:border-[rgba(255,255,255,0.08)]">
            <div className="flex items-end gap-2 border-brutal border-border dark:border-[rgba(255,255,255,0.1)] rounded-brutal bg-background-primary dark:bg-[rgba(0,0,0,0.3)] p-2">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask Peony..."
                rows={1}
                className="flex-1 resize-none bg-transparent text-sm text-text-primary placeholder-text-muted outline-none px-2 py-1.5 max-h-24 overflow-auto font-sans"
                style={{ minHeight: '36px' }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className={clsx(
                  'w-9 h-9 flex items-center justify-center border-brutal border-accent/40 rounded-brutal transition-all shrink-0',
                  input.trim() && !isLoading
                    ? 'bg-accent text-white shadow-brutal-sm hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-none'
                    : 'bg-background-tertiary text-text-muted cursor-not-allowed'
                )}
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
            <p className="font-mono text-[9px] uppercase tracking-widest text-text-muted mt-2 text-center">
              Enter to send · Shift+Enter for newline
            </p>
          </div>
        </>
      )}

      {/* ---- History View ---- */}
      {view === 'history' && (
        <div className="flex-1 overflow-auto">
          {sortedSessions.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-8">
              <History className="w-8 h-8 text-text-muted/40 mb-3" />
              <p className="text-sm text-text-muted">No conversations yet</p>
              <button
                onClick={handleNewSession}
                className="mt-4 font-mono text-xs uppercase tracking-widest text-accent hover:text-accent-dark transition-colors"
              >
                Start a conversation
              </button>
            </div>
          ) : (
            <div className="divide-y divide-border/40 dark:divide-[rgba(255,255,255,0.06)]">
              {sortedSessions.map(s => (
                <button
                  key={s.sessionId}
                  onClick={() => handleSwitchSession(s.sessionId)}
                  className={clsx(
                    'w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-accent/[0.06] transition-colors group',
                    s.sessionId === activeSessionId && 'bg-accent/[0.06]'
                  )}
                >
                  <div className="flex-1 min-w-0">
                    <p className={clsx(
                      'text-sm truncate',
                      s.sessionId === activeSessionId ? 'text-accent font-medium' : 'text-text-primary'
                    )}>
                      {s.name}
                    </p>
                    <p className="font-mono text-[10px] text-text-muted mt-0.5">
                      {relativeTime(s.updatedAt)} · {s.messageCount} msg{s.messageCount !== 1 ? 's' : ''}
                    </p>
                  </div>
                  <button
                    onClick={(e) => handleDeleteSession(e, s.sessionId)}
                    className="w-7 h-7 flex items-center justify-center text-text-muted/0 group-hover:text-text-muted hover:!text-status-error transition-colors shrink-0 rounded-brutal"
                    title="Delete session"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
