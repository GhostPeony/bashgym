import { useState, useRef, useEffect, useCallback } from 'react'
import { X, Send, MessageSquare, Loader2, AlertCircle } from 'lucide-react'
import { useUIStore } from '../../stores'
import { agentApi } from '../../services/api'
import { clsx } from 'clsx'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
}

export function AgentChat() {
  const { isAgentChatOpen, setAgentChatOpen } = useUIStore()
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'I\'m your Gym Agent. I can help you plan training runs, check system status, schedule data generation, or answer questions about your traces and models. What would you like to do?',
      timestamp: Date.now(),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  useEffect(() => {
    if (isAgentChatOpen) {
      // Focus input when panel opens
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isAgentChatOpen])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || isLoading) return

    setError(null)
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      // Build conversation history for context
      const history = messages
        .filter(m => m.id !== 'welcome' || messages.length === 1)
        .map(m => ({ role: m.role, content: m.content }))
      history.push({ role: 'user', content: text })

      const result = await agentApi.chat(text, history)

      if (result.ok && result.data) {
        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: result.data.response,
          timestamp: Date.now(),
        }
        setMessages(prev => [...prev, assistantMsg])
      } else {
        setError(result.error || 'Failed to get response')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const sendQuickMessage = async (text: string) => {
    if (isLoading) return
    setError(null)
    setInput('')
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const history = messages
        .filter(m => m.id !== 'welcome' || messages.length === 1)
        .map(m => ({ role: m.role, content: m.content }))
      history.push({ role: 'user', content: text })
      const result = await agentApi.chat(text, history)
      if (result.ok && result.data) {
        setMessages(prev => [...prev, {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: result.data.response,
          timestamp: Date.now(),
        }])
      } else {
        setError(result.error || 'Failed to get response')
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setIsLoading(false)
    }
  }

  if (!isAgentChatOpen) return null

  return (
    <div className="fixed inset-y-0 right-0 w-[420px] max-w-full z-50 flex flex-col bg-background-primary border-l-[3px] border-border shadow-[-6px_0_0_0_var(--border)]">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b-[2px] border-border bg-background-secondary">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 border-brutal border-border rounded-brutal bg-accent-light flex items-center justify-center">
            <MessageSquare className="w-4 h-4 text-accent-dark" />
          </div>
          <div>
            <h2 className="font-brand text-base text-text-primary leading-tight">Gym Agent</h2>
            <p className="font-mono text-[10px] uppercase tracking-widest text-text-muted">System-Aware Assistant</p>
          </div>
        </div>
        <button
          onClick={() => setAgentChatOpen(false)}
          className="w-8 h-8 flex items-center justify-center border-brutal border-border rounded-brutal hover:bg-accent-light transition-colors"
        >
          <X className="w-4 h-4 text-text-secondary" />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto px-5 py-4 space-y-4">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={clsx(
              'flex',
              msg.role === 'user' ? 'justify-end' : 'justify-start'
            )}
          >
            <div
              className={clsx(
                'max-w-[85%] px-4 py-3 text-sm leading-relaxed',
                msg.role === 'user'
                  ? 'bg-accent text-white border-[2px] border-border rounded-brutal shadow-brutal-sm'
                  : 'bg-background-secondary text-text-primary border-[2px] border-border rounded-brutal'
              )}
            >
              {/* Render markdown-ish content with line breaks */}
              {msg.content.split('\n').map((line, i) => (
                <span key={i}>
                  {line.startsWith('- ') ? (
                    <span className="block pl-2 border-l-2 border-accent-light my-1">
                      {line.slice(2)}
                    </span>
                  ) : line.startsWith('**') && line.endsWith('**') ? (
                    <strong className="block font-semibold mt-2 mb-1">
                      {line.slice(2, -2)}
                    </strong>
                  ) : line.startsWith('`') && line.endsWith('`') ? (
                    <code className="font-mono text-xs bg-background-tertiary px-1.5 py-0.5 rounded border border-border">
                      {line.slice(1, -1)}
                    </code>
                  ) : (
                    <span>{line}</span>
                  )}
                  {i < msg.content.split('\n').length - 1 && <br />}
                </span>
              ))}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-background-secondary border-[2px] border-border rounded-brutal px-4 py-3 flex items-center gap-2">
              <Loader2 className="w-4 h-4 text-accent animate-spin" />
              <span className="font-mono text-xs text-text-muted uppercase tracking-widest">Thinking</span>
            </div>
          </div>
        )}

        {error && (
          <div className="flex justify-start">
            <div className="bg-status-error/10 border-[2px] border-status-error/30 rounded-brutal px-4 py-3 flex items-center gap-2 max-w-[85%]">
              <AlertCircle className="w-4 h-4 text-status-error shrink-0" />
              <span className="text-sm text-status-error">{error}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Quick Actions */}
      <div className="px-5 pb-2 flex gap-2 flex-wrap">
        {messages.length <= 1 && (
          <>
            <button
              onClick={() => sendQuickMessage('What\'s my current system status?')}
              className="tag text-[10px] cursor-pointer hover:bg-accent-light transition-colors"
            >
              <span>System Status</span>
            </button>
            <button
              onClick={() => sendQuickMessage('Help me plan a training run')}
              className="tag text-[10px] cursor-pointer hover:bg-accent-light transition-colors"
            >
              <span>Plan Training</span>
            </button>
            <button
              onClick={() => sendQuickMessage('How should I generate more training data?')}
              className="tag text-[10px] cursor-pointer hover:bg-accent-light transition-colors"
            >
              <span>Data Generation</span>
            </button>
          </>
        )}
      </div>

      {/* Input */}
      <div className="px-5 pb-5 pt-2">
        <div className="flex items-end gap-2 border-[2px] border-border rounded-brutal bg-background-secondary p-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your agent..."
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm text-text-primary placeholder-text-muted outline-none px-2 py-1.5 max-h-24 overflow-auto font-sans"
            style={{ minHeight: '36px' }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className={clsx(
              'w-9 h-9 flex items-center justify-center border-brutal border-border rounded-brutal transition-all shrink-0',
              input.trim() && !isLoading
                ? 'bg-accent text-white shadow-brutal-sm hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-none'
                : 'bg-background-tertiary text-text-muted cursor-not-allowed'
            )}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="font-mono text-[9px] uppercase tracking-widest text-text-muted mt-2 text-center">
          Enter to send / Shift+Enter for newline
        </p>
      </div>
    </div>
  )
}
