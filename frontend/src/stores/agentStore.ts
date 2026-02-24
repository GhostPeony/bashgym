import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { agentApi } from '../services/api'
import type { AgentSessionMeta, AgentSessionMessage, PendingAction } from '../services/api'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  contextUsed: string[]
}

interface SessionMeta {
  sessionId: string
  name: string
  createdAt: string
  updatedAt: string
  messageCount: number
}

interface AgentState {
  sessions: SessionMeta[]
  activeSessionId: string | null
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
  pendingAction: PendingAction | null

  createSession: (name?: string) => string
  switchSession: (sessionId: string) => Promise<void>
  renameSession: (sessionId: string, name: string) => void
  deleteSession: (sessionId: string) => Promise<void>
  addUserMessage: (content: string) => ChatMessage
  addAssistantMessage: (content: string, contextUsed: string[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  setPendingAction: (action: PendingAction | null) => void
  saveActiveSession: () => Promise<void>
  loadSessions: () => Promise<void>
  initializeDefaultSession: () => void
}

function generateId(): string {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function formatSessionDate(date: Date): string {
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}

function toApiMessages(messages: ChatMessage[]): AgentSessionMessage[] {
  return messages.map(m => ({
    id: m.id,
    role: m.role,
    content: m.content,
    timestamp: m.timestamp,
    context_used: m.contextUsed,
  }))
}

function fromApiMessages(msgs: AgentSessionMessage[]): ChatMessage[] {
  return msgs.map(m => ({
    id: m.id,
    role: m.role as 'user' | 'assistant',
    content: m.content,
    timestamp: m.timestamp,
    contextUsed: m.context_used ?? [],
  }))
}

function fromApiSessionMeta(s: AgentSessionMeta): SessionMeta {
  return {
    sessionId: s.session_id,
    name: s.name,
    createdAt: s.created_at,
    updatedAt: s.updated_at,
    messageCount: s.message_count,
  }
}

export const useAgentStore = create<AgentState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,
      messages: [],
      isLoading: false,
      error: null,
      pendingAction: null,

      createSession: (name?: string) => {
        const sessionId = generateId()
        const now = new Date().toISOString()
        const sessionName = name ?? formatSessionDate(new Date())
        const meta: SessionMeta = {
          sessionId,
          name: sessionName,
          createdAt: now,
          updatedAt: now,
          messageCount: 0,
        }
        set(state => ({
          sessions: [meta, ...state.sessions],
          activeSessionId: sessionId,
          messages: [],
          error: null,
        }))
        return sessionId
      },

      switchSession: async (sessionId: string) => {
        const { activeSessionId } = get()
        if (sessionId === activeSessionId) return

        set({ activeSessionId: sessionId, messages: [], error: null })

        // Try loading from backend
        try {
          const result = await agentApi.loadSession(sessionId)
          if (result.ok && result.data) {
            set({ messages: fromApiMessages(result.data) })
          }
        } catch {
          // Session may not exist on backend yet — that's fine
        }
      },

      renameSession: (sessionId: string, name: string) => {
        set(state => ({
          sessions: state.sessions.map(s =>
            s.sessionId === sessionId ? { ...s, name, updatedAt: new Date().toISOString() } : s
          ),
        }))
      },

      deleteSession: async (sessionId: string) => {
        const { sessions, activeSessionId, createSession } = get()

        // Remove from local state
        const remaining = sessions.filter(s => s.sessionId !== sessionId)
        set({ sessions: remaining })

        // If we deleted the active session, switch
        if (activeSessionId === sessionId) {
          if (remaining.length > 0) {
            await get().switchSession(remaining[0].sessionId)
          } else {
            createSession()
          }
        }

        // Delete from backend (fire and forget)
        agentApi.deleteSession(sessionId).catch(() => {})
      },

      addUserMessage: (content: string) => {
        const msg: ChatMessage = {
          id: `user-${Date.now()}`,
          role: 'user',
          content,
          timestamp: Date.now(),
          contextUsed: [],
        }
        set(state => ({ messages: [...state.messages, msg] }))
        return msg
      },

      addAssistantMessage: (content: string, contextUsed: string[]) => {
        const msg: ChatMessage = {
          id: `asst-${Date.now()}`,
          role: 'assistant',
          content,
          timestamp: Date.now(),
          contextUsed,
        }
        set(state => ({
          messages: [...state.messages, msg],
          sessions: state.sessions.map(s =>
            s.sessionId === state.activeSessionId
              ? { ...s, messageCount: state.messages.length + 1, updatedAt: new Date().toISOString() }
              : s
          ),
        }))
      },

      setLoading: (loading: boolean) => set({ isLoading: loading }),
      setError: (error: string | null) => set({ error }),
      setPendingAction: (action: PendingAction | null) => set({ pendingAction: action }),

      saveActiveSession: async () => {
        const { activeSessionId, sessions, messages } = get()
        if (!activeSessionId || messages.length === 0) return

        const session = sessions.find(s => s.sessionId === activeSessionId)
        if (!session) return

        try {
          await agentApi.saveSession(
            activeSessionId,
            session.name,
            toApiMessages(messages),
          )
        } catch {
          // Silently fail — localStorage is the primary store
        }
      },

      loadSessions: async () => {
        try {
          const result = await agentApi.listSessions()
          if (result.ok && result.data && result.data.length > 0) {
            const backendSessions = result.data.map(fromApiSessionMeta)
            // Merge: keep local sessions not on backend, add backend sessions
            set(state => {
              const localIds = new Set(state.sessions.map(s => s.sessionId))
              const merged = [...state.sessions]
              for (const bs of backendSessions) {
                if (!localIds.has(bs.sessionId)) {
                  merged.push(bs)
                }
              }
              return { sessions: merged }
            })
          }
        } catch {
          // Backend unavailable — that's fine, we have localStorage
        }
      },

      initializeDefaultSession: () => {
        const { sessions, createSession } = get()
        if (sessions.length === 0) {
          createSession()
        }
      },
    }),
    {
      name: 'bashgym-peony-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
)
