import { create } from 'zustand'

export interface HermesStreamMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface HermesStreamSnapshot {
  panelId: string
  label: string
  endpointId: string
  messages: HermesStreamMessage[]
  sending: boolean
  activity: string | null
  error: string | null
}

interface AgentStreamState {
  hermesStreams: Map<string, HermesStreamSnapshot>
  version: number
  publishHermesStream: (snapshot: HermesStreamSnapshot) => void
  removeHermesStream: (panelId: string) => void
}

type HermesPromptSender = (prompt: string) => void | Promise<void>
const hermesPromptSenders = new Map<string, HermesPromptSender>()

export const useAgentStreamStore = create<AgentStreamState>((set) => ({
  hermesStreams: new Map(),
  version: 0,

  publishHermesStream: (snapshot) => set((state) => {
    const next = new Map(state.hermesStreams)
    next.set(snapshot.panelId, snapshot)
    return { hermesStreams: next, version: state.version + 1 }
  }),

  removeHermesStream: (panelId) => set((state) => {
    if (!state.hermesStreams.has(panelId)) return state
    const next = new Map(state.hermesStreams)
    next.delete(panelId)
    return { hermesStreams: next, version: state.version + 1 }
  })
}))

export function registerHermesPromptSender(panelId: string, sender: HermesPromptSender): () => void {
  hermesPromptSenders.set(panelId, sender)
  return () => {
    if (hermesPromptSenders.get(panelId) === sender) hermesPromptSenders.delete(panelId)
  }
}

export function sendHermesStreamPrompt(panelId: string, prompt: string): boolean {
  const sender = hermesPromptSenders.get(panelId)
  if (!sender) return false
  void sender(prompt)
  return true
}
