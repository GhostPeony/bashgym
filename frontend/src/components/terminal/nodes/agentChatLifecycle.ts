export interface AgentChatSurfaceActions {
  dismiss: () => void
  stop: () => void
}

interface AgentChatSurfaceActionDependencies {
  abort: () => void
  hide: () => void
}

export function createAgentChatSurfaceActions({
  abort,
  hide
}: AgentChatSurfaceActionDependencies): AgentChatSurfaceActions {
  return {
    dismiss: hide,
    stop: abort
  }
}
