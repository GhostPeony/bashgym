export const AUTO_RESEARCH_CODEX_TERMINAL_TITLE = 'AutoResearch Codex'
export const CAMPAIGN_AGENT_ADOPTION_CLEANED_ERROR =
  'Codex launched, but its terminal could not be added to the launch workspace. The launched session was closed.'
export const CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR =
  'Codex launched, but its terminal could not be added to the launch workspace, and cleanup failed. The invisible session may still be running.'

interface CampaignAgentWorkspaceStore {
  adoptTerminalIntoWorkspace: (
    workspaceId: string,
    terminal: { terminalId: string; title: string; cwd: string },
  ) => boolean
}

interface MainOwnedCampaignAgentTerminal {
  terminalId: string
  cwd: string
}

interface CampaignAgentLaunchLifecycle {
  adopt: () => boolean
  kill: (terminalId: string) => Promise<boolean>
  selectTerminal: () => void
  reconcile: () => Promise<void>
}

export type CampaignAgentLaunchLifecycleOutcome =
  | { status: 'adopted' }
  | { status: 'adoption_failed_cleaned'; durableError: string }
  | { status: 'cleanup_failed'; durableError: string }

/** Makes a main-owned campaign-agent PTY visible without starting another command. */
export function adoptCampaignAgentTerminal(
  workspaceStore: CampaignAgentWorkspaceStore,
  workspaceId: string,
  terminal: MainOwnedCampaignAgentTerminal,
): boolean {
  return workspaceStore.adoptTerminalIntoWorkspace(workspaceId, {
    terminalId: terminal.terminalId,
    title: AUTO_RESEARCH_CODEX_TERMINAL_TITLE,
    cwd: terminal.cwd,
  })
}

export async function finalizeSuccessfulCampaignAgentLaunch(
  terminalId: string,
  lifecycle: CampaignAgentLaunchLifecycle,
): Promise<CampaignAgentLaunchLifecycleOutcome> {
  let adopted = false
  try {
    adopted = lifecycle.adopt()
  } catch {
    // A persistence or workspace-store failure is still a failed adoption. The
    // main-owned child must not outlive its visible Workspace ownership.
  }
  if (adopted) {
    lifecycle.selectTerminal()
    await lifecycle.reconcile()
    return { status: 'adopted' }
  }

  try {
    if (await lifecycle.kill(terminalId)) {
      return {
        status: 'adoption_failed_cleaned',
        durableError: CAMPAIGN_AGENT_ADOPTION_CLEANED_ERROR,
      }
    }
  } catch {
    // Cleanup failures use the same durable operator-facing recovery message.
  }

  return {
    status: 'cleanup_failed',
    durableError: CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR,
  }
}

export function preserveDurableCampaignAgentError(
  durableError: string | null,
  reconciledError: string | null,
): string | null {
  return durableError ?? reconciledError
}
