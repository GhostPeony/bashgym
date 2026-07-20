import assert from 'node:assert/strict'
import test from 'node:test'

import {
  adoptCampaignAgentTerminal,
  CAMPAIGN_AGENT_ADOPTION_CLEANED_ERROR,
  CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR,
  AUTO_RESEARCH_CODEX_TERMINAL_TITLE,
  finalizeSuccessfulCampaignAgentLaunch,
  preserveDurableCampaignAgentError
} from './campaignAgentTerminalAdoption'

test('binds the main-owned Codex PTY adoption to the launch workspace', () => {
  const calls: unknown[] = []
  const workspaceStore = {
    adoptTerminalIntoWorkspace: (workspaceId: string, terminal: unknown) => {
      calls.push({ workspaceId, terminal })
      return true
    }
  }

  const adopted = adoptCampaignAgentTerminal(workspaceStore, 'workspace-a', {
    terminalId: 'terminal-codex-1',
    cwd: 'C:\\workspace'
  })

  assert.equal(adopted, true)
  assert.deepEqual(calls, [
    {
      workspaceId: 'workspace-a',
      terminal: {
        terminalId: 'terminal-codex-1',
        title: AUTO_RESEARCH_CODEX_TERMINAL_TITLE,
        cwd: 'C:\\workspace'
      }
    }
  ])
  assert.equal('launchCommand' in (calls[0] as { terminal: object }).terminal, false)
})

test('kills an invisible launched terminal and skips selection and reconciliation when adoption fails', async () => {
  for (const adopt of [
    () => false,
    () => {
      throw new Error('workspace persistence failed')
    }
  ]) {
    const killed: string[] = []
    let selected = 0
    let reconciled = 0

    const outcome = await finalizeSuccessfulCampaignAgentLaunch('terminal-codex-1', {
      adopt,
      kill: async (terminalId) => {
        killed.push(terminalId)
        return true
      },
      selectTerminal: () => {
        selected += 1
      },
      reconcile: async () => {
        reconciled += 1
      }
    })

    assert.deepEqual(killed, ['terminal-codex-1'])
    assert.equal(selected, 0)
    assert.equal(reconciled, 0)
    assert.deepEqual(outcome, {
      status: 'adoption_failed_cleaned',
      durableError: CAMPAIGN_AGENT_ADOPTION_CLEANED_ERROR
    })
    assert.equal(
      preserveDurableCampaignAgentError(outcome.durableError, null),
      CAMPAIGN_AGENT_ADOPTION_CLEANED_ERROR
    )
  }
})

test('reports a distinct durable cleanup error when terminal kill returns false or throws', async () => {
  for (const kill of [
    async () => false,
    async () => {
      throw new Error('kill failed')
    }
  ]) {
    let reconciled = 0
    const outcome = await finalizeSuccessfulCampaignAgentLaunch('terminal-codex-2', {
      adopt: () => false,
      kill,
      selectTerminal: () => {
        throw new Error('invisible terminal must not be selected')
      },
      reconcile: async () => {
        reconciled += 1
      }
    })

    assert.equal(reconciled, 0)
    assert.deepEqual(outcome, {
      status: 'cleanup_failed',
      durableError: CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR
    })
    assert.equal(
      preserveDurableCampaignAgentError(outcome.durableError, null),
      CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR
    )
    assert.equal(
      preserveDurableCampaignAgentError(outcome.durableError, 'poll succeeded'),
      CAMPAIGN_AGENT_CLEANUP_FAILED_ERROR
    )
  }
})

test('selects and reconciles an adopted terminal without calling cleanup', async () => {
  const events: string[] = []
  const outcome = await finalizeSuccessfulCampaignAgentLaunch('terminal-codex-3', {
    adopt: () => true,
    kill: async () => {
      events.push('kill')
      return true
    },
    selectTerminal: () => {
      events.push('select')
    },
    reconcile: async () => {
      events.push('reconcile')
    }
  })

  assert.deepEqual(events, ['select', 'reconcile'])
  assert.deepEqual(outcome, { status: 'adopted' })
})
