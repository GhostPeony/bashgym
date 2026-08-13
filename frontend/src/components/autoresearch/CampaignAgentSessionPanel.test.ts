import assert from 'node:assert/strict'
import test from 'node:test'
import { Children, createElement, isValidElement, type ReactElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  CampaignAgentSessionPanel,
  type CampaignAgentSessionPanelProps
} from './CampaignAgentSessionPanel'
import { parseCampaignAgentPublicView } from './campaignAgentModel'

function publicView(attached = false) {
  return parseCampaignAgentPublicView({
    schema_version: 'campaign_agent_public_view.v1',
    observed_at: '2026-07-16T20:01:00Z',
    scope: { workspace_id: 'workspace-a', campaign_id: 'campaign-a' },
    attachment: attached
      ? {
          schema_version: 'campaign_agent_public_attachment.v1',
          attachment_id: 'attachment-1',
          attachment_version: 3,
          status: 'attached',
          requested_capabilities: ['campaign_observe', 'artifact_read'],
          granted_capabilities: ['campaign_observe'],
          receipt_window: { from_version: 3, through_version: 3, has_earlier: true },
          provenance: {
            agent_family: 'codex',
            agent_origin: 'desktop-safe-origin',
            agent_origin_status: 'verified',
            session_id: 'desktop-safe-session',
            agent_principal_id: 'desktop-safe-principal',
            attached_at: '2026-07-16T20:00:30Z',
            attached_by: 'human-operator',
            grant_receipt_id: 'grant-receipt-7',
            grant_receipt_digest: `sha256:${'a'.repeat(64)}`,
            credential_issued_at: '2026-07-16T20:00:30Z',
            credential_expires_at: '2026-07-16T21:00:30Z',
            credential_status: 'active',
            credential_revocation_revision: 3,
            credential_status_source: 'campaign_authority',
            liveness: 'live',
            resume_cursor: null,
            revoked_at: null,
            revoked_by: null
          },
          receipts: [
            {
              receipt_id: 'attach-receipt-1',
              kind: 'attach',
              actor_id: 'human-operator',
              occurred_at: '2026-07-16T20:00:30Z',
              idempotency_key: 'attach-campaign-a-01',
              attachment_version: 3,
              receipt_digest: `sha256:${'b'.repeat(64)}`
            }
          ]
        }
      : null,
    audit_events: []
  })
}

function props(
  overrides: Partial<CampaignAgentSessionPanelProps> = {}
): CampaignAgentSessionPanelProps {
  return {
    scope: { workspaceId: 'workspace-a', campaignId: 'campaign-a' },
    freshness: 'live',
    hostState: 'ready',
    hostError: null,
    eligibleSessions: [
      { terminalId: 'terminal-codex-1', family: 'codex', state: 'eligible' },
      { terminalId: 'terminal-hermes-2', family: 'hermes', state: 'registered' }
    ],
    selectedTerminalId: 'terminal-codex-1',
    requestedCapabilities: ['campaign_observe', 'artifact_read'],
    grantedCapabilities: ['campaign_observe'],
    publicView: publicView(false),
    pendingAction: null,
    onSelectTerminal: () => {},
    onRequestedCapabilityChange: () => {},
    onGrantedCapabilityChange: () => {},
    onRefresh: () => {},
    onLaunchCodex: () => {},
    onRegister: () => {},
    onApproveAttachment: () => {},
    onActivate: () => {},
    onRevoke: () => {},
    ...overrides
  }
}

interface InteractiveProps {
  'aria-label'?: string
  children?: ReactNode
  disabled?: boolean
  onChange?: (event: { target: { value?: string; checked?: boolean } }) => void
  onClick?: () => void
}

function findByLabel(node: ReactNode, label: string): ReactElement {
  if (isValidElement(node)) {
    const nodeProps = node.props as InteractiveProps
    if (nodeProps['aria-label'] === label) return node
    for (const child of Children.toArray(nodeProps.children)) {
      try {
        return findByLabel(child, label)
      } catch {
        /* keep walking */
      }
    }
  }
  throw new Error(`Element not found: ${label}`)
}

function interactive(node: ReactElement): InteractiveProps {
  return node.props as InteractiveProps
}

test('renders only safe live-session descriptors and never asks for a trust identity', () => {
  const html = renderToStaticMarkup(createElement(CampaignAgentSessionPanel, props()))
  assert.match(html, /Campaign agent session/)
  assert.match(html, /terminal-codex-1/)
  assert.match(html, /terminal-hermes-2/)
  assert.match(html, /Codex/)
  assert.match(html, /Hermes/)
  assert.doesNotMatch(html, /Claude/)
  assert.doesNotMatch(html, /Agent origin|Agent principal|Agent session ID|Brokered origin/i)
  assert.doesNotMatch(html, /desktop-safe-origin|desktop-safe-principal|desktop-safe-session/)
  assert.doesNotMatch(html, /bgag\.|credential token|raw token/i)
})

test('keeps a compact read-only shell and guided steps visible for offline, error, loading, and empty host states', () => {
  for (const [hostState, hostError] of [
    ['loading', null],
    ['offline', 'Desktop host unavailable.'],
    ['error', 'Host request failed.'],
    ['ready', null]
  ] as const) {
    const eligibleSessions = hostState === 'ready' ? [] : props().eligibleSessions
    const tree = CampaignAgentSessionPanel(
      props({ hostState, hostError, eligibleSessions, selectedTerminalId: '' })
    )
    const html = renderToStaticMarkup(tree)
    assert.match(html, /Campaign agent session/)
    assert.match(html, /Campaign capabilities/)
    assert.match(
      html,
      /No eligible Codex terminal|Desktop host unavailable|Host request failed|Discovering the desktop campaign-agent host/
    )
    assert.match(html, /Launch Codex/)
    assert.match(html, /Register/)
    assert.match(html, /Approve scope/)
    assert.match(html, /Activate/)
    for (const label of [
      'Register selected session',
      'Approve campaign attachment',
      'Activate campaign agent'
    ]) {
      assert.equal(interactive(findByLabel(tree, label)).disabled, true, `${hostState}: ${label}`)
    }
    assert.equal(
      interactive(findByLabel(tree, 'Launch Codex campaign agent')).disabled,
      hostState !== 'ready'
    )
  }
})

test('launches only Codex through the optional high-level host callback', () => {
  let launches = 0
  const tree = CampaignAgentSessionPanel(
    props({
      eligibleSessions: [],
      selectedTerminalId: '',
      onLaunchCodex: () => {
        launches += 1
      }
    })
  )
  const launch = interactive(findByLabel(tree, 'Launch Codex campaign agent'))
  assert.equal(launch.disabled, false)
  launch.onClick?.()
  assert.equal(launches, 1)

  const unavailable = CampaignAgentSessionPanel(props({ onLaunchCodex: undefined }))
  assert.equal(interactive(findByLabel(unavailable, 'Launch Codex campaign agent')).disabled, true)
  assert.match(renderToStaticMarkup(unavailable), /Direct Hermes launch is not available yet/)
})

test('exposes an explicit register then approve then activate progression', () => {
  const eligible = CampaignAgentSessionPanel(props())
  assert.equal(interactive(findByLabel(eligible, 'Register selected session')).disabled, false)
  assert.equal(interactive(findByLabel(eligible, 'Approve campaign attachment')).disabled, true)
  assert.equal(interactive(findByLabel(eligible, 'Activate campaign agent')).disabled, true)

  const registered = CampaignAgentSessionPanel(
    props({
      eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'registered' }]
    })
  )
  assert.equal(interactive(findByLabel(registered, 'Register selected session')).disabled, true)
  assert.equal(interactive(findByLabel(registered, 'Approve campaign attachment')).disabled, false)
  assert.equal(interactive(findByLabel(registered, 'Activate campaign agent')).disabled, true)

  const attached = CampaignAgentSessionPanel(
    props({
      eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'authorized' }],
      publicView: publicView(true)
    })
  )
  assert.equal(interactive(findByLabel(attached, 'Approve campaign attachment')).disabled, true)
  assert.equal(interactive(findByLabel(attached, 'Activate campaign agent')).disabled, false)

  const retry = CampaignAgentSessionPanel(
    props({
      eligibleSessions: [
        { terminalId: 'terminal-codex-1', family: 'codex', state: 'credential_ready' }
      ],
      publicView: publicView(true)
    })
  )
  const active = CampaignAgentSessionPanel(
    props({
      eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'active' }],
      publicView: publicView(true)
    })
  )
  assert.equal(interactive(findByLabel(retry, 'Activate campaign agent')).disabled, false)
  assert.equal(interactive(findByLabel(active, 'Activate campaign agent')).disabled, true)
})

test('fails closed for stale campaign state, failed sessions, unsupported capabilities, and mismatched scope', () => {
  const cases: Partial<CampaignAgentSessionPanelProps>[] = [
    { freshness: 'stale' },
    { eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'failed' }] },
    { grantedCapabilities: ['training_launch'], requestedCapabilities: ['campaign_observe'] },
    {
      publicView: parseCampaignAgentPublicView({
        schema_version: 'campaign_agent_public_view.v1',
        observed_at: '2026-07-16T20:01:00Z',
        scope: { workspace_id: 'workspace-a', campaign_id: 'campaign-other' },
        attachment: null,
        audit_events: []
      })
    }
  ]
  for (const item of cases) {
    const tree = CampaignAgentSessionPanel(props(item))
    for (const label of [
      'Register selected session',
      'Approve campaign attachment',
      'Activate campaign agent'
    ]) {
      assert.equal(interactive(findByLabel(tree, label)).disabled, true, label)
    }
  }
})

test('changes only terminal selection and supported read-only capability declarations', () => {
  let selected = ''
  const requested: Array<[string, boolean]> = []
  const granted: Array<[string, boolean]> = []
  const tree = CampaignAgentSessionPanel(
    props({
      onSelectTerminal: (terminalId) => {
        selected = terminalId
      },
      onRequestedCapabilityChange: (capability, checked) => {
        requested.push([capability, checked])
      },
      onGrantedCapabilityChange: (capability, checked) => {
        granted.push([capability, checked])
      }
    })
  )
  interactive(findByLabel(tree, 'Eligible live terminal')).onChange?.({
    target: { value: 'terminal-hermes-2' }
  })
  interactive(findByLabel(tree, 'Request artifact read')).onChange?.({ target: { checked: true } })
  interactive(findByLabel(tree, 'Grant campaign observe')).onChange?.({
    target: { checked: false }
  })
  assert.equal(selected, 'terminal-hermes-2')
  assert.deepEqual(requested, [['artifact_read', true]])
  assert.deepEqual(granted, [['campaign_observe', false]])
})

test('makes only observation and artifact reading selectable while showing mutation capabilities as unavailable', () => {
  const tree = CampaignAgentSessionPanel(props())
  const html = renderToStaticMarkup(tree)
  assert.ok(findByLabel(tree, 'Request campaign observe'))
  assert.ok(findByLabel(tree, 'Grant campaign observe'))
  assert.ok(findByLabel(tree, 'Request artifact read'))
  assert.ok(findByLabel(tree, 'Grant artifact read'))
  assert.throws(() => findByLabel(tree, 'Request training launch'))
  assert.throws(() => findByLabel(tree, 'Request training pause self'))
  assert.throws(() => findByLabel(tree, 'Request artifact propose'))
  assert.match(html, /Training launch/)
  assert.match(html, /Pause training/)
  assert.match(html, /Propose artifacts/)
  assert.match(html, /Unavailable in this direct agent path/)
})

test('revoke stays visible but requires a live authoritative attachment', () => {
  const unattached = CampaignAgentSessionPanel(props())
  const attached = CampaignAgentSessionPanel(
    props({
      publicView: publicView(true),
      eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'authorized' }]
    })
  )
  const orphaned = CampaignAgentSessionPanel(
    props({
      publicView: publicView(true),
      eligibleSessions: [],
      selectedTerminalId: ''
    })
  )
  const stale = CampaignAgentSessionPanel(
    props({
      publicView: publicView(true),
      freshness: 'stale',
      eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'authorized' }]
    })
  )
  assert.equal(interactive(findByLabel(unattached, 'Revoke campaign attachment')).disabled, true)
  assert.equal(interactive(findByLabel(attached, 'Revoke campaign attachment')).disabled, false)
  assert.equal(interactive(findByLabel(orphaned, 'Revoke campaign attachment')).disabled, true)
  assert.equal(interactive(findByLabel(stale, 'Revoke campaign attachment')).disabled, true)
})

test('shows the credential lifecycle without exposing credential material', () => {
  const ready = renderToStaticMarkup(
    createElement(
      CampaignAgentSessionPanel,
      props({
        eligibleSessions: [
          { terminalId: 'terminal-codex-1', family: 'codex', state: 'credential_ready' }
        ],
        publicView: publicView(true)
      })
    )
  )
  const consumed = renderToStaticMarkup(
    createElement(
      CampaignAgentSessionPanel,
      props({
        eligibleSessions: [
          { terminalId: 'terminal-codex-1', family: 'codex', state: 'credential_consumed' }
        ],
        publicView: publicView(true)
      })
    )
  )
  const active = renderToStaticMarkup(
    createElement(
      CampaignAgentSessionPanel,
      props({
        eligibleSessions: [{ terminalId: 'terminal-codex-1', family: 'codex', state: 'active' }],
        publicView: publicView(true)
      })
    )
  )
  assert.match(ready, /Activate again to retry/)
  assert.match(active, /Read-only campaign tools are active/)
  assert.match(consumed, /Credential authority was consumed by a main-owned process/)
  assert.doesNotMatch(`${ready}${active}${consumed}`, /bgag\.|raw token|bearer\s/i)
})
