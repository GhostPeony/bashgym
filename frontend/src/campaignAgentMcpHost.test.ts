import assert from 'node:assert/strict'
import test from 'node:test'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import { CampaignAgentMcpHost } from '../electron/campaignAgentMcpHost'

const observation = {
  schemaVersion: 'campaign_agent_observation.v1',
  scope: { workspaceId: 'workspace-a', campaignId: 'campaign-1' },
  campaign: {
    status: 'active',
    version: 3,
    manifestRevision: 2,
    activeStudyId: 'study-1',
    activeActionId: null,
    latestEventCursor: 12
  },
  agent: {
    attachmentId: 'attachment-1',
    attachmentVersion: 1,
    agentFamily: 'codex',
    agentPrincipalId: 'codex-agent',
    authorizedCapability: 'campaign_observe'
  }
}

const artifactPage = {
  schemaVersion: 'campaign_agent_artifact_page.v1',
  scope: { workspaceId: 'workspace-a', campaignId: 'campaign-1' },
  items: [
    {
      artifactId: 'artifact-1',
      producerActionId: null,
      sha256: 'a'.repeat(64),
      sizeBytes: 42,
      schemaName: 'campaign_training_log.v1',
      sealed: true,
      valid: true,
      createdAt: '2026-07-17T00:00:00Z'
    }
  ],
  nextCursor: null,
  hasMore: false
}

async function connect(host: CampaignAgentMcpHost) {
  const launch = await host.start()
  const transport = new StreamableHTTPClientTransport(new URL(launch.url), {
    requestInit: { headers: launch.headers }
  })
  const client = new Client({ name: 'campaign-agent-host-test', version: '1.0.0' })
  await client.connect(transport)
  return { client, launch }
}

function hostOptions(
  overrides: Partial<ConstructorParameters<typeof CampaignAgentMcpHost>[0]> = {}
) {
  return {
    terminalId: 'terminal-1',
    generation: 'generation-1',
    scope: { workspaceId: 'workspace-a', campaignId: 'campaign-1' },
    observe: async () => observation,
    artifacts: async () => artifactPage,
    ...overrides
  }
}

test('serves only the two fixed read-only campaign tools over authenticated loopback MCP', async (t) => {
  const artifactCalls: unknown[] = []
  const host = new CampaignAgentMcpHost({
    terminalId: 'terminal-1',
    generation: 'generation-1',
    scope: { workspaceId: 'workspace-a', campaignId: 'campaign-1' },
    observe: async () => observation,
    artifacts: async (args) => {
      artifactCalls.push(args)
      return artifactPage
    }
  })
  t.after(() => host.close())
  const { client, launch } = await connect(host)
  t.after(() => client.close())

  assert.match(launch.url, /^http:\/\/127\.0\.0\.1:\d+\/[A-Za-z0-9_-]{32,}$/)
  assert.deepEqual(Object.keys(launch.headers), ['X-BashGym-MCP-Launch'])
  assert.match(launch.headers['X-BashGym-MCP-Launch'], /^[A-Za-z0-9_-]{43}$/)
  assert.deepEqual(
    (await client.listTools()).tools.map((tool) => tool.name),
    ['campaign_observe', 'campaign_artifacts']
  )

  const observed = await client.callTool({ name: 'campaign_observe', arguments: {} })
  assert.deepEqual(observed.structuredContent, observation)
  const artifacts = await client.callTool({
    name: 'campaign_artifacts',
    arguments: { afterCursor: 'a1.ABCDEFGHIJK', limit: 10 }
  })
  assert.deepEqual(artifacts.structuredContent, artifactPage)
  assert.deepEqual(artifactCalls, [{ afterCursor: 'a1.ABCDEFGHIJK', limit: 10 }])
})

test('rejects unauthenticated requests and a second MCP client for the same PTY generation', async (t) => {
  const host = new CampaignAgentMcpHost(hostOptions())
  t.after(() => host.close())
  const { client, launch } = await connect(host)
  t.after(() => client.close())

  assert.equal((await fetch(launch.url)).status, 401)
  assert.equal(
    (
      await fetch(launch.url.replace(/[^/]+$/, 'wrong-route'), {
        headers: launch.headers
      })
    ).status,
    404
  )

  const second = new Client({ name: 'second-client', version: '1.0.0' })
  const secondTransport = new StreamableHTTPClientTransport(new URL(launch.url), {
    requestInit: { headers: launch.headers }
  })
  await assert.rejects(second.connect(secondTransport))
  await second.close().catch(() => undefined)
})

test('accepts backend-sized identifiers and only canonical UTC artifact timestamps', async (t) => {
  let page = {
    ...artifactPage,
    items: [{ ...artifactPage.items[0], artifactId: 'a'.repeat(160) }]
  }
  const host = new CampaignAgentMcpHost(hostOptions({ artifacts: async () => page }))
  t.after(() => host.close())
  const { client } = await connect(host)
  t.after(() => client.close())

  const valid = await client.callTool({ name: 'campaign_artifacts', arguments: {} })
  assert.equal(valid.isError, undefined)
  assert.equal((valid.structuredContent as typeof artifactPage).items[0].artifactId.length, 160)

  page = {
    ...artifactPage,
    items: [{ ...artifactPage.items[0], createdAt: '2026-07-16T17:00:00-07:00' }]
  }
  const offsetTimestamp = await client.callTool({ name: 'campaign_artifacts', arguments: {} })
  assert.equal(offsetTimestamp.isError, true)

  page = {
    ...artifactPage,
    items: [{ ...artifactPage.items[0], artifactId: 'a'.repeat(161) }]
  }
  const oversizedIdentifier = await client.callTool({ name: 'campaign_artifacts', arguments: {} })
  assert.equal(oversizedIdentifier.isError, true)
})

test('rejects unbounded arguments and private backend response fields without leaking them', async (t) => {
  const canary = 'PRIVATE-ARTIFACT-URI-CANARY'
  let artifactCalls = 0
  const host = new CampaignAgentMcpHost(
    hostOptions({
      artifacts: async () => {
        artifactCalls += 1
        return {
          ...artifactPage,
          items: [{ ...artifactPage.items[0], uri: `file://${canary}` }]
        }
      }
    })
  )
  t.after(() => host.close())
  const { client } = await connect(host)
  t.after(() => client.close())

  const invalidArgs = await client.callTool({
    name: 'campaign_artifacts',
    arguments: { limit: 51, target: 'http://not-allowlisted.invalid' }
  })
  assert.equal(invalidArgs.isError, true)
  assert.equal(artifactCalls, 0)

  const invalidProjection = await client.callTool({
    name: 'campaign_artifacts',
    arguments: { limit: 1 }
  })
  assert.equal(invalidProjection.isError, true)
  assert.doesNotMatch(JSON.stringify(invalidProjection), new RegExp(canary))
  assert.equal(artifactCalls, 1)

  const genericTool = await client.callTool({
    name: 'fetch',
    arguments: { url: 'http://not-allowlisted.invalid', method: 'POST', body: {} }
  })
  assert.equal(genericTool.isError, true)
})

test('lock fails closed with an authority-unavailable tool error and close is idempotent', async (t) => {
  let observations = 0
  const host = new CampaignAgentMcpHost(
    hostOptions({
      observe: async () => {
        observations += 1
        return observation
      }
    })
  )
  t.after(() => host.close())
  const { client } = await connect(host)
  t.after(() => client.close())

  host.lock()
  const result = await client.callTool({ name: 'campaign_observe', arguments: {} })
  assert.equal(result.isError, true)
  assert.match(JSON.stringify(result), /Campaign authority is unavailable/)
  assert.equal(observations, 0)
  await host.close()
  await host.close()
})

test('allows only one active host for a terminal PTY generation', async (t) => {
  const first = new CampaignAgentMcpHost(hostOptions())
  const duplicate = new CampaignAgentMcpHost(hostOptions())
  t.after(() => Promise.all([first.close(), duplicate.close()]))
  await first.start()
  await assert.rejects(duplicate.start(), /PTY generation already has an active MCP host/)

  await first.close()
  const replacement = new CampaignAgentMcpHost(hostOptions())
  t.after(() => replacement.close())
  await replacement.start()
})
