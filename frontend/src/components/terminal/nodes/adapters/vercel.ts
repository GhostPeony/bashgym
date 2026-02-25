import { Triangle, RefreshCw, FileCode, Send, Rocket, Eye } from 'lucide-react'
import type { NodeAdapter, ContextPayload, NodeAction, ConfigField } from '../types'
import { registerAdapter } from '../IntegrationNode'
import { routeToLinkedTerminals } from '../../../../utils/edgeRouting'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface V0File {
  name: string
  content: string
}

/**
 * Format v0-generated files as fenced markdown code blocks.
 */
function formatV0Files(files: V0File[]): string {
  return files
    .map(f => {
      const ext = f.name.split('.').pop() ?? ''
      return `### ${f.name}\n\`\`\`${ext}\n${f.content}\n\`\`\``
    })
    .join('\n\n')
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function createVercelAdapter(
  config: Record<string, unknown>,
  onChange: (key: string, value: unknown) => void
): NodeAdapter {
  const projectName = (config.projectName as string) || ''
  const deployStatus = (config.deployStatus as string) || ''
  const deployUrl = (config.deployUrl as string) || ''
  const deployId = (config.deployId as string) || ''
  const buildLogs = (config.buildLogs as string) || ''
  const v0Files = (config.v0Files as V0File[]) || []
  const v0PreviewUrl = (config.v0PreviewUrl as string) || ''
  const v0ChatId = (config.v0ChatId as string) || ''
  const v0Prompt = (config.v0Prompt as string) || ''
  const panelId = config._panelId as string | undefined

  return {
    type: 'vercel',
    icon: Triangle,
    label: 'Vercel',

    getContext(): ContextPayload {
      let content: string
      if (buildLogs) {
        content = buildLogs
      } else if (v0Files.length > 0) {
        content = formatV0Files(v0Files)
      } else {
        content = projectName
          ? `Project: ${projectName}\nStatus: ${deployStatus || 'unknown'}\nURL: ${deployUrl || 'n/a'}`
          : '(not configured)'
      }

      const summary = projectName
        ? `${projectName}: ${deployStatus || 'no deploy'}`
        : 'Vercel: not configured'

      const tokenEstimate = Math.ceil(content.length / 4)

      return { summary, content, tokenEstimate }
    },

    getActions(): NodeAction[] {
      const actions: NodeAction[] = []

      // 1. Refresh deploy status
      actions.push({
        id: 'refresh',
        label: 'Refresh',
        icon: RefreshCw,
        async handler() {
          const tokenResult = await window.bashgym?.credentials.read('vercel-token')
          if (!tokenResult?.value) return

          const response = await fetch(
            `https://api.vercel.com/v13/deployments?projectId=${encodeURIComponent(projectName)}&limit=1`,
            {
              headers: { Authorization: `Bearer ${tokenResult.value}` }
            }
          )

          const text = await response.text()
          try {
            const data = JSON.parse(text)
            const latest = data?.deployments?.[0]
            if (latest) {
              onChange('deployStatus', latest.state ?? latest.readyState ?? 'unknown')
              onChange('deployUrl', latest.url ? `https://${latest.url}` : '')
              onChange('deployId', latest.uid ?? latest.id ?? '')
            }
          } catch {
            // Non-JSON response -- leave config unchanged
          }
        }
      })

      // 2. Get build logs
      actions.push({
        id: 'get-logs',
        label: 'Get Logs',
        icon: FileCode,
        async handler() {
          if (!deployId) return
          const tokenResult = await window.bashgym?.credentials.read('vercel-token')
          if (!tokenResult?.value) return

          const response = await fetch(
            `https://api.vercel.com/v12/deployments/${encodeURIComponent(deployId)}/events`,
            {
              headers: { Authorization: `Bearer ${tokenResult.value}` }
            }
          )

          const text = await response.text()
          try {
            const events = JSON.parse(text) as Array<{
              type?: string
              text?: string
              payload?: { text?: string }
            }>
            const logLines = events
              .filter(e => e.type === 'stdout' || e.type === 'stderr')
              .slice(-50)
              .map(e => e.text ?? e.payload?.text ?? '')
              .join('\n')

            onChange('buildLogs', logLines)
          } catch {
            // Non-JSON response -- leave config unchanged
          }
        }
      })

      // 3. Send logs to linked terminals
      actions.push({
        id: 'send-logs',
        label: 'Send Logs',
        icon: Send,
        async handler() {
          if (panelId && buildLogs) {
            await routeToLinkedTerminals(panelId, buildLogs, 'md')
          }
        }
      })

      // 4. Generate with v0
      actions.push({
        id: 'generate',
        label: 'Generate',
        icon: Rocket,
        async handler() {
          const apiKeyResult = await window.bashgym?.credentials.read('v0-api-key')
          if (!apiKeyResult?.value || !v0Prompt) return

          const { createClient } = await import('v0-sdk')
          const client = createClient({ apiKey: apiKeyResult.value })

          const chat = await client.chats.create({ message: v0Prompt })

          // chats.create can return a stream or a ChatDetail; we only handle
          // the synchronous ChatDetail shape here.
          if (chat && typeof chat === 'object' && 'id' in chat) {
            const detail = chat as {
              id: string
              latestVersion?: {
                demoUrl?: string
                files?: Array<{ name: string; content: string }>
              }
            }
            const files: V0File[] = (detail.latestVersion?.files ?? []).map(f => ({
              name: f.name,
              content: f.content
            }))

            onChange('v0Files', files)
            onChange('v0PreviewUrl', detail.latestVersion?.demoUrl ?? '')
            onChange('v0ChatId', detail.id)
          }
        }
      })

      // 5. Preview -- push v0 demo URL to linked browser panels
      actions.push({
        id: 'preview',
        label: 'Preview',
        icon: Eye,
        async handler() {
          if (!v0PreviewUrl || !panelId) return

          const { useTerminalStore } = await import('../../../../stores')
          const { canvasEdges, panels, setPanelUrl } = useTerminalStore.getState()

          const connectedEdges = canvasEdges.filter(
            e => e.source === panelId || e.target === panelId
          )

          for (const edge of connectedEdges) {
            const targetPanelId = edge.source === panelId ? edge.target : edge.source
            const targetPanel = panels.find(
              p => p.id === targetPanelId && p.type === 'browser'
            )
            if (targetPanel) {
              setPanelUrl(targetPanel.id, v0PreviewUrl)
            }
          }
        }
      })

      // 6. Send generated code to linked terminals
      actions.push({
        id: 'send-code',
        label: 'Send Code',
        icon: Send,
        async handler() {
          if (panelId && v0Files.length > 0) {
            const formatted = formatV0Files(v0Files)
            await routeToLinkedTerminals(panelId, formatted, 'md')
          }
        }
      })

      return actions
    },

    getConfigFields(): ConfigField[] {
      return [
        {
          key: 'projectName',
          label: 'Project',
          type: 'text',
          placeholder: 'prj_xxx or project name'
        },
        {
          key: 'v0Prompt',
          label: 'v0 Prompt',
          type: 'textarea',
          placeholder: 'Build a settings page with...'
        }
      ]
    }
  }
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerAdapter('vercel', createVercelAdapter)
