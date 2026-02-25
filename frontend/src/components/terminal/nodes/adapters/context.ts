import { FileText, Globe, Code, StickyNote, RefreshCw } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { NodeAdapter, ContextPayload, NodeAction, ConfigField } from '../types'
import { registerAdapter } from '../IntegrationNode'

// ---------------------------------------------------------------------------
// Mode icon mapping
// ---------------------------------------------------------------------------

type ContextMode = 'text' | 'file' | 'url' | 'snippet'

const modeIcons: Record<ContextMode, LucideIcon> = {
  text: StickyNote,
  file: FileText,
  url: Globe,
  snippet: Code
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function createContextAdapter(
  config: Record<string, unknown>,
  onChange: (key: string, value: unknown) => void
): NodeAdapter {
  const mode = (config.mode as ContextMode) || 'text'
  const title = config.title as string | undefined
  const content = (config.content as string) || ''
  const filePath = config.filePath as string | undefined
  const url = config.url as string | undefined

  return {
    type: 'context',
    icon: modeIcons[mode] ?? StickyNote,
    label: 'Context',

    getContext(): ContextPayload {
      const tokenEstimate = Math.ceil(content.length / 4)

      const modeLabels: Record<ContextMode, string> = {
        text: 'Text',
        file: 'File',
        url: 'URL',
        snippet: 'Snippet'
      }

      const summary = title || modeLabels[mode] || 'Context'

      return { summary, content, tokenEstimate }
    },

    getActions(): NodeAction[] {
      const actions: NodeAction[] = []

      if (mode === 'file' && filePath) {
        actions.push({
          id: 'reload-file',
          label: 'Reload',
          icon: RefreshCw,
          async handler() {
            const result = await window.bashgym?.files.readFile(filePath)
            if (result?.success && result.content != null) {
              onChange('content', result.content)
            }
          }
        })
      }

      if (mode === 'url' && url) {
        actions.push({
          id: 'fetch-url',
          label: 'Fetch',
          icon: RefreshCw,
          async handler() {
            try {
              const response = await fetch(url)
              const text = await response.text()
              onChange('content', text)
              onChange('lastFetched', Date.now())
            } catch {
              // Fetch failed - leave content unchanged
            }
          }
        })
      }

      return actions
    },

    getConfigFields(): ConfigField[] {
      const fields: ConfigField[] = [
        {
          key: 'mode',
          label: 'Mode',
          type: 'select',
          options: [
            { label: 'Text', value: 'text' },
            { label: 'File', value: 'file' },
            { label: 'URL', value: 'url' },
            { label: 'Snippet', value: 'snippet' }
          ]
        },
        {
          key: 'title',
          label: 'Title',
          type: 'text',
          placeholder: 'Optional title'
        }
      ]

      if (mode === 'file') {
        fields.push({
          key: 'filePath',
          label: 'File Path',
          type: 'text',
          placeholder: '/path/to/file'
        })
      }

      if (mode === 'url') {
        fields.push({
          key: 'url',
          label: 'URL',
          type: 'text',
          placeholder: 'https://...'
        })
      }

      fields.push({
        key: 'content',
        label: mode === 'snippet' ? 'Snippet' : 'Content',
        type: 'textarea',
        placeholder: 'Paste or type content...'
      })

      fields.push({
        key: 'autoSend',
        label: 'Auto-Send',
        type: 'toggle'
      })

      return fields
    }
  }
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerAdapter('context', createContextAdapter)
