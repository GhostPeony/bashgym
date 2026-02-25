import { Database, RefreshCw, Send, Play } from 'lucide-react'
import type { NodeAdapter, ContextPayload, NodeAction, ConfigField } from '../types'
import { registerAdapter } from '../IntegrationNode'
import { routeToLinkedTerminals } from '../../../../utils/edgeRouting'

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function createNeonAdapter(
  config: Record<string, unknown>,
  onChange: (key: string, value: unknown) => void
): NodeAdapter {
  const connected = config.connected as boolean | undefined
  const schema = (config.schema as string) || ''
  const tableCount = (config.tableCount as number) || 0
  const projectName = (config.projectName as string) || ''
  const branchName = (config.branchName as string) || 'main'
  const query = (config.query as string) || ''
  const queryResult = (config.queryResult as string) || ''
  const panelId = config._panelId as string | undefined

  return {
    type: 'neon',
    icon: Database,
    label: 'Neon',

    getContext(): ContextPayload {
      const content = connected ? schema : '(not connected)'
      const summary = connected
        ? `${projectName} (${branchName}) ${tableCount} tables`
        : '(not connected)'
      const tokenEstimate = Math.ceil(content.length / 4)

      return { summary, content, tokenEstimate }
    },

    getActions(): NodeAction[] {
      const actions: NodeAction[] = []

      // 1. Connect / Refresh
      actions.push({
        id: 'connect',
        label: connected ? 'Refresh' : 'Connect',
        icon: RefreshCw,
        async handler() {
          try {
            const apiKeyResult = await window.bashgym?.credentials.read('neon-api-key')
            const connStringResult = await window.bashgym?.credentials.read('neon-connection-string')

            if (!apiKeyResult?.value || !connStringResult?.value) {
              onChange('connected', false)
              return
            }

            const connectionString = connStringResult.value

            const { neon } = await import('@neondatabase/serverless')
            const sql = neon(connectionString)

            // Query tables
            const tablesResult = await sql`
              SELECT table_name
              FROM information_schema.tables
              WHERE table_schema = 'public'
              ORDER BY table_name
            `

            // Query columns
            const columnsResult = await sql`
              SELECT table_name, column_name, data_type
              FROM information_schema.columns
              WHERE table_schema = 'public'
              ORDER BY table_name, ordinal_position
            `

            // Build compact schema: tableName: col1(type) col2(type) ...
            const tableMap = new Map<string, string[]>()
            for (const row of tablesResult) {
              tableMap.set(row.table_name as string, [])
            }
            for (const row of columnsResult) {
              const tableName = row.table_name as string
              const colName = row.column_name as string
              const dataType = row.data_type as string
              const cols = tableMap.get(tableName)
              if (cols) {
                cols.push(`${colName}(${dataType})`)
              }
            }

            const lines: string[] = [`## DB: ${projectName} (neon, branch: ${branchName})`]
            for (const [table, cols] of tableMap) {
              lines.push(`${table}: ${cols.join(' ')}`)
            }
            const schemaText = lines.join('\n')

            onChange('connected', true)
            onChange('schema', schemaText)
            onChange('tableCount', tableMap.size)
          } catch (error) {
            onChange('connected', false)
            onChange('errorMessage', String(error))
          }
        }
      })

      // 2. Send Schema
      actions.push({
        id: 'send-schema',
        label: 'Send Schema',
        icon: Send,
        async handler() {
          if (panelId && schema) {
            await routeToLinkedTerminals(panelId, schema, 'md')
          }
        }
      })

      // 3. Run Query
      actions.push({
        id: 'run-query',
        label: 'Run Query',
        icon: Play,
        async handler() {
          try {
            const connStringResult = await window.bashgym?.credentials.read('neon-connection-string')
            if (!connStringResult?.value || !query) return

            const { neon } = await import('@neondatabase/serverless')
            const sql = neon(connStringResult.value, { fullResults: true })

            const result = await sql(query)

            // Format as markdown table
            const fields = result.fields ?? []
            const rows = result.rows ?? []

            if (fields.length === 0) {
              onChange('queryResult', '(no results)')
              return
            }

            const headers = fields.map((f: { name: string }) => f.name)
            const separator = headers.map(() => '---')
            const dataRows = rows.map((row: Record<string, unknown>) =>
              headers.map(h => String(row[h] ?? ''))
            )

            const mdLines = [
              `| ${headers.join(' | ')} |`,
              `| ${separator.join(' | ')} |`,
              ...dataRows.map((r: string[]) => `| ${r.join(' | ')} |`)
            ]

            onChange('queryResult', mdLines.join('\n'))
          } catch (error) {
            onChange('queryResult', `Error: ${String(error)}`)
          }
        }
      })

      // 4. Send Results
      actions.push({
        id: 'send-results',
        label: 'Send Results',
        icon: Send,
        async handler() {
          if (panelId && queryResult) {
            await routeToLinkedTerminals(panelId, queryResult, 'md')
          }
        }
      })

      return actions
    },

    getConfigFields(): ConfigField[] {
      return [
        {
          key: 'projectName',
          label: 'Project Name',
          type: 'text'
        },
        {
          key: 'branchName',
          label: 'Branch Name',
          type: 'text',
          placeholder: 'main'
        },
        {
          key: 'query',
          label: 'Query',
          type: 'textarea',
          placeholder: 'SELECT * FROM users LIMIT 5'
        }
      ]
    }
  }
}

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

registerAdapter('neon', createNeonAdapter)
