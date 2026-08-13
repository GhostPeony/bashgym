import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  Filter,
  FilePlus2,
  Loader2,
  Network,
  RefreshCw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Terminal,
  Wrench
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { clsx } from 'clsx'
import {
  API_BASE,
  toolkitApi,
  type ToolkitEndpointCapability,
  type ToolkitInventoryResponse,
  type ToolkitSkill,
  type ToolkitTool
} from '../../../services/api'
import { useTerminalStore, type TerminalSession } from '../../../stores'
import { DataNodeShell } from './DataNodeShell'
import { findDynamicNodePosition } from '../canvasPlacement'
import { hueFor } from './dataPanels'
import { isCatalogSkillActive, skillIdFor } from './skillLabModel'
import { ConfigRow, ConfigRows, ConfigSection, NodeConfigModal } from './NodeConfigModal'
import type { DataNodeData } from './types'

export type ToolKitNodeType = Node<DataNodeData, 'toolkit'>

type ViewMode = 'skills' | 'tools' | 'agents'
type ScopeMode = 'linked' | 'all'
type HarnessKind = 'claude' | 'codex' | 'hermes' | 'peony' | 'terminal'

interface LinkedInstance {
  panelId: string
  title: string
  kind: HarnessKind
  label: string
  model?: string
}

const SKILL_SOURCES_BY_HARNESS: Record<HarnessKind, string[]> = {
  claude: ['agents', 'claude', 'workspace', 'peony'],
  codex: ['codex', 'codex-system', 'workspace', 'peony'],
  hermes: ['hermes', 'workspace', 'peony'],
  peony: ['workspace', 'peony'],
  terminal: ['workspace', 'peony']
}

function formatDate(value?: string): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
}

function CountTile({
  icon: Icon,
  label,
  value
}: {
  icon: LucideIcon
  label: string
  value: number
}) {
  return (
    <div className="border-brutal border-border-subtle rounded-brutal px-2 py-1.5 bg-background-card min-w-0">
      <div className="flex items-center gap-1 text-[9px] font-mono text-text-muted uppercase">
        <Icon className="w-3 h-3" />
        <span className="truncate">{label}</span>
      </div>
      <div className="text-sm font-mono font-semibold text-text-primary">{value}</div>
    </div>
  )
}

function matchQuery(value: string, query: string): boolean {
  return value.toLowerCase().includes(query.trim().toLowerCase())
}

function skillText(skill: ToolkitSkill): string {
  return `${skill.name} ${skill.description} ${skill.source} ${skill.path || ''}`
}

function toolText(tool: ToolkitTool): string {
  return `${tool.name} ${tool.description} ${tool.source}`
}

function endpointText(endpoint: ToolkitEndpointCapability): string {
  return `${endpoint.label} ${endpoint.endpoint_id} ${endpoint.kind} ${endpoint.skill_names.join(' ')} ${endpoint.toolset_names.join(' ')}`
}

function endpointSkillItems(endpoint: ToolkitEndpointCapability): ToolkitSkill[] {
  return endpoint.skill_names.map((name) => ({
    name,
    description: `${endpoint.label} endpoint skill`,
    source: endpoint.kind || 'agent',
    path: `agent:${endpoint.endpoint_id}`,
    remote_endpoint_id: endpoint.endpoint_id,
    resource_counts: { scripts: 0, references: 0, assets: 0 },
    tool_count: 0
  }))
}

function inferHarness(session?: TerminalSession): HarnessKind {
  if (session?.agentKind === 'claude') return 'claude'
  if (session?.agentKind === 'codex') return 'codex'

  const text = [
    session?.title,
    session?.model,
    session?.taskSummary,
    ...(session?.lastOutput || [])
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase()

  if (text.includes('claude')) return 'claude'
  if (text.includes('codex') || text.includes('gpt-')) return 'codex'
  if (text.includes('hermes')) return 'hermes'
  return 'terminal'
}

function labelForHarness(kind: HarnessKind): string {
  if (kind === 'claude') return 'Claude'
  if (kind === 'codex') return 'Codex'
  if (kind === 'hermes') return 'Hermes'
  if (kind === 'peony') return 'Peony'
  return 'Terminal'
}

function skillMatchesLinkedScope(skill: ToolkitSkill, linkedInstances: LinkedInstance[]): boolean {
  if (!linkedInstances.length) return ['workspace', 'peony'].includes(skill.source)
  if (skill.source.startsWith('env:')) return true

  const allowed = new Set<string>()
  for (const instance of linkedInstances) {
    for (const source of SKILL_SOURCES_BY_HARNESS[instance.kind]) allowed.add(source)
  }
  return [skill.source, ...(skill.available_sources || [])].some((source) => allowed.has(source))
}

function toolMatchesLinkedScope(tool: ToolkitTool, linkedInstances: LinkedInstance[]): boolean {
  if (!linkedInstances.length) return tool.source.startsWith('peony')
  return linkedInstances.some((instance) => {
    if (instance.kind === 'hermes') return tool.source.startsWith('peony')
    if (instance.kind === 'claude' || instance.kind === 'codex')
      return tool.source.startsWith('peony')
    return tool.source.startsWith('peony')
  })
}

function buildToolkitContext(
  inventory: ToolkitInventoryResponse | null,
  skillIdea: string,
  targetRoot: string,
  linkedInstances: LinkedInstance[],
  scopeMode: ScopeMode,
  visibleSkills: ToolkitSkill[],
  visibleTools: ToolkitTool[],
  visibleEndpoints: ToolkitEndpointCapability[]
): string {
  const lines: string[] = [
    '## BashGym Tool Kit inventory',
    '',
    `Generated: ${new Date().toISOString()}`,
    `Toolkit API: GET ${API_BASE}/agent/toolkit`,
    `Scope: ${scopeMode}`
  ]

  if (!inventory) {
    lines.push('', 'No inventory snapshot is loaded yet.')
    return lines.join('\n')
  }

  lines.push(
    '',
    '### Counts',
    `- visible skills: ${visibleSkills.length} of ${(inventory.counts.active_skills || inventory.counts.skills || 0) + (inventory.counts.endpoint_skills || 0)}`,
    `- visible tools: ${visibleTools.length} of ${inventory.counts.tools || 0}`,
    `- agent endpoints: ${inventory.counts.endpoints || 0}`,
    `- endpoint skills: ${inventory.counts.endpoint_skills || 0}`,
    `- endpoint toolsets: ${inventory.counts.endpoint_toolsets || 0}`,
    `- cached: ${inventory.cached ? 'yes' : 'no'}`
  )

  lines.push('', '### Linked instances')
  if (linkedInstances.length) {
    for (const instance of linkedInstances) {
      lines.push(
        `- ${instance.label}: ${instance.title}${instance.model ? ` (${instance.model})` : ''}`
      )
    }
  } else {
    lines.push('- none linked; showing workspace/Peony defaults')
  }

  lines.push('', '### Skill roots')
  for (const root of inventory.skill_roots.filter((root) => root.exists).slice(0, 10)) {
    lines.push(`- ${root.label}: ${root.path} (${root.skill_count})`)
  }

  lines.push('', '### Visible skills')
  for (const skill of visibleSkills.slice(0, 25)) {
    const resources = [
      skill.resource_counts.scripts ? `${skill.resource_counts.scripts} scripts` : '',
      skill.resource_counts.references ? `${skill.resource_counts.references} refs` : '',
      skill.resource_counts.assets ? `${skill.resource_counts.assets} assets` : '',
      skill.tool_count ? `${skill.tool_count} tools` : ''
    ]
      .filter(Boolean)
      .join(', ')
    lines.push(
      `- ${skill.name} [${skill.source}]${resources ? ` (${resources})` : ''}: ${skill.description}`
    )
  }

  lines.push('', '### Peony tools')
  for (const tool of visibleTools.slice(0, 25)) {
    lines.push(`- ${tool.name} [${tool.source}]: ${tool.description}`)
  }

  lines.push('', '### Connected agent endpoints')
  for (const endpoint of visibleEndpoints) {
    lines.push(
      `- ${endpoint.label} (${endpoint.endpoint_id}): ok=${endpoint.ok}, auth=${endpoint.auth_configured}, skills=${endpoint.skills}, toolsets=${endpoint.toolsets}, models=${endpoint.models}`
    )
    for (const name of endpoint.skill_names.slice(0, 10)) lines.push(`  - skill: ${name}`)
    for (const name of endpoint.toolset_names.slice(0, 10)) lines.push(`  - toolset: ${name}`)
    for (const warning of endpoint.warnings.slice(0, 3)) lines.push(`  - warning: ${warning}`)
  }

  if (skillIdea.trim()) {
    lines.push(
      '',
      '### Skill workshop request',
      `Target root: ${targetRoot || '(choose an appropriate skill root)'}`,
      `Idea: ${skillIdea.trim()}`,
      '',
      'Use skill-creator principles:',
      '- Keep SKILL.md concise and trigger-focused.',
      '- Add scripts, references, or assets only when they remove real repeated work.',
      '- Validate YAML frontmatter, naming, and resource layout before calling it done.',
      '- Prefer a terminal-agent implementation pass with explicit tests or forward checks.'
    )
  }

  if (inventory.warnings.length) {
    lines.push('', '### Inventory warnings')
    for (const warning of inventory.warnings.slice(0, 10)) lines.push(`- ${warning}`)
  }

  return lines.join('\n')
}

function SkillRow({
  skill,
  onSelect
}: {
  skill: ToolkitSkill
  onSelect: (skill: ToolkitSkill) => void
}) {
  const resources = skill.resource_counts
  const resourceBits = [
    resources.scripts ? `${resources.scripts}s` : '',
    resources.references ? `${resources.references}r` : '',
    resources.assets ? `${resources.assets}a` : '',
    skill.tool_count ? `${skill.tool_count}t` : ''
  ]
    .filter(Boolean)
    .join(' ')

  const content = (
    <>
      <div className="flex items-center gap-1.5 min-w-0">
        {skill.remote_endpoint_id ? (
          <Network className="h-2.5 w-2.5 flex-shrink-0 text-accent" />
        ) : (
          <span className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
        )}
        <span className="text-text-primary truncate">{skill.name}</span>
        <span className="text-text-muted flex-shrink-0">{skill.source}</span>
      </div>
      <div className="text-text-muted truncate pl-3">
        {skill.description || skill.path || 'No description'}
        {resourceBits ? ` | ${resourceBits}` : ''}
      </div>
    </>
  )

  if (skill.remote_endpoint_id) {
    return (
      <div
        className="block w-full min-w-0 rounded-brutal px-1 py-0.5 font-mono text-[10px]"
        title={`Reported by ${skill.remote_endpoint_id}; local instructions are not available`}
      >
        {content}
      </div>
    )
  }

  return (
    <button
      type="button"
      className="nodrag block w-full min-w-0 rounded-brutal px-1 py-0.5 text-left font-mono text-[10px] hover:bg-background-secondary"
      onClick={() => onSelect(skill)}
      title={`Open ${skill.name} in Skill Lab`}
    >
      {content}
    </button>
  )
}

function ToolRow({ tool }: { tool: ToolkitTool }) {
  return (
    <div className="min-w-0 text-[10px] font-mono">
      <div className="flex items-center gap-1.5 min-w-0">
        <Wrench className="w-2.5 h-2.5 text-accent flex-shrink-0" />
        <span className="text-text-primary truncate">{tool.name}</span>
        <span className="text-text-muted flex-shrink-0">{tool.source}</span>
      </div>
      <div className="text-text-muted truncate pl-4">{tool.description}</div>
    </div>
  )
}

function EndpointRow({ endpoint }: { endpoint: ToolkitEndpointCapability }) {
  const stateClass = endpoint.ok
    ? 'text-status-success'
    : endpoint.auth_configured
      ? 'text-status-warning'
      : 'text-text-muted'

  return (
    <div className="min-w-0 text-[10px] font-mono">
      <div className="flex items-center gap-1.5 min-w-0">
        {endpoint.ok ? (
          <CheckCircle2 className="w-2.5 h-2.5 text-status-success flex-shrink-0" />
        ) : (
          <ShieldCheck className={clsx('w-2.5 h-2.5 flex-shrink-0', stateClass)} />
        )}
        <span className="text-text-primary truncate">{endpoint.label}</span>
        <span className={clsx('flex-shrink-0', stateClass)}>{endpoint.endpoint_id}</span>
      </div>
      <div className="text-text-muted truncate pl-4">
        {endpoint.skills} skills | {endpoint.toolsets} toolsets | {endpoint.models} models
      </div>
      {(endpoint.skill_names.length > 0 || endpoint.toolset_names.length > 0) && (
        <div className="text-text-muted truncate pl-4">
          {[...endpoint.skill_names.slice(0, 3), ...endpoint.toolset_names.slice(0, 3)].join(', ')}
        </div>
      )}
      {endpoint.warnings[0] && (
        <div className="text-status-warning truncate pl-4">{endpoint.warnings[0]}</div>
      )}
    </div>
  )
}

export const ToolKitNode = memo(function ToolKitNode({
  data,
  selected
}: NodeProps<ToolKitNodeType>) {
  const [inventory, setInventory] = useState<ToolkitInventoryResponse | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('skills')
  const [scopeMode, setScopeMode] = useState<ScopeMode>('linked')
  const [query, setQuery] = useState('')
  const [skillIdea, setSkillIdea] = useState('')
  const [targetRoot, setTargetRoot] = useState('')
  const [includeRemote, setIncludeRemote] = useState(true)
  const [configOpen, setConfigOpen] = useState(false)
  const [workshopOpen, setWorkshopOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mountedRef = useRef(true)
  const panels = useTerminalStore((state) => state.panels)
  const canvasEdges = useTerminalStore((state) => state.canvasEdges)
  const sessionsVersion = useTerminalStore((state) => state.sessionsVersion)

  const loadInventory = useCallback(
    async (refresh = false) => {
      setLoading(true)
      const res = await toolkitApi.inventory({ includeRemote, refresh })
      if (!mountedRef.current) return
      setLoading(false)
      if (res.ok && res.data) {
        const snapshot = res.data
        setInventory(snapshot)
        setError(null)
        const workspaceRoot = snapshot.skill_roots.find(
          (root) => root.label === 'workspace' && root.exists
        )
        setTargetRoot(
          (current) =>
            current ||
            workspaceRoot?.path ||
            snapshot.skill_roots.find((root) => root.exists)?.path ||
            ''
        )
      } else {
        setError(res.error || 'Unable to load toolkit inventory')
      }
    },
    [includeRemote]
  )

  useEffect(() => {
    mountedRef.current = true
    void loadInventory(false)
    return () => {
      mountedRef.current = false
    }
  }, [loadInventory])

  const linkedInstances = useMemo(() => {
    const linkedPanelIds = new Set<string>()
    for (const edge of canvasEdges) {
      if (edge.source === data.panelId) linkedPanelIds.add(edge.target)
      if (edge.target === data.panelId) linkedPanelIds.add(edge.source)
    }

    const state = useTerminalStore.getState()
    void sessionsVersion
    return panels
      .filter((panel) => linkedPanelIds.has(panel.id))
      .map((panel): LinkedInstance | null => {
        if (panel.type === 'terminal' && panel.terminalId) {
          const session = state.sessions.get(panel.terminalId)
          const kind = inferHarness(session)
          return {
            panelId: panel.id,
            title: panel.title || session?.title || 'Terminal',
            kind,
            label: labelForHarness(kind),
            model: session?.model
          }
        }
        if (panel.type === 'agent') {
          return {
            panelId: panel.id,
            title: panel.title || 'Hermes Agent',
            kind: 'hermes',
            label: 'Hermes'
          }
        }
        return null
      })
      .filter((item): item is LinkedInstance => item !== null)
  }, [canvasEdges, data.panelId, panels, sessionsVersion])

  const hasHermesLink = useMemo(
    () => linkedInstances.some((instance) => instance.kind === 'hermes'),
    [linkedInstances]
  )

  const scopedEndpoints = useMemo(() => {
    const items = inventory?.endpoint_capabilities ?? []
    if (scopeMode === 'all') return items
    return hasHermesLink ? items : []
  }, [hasHermesLink, inventory, scopeMode])

  const endpointSkills = useMemo(
    () => scopedEndpoints.flatMap((endpoint) => endpointSkillItems(endpoint)),
    [scopedEndpoints]
  )

  const scopedSkills = useMemo(() => {
    const items = (inventory?.skills ?? []).filter(isCatalogSkillActive)
    if (scopeMode === 'all') return [...items, ...endpointSkills]
    if (hasHermesLink) return endpointSkills
    return items.filter((skill) => skillMatchesLinkedScope(skill, linkedInstances))
  }, [endpointSkills, hasHermesLink, inventory, linkedInstances, scopeMode])

  const scopedTools = useMemo(() => {
    const items = inventory?.tools ?? []
    return scopeMode === 'all'
      ? items
      : items.filter((tool) => toolMatchesLinkedScope(tool, linkedInstances))
  }, [inventory, linkedInstances, scopeMode])

  const filteredSkills = useMemo(() => {
    const items = scopedSkills
    if (!query.trim()) return items.slice(0, 12)
    return items.filter((skill) => matchQuery(skillText(skill), query)).slice(0, 12)
  }, [scopedSkills, query])

  const filteredTools = useMemo(() => {
    const items = scopedTools
    if (!query.trim()) return items.slice(0, 12)
    return items.filter((tool) => matchQuery(toolText(tool), query)).slice(0, 12)
  }, [scopedTools, query])

  const filteredEndpoints = useMemo(() => {
    const items = scopedEndpoints
    if (!query.trim()) return items
    return items.filter((endpoint) => matchQuery(endpointText(endpoint), query))
  }, [scopedEndpoints, query])

  const buildContext = useCallback(
    () =>
      buildToolkitContext(
        inventory,
        skillIdea,
        targetRoot,
        linkedInstances,
        scopeMode,
        filteredSkills,
        filteredTools,
        filteredEndpoints
      ),
    [
      filteredEndpoints,
      filteredSkills,
      filteredTools,
      inventory,
      linkedInstances,
      scopeMode,
      skillIdea,
      targetRoot
    ]
  )

  const statusBarClass = error
    ? 'bg-status-error'
    : inventory
      ? 'bg-status-success'
      : 'bg-background-tertiary'

  const roots = inventory?.skill_roots.filter((root) => root.exists) ?? []
  const linkedLabel = linkedInstances.length
    ? linkedInstances.map((instance) => instance.label).join(' + ')
    : 'Workspace'

  const openSkillLab = useCallback(
    (skill: ToolkitSkill) => {
      const state = useTerminalStore.getState()
      const existing = state.panels.find((panel) => panel.type === 'skilllab')
      const adapterConfig = {
        ...(existing?.adapterConfig || {}),
        selectedSkillId: skillIdFor(skill),
        selectedSkillName: skill.name,
        selectedSkillRevision: skill.revision,
        selectedSkillSource: skill.source,
        selectedSkillPath: skill.path,
        openRequestedAt: Date.now()
      }
      const panelId =
        existing?.id ??
        state.addPanel({
          type: 'skilllab',
          title: 'Skill Lab',
          adapterConfig
        })
      if (existing) {
        state.updatePanelConfig(panelId, adapterConfig)
      } else {
        const anchor = state.canvasNodes.get(data.panelId)?.position
        const occupied = Array.from(state.canvasNodes.values()).map((node) => node.position)
        state.updateCanvasNode(panelId, findDynamicNodePosition('skilllab', anchor, occupied))
      }
      const edgeExists = state.canvasEdges.some(
        (edge) =>
          (edge.source === data.panelId && edge.target === panelId) ||
          (edge.target === data.panelId && edge.source === panelId)
      )
      if (!edgeExists) {
        state.setCanvasEdges([
          ...state.canvasEdges,
          { id: `edge-${data.panelId}-${panelId}`, source: data.panelId, target: panelId }
        ])
      }
      state.setActivePanel(panelId)
    },
    [data.panelId]
  )

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={data.title}
        flowerVariant="toolkit"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? buildContext : undefined}
        statusBarClass={statusBarClass}
        hue={hueFor('toolkit')}
        onFocus={data.onFocus}
        onClose={data.onClose}
        headerRight={
          <>
            <div className="flex items-center gap-1 text-[9px] font-mono text-text-muted uppercase">
              {loading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Sparkles className="w-3 h-3" />
              )}
              <span>{inventory?.cached ? 'cached' : formatDate(inventory?.generated_at)}</span>
            </div>
            <button
              type="button"
              className="nodrag node-btn node-btn-accent"
              onClick={(event) => {
                event.stopPropagation()
                setConfigOpen(true)
              }}
              title="Configure Tool Kit node"
            >
              <SlidersHorizontal className="w-3 h-3" />
            </button>
            <button
              type="button"
              className="nodrag node-btn node-btn-accent"
              onClick={(event) => {
                event.stopPropagation()
                setWorkshopOpen(true)
              }}
              title="Open skill workshop"
            >
              <FilePlus2 className="w-3 h-3" />
            </button>
          </>
        }
      >
        <div className="space-y-2">
          <div className="grid grid-cols-3 gap-1.5">
            <CountTile icon={Brain} label="Skills" value={scopedSkills.length} />
            <CountTile icon={Wrench} label="Tools" value={scopedTools.length} />
            <CountTile icon={Network} label="Agents" value={scopedEndpoints.length} />
          </div>

          <div className="node-section !p-2">
            <div className="flex items-center gap-2 min-w-0">
              <Terminal className="w-3.5 h-3.5 text-accent flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <div className="node-field-label">Linked instance</div>
                <div className="text-[11px] font-mono text-text-primary truncate">
                  {scopeMode === 'all' ? 'All skill roots' : linkedLabel}
                </div>
              </div>
              <span className="text-[9px] font-mono text-text-muted uppercase">
                {includeRemote ? 'agents on' : 'agents off'}
              </span>
            </div>
            {linkedInstances.length > 0 && scopeMode === 'linked' && (
              <div className="mt-1 text-[9px] font-mono text-text-muted truncate">
                {linkedInstances.map((instance) => instance.title).join(' | ')}
              </div>
            )}
          </div>

          <div className="nodrag flex items-center gap-2">
            <div className="relative flex-1 min-w-0">
              <Search className="w-3 h-3 text-text-muted absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                className="input-brutal text-[11px] font-mono min-h-9 !pl-8"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search"
              />
            </div>
            <button
              type="button"
              className="node-btn h-9 w-9 justify-center flex-shrink-0"
              onClick={() => void loadInventory(true)}
              disabled={loading}
              title="Refresh"
            >
              {loading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <RefreshCw className="w-3 h-3" />
              )}
            </button>
          </div>

          <div className="nodrag grid grid-cols-4 gap-1">
            {(['skills', 'tools', 'agents'] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setViewMode(mode)}
                className={clsx(
                  'px-1.5 py-1 border-brutal rounded-brutal text-[9px] font-mono uppercase',
                  viewMode === mode
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-border-subtle text-text-muted hover:text-text-primary'
                )}
              >
                {mode}
              </button>
            ))}
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                setWorkshopOpen(true)
              }}
              className="border-brutal rounded-brutal border-border-subtle text-text-muted hover:text-text-primary px-1.5 py-1"
              title="Open skill workshop"
            >
              <FilePlus2 className="w-3 h-3 mx-auto" />
            </button>
          </div>

          {error && (
            <div className="border-brutal border-status-error/50 bg-status-error/10 rounded-brutal px-2 py-1 text-[10px] font-mono text-status-error">
              <AlertCircle className="inline w-3 h-3 mr-1" />
              {error.length > 130 ? `${error.slice(0, 127)}...` : error}
            </div>
          )}

          <div className="max-h-48 overflow-y-auto pr-1 space-y-1.5">
            {viewMode === 'skills' &&
              (filteredSkills.length ? (
                filteredSkills.map((skill) => (
                  <SkillRow
                    key={`${skill.source}-${skill.name}-${skill.path || ''}`}
                    skill={skill}
                    onSelect={openSkillLab}
                  />
                ))
              ) : (
                <div className="text-[10px] font-mono text-text-muted">No skills</div>
              ))}

            {viewMode === 'tools' &&
              (filteredTools.length ? (
                filteredTools.map((tool) => (
                  <ToolRow key={`${tool.source}-${tool.name}`} tool={tool} />
                ))
              ) : (
                <div className="text-[10px] font-mono text-text-muted">No tools</div>
              ))}

            {viewMode === 'agents' &&
              (filteredEndpoints.length ? (
                filteredEndpoints.map((endpoint) => (
                  <EndpointRow key={endpoint.endpoint_id} endpoint={endpoint} />
                ))
              ) : (
                <div className="text-[10px] font-mono text-text-muted">No agent endpoints</div>
              ))}
          </div>
        </div>
      </DataNodeShell>
      <NodeConfigModal
        isOpen={configOpen}
        onClose={() => setConfigOpen(false)}
        title={`${data.title} Config`}
        description="Capability inventory scope"
        size="lg"
      >
        <ConfigSection title="Inventory State">
          <ConfigRows>
            <ConfigRow
              label="Generated"
              value={
                inventory?.generated_at
                  ? new Date(inventory.generated_at).toLocaleString()
                  : undefined
              }
            />
            <ConfigRow label="Cached" value={inventory?.cached ? 'yes' : 'no'} />
            <ConfigRow label="Visible skills" value={scopedSkills.length} />
            <ConfigRow label="Visible tools" value={scopedTools.length} />
            <ConfigRow label="Visible agents" value={scopedEndpoints.length} />
            <ConfigRow
              label="Linked"
              value={
                linkedInstances.length
                  ? linkedInstances.map((instance) => instance.title).join(', ')
                  : 'workspace defaults'
              }
            />
            <ConfigRow label="Error" value={error} />
          </ConfigRows>
        </ConfigSection>

        <ConfigSection title="Scope">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <label className="node-field">
              <span className="node-field-label">Capability scope</span>
              <select
                className="input-brutal text-[11px] font-mono min-h-9"
                value={scopeMode}
                onChange={(event) => setScopeMode(event.target.value as ScopeMode)}
              >
                <option value="linked">Linked instance</option>
                <option value="all">All skill roots</option>
              </select>
            </label>
            <label className="node-field">
              <span className="node-field-label">Agent endpoint probe</span>
              <button
                type="button"
                className={clsx(
                  'node-btn node-btn-wide min-h-9 justify-center',
                  includeRemote ? 'node-btn-success' : 'node-btn-accent'
                )}
                onClick={() => setIncludeRemote((current) => !current)}
              >
                <Filter className="w-3 h-3" />
                <span>{includeRemote ? 'Enabled' : 'Disabled'}</span>
              </button>
            </label>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="node-btn node-btn-wide node-btn-accent justify-center"
              onClick={() => void loadInventory(true)}
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <RefreshCw className="w-3 h-3" />
              )}
              <span>Refresh Inventory</span>
            </button>
          </div>
        </ConfigSection>

        <ConfigSection title="Live Handles">
          <ConfigRows>
            <ConfigRow label="Toolkit API" value={`${API_BASE}/agent/toolkit`} />
            <ConfigRow label="Workspace API" value={`${API_BASE}/workspace/context?format=json`} />
          </ConfigRows>
        </ConfigSection>
      </NodeConfigModal>

      <NodeConfigModal
        isOpen={workshopOpen}
        onClose={() => setWorkshopOpen(false)}
        title={`${data.title} Workshop`}
        description="Skill context package"
        size="lg"
      >
        <ConfigSection title="Skill Workshop">
          <div className="grid grid-cols-1 gap-3">
            <label className="node-field">
              <span className="node-field-label">Target root</span>
              <select
                className="input-brutal text-[11px] font-mono min-h-9"
                value={targetRoot}
                onChange={(event) => setTargetRoot(event.target.value)}
              >
                <option value="">Choose root</option>
                {roots.map((root) => (
                  <option key={`${root.label}-${root.path}`} value={root.path}>
                    {root.label} ({root.skill_count})
                  </option>
                ))}
              </select>
            </label>
            <label className="node-field">
              <span className="node-field-label">Skill idea</span>
              <textarea
                className="input-brutal nowheel text-[11px] font-mono min-h-[144px]"
                value={skillIdea}
                onChange={(event) => setSkillIdea(event.target.value)}
                placeholder="Skill idea"
              />
            </label>
          </div>
        </ConfigSection>

        <ConfigSection title="Context">
          <ConfigRows>
            <ConfigRow label="Target" value={targetRoot} />
            <ConfigRow label="Idea ready" value={skillIdea.trim() ? 'yes' : 'no'} />
            <ConfigRow
              label="Send context"
              value={data.hasConnections ? 'linked terminal available' : 'link a terminal first'}
            />
          </ConfigRows>
        </ConfigSection>
      </NodeConfigModal>
    </>
  )
})
