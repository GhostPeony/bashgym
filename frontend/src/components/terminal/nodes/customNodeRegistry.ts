import type { Panel } from '../../../stores'
import { useWorkspaceStore } from '../../../stores/workspaceStore'
import type { CanvasGraphIndex } from '../canvasPerformance'
import { ActivityFeedNode } from './ActivityFeedNode'
import { AgentEndpointNode } from './AgentEndpointNode'
import { CampaignNode } from './CampaignNode'
import { DataDesignerNode } from './DataDesignerNode'
import { EvalNode } from './EvalNode'
import { HFInventoryNode } from './HFInventoryNode'
import { IntegrationNode } from './IntegrationNode'
import { KnowledgeBaseNode } from './KnowledgeBaseNode'
import { McpWorkbenchNode } from './McpWorkbenchNode'
import { SkillLabNode } from './SkillLabNode'
import { ToolKitNode } from './ToolKitNode'
import { TrainingRunNode } from './TrainingRunNode'
import type { DataNodeData, IntegrationNodeData } from './types'
import type { CustomNodeType } from './customNodeTypes'

// Adapter modules register themselves as import side effects. Keeping these next
// to the shared registry makes Grid and Canvas independent entry surfaces.
import './adapters/context'
import './adapters/neon'
import './adapters/vercel'

export const customNodeTypes = {
  context: IntegrationNode,
  neon: IntegrationNode,
  vercel: IntegrationNode,
  activity: ActivityFeedNode,
  training: TrainingRunNode,
  campaign: CampaignNode,
  evals: EvalNode,
  designer: DataDesignerNode,
  huggingface: HFInventoryNode,
  agent: AgentEndpointNode,
  toolkit: ToolKitNode,
  skilllab: SkillLabNode,
  mcp: McpWorkbenchNode,
  knowledge: KnowledgeBaseNode
} satisfies Record<CustomNodeType, unknown>

export function buildCustomNodeData(
  panel: Panel,
  graph: CanvasGraphIndex,
  onFocus: (panelId: string) => void,
  onClose: (panelId: string) => void
): IntegrationNodeData | DataNodeData {
  const hasConnections = graph.connectedPanelIds.has(panel.id)

  if (panel.type === 'context' || panel.type === 'neon' || panel.type === 'vercel') {
    return {
      panelId: panel.id,
      title: panel.title,
      adapterType: panel.type,
      adapterConfig: { ...panel.adapterConfig, _panelId: panel.id },
      hasConnections,
      onFocus,
      onClose
    }
  }

  const linkedPanels = graph.linkedPanelsById.get(panel.id) ?? []
  return {
    panelId: panel.id,
    title: panel.title,
    workspaceId: useWorkspaceStore.getState().activeWorkspaceId,
    adapterConfig: panel.adapterConfig,
    hasConnections,
    hasTerminalConnections: graph.terminalConnectedPanelIds.has(panel.id),
    linkedHuggingFace: linkedPanels
      .filter((candidate) => candidate.type === 'huggingface')
      .map((candidate) => ({
        panelId: candidate.id,
        title: candidate.title,
        adapterConfig: candidate.adapterConfig
      })),
    linkedKnowledgeBases: linkedPanels
      .filter((candidate) => candidate.type === 'knowledge')
      .map((candidate) => ({
        panelId: candidate.id,
        title: candidate.title,
        adapterConfig: candidate.adapterConfig
      })),
    linkedEvals: linkedPanels
      .filter((candidate) => candidate.type === 'evals')
      .map((candidate) => ({ panelId: candidate.id, title: candidate.title })),
    onFocus,
    onClose
  }
}
