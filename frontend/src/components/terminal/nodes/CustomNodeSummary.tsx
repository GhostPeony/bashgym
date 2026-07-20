import type { ComponentType } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import type { Panel } from '../../../stores'
import type { CanvasGraphIndex } from '../canvasPerformance'
import { NodeSurfaceProvider } from './NodeSurfaceProvider'
import { buildCustomNodeData, customNodeTypes } from './customNodeRegistry'
import { isCustomNodeType, type CustomNodeType } from './customNodeTypes'
import type { DataNodeData, IntegrationNodeData } from './types'

type SummaryNode = Node<IntegrationNodeData | DataNodeData, CustomNodeType>

export function CustomNodeSummary({
  panel,
  graph,
  selected,
  onFocus,
  onClose
}: {
  panel: Panel
  graph: CanvasGraphIndex
  selected: boolean
  onFocus: (panelId: string) => void
  onClose: (panelId: string) => void
}) {
  if (!isCustomNodeType(panel.type)) return null

  const Renderer = customNodeTypes[panel.type] as unknown as ComponentType<NodeProps<SummaryNode>>
  const props = {
    data: buildCustomNodeData(panel, graph, onFocus, onClose),
    selected
  } as NodeProps<SummaryNode>

  return (
    <NodeSurfaceProvider surface="grid">
      <Renderer {...props} />
    </NodeSurfaceProvider>
  )
}
