import type { PanelType } from '../../../stores'

export const CUSTOM_NODE_TYPES = [
  'context',
  'neon',
  'vercel',
  'activity',
  'training',
  'campaign',
  'evals',
  'designer',
  'huggingface',
  'agent',
  'toolkit',
  'skilllab',
  'mcp',
  'knowledge'
] as const satisfies readonly PanelType[]

export type CustomNodeType = (typeof CUSTOM_NODE_TYPES)[number]

const CUSTOM_NODE_TYPE_SET = new Set<PanelType>(CUSTOM_NODE_TYPES)

export function isCustomNodeType(type: PanelType): type is CustomNodeType {
  return CUSTOM_NODE_TYPE_SET.has(type)
}
