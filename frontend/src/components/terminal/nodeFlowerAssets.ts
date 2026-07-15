import type { PanelType } from '../../stores/terminalStore'
import {
  ghostPeonyIconPath,
  type GhostPeonyIconName
} from '../common/ghostPeonyIconAssets'

export type NodeFlowerVariant = PanelType | 'database' | 'integration' | 'supabase'

const FLOWER_ASSETS: Record<NodeFlowerVariant, GhostPeonyIconName> = {
  terminal: 'terminal',
  browser: 'browser',
  preview: 'preview',
  files: 'preview',
  context: 'context',
  database: 'database',
  neon: 'neon',
  supabase: 'supabase',
  vercel: 'vercel',
  integration: 'integration',
  activity: 'activity',
  training: 'training',
  campaign: 'training',
  evals: 'evals',
  designer: 'designer',
  huggingface: 'huggingface',
  agent: 'agent',
  toolkit: 'toolkit',
  skilllab: 'skilllab',
  mcp: 'mcp',
  knowledge: 'database'
}

export function flowerVariantForPanelType(type: PanelType): NodeFlowerVariant {
  return type
}

export function nodeFlowerIconName(variant: NodeFlowerVariant): GhostPeonyIconName {
  return FLOWER_ASSETS[variant] ?? 'app'
}

export function nodeFlowerAssetPath(variant: NodeFlowerVariant): string {
  return ghostPeonyIconPath(nodeFlowerIconName(variant), 'color')
}
