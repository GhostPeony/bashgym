import type { PanelType } from '../../stores/terminalStore'
import {
  ghostPeonyIconPath,
  type GhostPeonyIconName
} from '../common/ghostPeonyIconAssets'

export type NodeFlowerVariant = PanelType | 'integration'

const FLOWER_ASSETS: Record<NodeFlowerVariant, GhostPeonyIconName> = {
  terminal: 'terminal',
  browser: 'browser',
  preview: 'preview',
  files: 'preview',
  context: 'context',
  neon: 'neon',
  vercel: 'vercel',
  integration: 'integration',
  activity: 'activity',
  training: 'training',
  evals: 'evals',
  designer: 'designer',
  huggingface: 'huggingface',
  agent: 'agent',
  toolkit: 'toolkit'
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
