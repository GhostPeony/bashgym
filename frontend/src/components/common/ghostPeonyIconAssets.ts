export type GhostPeonyIconName =
  | 'activity'
  | 'agent'
  | 'app'
  | 'browser'
  | 'context'
  | 'designer'
  | 'evals'
  | 'files'
  | 'huggingface'
  | 'integration'
  | 'neon'
  | 'preview'
  | 'terminal'
  | 'toolkit'
  | 'training'
  | 'vercel'

const ICON_ASSET: Partial<Record<GhostPeonyIconName, string>> = {
  app: '/bashgym-peony.png',
  activity: '/node-icons/node-activity.png',
  designer: '/node-icons/node-designer.png',
  huggingface: '/node-icons/node-huggingface.png',
  training: '/node-icons/node-training.png',
  toolkit: '/node-icons/node-toolkit.png'
}

const NEUTRAL_ICON_ASSET: Partial<Record<GhostPeonyIconName, string>> = {
  app: '/node-icons/node-app-neutral.png',
  activity: '/node-icons/node-activity-neutral.png',
  designer: '/node-icons/node-designer-neutral.png',
  huggingface: '/node-icons/node-huggingface-neutral.png',
  training: '/node-icons/node-training-neutral.png',
  toolkit: '/node-icons/node-toolkit-neutral.png'
}

export type GhostPeonyIconTone = 'neutral' | 'color' | 'accent' | 'node'

export function ghostPeonyIconPath(
  name: GhostPeonyIconName,
  tone: GhostPeonyIconTone = 'neutral'
): string {
  return tone === 'color'
    ? ICON_ASSET[name] ?? ICON_ASSET.app!
    : NEUTRAL_ICON_ASSET[name] ?? NEUTRAL_ICON_ASSET.app!
}
