export type GhostPeonyIconName =
  | 'activity'
  | 'agent'
  | 'app'
  | 'browser'
  | 'context'
  | 'database'
  | 'designer'
  | 'evals'
  | 'files'
  | 'huggingface'
  | 'integration'
  | 'mcp'
  | 'neon'
  | 'preview'
  | 'supabase'
  | 'terminal'
  | 'toolkit'
  | 'skilllab'
  | 'training'
  | 'vercel'

const ICON_ASSET: Partial<Record<GhostPeonyIconName, string>> = {
  app: '/bashgym-peony.png',
  activity: '/node-icons/node-activity.png',
  database: '/node-icons/node-database.png',
  designer: '/node-icons/node-designer.png',
  evals: '/node-icons/node-evals.png',
  huggingface: '/node-icons/node-huggingface.png',
  neon: '/node-icons/node-database.png',
  supabase: '/node-icons/node-database.png',
  terminal: '/node-icons/node-terminal.png',
  training: '/node-icons/node-training.png',
  toolkit: '/node-icons/node-toolkit.png',
  skilllab: '/node-icons/node-toolkit.png',
  mcp: '/node-icons/node-mcp.png'
}

const NEUTRAL_ICON_ASSET: Partial<Record<GhostPeonyIconName, string>> = {
  app: '/node-icons/node-app-neutral.png',
  activity: '/node-icons/node-activity-neutral.png',
  database: '/node-icons/node-database-neutral.png',
  designer: '/node-icons/node-designer-neutral.png',
  evals: '/node-icons/node-evals-neutral.png',
  huggingface: '/node-icons/node-huggingface-neutral.png',
  neon: '/node-icons/node-database-neutral.png',
  supabase: '/node-icons/node-database-neutral.png',
  terminal: '/node-icons/node-terminal-neutral.png',
  training: '/node-icons/node-training-neutral.png',
  toolkit: '/node-icons/node-toolkit-neutral.png',
  skilllab: '/node-icons/node-toolkit-neutral.png',
  mcp: '/node-icons/node-mcp-neutral.png'
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
