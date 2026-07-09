/**
 * One-click canvas workspace presets.
 *
 * A preset describes a set of panels (terminals may auto-launch an agent CLI),
 * their canvas positions, and the edges between them. Built-ins cover common
 * missions; custom presets are captured from the live canvas and persisted to
 * localStorage.
 */

import { useTerminalStore } from '../../stores'
import type { Panel, PanelType } from '../../stores/terminalStore'

export interface PresetPanel {
  type: PanelType
  title: string
  /** Terminal panels only: command auto-typed into the fresh PTY */
  launch?: string
  position: { x: number; y: number }
}

export interface CanvasPreset {
  id: string
  name: string
  panels: PresetPanel[]
  /** Edges as [sourceIndex, targetIndex] into `panels` */
  links: Array<[number, number]>
}

export const BUILTIN_PRESETS: CanvasPreset[] = [
  {
    id: 'training-mission',
    name: 'Training Mission Control',
    panels: [
      { type: 'terminal', title: 'Claude Code', launch: 'claude', position: { x: 60, y: 280 } },
      { type: 'training', title: 'Training Run', position: { x: 540, y: 40 } },
      { type: 'evals', title: 'Evals', position: { x: 540, y: 330 } },
      { type: 'designer', title: 'Data Designer', position: { x: 540, y: 580 } },
      { type: 'activity', title: 'Activity', position: { x: 60, y: 620 } }
    ],
    links: [[0, 1], [0, 2], [0, 3]]
  },
  {
    id: 'new-project',
    name: 'New Project',
    panels: [
      { type: 'terminal', title: 'Claude Code', launch: 'claude', position: { x: 60, y: 60 } },
      { type: 'terminal', title: 'Shell', position: { x: 60, y: 400 } },
      { type: 'browser', title: 'Browser', position: { x: 540, y: 60 } },
      { type: 'activity', title: 'Activity', position: { x: 540, y: 430 } }
    ],
    links: [[2, 0]]
  },
  {
    id: 'web-dev',
    name: 'Landing Page',
    panels: [
      { type: 'terminal', title: 'Claude Code', launch: 'claude', position: { x: 60, y: 200 } },
      { type: 'browser', title: 'Preview', position: { x: 540, y: 100 } }
    ],
    links: [[1, 0]]
  }
]

const CUSTOM_PRESETS_KEY = 'bashgym_canvas_presets'

export function loadCustomPresets(): CanvasPreset[] {
  try {
    const stored = localStorage.getItem(CUSTOM_PRESETS_KEY)
    if (stored) return JSON.parse(stored)
  } catch {
    // Ignore
  }
  return []
}

export function saveCustomPreset(preset: CanvasPreset): void {
  const all = loadCustomPresets().filter((p) => p.id !== preset.id)
  all.push(preset)
  try {
    localStorage.setItem(CUSTOM_PRESETS_KEY, JSON.stringify(all))
  } catch {
    // Ignore storage errors
  }
}

export function deleteCustomPreset(id: string): void {
  try {
    localStorage.setItem(
      CUSTOM_PRESETS_KEY,
      JSON.stringify(loadCustomPresets().filter((p) => p.id !== id))
    )
  } catch {
    // Ignore
  }
}

/** Panel types a captured preset can restore (terminals restart fresh, optionally with an agent) */
const CAPTURABLE_TYPES: PanelType[] = [
  'terminal', 'browser', 'activity', 'training', 'evals', 'designer', 'huggingface', 'context', 'neon', 'vercel'
]

/** Snapshot the live canvas (panels, positions, edges) as a reusable preset */
export function captureCurrentAsPreset(name: string): CanvasPreset {
  const st = useTerminalStore.getState()
  const panels: PresetPanel[] = []
  const idToIndex = new Map<string, number>()

  st.panels.forEach((p: Panel, index: number) => {
    if (!CAPTURABLE_TYPES.includes(p.type)) return
    const pos = st.canvasNodes.get(p.id)?.position ?? {
      x: 60 + (index % 3) * 420,
      y: 60 + Math.floor(index / 3) * 300
    }
    let launch: string | undefined
    if (p.type === 'terminal' && p.terminalId) {
      const kind = st.sessions.get(p.terminalId)?.agentKind
      launch = kind === 'claude' ? 'claude' : kind === 'codex' ? 'codex' : undefined
    }
    idToIndex.set(p.id, panels.length)
    panels.push({ type: p.type, title: p.title, launch, position: pos })
  })

  const links: Array<[number, number]> = []
  st.canvasEdges.forEach((e) => {
    const a = idToIndex.get(e.source)
    const b = idToIndex.get(e.target)
    if (a !== undefined && b !== undefined) links.push([a, b])
  })

  return { id: `custom-${Date.now()}`, name, panels, links }
}

/**
 * Spawn a preset's panels at their positions.
 * Returns panel-id pairs for the caller to connect as edges.
 */
export function applyPreset(
  preset: CanvasPreset,
  setPosition: (panelId: string, pos: { x: number; y: number }) => void
): Array<[string, string]> {
  const ids: string[] = []
  for (const p of preset.panels) {
    const store = useTerminalStore.getState()
    let panelId: string
    if (p.type === 'terminal') {
      const terminalId = store.createTerminal(undefined, p.title, p.launch)
      const panel = useTerminalStore.getState().panels.find((pp) => pp.terminalId === terminalId)
      if (!panel) continue
      panelId = panel.id
    } else {
      panelId = store.addPanel({ type: p.type, title: p.title, adapterConfig: {} })
    }
    ids.push(panelId)
    setPosition(panelId, p.position)
  }
  return preset.links
    .filter(([a, b]) => ids[a] !== undefined && ids[b] !== undefined)
    .map(([a, b]) => [ids[a], ids[b]])
}
