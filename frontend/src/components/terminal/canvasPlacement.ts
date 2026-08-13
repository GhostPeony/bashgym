import type { PanelType } from '../../stores/terminalStore'

export interface CanvasPosition {
  x: number
  y: number
}

const NODE_CLEARANCE_X = 392
const NODE_CLEARANCE_Y = 260

const PREFERRED_OFFSETS: Partial<Record<PanelType, CanvasPosition>> = {
  training: { x: 420, y: -40 },
  campaign: { x: -420, y: -280 },
  designer: { x: 420, y: 0 },
  evals: { x: 420, y: 280 },
  huggingface: { x: 420, y: 280 },
  toolkit: { x: 0, y: 300 },
  // Keep the lab out from under the docked Master Control surface on the right.
  skilllab: { x: -420, y: 0 },
  activity: { x: 0, y: 300 }
}

const NEARBY_OFFSETS: CanvasPosition[] = [
  { x: 420, y: 0 },
  { x: 420, y: 280 },
  { x: 0, y: 300 },
  { x: -420, y: 280 },
  { x: -420, y: 0 },
  { x: -420, y: -280 },
  { x: 0, y: -300 },
  { x: 420, y: -280 }
]

function overlaps(candidate: CanvasPosition, occupied: CanvasPosition): boolean {
  return (
    Math.abs(candidate.x - occupied.x) < NODE_CLEARANCE_X &&
    Math.abs(candidate.y - occupied.y) < NODE_CLEARANCE_Y
  )
}

/** Find the nearest open lane around the active canvas anchor. */
export function findDynamicNodePosition(
  type: PanelType,
  anchor: CanvasPosition | undefined,
  occupied: readonly CanvasPosition[]
): CanvasPosition {
  if (!anchor && occupied.length === 0) return { x: 80, y: 80 }

  const base = anchor || occupied.at(-1) || { x: 80, y: 80 }
  const preferred = PREFERRED_OFFSETS[type] || NEARBY_OFFSETS[0]
  const offsets = [
    preferred,
    ...NEARBY_OFFSETS.filter((offset) => offset.x !== preferred.x || offset.y !== preferred.y)
  ]

  for (let ring = 1; ring <= 4; ring += 1) {
    for (const offset of offsets) {
      const candidate = {
        x: base.x + offset.x * ring,
        y: base.y + offset.y * ring
      }
      if (!occupied.some((position) => overlaps(candidate, position))) return candidate
    }
  }

  return {
    x: base.x + 420,
    y: base.y + 300 * (occupied.length + 1)
  }
}
