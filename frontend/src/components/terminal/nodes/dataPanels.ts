import { Activity, Bot, Boxes, Cloud, Dumbbell, Factory, FlaskConical } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { PanelType } from '../../../stores/terminalStore'

export interface DataPanelDef {
  type: PanelType
  title: string
  icon: LucideIcon
  /** Identity hue from the platform accent palette (accentStore spectrum) */
  hue: number
}

/** Live-data canvas nodes. MasterControlPanel renders one add-button per entry. */
export const DATA_PANEL_DEFS: DataPanelDef[] = [
  { type: 'activity', title: 'Activity', icon: Activity, hue: 308 },   // Orchid
  { type: 'training', title: 'Training Run', icon: Dumbbell, hue: 35 }, // Marigold
  { type: 'evals', title: 'Evals', icon: FlaskConical, hue: 140 },      // Moss
  { type: 'designer', title: 'Data Designer', icon: Factory, hue: 210 }, // Sky
  { type: 'huggingface', title: 'Hugging Face', icon: Cloud, hue: 52 },  // Sunflower
  { type: 'agent', title: 'Hermes Agent', icon: Bot, hue: 270 },         // Wisteria
  { type: 'toolkit', title: 'Tool Kit', icon: Boxes, hue: 10 }           // Poppy
]

export const DATA_NODE_TYPES: PanelType[] = DATA_PANEL_DEFS.map((d) => d.type)

export function hueFor(type: PanelType): number | undefined {
  return DATA_PANEL_DEFS.find((d) => d.type === type)?.hue
}
