import { Activity } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { PanelType } from '../../../stores/terminalStore'

export interface DataPanelDef {
  type: PanelType
  title: string
  icon: LucideIcon
}

/** Live-data canvas nodes. MasterControlPanel renders one add-button per entry. */
export const DATA_PANEL_DEFS: DataPanelDef[] = [
  { type: 'activity', title: 'Activity', icon: Activity },
]

export const DATA_NODE_TYPES: PanelType[] = DATA_PANEL_DEFS.map((d) => d.type)
