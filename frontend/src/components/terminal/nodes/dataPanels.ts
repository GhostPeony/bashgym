import { Activity, Dumbbell, Factory, FlaskConical } from 'lucide-react'
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
  { type: 'training', title: 'Training Run', icon: Dumbbell },
  { type: 'evals', title: 'Evals', icon: FlaskConical },
  { type: 'designer', title: 'Data Designer', icon: Factory },
]

export const DATA_NODE_TYPES: PanelType[] = DATA_PANEL_DEFS.map((d) => d.type)
