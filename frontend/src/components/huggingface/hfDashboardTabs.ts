import { Database, FileText, HardDrive, Layers, Package, Search, Server } from 'lucide-react'

export type HFDashboardTab =
  'training' | 'spaces' | 'datasets' | 'models' | 'buckets' | 'research' | 'traces'

export function hfDashboardTabs() {
  return [
    { id: 'training' as HFDashboardTab, label: 'Jobs', icon: Server, requiresPro: false },
    { id: 'spaces' as HFDashboardTab, label: 'ZeroGPU Spaces', icon: Layers, requiresPro: true },
    { id: 'datasets' as HFDashboardTab, label: 'Datasets', icon: Database, requiresPro: false },
    { id: 'models' as HFDashboardTab, label: 'My Models', icon: Package, requiresPro: false },
    { id: 'buckets' as HFDashboardTab, label: 'Buckets', icon: HardDrive, requiresPro: false },
    { id: 'research' as HFDashboardTab, label: 'Research', icon: Search, requiresPro: false },
    { id: 'traces' as HFDashboardTab, label: 'Traces', icon: FileText, requiresPro: false }
  ]
}
