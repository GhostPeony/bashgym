import { Zap } from 'lucide-react'
import { AutoResearchPanel } from '../training/AutoResearchPanel'
import { useAutoResearchStore } from '../../stores/autoresearchStore'

export function AutoResearchDashboard() {
  const { status, traceStatus, activeMode } = useAutoResearchStore()

  const activeStatus = activeMode === 'hyperparam' ? status : traceStatus

  return (
    <div className="h-full overflow-auto">
      <div className="max-w-[1200px] mx-auto px-6 py-8">
        {/* Page Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-3">
              <Zap className="w-6 h-6 text-accent" />
              <h1 className="font-serif text-2xl">AutoResearch</h1>
              {activeStatus !== 'idle' && (
                <span className={`px-2 py-0.5 text-xs font-mono uppercase tracking-wider rounded-sm ${
                  activeStatus === 'running' ? 'bg-status-success text-white' :
                  activeStatus === 'paused' ? 'bg-status-warning text-white' :
                  activeStatus === 'completed' ? 'bg-accent text-white' :
                  activeStatus === 'failed' ? 'bg-status-error text-white' :
                  'bg-background-tertiary text-text-muted'
                }`}>
                  {activeStatus}
                </span>
              )}
            </div>
            <p className="text-text-muted text-sm mt-1">
              Automated hyperparameter search and trace data optimization
            </p>
          </div>
        </div>

        {/* Main Content */}
        <AutoResearchPanel />
      </div>
    </div>
  )
}
