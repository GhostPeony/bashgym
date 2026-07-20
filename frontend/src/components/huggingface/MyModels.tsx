import { ExternalLink, Download, Heart, Lock, Globe, Package } from 'lucide-react'
import { hfMyModelsResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'

export function MyModels() {
  const { data, loading, refreshing, error, refresh } = useSessionResource(hfMyModelsResource)
  const models = data ?? []

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin w-6 h-6 border-2 border-accent border-t-transparent rounded-full" />
      </div>
    )
  }

  if (error && data === null) {
    return (
      <div className="card p-6 text-center">
        <p className="text-text-muted">{error}</p>
        <button onClick={() => refresh()} className="btn-secondary mt-3">Retry</button>
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="card p-8 text-center">
        <Package className="w-10 h-10 text-text-muted mx-auto mb-3" />
        <p className="text-text-secondary font-serif text-lg">No models on Hub yet</p>
        <p className="text-text-muted text-sm mt-1">
          Push your first model from the Models page or enable auto-push in Training Config.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs uppercase tracking-widest text-text-muted">
          {models.length} model{models.length !== 1 ? 's' : ''} on Hub
        </span>
        <button onClick={() => refresh()} className="btn-secondary text-xs">
          {refreshing ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {models.map((model) => (
          <div key={model.id} className="card p-4 border-2 hover:border-accent transition-colors shadow-brutal-sm">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="font-mono text-sm font-semibold truncate">{model.id}</h3>
                  {model.private ? (
                    <Lock className="w-3 h-3 text-text-muted flex-shrink-0" />
                  ) : (
                    <Globe className="w-3 h-3 text-text-muted flex-shrink-0" />
                  )}
                </div>
                {model.pipeline_tag && (
                  <span className="font-mono text-[0.65rem] uppercase tracking-wider text-accent">{model.pipeline_tag}</span>
                )}
              </div>
              <a
                href={model.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-accent transition-colors flex-shrink-0"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </div>

            <div className="flex items-center gap-4 mt-2 text-xs text-text-muted font-mono">
              <span className="flex items-center gap-1">
                <Download className="w-3 h-3" />
                {model.downloads.toLocaleString()}
              </span>
              <span className="flex items-center gap-1">
                <Heart className="w-3 h-3" />
                {model.likes}
              </span>
              {model.last_modified && (
                <span>{new Date(model.last_modified).toLocaleDateString()}</span>
              )}
            </div>

            {model.tags.includes('bashgym') && (
              <span className="inline-block mt-2 px-2 py-0.5 text-[0.6rem] font-mono uppercase tracking-wider bg-accent-light text-accent-dark border border-accent rounded-sm">
                BashGym
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
