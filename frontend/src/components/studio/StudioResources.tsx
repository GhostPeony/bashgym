import { lazy, Suspense, useState, type ReactNode } from 'react'

const DataDesignerTab = lazy(() =>
  import('../factory/DataDesignerTab').then((m) => ({ default: m.DataDesignerTab }))
)
const EvaluatorDashboard = lazy(() =>
  import('../evaluator/EvaluatorDashboard').then((m) => ({ default: m.EvaluatorDashboard }))
)
const ModelBrowser = lazy(() => import('../models').then((m) => ({ default: m.ModelBrowser })))
const ModelProfilePage = lazy(() =>
  import('../models').then((m) => ({ default: m.ModelProfilePage }))
)
const ModelComparison = lazy(() =>
  import('../models').then((m) => ({ default: m.ModelComparison }))
)
const DeviceManager = lazy(() =>
  import('../training/DeviceManager').then((m) => ({ default: m.DeviceManager }))
)

export function StudioResources({ children }: { children: ReactNode }) {
  const [tool, setTool] = useState('materials')
  const [modelId, setModelId] = useState<string | null>(null)
  const [comparison, setComparison] = useState<string[]>([])
  return (
    <section aria-label="Research resources">
      <nav className="studio-resource-tabs" aria-label="Resource tools">
        {[
          ['materials', 'Campaign materials'],
          ['data', 'Data Designer'],
          ['evaluation', 'Evaluator'],
          ['models', 'Models'],
          ['environment', 'Environment']
        ].map(([id, label]) => (
          <button key={id} onClick={() => setTool(id)} aria-pressed={tool === id}>
            {label}
          </button>
        ))}
      </nav>
      <Suspense fallback={<p role="status">Loading resource tools…</p>}>
        {tool === 'materials' && children}
        {tool === 'data' && <DataDesignerTab campaignMode />}
        {tool === 'evaluation' && <EvaluatorDashboard />}
        {tool === 'models' &&
          (comparison.length >= 2 ? (
            <ModelComparison
              modelIds={comparison}
              onBack={() => setComparison([])}
              onAddModel={() => {
                setComparison([])
                setModelId(null)
              }}
              onRemoveModel={(id) =>
                setComparison((current) => current.filter((item) => item !== id))
              }
            />
          ) : modelId ? (
            <ModelProfilePage
              modelId={modelId}
              onBack={() => setModelId(null)}
              onCompare={setComparison}
            />
          ) : (
            <ModelBrowser
              onSelectModel={setModelId}
              onCompare={setComparison}
              onTrainNew={() => {
                window.location.hash = 'setup'
              }}
            />
          ))}
        {tool === 'environment' && <DeviceManager />}
      </Suspense>
    </section>
  )
}
