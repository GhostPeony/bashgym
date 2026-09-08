import type { DesignerPipelinesResponse } from '../../services/api'

export function DesignerReadinessPanel({
  data,
  error,
  checking,
  onRefresh,
  showSetupLink = true
}: {
  data: DesignerPipelinesResponse | null
  error: string | null
  checking: boolean
  onRefresh: () => void
  showSetupLink?: boolean
}) {
  const readiness = data?.readiness
  return (
    <section
      className="card p-4 mb-5 text-xs leading-5 text-text-secondary"
      aria-label="Data Designer readiness"
    >
      <h3 className="font-brand text-lg text-text-primary">Optional Data Designer</h3>
      <p>
        Use an existing dataset for campaign preparation, or explicitly generate synthetic data
        here. Generation teachers are separate from your research agent and the learner being
        trained.
      </p>
      {error ? (
        <p role="alert" className="mt-2 text-status-warning">
          Check failed: {error}. Previous evidence may be stale.
        </p>
      ) : null}
      {!data ? (
        <p role="status">
          {checking ? 'Checking backend dependencies…' : 'Backend readiness has not been checked.'}
        </p>
      ) : null}
      {readiness ? (
        <>
          <p className="mt-2">
            Backend process imports: Data Designer{' '}
            {readiness.data_designer_importable ? 'available' : 'unavailable'}; pandas{' '}
            {readiness.pandas_importable ? 'available' : 'unavailable'}; pipeline dependencies{' '}
            {readiness.pipeline_builders_importable ? 'available' : 'unavailable'}.
          </p>
          <p>
            Checked <time dateTime={readiness.checked_at}>{readiness.checked_at}</time>. This check
            covers imports in the running backend process and credential presence only. It does not
            verify provider access, generated data or campaign recipe readiness.
          </p>
          <p className="mt-2">
            This browser form uses NVIDIA NIM. NVIDIA_API_KEY is{' '}
            {readiness.credential_configured ? 'present' : 'missing'} in the backend environment.{' '}
            Configure it in the backend environment before requesting generation. Other providers
            require an explicit provider and endpoint through the Data Designer API.
          </p>
        </>
      ) : data ? (
        <p className="mt-2">
          This backend did not return scoped readiness evidence. Refresh after updating the backend.
        </p>
      ) : null}
      {data && !data.available ? (
        <p className="mt-2">
          Install the optional dependencies in the environment running the backend with{' '}
          <code>pip install &quot;bashgym[data-designer]&quot;</code>, restart that backend, then
          recheck. Installing the package alone does not configure a generation provider.
        </p>
      ) : null}
      {data?.available && data.pipelines.length === 0 ? (
        <p className="mt-2">
          No pipeline definitions are available. Check the optional package compatibility in the
          backend environment, restart after changes, then recheck.
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button className="btn-secondary" type="button" disabled={checking} onClick={onRefresh}>
          {checking ? 'Checking…' : 'Recheck dependencies'}
        </button>
        {showSetupLink ? (
          <a className="text-accent-dark underline" href="#setup">
            Continue setup with existing data
          </a>
        ) : null}
      </div>
    </section>
  )
}
