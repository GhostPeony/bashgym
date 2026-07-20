import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  CloudDownload,
  Database,
  ExternalLink,
  FileJson,
  Loader2,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  sourcesApi,
  type SourceCard,
  type SourcePrepareResponse,
  type SourceUse,
} from '../../services/api'
import { useTrainingStore } from '../../stores'
import { useKeyedSessionResource, useSessionResource } from '../../stores/sessionResource'
import {
  sourceCatalogResource,
  sourceRecommendationsResource,
} from '../../stores/factoryResources'

const SOURCE_GOALS: Array<{ value: SourceUse; label: string; detail: string }> = [
  { value: 'sft', label: 'SFT', detail: 'conversation examples' },
  { value: 'dpo', label: 'DPO', detail: 'chosen/rejected pairs' },
  { value: 'reward_model', label: 'Reward model', detail: 'scored response examples' },
  { value: 'process_reward', label: 'Process reward', detail: 'step-level reward examples' },
  { value: 'terminal_rl', label: 'Terminal RL', detail: 'environment specs' },
  { value: 'evaluation', label: 'Evaluation', detail: 'eval manifests and heldouts' },
  { value: 'raw_reference', label: 'Raw reference', detail: 'reference corpus handoff' },
]

const EMPTY_SOURCE_CARDS: SourceCard[] = []
const SOURCE_FETCH_APPROVAL_LIMIT = 1000

type PrepareState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; error: string }
  | { status: 'success'; report: SourcePrepareResponse; artifactPathApplied: boolean }

function goalLabel(goal: SourceUse): string {
  return SOURCE_GOALS.find((item) => item.value === goal)?.label ?? goal
}

function formatList(items: string[]): string {
  return items.length > 0 ? items.join(', ') : 'None listed'
}

function defaultOutputDir(sourceId: string, goal: SourceUse): string {
  return `data/sources/${sourceId}/${goal}`
}

function responseError(error: unknown): string {
  if (typeof error === 'string') return error
  try {
    return JSON.stringify(error)
  } catch {
    return String(error)
  }
}

function SourceStatus({ source }: { source: SourceCard }) {
  if (source.eval_only) {
    return (
      <span className="tag text-[10px] text-status-warning">
        <span>Eval only</span>
      </span>
    )
  }
  if (source.training_eligible) {
    return (
      <span className="tag text-[10px] text-status-success">
        <span>Training eligible</span>
      </span>
    )
  }
  return (
    <span className="tag text-[10px] text-text-muted">
      <span>Review required</span>
    </span>
  )
}

function SourceCardButton({
  source,
  selected,
  recommended,
  onSelect,
}: {
  source: SourceCard
  selected: boolean
  recommended: boolean
  onSelect: (sourceId: string) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(source.id)}
      className={clsx(
        'card p-4 text-left transition-press',
        selected ? 'border-accent bg-accent-light' : 'hover:border-border'
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p
            className={clsx(
              'font-mono text-sm font-bold truncate',
              selected ? 'text-accent-dark' : 'text-text-primary'
            )}
          >
            {source.name}
          </p>
          <p className="text-xs text-text-secondary mt-1 line-clamp-2">{source.task_family}</p>
        </div>
        <SourceStatus source={source} />
      </div>
      <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-text-muted mt-3">
        {source.domain} / {source.adapter}
      </p>
      <p className="text-xs text-text-muted mt-2">
        Artifacts: {formatList(source.artifact_types)}
      </p>
      {recommended ? (
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-accent-dark mt-3">
          Recommended for current goal
        </p>
      ) : null}
    </button>
  )
}

function SourceDetails({ source }: { source: SourceCard }) {
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="font-brand text-xl text-text-primary">{source.name}</h3>
          <p className="font-mono text-xs text-text-muted mt-1">{source.id}</p>
        </div>
        <SourceStatus source={source} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4 text-sm">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">Domain</p>
          <p className="text-text-primary">{source.domain}</p>
        </div>
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">License</p>
          <p className="text-text-primary">{source.license}</p>
        </div>
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">Input</p>
          <p className="text-text-primary">{source.input_format}</p>
        </div>
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">Size</p>
          <p className="text-text-primary">{source.data_size ?? 'Not listed'}</p>
        </div>
      </div>

      <div className="border-t-2 border-border mt-4 pt-4 space-y-3 text-sm text-text-secondary">
        <p>
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
            Recommended:
          </span>{' '}
          {formatList(source.recommended_uses.map(goalLabel))}
        </p>
        <p>
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
            Avoid:
          </span>{' '}
          {formatList(source.not_recommended_for.map(goalLabel))}
        </p>
        <p>
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
            Risks:
          </span>{' '}
          {formatList(source.known_risks)}
        </p>
        <p>
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
            Split:
          </span>{' '}
          {source.split_policy || 'No split policy listed'}
        </p>
        <p>
          <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
            Decontam:
          </span>{' '}
          {source.decontam_notes || 'No decontamination notes listed'}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2 mt-4">
        <a
          href={source.homepage}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary flex items-center gap-2 text-xs"
        >
          Homepage
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
        {source.repo ? (
          <a
            href={source.repo}
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary flex items-center gap-2 text-xs"
          >
            Repository
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        ) : null}
      </div>
    </div>
  )
}

export function SourceLibraryPanel() {
  const [prepareState, setPrepareState] = useState<PrepareState>({ status: 'idle' })
  const [selectedSourceId, setSelectedSourceId] = useState('')
  const [goal, setGoal] = useState<SourceUse>('dpo')
  const [domainFilter, setDomainFilter] = useState('')
  const [includeEvalOnly, setIncludeEvalOnly] = useState(false)
  const [inputPath, setInputPath] = useState('')
  const [fetchRemote, setFetchRemote] = useState(false)
  const [splitInput, setSplitInput] = useState('train')
  const [subsetInput, setSubsetInput] = useState('')
  const [revisionInput, setRevisionInput] = useState('')
  const [outputDir, setOutputDir] = useState('')
  const [limitInput, setLimitInput] = useState('')
  const [fetchApprovalReason, setFetchApprovalReason] = useState('')
  const [forceRefresh, setForceRefresh] = useState(false)
  const [allowEvalOnly, setAllowEvalOnly] = useState(false)
  const [overrideReason, setOverrideReason] = useState('')
  const setDatasetPathOverride = useTrainingStore((state) => state.setDatasetPathOverride)

  const {
    data: catalog,
    loading: catalogLoading,
    error: catalogError,
    refresh: refreshCatalog,
  } = useSessionResource(sourceCatalogResource)

  useEffect(() => {
    if (catalog) setSelectedSourceId((current) => current || catalog.sources[0]?.id || '')
  }, [catalog])

  const sources = catalog?.sources ?? EMPTY_SOURCE_CARDS

  const domains = useMemo(() => {
    const names = new Set<string>()
    for (const source of sources) names.add(source.domain)
    return Array.from(names).sort()
  }, [sources])

  const selectedSource = useMemo(
    () => sources.find((source) => source.id === selectedSourceId) ?? null,
    [selectedSourceId, sources]
  )
  const canFetchRemote = selectedSource?.huggingface_id !== null && selectedSource?.huggingface_id !== undefined

  useEffect(() => {
    if (selectedSourceId) setOutputDir(defaultOutputDir(selectedSourceId, goal))
  }, [selectedSourceId, goal])

  useEffect(() => {
    if (!canFetchRemote) setFetchRemote(false)
  }, [canFetchRemote])

  const recommendationsKey = JSON.stringify([domainFilter, goal, includeEvalOnly])
  const {
    data: recommendations,
    loading: recommendationsLoading,
    refreshing: recommendationsRefreshing,
    error: recommendationsError,
    refresh: refreshRecommendations,
  } = useKeyedSessionResource(sourceRecommendationsResource, recommendationsKey)

  const recommendedSourceIds = useMemo(() => {
    if (!recommendations) return new Set<string>()
    return new Set(recommendations.map((item) => item.source.id))
  }, [recommendations])

  const visibleSources = useMemo(() => {
    if (!domainFilter) return sources
    return sources.filter((source) => source.domain === domainFilter)
  }, [domainFilter, sources])

  const handlePrepare = useCallback(async () => {
    if (!selectedSourceId) return
    const trimmedLimit = limitInput.trim()
    const limit = trimmedLimit ? Number(trimmedLimit) : undefined
    if (limit !== undefined && (!Number.isFinite(limit) || limit < 1)) {
      setPrepareState({ status: 'error', error: 'Limit must be a positive number.' })
      return
    }
    if (
      fetchRemote &&
      limit !== undefined &&
      limit > SOURCE_FETCH_APPROVAL_LIMIT &&
      !fetchApprovalReason.trim()
    ) {
      setPrepareState({
        status: 'error',
        error: `Fetching more than ${SOURCE_FETCH_APPROVAL_LIMIT} records requires an approval reason.`,
      })
      return
    }
    setPrepareState({ status: 'loading' })
    const response = await sourcesApi.prepare(selectedSourceId, {
      goal,
      output_dir: outputDir.trim() || undefined,
      input_path: fetchRemote ? undefined : inputPath.trim() || undefined,
      fetch: fetchRemote,
      split: fetchRemote ? splitInput.trim() || 'train' : undefined,
      subset: fetchRemote ? subsetInput.trim() || undefined : undefined,
      revision: fetchRemote ? revisionInput.trim() || undefined : undefined,
      limit,
      fetch_approval_reason: fetchRemote ? fetchApprovalReason.trim() || undefined : undefined,
      force_refresh: fetchRemote ? forceRefresh : undefined,
      allow_eval_only: allowEvalOnly,
      override_reason: allowEvalOnly ? overrideReason.trim() || undefined : undefined,
    })
    if (response.ok && response.data) {
      setPrepareState({
        status: 'success',
        report: response.data,
        artifactPathApplied: false,
      })
    } else {
      setPrepareState({ status: 'error', error: responseError(response.error) })
    }
  }, [
    selectedSourceId,
    limitInput,
    goal,
    outputDir,
    inputPath,
    fetchRemote,
    splitInput,
    subsetInput,
    revisionInput,
    fetchApprovalReason,
    forceRefresh,
    allowEvalOnly,
    overrideReason,
  ])

  const firstArtifactPath =
    prepareState.status === 'success' ? prepareState.report.artifacts?.[0]?.path ?? null : null

  const handleUseArtifact = useCallback(() => {
    if (prepareState.status !== 'success') return
    const path = prepareState.report.artifacts?.[0]?.path
    if (!path) return
    setDatasetPathOverride(path)
    setPrepareState({ ...prepareState, artifactPathApplied: true })
  }, [prepareState, setDatasetPathOverride])

  const trainingGoalBlocked =
    selectedSource !== null &&
    selectedSource.eval_only &&
    goal !== 'evaluation' &&
    goal !== 'raw_reference' &&
    !allowEvalOnly
  const parsedLimit = Number(limitInput.trim())
  const largeRemoteFetchSelected =
    fetchRemote &&
    limitInput.trim() !== '' &&
    Number.isFinite(parsedLimit) &&
    parsedLimit > SOURCE_FETCH_APPROVAL_LIMIT

  if (catalogLoading) {
    return (
      <div className="p-6 max-w-6xl mx-auto">
        <div className="card p-8 text-center">
          <Loader2 className="w-8 h-8 animate-spin text-accent mx-auto mb-3" />
          <p className="font-mono text-xs uppercase tracking-[0.15em] text-text-muted">
            Loading sources
          </p>
        </div>
      </div>
    )
  }

  if (!catalog) {
    return (
      <div className="p-6 max-w-5xl mx-auto">
        <div className="card p-4 border-l-4 border-l-status-error bg-status-error/10">
          <p className="font-mono text-xs text-status-error">{catalogError || 'Failed to load sources'}</p>
          <button
            onClick={() => void refreshCatalog()}
            className="btn-secondary flex items-center gap-2 mt-3 text-xs"
          >
            <RefreshCw className="w-4 h-4" />
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="mb-6 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="font-brand text-3xl text-text-primary flex items-center gap-2">
            <Database className="w-6 h-6 text-accent" />
            Source Library
          </h2>
          <p className="font-mono text-xs text-text-muted mt-1">
            Public source cards, provenance guardrails, and local artifact preparation
          </p>
        </div>
        <span className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted">
          {catalog.count} sources / {catalog.ok ? 'catalog valid' : 'review catalog'}
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.05fr)_minmax(360px,0.95fr)] gap-5">
        <div className="space-y-5">
          <div className="card p-4">
            <div className="flex items-center justify-between mb-4 gap-3">
              <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary">
                Match Sources
              </h3>
              <button
                onClick={() => void refreshRecommendations()}
                className="font-mono text-[10px] uppercase tracking-[0.12em] text-text-muted hover:text-accent-dark flex items-center gap-1 transition-colors"
                title="Refresh source recommendations"
              >
                <RefreshCw
                  className={clsx(
                    'w-3 h-3',
                    (recommendationsLoading || recommendationsRefreshing) && 'animate-spin'
                  )}
                />
                Recommend
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Goal
                </span>
                <select
                  value={goal}
                  onChange={(event) => setGoal(event.target.value as SourceUse)}
                  className="input w-full"
                >
                  {SOURCE_GOALS.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label} - {item.detail}
                    </option>
                  ))}
                </select>
              </label>

              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Domain
                </span>
                <select
                  value={domainFilter}
                  onChange={(event) => setDomainFilter(event.target.value)}
                  className="input w-full"
                >
                  <option value="">All domains</option>
                  {domains.map((domain) => (
                    <option key={domain} value={domain}>
                      {domain}
                    </option>
                  ))}
                </select>
              </label>

              <button
                type="button"
                onClick={() => setIncludeEvalOnly((current) => !current)}
                className={clsx(
                  'mt-5 flex items-center justify-center gap-2 px-3 py-2 border-2 transition-press font-mono text-[11px] uppercase tracking-[0.12em]',
                  includeEvalOnly
                    ? 'border-status-warning bg-status-warning/10 text-status-warning'
                    : 'border-border text-text-secondary bg-background-card'
                )}
                title="Include eval-only benchmark sources in recommendations"
              >
                {includeEvalOnly ? (
                  <ShieldAlert className="w-4 h-4" />
                ) : (
                  <ShieldCheck className="w-4 h-4" />
                )}
                Eval-only {includeEvalOnly ? 'shown' : 'hidden'}
              </button>
            </div>

            {recommendationsError ? (
              <p className="font-mono text-xs text-status-error mt-3">{recommendationsError}</p>
            ) : null}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {visibleSources.map((source) => (
              <SourceCardButton
                key={source.id}
                source={source}
                selected={source.id === selectedSourceId}
                recommended={recommendedSourceIds.has(source.id)}
                onSelect={setSelectedSourceId}
              />
            ))}
          </div>
        </div>

        <div className="space-y-5">
          {selectedSource ? <SourceDetails source={selectedSource} /> : null}

          <div className="card p-4">
            <div className="flex items-center gap-2 mb-4">
              <FileJson className="w-4 h-4 text-accent" />
              <h3 className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-secondary">
                Prepare Artifacts
              </h3>
            </div>

            <div className="space-y-4">
              <button
                type="button"
                onClick={() => setFetchRemote((current) => !current)}
                disabled={!canFetchRemote}
                className={clsx(
                  'w-full flex items-center justify-center gap-2 px-3 py-2 border-2 transition-press font-mono text-[11px] uppercase',
                  fetchRemote
                    ? 'border-accent bg-accent-light text-accent-dark'
                    : 'border-border text-text-secondary bg-background-card',
                  !canFetchRemote ? 'opacity-60 cursor-not-allowed' : ''
                )}
                title={
                  canFetchRemote
                    ? 'Fetch Hugging Face source records before preparing artifacts'
                    : 'This source card does not define a Hugging Face dataset'
                }
              >
                {fetchRemote ? (
                  <CloudDownload className="w-4 h-4" />
                ) : (
                  <FileJson className="w-4 h-4" />
                )}
                {fetchRemote ? 'Hugging Face fetch on' : canFetchRemote ? 'Use local input' : 'Local input only'}
              </button>

              {fetchRemote ? (
                <>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <label className="block">
                      <span className="block font-mono text-[11px] uppercase text-text-muted mb-1">
                        Split
                      </span>
                      <input
                        value={splitInput}
                        onChange={(event) => setSplitInput(event.target.value)}
                        placeholder="train"
                        className="input w-full font-mono text-xs"
                      />
                    </label>

                    <label className="block">
                      <span className="block font-mono text-[11px] uppercase text-text-muted mb-1">
                        Subset
                      </span>
                      <input
                        value={subsetInput}
                        onChange={(event) => setSubsetInput(event.target.value)}
                        placeholder="Optional"
                        className="input w-full font-mono text-xs"
                      />
                    </label>

                    <label className="block">
                      <span className="block font-mono text-[11px] uppercase text-text-muted mb-1">
                        Revision
                      </span>
                      <input
                        value={revisionInput}
                        onChange={(event) => setRevisionInput(event.target.value)}
                        placeholder="Optional"
                        className="input w-full font-mono text-xs"
                      />
                    </label>
                  </div>

                  <button
                    type="button"
                    onClick={() => setForceRefresh((current) => !current)}
                    className={clsx(
                      'w-full flex items-center justify-center gap-2 px-3 py-2 border-2 transition-press font-mono text-[11px] uppercase tracking-[0.12em]',
                      forceRefresh
                        ? 'border-status-warning bg-status-warning/10 text-status-warning'
                        : 'border-border text-text-secondary bg-background-card'
                    )}
                    title="Bypass a matching local source fetch cache"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Cache {forceRefresh ? 'refresh' : 'reuse'}
                  </button>

                  <label className="block">
                    <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                      Fetch approval reason
                    </span>
                    <textarea
                      value={fetchApprovalReason}
                      onChange={(event) => setFetchApprovalReason(event.target.value)}
                      className="input w-full min-h-20 text-sm"
                      placeholder={`Required above ${SOURCE_FETCH_APPROVAL_LIMIT} records`}
                    />
                  </label>

                  {largeRemoteFetchSelected && !fetchApprovalReason.trim() ? (
                    <div className="border-2 border-status-warning bg-status-warning/10 p-3">
                      <p className="font-mono text-xs text-status-warning flex items-center gap-2">
                        <AlertCircle className="w-4 h-4" />
                        Larger source fetches need an approval reason.
                      </p>
                    </div>
                  ) : null}
                </>
              ) : (
                <label className="block">
                  <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                    Local JSON/JSONL input
                  </span>
                  <input
                    value={inputPath}
                    onChange={(event) => setInputPath(event.target.value)}
                    placeholder="Optional path to source records"
                    className="input w-full font-mono text-xs"
                  />
                </label>
              )}

              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Output directory
                </span>
                <input
                  value={outputDir}
                  onChange={(event) => setOutputDir(event.target.value)}
                  className="input w-full font-mono text-xs"
                />
              </label>

              <label className="block">
                <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                  Limit
                </span>
                <input
                  type="number"
                  min={1}
                  value={limitInput}
                  onChange={(event) => setLimitInput(event.target.value)}
                  placeholder="Optional"
                  className="input w-full"
                />
              </label>

              <button
                type="button"
                onClick={() => setAllowEvalOnly((current) => !current)}
                className={clsx(
                  'w-full flex items-center justify-center gap-2 px-3 py-2 border-2 transition-press font-mono text-[11px] uppercase tracking-[0.12em]',
                  allowEvalOnly
                    ? 'border-status-warning bg-status-warning/10 text-status-warning'
                    : 'border-border text-text-secondary bg-background-card'
                )}
                title="Allow an eval-only source for a training goal and record a reason"
              >
                {allowEvalOnly ? (
                  <ShieldAlert className="w-4 h-4" />
                ) : (
                  <ShieldCheck className="w-4 h-4" />
                )}
                Eval-only override {allowEvalOnly ? 'on' : 'off'}
              </button>

              {allowEvalOnly ? (
                <label className="block">
                  <span className="block font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-1">
                    Override reason
                  </span>
                  <textarea
                    value={overrideReason}
                    onChange={(event) => setOverrideReason(event.target.value)}
                    className="input w-full min-h-20 text-sm"
                    placeholder="Required when overriding eval-only guardrails"
                  />
                </label>
              ) : null}

              {trainingGoalBlocked ? (
                <div className="border-2 border-status-warning bg-status-warning/10 p-3">
                  <p className="font-mono text-xs text-status-warning flex items-center gap-2">
                    <AlertCircle className="w-4 h-4" />
                    This source is eval-only for training goals.
                  </p>
                </div>
              ) : null}

              <button
                onClick={handlePrepare}
                disabled={!selectedSourceId || prepareState.status === 'loading'}
                className="btn-primary flex items-center justify-center gap-2 w-full"
              >
                {prepareState.status === 'loading' ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <FileJson className="w-4 h-4" />
                )}
                Prepare {fetchRemote ? 'Remote Artifacts' : inputPath.trim() ? 'Artifacts' : 'Manifest'}
              </button>
            </div>
          </div>

          {prepareState.status === 'error' ? (
            <div className="card p-4 border-l-4 border-l-status-error bg-status-error/10">
              <p className="font-mono text-xs text-status-error">{prepareState.error}</p>
            </div>
          ) : null}

          {prepareState.status === 'success' ? (
            <div className="card p-4">
              <div className="flex items-center justify-between gap-3 mb-3">
                <div>
                  <h3 className="font-brand text-lg text-text-primary">Prepared</h3>
                  <p className="font-mono text-xs text-text-muted">
                    {prepareState.report.source_id ?? prepareState.report.source?.id} /{' '}
                    {prepareState.report.goal ? goalLabel(prepareState.report.goal) : goalLabel(goal)}
                  </p>
                </div>
                {prepareState.report.ok ? (
                  <CheckCircle2 className="w-5 h-5 text-status-success" />
                ) : (
                  <AlertCircle className="w-5 h-5 text-status-error" />
                )}
              </div>

              <div className="space-y-2 font-mono text-xs text-text-secondary">
                {prepareState.report.output_dir ? <p>Output: {prepareState.report.output_dir}</p> : null}
                {prepareState.report.fetch_report ? (
                  <p>Fetched: {prepareState.report.fetch_report.records_path}</p>
                ) : null}
                {prepareState.report.fetch_report?.report_path ? (
                  <p>Fetch report: {prepareState.report.fetch_report.report_path}</p>
                ) : null}
                {prepareState.report.fetch_report?.cache_hit !== undefined ? (
                  <p>Fetch cache: {prepareState.report.fetch_report.cache_hit ? 'hit' : 'miss'}</p>
                ) : null}
                {prepareState.report.fetch_report?.approval_required ? (
                  <p>
                    Fetch approval:{' '}
                    {prepareState.report.fetch_report.approval_granted ? 'recorded' : 'required'}
                  </p>
                ) : null}
                {prepareState.report.report_path ? <p>Report: {prepareState.report.report_path}</p> : null}
                {prepareState.report.manifest_path ? <p>Manifest: {prepareState.report.manifest_path}</p> : null}
                {prepareState.report.source_manifest?.manifest_path ? (
                  <p>Manifest: {prepareState.report.source_manifest.manifest_path}</p>
                ) : null}
                {prepareState.report.record_count !== undefined ? (
                  <p>
                    Records: {prepareState.report.record_count} / converted:{' '}
                    {prepareState.report.converted_count ?? 0}
                  </p>
                ) : null}
                {prepareState.report.source_schema_mapping ? (
                  <p>
                    Schema mapper: {prepareState.report.source_schema_mapping.mapper} / normalized:{' '}
                    {prepareState.report.source_schema_mapping.normalized_records}
                  </p>
                ) : null}
              </div>

              {prepareState.report.artifacts && prepareState.report.artifacts.length > 0 ? (
                <div className="border-t-2 border-border mt-4 pt-4">
                  <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-text-muted mb-2">
                    Artifacts
                  </p>
                  <ul className="space-y-2">
                    {prepareState.report.artifacts.map((artifact) => (
                      <li key={`${artifact.artifact_type}-${artifact.path}`} className="text-sm text-text-secondary">
                        <span className="font-mono text-text-primary">{artifact.artifact_type}</span> -{' '}
                        {artifact.path}
                        {artifact.validation ? (
                          <span
                            className={clsx(
                              'font-mono text-xs ml-2',
                              artifact.validation.ok ? 'text-status-success' : 'text-status-error'
                            )}
                          >
                            validation {artifact.validation.ok ? 'ok' : 'failed'}
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {firstArtifactPath ? (
                <button
                  onClick={handleUseArtifact}
                  className="btn-secondary flex items-center gap-2 mt-4 text-xs"
                >
                  {prepareState.artifactPathApplied ? 'Training path set' : 'Use first artifact in Training'}
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
