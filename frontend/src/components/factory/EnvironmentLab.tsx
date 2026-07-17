import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  AlertCircle,
  Bot,
  Boxes,
  CheckCircle2,
  ClipboardPaste,
  Database,
  FileJson,
  FolderInput,
  Gauge,
  GitCompareArrows,
  Loader2,
  Package,
  PlayCircle,
  Server,
  ShieldCheck,
  Shuffle,
  Terminal,
  Upload,
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  evalAdvancedApi,
  environmentApi,
  type DPPOSmokeLaunchPlan,
  type DPPOTrainLogprobsSpec,
  type EnvironmentAttemptPayload,
  type EnvironmentCanarySuiteResponse,
  type EnvironmentCommandAttemptPayload,
  type EnvironmentDppoReplaySummary,
  type EnvironmentHoldoutComparisonResponse,
  type EnvironmentHoldoutGateResponse,
  type EnvironmentMixReport,
  type EnvironmentNormalizeResponse,
  type EnvironmentPassKResponse,
  type EnvironmentPipelinesResponse,
  type EnvironmentRolloutPassKResponse,
  type EnvironmentSpuriousRewardControlResponse,
  type TerminalEnvironmentSpec,
} from '../../services/api'
import { BaseModelSelect } from '../common/BaseModelSelect'
import { ModelSelect } from '../common/ModelSelect'

type LabStatus =
  | { kind: 'idle'; message: string }
  | { kind: 'loading'; message: string }
  | { kind: 'ready'; message: string }
  | { kind: 'error'; message: string }

const EMPTY_ENVIRONMENTS: TerminalEnvironmentSpec[] = []

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isAttemptRecord(value: unknown): value is EnvironmentAttemptPayload {
  return (
    isRecord(value) &&
    typeof value.environment_id === 'string' &&
    typeof value.attempt_index === 'number' &&
    typeof value.passed === 'boolean'
  )
}

function isCommandAttemptRecord(value: unknown): value is EnvironmentCommandAttemptPayload {
  return (
    isRecord(value) &&
    typeof value.environment_id === 'string' &&
    typeof value.attempt_index === 'number' &&
    Array.isArray(value.commands) &&
    value.commands.every((command) => typeof command === 'string')
  )
}

function isDppoTrainLogprobsRecord(value: unknown): value is DPPOTrainLogprobsSpec {
  return (
    isRecord(value) &&
    typeof value.environment_id === 'string' &&
    typeof value.attempt_index === 'number' &&
    Array.isArray(value.token_logprobs) &&
    value.token_logprobs.every((logprob) => typeof logprob === 'number')
  )
}

function coerceRecords(payload: unknown): Record<string, unknown>[] {
  if (Array.isArray(payload)) return payload.filter(isRecord)
  if (isRecord(payload) && Array.isArray(payload.data)) return payload.data.filter(isRecord)
  if (isRecord(payload)) return [payload]
  return []
}

function parseEnvironmentRecords(text: string, fileName: string): Record<string, unknown>[] {
  const lowerName = fileName.toLowerCase()
  if (lowerName.endsWith('.jsonl') || lowerName.endsWith('.ndjson')) {
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter(isRecord)
  }

  try {
    return coerceRecords(JSON.parse(text) as unknown)
  } catch (error) {
    const records = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter(isRecord)
    if (records.length > 0) return records
    throw error
  }
}

function coerceAttemptRecords(payload: unknown): EnvironmentAttemptPayload[] {
  if (Array.isArray(payload)) return payload.filter(isAttemptRecord)
  if (isRecord(payload) && Array.isArray(payload.attempts)) {
    return payload.attempts.filter(isAttemptRecord)
  }
  if (isAttemptRecord(payload)) return [payload]
  return []
}

function coerceCommandAttemptRecords(payload: unknown): EnvironmentCommandAttemptPayload[] {
  if (Array.isArray(payload)) return payload.filter(isCommandAttemptRecord)
  if (isRecord(payload) && Array.isArray(payload.command_attempts)) {
    return payload.command_attempts.filter(isCommandAttemptRecord)
  }
  if (isCommandAttemptRecord(payload)) return [payload]
  return []
}

function coerceDppoTrainLogprobsRecords(payload: unknown): DPPOTrainLogprobsSpec[] {
  if (Array.isArray(payload)) return payload.filter(isDppoTrainLogprobsRecord)
  if (isRecord(payload) && Array.isArray(payload.train_logprobs)) {
    return payload.train_logprobs.filter(isDppoTrainLogprobsRecord)
  }
  if (isDppoTrainLogprobsRecord(payload)) return [payload]
  return []
}

function parseAttemptRecords(text: string): EnvironmentAttemptPayload[] {
  const trimmed = text.trim()
  if (trimmed.length === 0) return []
  try {
    return coerceAttemptRecords(JSON.parse(trimmed) as unknown)
  } catch {
    return trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter(isAttemptRecord)
  }
}

function parseCommandAttemptRecords(text: string): EnvironmentCommandAttemptPayload[] {
  const trimmed = text.trim()
  if (trimmed.length === 0) return []
  try {
    return coerceCommandAttemptRecords(JSON.parse(trimmed) as unknown)
  } catch {
    return trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter(isCommandAttemptRecord)
  }
}

function parseDppoTrainLogprobs(text: string): DPPOTrainLogprobsSpec[] {
  const trimmed = text.trim()
  if (trimmed.length === 0) return []
  try {
    return coerceDppoTrainLogprobsRecords(JSON.parse(trimmed) as unknown)
  } catch {
    return trimmed
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter(isDppoTrainLogprobsRecord)
  }
}

function parseKValues(text: string): number[] {
  const values = text
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isInteger(value) && value > 0)
  return values.length > 0 ? Array.from(new Set(values)).sort((a, b) => a - b) : [1]
}

function parseCommaValues(text: string): string[] {
  return Array.from(
    new Set(
      text
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
    )
  )
}

function baselineAttempts(
  environments: TerminalEnvironmentSpec[],
  nSamples: number
): EnvironmentAttemptPayload[] {
  return environments.flatMap((environment) =>
    Array.from({ length: nSamples }, (_, attemptIndex) => ({
      environment_id: environment.id,
      attempt_index: attemptIndex,
      passed: false,
      reward: 0,
      verifier_status: 'failed',
      timeout: false,
      tool_calls: 0,
      tokens: 0,
      action_tokens: 0,
      observation_tokens: 0,
    }))
  )
}

function seededCommandAttempts(
  environment: TerminalEnvironmentSpec | null
): EnvironmentCommandAttemptPayload[] {
  if (!environment) return []
  return [
    {
      environment_id: environment.id,
      attempt_index: 0,
      commands: ['python -c "from pathlib import Path; print(Path.cwd())"'],
      metadata: { source: 'environment_lab_seed' },
    },
  ]
}

function formatPercent(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return '0%'
  return `${Math.round(value * 100)}%`
}

function formatSignedPercent(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return '+0%'
  const rounded = Math.round(value * 100)
  return `${rounded >= 0 ? '+' : ''}${rounded}%`
}

function topEntries(values: Record<string, number> | undefined): Array<[string, number]> {
  return Object.entries(values || {})
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 5)
}

function metadataTamperHint(metadata: Record<string, unknown>): string {
  const audit = isRecord(metadata.tamper_audit) ? metadata.tamper_audit : null
  const tamperedPaths =
    audit && Array.isArray(audit.tampered_paths)
      ? audit.tampered_paths.filter((path): path is string => typeof path === 'string')
      : []
  const missingPaths =
    audit && Array.isArray(audit.missing_paths)
      ? audit.missing_paths.filter((path): path is string => typeof path === 'string')
      : []
  const changedPaths = [...tamperedPaths, ...missingPaths]
  return changedPaths.length > 0 ? changedPaths.join(', ') : 'protected file changed'
}

function MetricTile({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div className="border-brutal border-border rounded-brutal bg-background-card p-4">
      <p className="font-mono text-xs uppercase text-text-muted">{label}</p>
      <p className="font-brand text-3xl text-text-primary mt-1">{value}</p>
      {hint && <p className="text-xs text-text-secondary mt-1 truncate">{hint}</p>}
    </div>
  )
}

function DistributionList({
  title,
  values,
}: {
  title: string
  values: Record<string, number> | undefined
}) {
  const entries = topEntries(values)
  return (
    <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
      <p className="font-mono text-xs uppercase text-text-muted mb-2">{title}</p>
      {entries.length === 0 ? (
        <p className="font-mono text-xs text-text-muted">No values</p>
      ) : (
        <div className="space-y-2">
          {entries.map(([name, count]) => (
            <div key={name} className="flex items-center justify-between gap-3">
              <span className="text-sm text-text-primary truncate">{name}</span>
              <span className="font-mono text-xs text-text-muted">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function reportSummary(report: EnvironmentMixReport | null) {
  if (!report) {
    return {
      total: '0',
      domains: '0',
      skills: '0',
      balance: '0%',
    }
  }
  return {
    total: String(report.total),
    domains: String(Object.keys(report.domain_distribution || {}).length),
    skills: String(Object.keys(report.skill_distribution || {}).length),
    balance: formatPercent(report.axis_balance?.domain),
  }
}

function SourcePanel({
  sourcePath,
  preserveRaw,
  onPathChange,
  onPreserveRawChange,
  onFileChange,
  onImportPath,
  loading,
}: {
  sourcePath: string
  preserveRaw: boolean
  onPathChange: (value: string) => void
  onPreserveRawChange: (value: boolean) => void
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void
  onImportPath: () => void
  loading: boolean
}) {
  return (
    <section className="card p-4 space-y-4">
      <div className="flex items-center gap-2">
        <FileJson className="w-4 h-4 text-accent" />
        <h3 className="font-brand text-lg text-text-primary">Import</h3>
      </div>

      <label className="block">
        <span className="block font-mono text-xs uppercase text-text-muted mb-1">
          Local JSON or JSONL
        </span>
        <input
          type="file"
          accept=".json,.jsonl,.ndjson,application/json"
          onChange={onFileChange}
          className="input w-full text-sm"
        />
      </label>

      <label className="block">
        <span className="block font-mono text-xs uppercase text-text-muted mb-1">
          Server path
        </span>
        <div className="flex gap-2">
          <input
            type="text"
            value={sourcePath}
            onChange={(event) => onPathChange(event.target.value)}
            placeholder="tests/fixtures/tmax_envs/tmax_demo_001.jsonl"
            className="input w-full font-mono text-xs"
          />
          <button
            type="button"
            onClick={onImportPath}
            disabled={loading || sourcePath.trim().length === 0}
            className="btn-icon flex items-center justify-center"
            title="Import server path"
          >
            <FolderInput className="w-4 h-4" />
          </button>
        </div>
      </label>

      <label className="flex items-center gap-2 text-sm text-text-secondary">
        <input
          type="checkbox"
          checked={preserveRaw}
          onChange={(event) => onPreserveRawChange(event.target.checked)}
          className="rounded-brutal"
        />
        Preserve raw records
      </label>
    </section>
  )
}

function EnvironmentList({
  environments,
  selectedId,
  onSelect,
}: {
  environments: TerminalEnvironmentSpec[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  if (environments.length === 0) {
    return (
      <div className="card p-8 text-center">
        <Boxes className="w-9 h-9 text-text-muted mx-auto mb-3" />
        <h3 className="font-brand text-xl text-text-primary">No environments loaded</h3>
        <p className="font-mono text-xs text-text-muted mt-1">Import JSON or JSONL to inspect the mix</p>
      </div>
    )
  }

  return (
    <div className="card p-0 overflow-hidden">
      <div className="grid grid-cols-[minmax(0,1.6fr)_120px_160px_120px] gap-3 px-4 py-2 border-b-2 border-border bg-background-secondary font-mono text-xs uppercase text-text-muted">
        <span>Environment</span>
        <span>Domain</span>
        <span>Skills</span>
        <span>Verifier</span>
      </div>
      <div className="divide-y-2 divide-border-subtle">
        {environments.map((env) => (
          <button
            key={env.id}
            type="button"
            onClick={() => onSelect(env.id)}
            className={clsx(
              'w-full grid grid-cols-[minmax(0,1.6fr)_120px_160px_120px] gap-3 px-4 py-3 text-left transition-colors',
              selectedId === env.id ? 'bg-accent-light' : 'bg-background-card hover:bg-background-secondary'
            )}
          >
            <span className="min-w-0">
              <span className="block font-mono text-sm font-semibold text-text-primary truncate">
                {env.id}
              </span>
              <span className="block text-xs text-text-muted truncate">{env.instruction}</span>
            </span>
            <span className="font-mono text-xs text-text-secondary truncate">{env.domain}</span>
            <span className="text-xs text-text-secondary truncate">{env.skills.join(', ') || 'none'}</span>
            <span className="font-mono text-xs text-text-secondary truncate">
              {env.verifier?.kind || 'unknown'}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

function EnvironmentDetail({
  environment,
  outputDir,
  overwrite,
  materializeResult,
  onOutputDirChange,
  onOverwriteChange,
  onMaterialize,
  loading,
}: {
  environment: TerminalEnvironmentSpec | null
  outputDir: string
  overwrite: boolean
  materializeResult: string
  onOutputDirChange: (value: string) => void
  onOverwriteChange: (value: boolean) => void
  onMaterialize: () => void
  loading: boolean
}) {
  if (!environment) return null

  return (
    <section className="card p-4 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="font-brand text-lg text-text-primary truncate">{environment.id}</h3>
          <p className="text-sm text-text-secondary mt-1">{environment.instruction}</p>
        </div>
        <span className="tag shrink-0">
          <span>{environment.source}</span>
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
          <p className="font-mono text-xs uppercase text-text-muted">Verifier</p>
          <p className="font-mono text-xs text-text-primary mt-1 break-all">
            {environment.verifier?.command || 'none'}
          </p>
        </div>
        <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
          <p className="font-mono text-xs uppercase text-text-muted">Rollout</p>
          <p className="font-mono text-xs text-text-primary mt-1">
            {environment.rollout?.max_tool_calls ?? 0} tool calls
          </p>
        </div>
        <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
          <p className="font-mono text-xs uppercase text-text-muted">Files</p>
          <p className="font-mono text-xs text-text-primary mt-1">
            {Object.keys(environment.files || {}).length} embedded
          </p>
        </div>
      </div>

      {environment.axes.length > 0 && (
        <div>
          <p className="font-mono text-xs uppercase text-text-muted mb-2">Axes</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {environment.axes.slice(0, 8).map((axis) => (
              <div
                key={`${axis.name}:${axis.value}`}
                className="flex items-center justify-between gap-3 border-brutal border-border-subtle rounded-brutal bg-background-secondary px-3 py-2"
              >
                <span className="font-mono text-xs text-text-muted truncate">{axis.name}</span>
                <span className="text-xs text-text-primary truncate">{axis.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="border-t-2 border-border pt-4">
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-3">
          <label className="block">
            <span className="block font-mono text-xs uppercase text-text-muted mb-1">
              Output directory
            </span>
            <input
              type="text"
              value={outputDir}
              onChange={(event) => onOutputDirChange(event.target.value)}
              className="input w-full font-mono text-xs"
            />
          </label>
          <div className="flex items-end gap-3">
            <label className="flex items-center gap-2 text-sm text-text-secondary pb-2">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(event) => onOverwriteChange(event.target.checked)}
                className="rounded-brutal"
              />
              Overwrite
            </label>
            <button
              type="button"
              onClick={onMaterialize}
              disabled={loading || outputDir.trim().length === 0}
              className="btn-primary flex items-center gap-2"
            >
              <Package className="w-4 h-4" />
              Materialize
            </button>
          </div>
        </div>
        {materializeResult && (
          <p className="font-mono text-xs text-status-success mt-3 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {materializeResult}
          </p>
        )}
      </div>
    </section>
  )
}

function PassKPanel({
  environments,
  attemptText,
  kValuesText,
  result,
  loading,
  onAttemptTextChange,
  onKValuesTextChange,
  onSeedBaseline,
  onEvaluate,
}: {
  environments: TerminalEnvironmentSpec[]
  attemptText: string
  kValuesText: string
  result: EnvironmentPassKResponse | null
  loading: boolean
  onAttemptTextChange: (value: string) => void
  onKValuesTextChange: (value: string) => void
  onSeedBaseline: () => void
  onEvaluate: () => void
}) {
  const passEntries = useMemo(
    () => Object.entries(result?.report.pass_at_k || {}),
    [result]
  )
  const statusEntries = useMemo(
    () => Object.entries(result?.report.attempt_summary.verifier_status_distribution || {}),
    [result]
  )

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Environment Pass@k</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Verifier-backed attempt outcomes for loaded environments
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={kValuesText}
            onChange={(event) => onKValuesTextChange(event.target.value)}
            className="input w-28 font-mono text-xs"
            aria-label="Pass k values"
          />
          <button
            type="button"
            onClick={onSeedBaseline}
            disabled={loading || environments.length === 0}
            className="btn-icon flex items-center justify-center"
            title="Seed failed baseline attempts"
          >
            <ClipboardPaste className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={onEvaluate}
            disabled={loading || environments.length === 0 || attemptText.trim().length === 0}
            className="btn-primary flex items-center gap-2"
          >
            <Gauge className="w-4 h-4" />
            Compute
          </button>
        </div>
      </div>

      <textarea
        value={attemptText}
        onChange={(event) => onAttemptTextChange(event.target.value)}
        placeholder='[{"environment_id":"env_a","attempt_index":0,"passed":true,"tool_calls":3,"action_tokens":120,"observation_tokens":400}]'
        className="input w-full min-h-28 font-mono text-xs resize-y"
      />

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {passEntries.map(([label, value]) => (
            <MetricTile key={label} label={label} value={formatPercent(value)} hint="mean over envs" />
          ))}
          <MetricTile
            label="Attempts"
            value={String(result.report.n_attempts)}
            hint={`${result.report.n_environments} environments`}
          />
          <MetricTile
            label="Timeouts"
            value={formatPercent(result.report.attempt_summary.timeout_rate)}
            hint="attempt rate"
          />
        </div>
      ) : null}

      {result ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Token telemetry</p>
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              <span className="text-text-muted">action</span>
              <span className="text-text-primary text-right">
                {result.report.attempt_summary.mean_action_tokens ?? 'n/a'}
              </span>
              <span className="text-text-muted">observation</span>
              <span className="text-text-primary text-right">
                {result.report.attempt_summary.mean_observation_tokens ?? 'n/a'}
              </span>
            </div>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Verifier status</p>
            <div className="space-y-1">
              {statusEntries.length > 0 ? (
                statusEntries.map(([name, count]) => (
                  <div key={name} className="flex items-center justify-between gap-3 text-xs">
                    <span className="text-text-primary truncate">{name}</span>
                    <span className="font-mono text-text-muted">{count}</span>
                  </div>
                ))
              ) : (
                <p className="font-mono text-xs text-text-muted">No attempts</p>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </section>
  )
}

function HoldoutGatePanel({
  environments,
  attemptText,
  splitBy,
  fraction,
  seed,
  minPassAt1,
  maxTimeoutRate,
  maxTamperRate,
  requireNoContamination,
  modelId,
  recordToRegistry,
  result,
  loading,
  onSplitByChange,
  onFractionChange,
  onSeedChange,
  onMinPassAt1Change,
  onMaxTimeoutRateChange,
  onMaxTamperRateChange,
  onRequireNoContaminationChange,
  onModelIdChange,
  onRecordToRegistryChange,
  onRun,
}: {
  environments: TerminalEnvironmentSpec[]
  attemptText: string
  splitBy: string
  fraction: string
  seed: string
  minPassAt1: string
  maxTimeoutRate: string
  maxTamperRate: string
  requireNoContamination: boolean
  modelId: string
  recordToRegistry: boolean
  result: EnvironmentHoldoutGateResponse | null
  loading: boolean
  onSplitByChange: (value: string) => void
  onFractionChange: (value: string) => void
  onSeedChange: (value: string) => void
  onMinPassAt1Change: (value: string) => void
  onMaxTimeoutRateChange: (value: string) => void
  onMaxTamperRateChange: (value: string) => void
  onRequireNoContaminationChange: (value: boolean) => void
  onModelIdChange: (value: string) => void
  onRecordToRegistryChange: (value: boolean) => void
  onRun: () => void
}) {
  const gate = result?.result.gate || null
  const report = result?.result.report || null
  const split = result?.result.split || null
  const reasons = gate?.reasons || []
  const holdoutGroupsText = useMemo(
    () => split?.holdout_group_keys.join(', ') || 'none',
    [split]
  )
  const recordedText = useMemo(
    () => result?.recorded.join(', ') || 'not recorded',
    [result]
  )
  const canRun = environments.length > 0 && attemptText.trim().length > 0 && !loading

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Environment Holdout Gate</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Split manifest, leakage check, and pass@k release verdict
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <ShieldCheck className="w-4 h-4" />
          Gate
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Split by</span>
          <select
            value={splitBy}
            onChange={(event) => onSplitByChange(event.target.value)}
            className="input w-full font-mono text-xs"
          >
            <option value="task_family">Task family</option>
            <option value="domain">Domain</option>
            <option value="source">Source</option>
            <option value="source_uri">Source URI</option>
            <option value="repo">Repo</option>
            <option value="generator_seed">Seed</option>
          </select>
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Fraction</span>
          <input
            type="number"
            min={0.05}
            max={0.95}
            step={0.05}
            value={fraction}
            onChange={(event) => onFractionChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Seed</span>
          <input
            type="number"
            value={seed}
            onChange={(event) => onSeedChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Min p@1</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minPassAt1}
            onChange={(event) => onMinPassAt1Change(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Timeout max</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={maxTimeoutRate}
            onChange={(event) => onMaxTimeoutRateChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Tamper max</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={maxTamperRate}
            onChange={(event) => onMaxTamperRateChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      <label className="flex items-center gap-2 text-sm text-text-secondary">
        <input
          type="checkbox"
          checked={requireNoContamination}
          onChange={(event) => onRequireNoContaminationChange(event.target.checked)}
          className="rounded-brutal"
        />
        Require clean split
      </label>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-3 items-end">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Registry model</span>
          <input
            value={modelId}
            onChange={(event) => onModelIdChange(event.target.value)}
            className="input w-full font-mono text-xs"
            placeholder="optional model_id"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-text-secondary pb-2">
          <input
            type="checkbox"
            checked={recordToRegistry}
            onChange={(event) => onRecordToRegistryChange(event.target.checked)}
            className="rounded-brutal"
          />
          Record gate
        </label>
      </div>

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="Verdict"
            value={gate?.ship ? 'ship' : 'hold'}
            hint={reasons.length > 0 ? reasons[0] : 'gate clear'}
          />
          <MetricTile
            label="Holdout"
            value={String(split?.n_holdout ?? 0)}
            hint={`${split?.n_train ?? 0} train`}
          />
          <MetricTile
            label="Pass@1"
            value={formatPercent(report?.pass_at_k['pass@1'])}
            hint={`${report?.n_attempts ?? 0} attempts`}
          />
          <MetricTile
            label="Leakage"
            value={String(result.result.contamination.length)}
            hint={`${split?.holdout_hashes.length ?? 0} hashes`}
          />
          <MetricTile
            label="Registry"
            value={result.recorded.length > 0 ? String(result.recorded.length) : 'off'}
            hint={recordedText}
          />
        </div>
      ) : null}

      {result ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Holdout groups</p>
            <p className="font-mono text-xs text-text-primary break-words">
              {holdoutGroupsText}
            </p>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Gate reasons</p>
            {reasons.length > 0 ? (
              <div className="space-y-1">
                {reasons.map((reason) => (
                  <p key={reason} className="font-mono text-xs text-text-primary break-words">
                    {reason}
                  </p>
                ))}
              </div>
            ) : (
              <p className="font-mono text-xs text-text-primary">clear</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function HoldoutComparisonPanel({
  environments,
  baseAttemptText,
  candidateAttemptText,
  clusterBy,
  compareK,
  minDelta,
  minCandidatePassAt1,
  nResamples,
  requireCiExcludesZero,
  result,
  loading,
  onBaseAttemptTextChange,
  onClusterByChange,
  onCompareKChange,
  onMinDeltaChange,
  onMinCandidatePassAt1Change,
  onNResamplesChange,
  onRequireCiExcludesZeroChange,
  onRun,
}: {
  environments: TerminalEnvironmentSpec[]
  baseAttemptText: string
  candidateAttemptText: string
  clusterBy: string
  compareK: string
  minDelta: string
  minCandidatePassAt1: string
  nResamples: string
  requireCiExcludesZero: boolean
  result: EnvironmentHoldoutComparisonResponse | null
  loading: boolean
  onBaseAttemptTextChange: (value: string) => void
  onClusterByChange: (value: string) => void
  onCompareKChange: (value: string) => void
  onMinDeltaChange: (value: string) => void
  onMinCandidatePassAt1Change: (value: string) => void
  onNResamplesChange: (value: string) => void
  onRequireCiExcludesZeroChange: (value: boolean) => void
  onRun: () => void
}) {
  const gate = result?.result.gate || null
  const bootstrap = result?.result.bootstrap || null
  const reasons = gate?.reasons || []
  const canRun =
    environments.length > 0 &&
    baseAttemptText.trim().length > 0 &&
    candidateAttemptText.trim().length > 0 &&
    !loading
  const ciText =
    bootstrap !== null
      ? `${formatSignedPercent(bootstrap.ci_low)}..${formatSignedPercent(bootstrap.ci_high)}`
      : '+0%..+0%'

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <GitCompareArrows className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Holdout Comparison Gate</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Base vs candidate paired bootstrap over the current holdout split
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <GitCompareArrows className="w-4 h-4" />
          Compare
        </button>
      </div>

      <textarea
        value={baseAttemptText}
        onChange={(event) => onBaseAttemptTextChange(event.target.value)}
        placeholder='Base attempts: [{"environment_id":"env_a","attempt_index":0,"passed":false}]'
        className="input w-full min-h-24 font-mono text-xs resize-y"
      />

      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Cluster by</span>
          <select
            value={clusterBy}
            onChange={(event) => onClusterByChange(event.target.value)}
            className="input w-full font-mono text-xs"
          >
            <option value="task_family">Task family</option>
            <option value="domain">Domain</option>
            <option value="source">Source</option>
            <option value="source_uri">Source URI</option>
            <option value="repo">Repo</option>
            <option value="generator_seed">Seed</option>
          </select>
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Compare k</span>
          <input
            type="number"
            min={1}
            value={compareK}
            onChange={(event) => onCompareKChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Min delta</span>
          <input
            type="number"
            min={-1}
            max={1}
            step={0.05}
            value={minDelta}
            onChange={(event) => onMinDeltaChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Min cand p@1</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minCandidatePassAt1}
            onChange={(event) => onMinCandidatePassAt1Change(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Resamples</span>
          <input
            type="number"
            min={1}
            max={10000}
            value={nResamples}
            onChange={(event) => onNResamplesChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-text-secondary pt-6">
          <input
            type="checkbox"
            checked={requireCiExcludesZero}
            onChange={(event) => onRequireCiExcludesZeroChange(event.target.checked)}
            className="rounded-brutal"
          />
          Require CI
        </label>
      </div>

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="Verdict"
            value={gate?.ship ? 'ship' : 'hold'}
            hint={reasons.length > 0 ? reasons[0] : 'comparison clear'}
          />
          <MetricTile
            label={result.result.compare_metric}
            value={formatSignedPercent(bootstrap?.mean)}
            hint={ciText}
          />
          <MetricTile
            label="Candidate p@1"
            value={formatPercent(gate?.observed.candidate_pass_at_1)}
            hint={`${result.result.candidate_report.n_attempts} attempts`}
          />
          <MetricTile
            label="Clusters"
            value={String(bootstrap?.n_clusters ?? 0)}
            hint={`${bootstrap?.n ?? 0} paired envs`}
          />
        </div>
      ) : null}

      {result ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Base vs candidate</p>
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              <span className="text-text-muted">base p@1</span>
              <span className="text-text-primary text-right">
                {formatPercent(result.result.base_report.pass_at_k['pass@1'])}
              </span>
              <span className="text-text-muted">candidate p@1</span>
              <span className="text-text-primary text-right">
                {formatPercent(result.result.candidate_report.pass_at_k['pass@1'])}
              </span>
              <span className="text-text-muted">bootstrap</span>
              <span className="text-text-primary text-right">
                {bootstrap?.better ? 'better' : 'unclear'}
              </span>
            </div>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Gate reasons</p>
            {reasons.length > 0 ? (
              <div className="space-y-1">
                {reasons.map((reason) => (
                  <p key={reason} className="font-mono text-xs text-text-primary break-words">
                    {reason}
                  </p>
                ))}
              </div>
            ) : (
              <p className="font-mono text-xs text-text-primary">clear</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function SpuriousRewardControlPanel({
  environments,
  attemptText,
  trials,
  randomPassProbability,
  minObservedPassAt1,
  maxControlPassAt1,
  minLiftOverControl,
  result,
  loading,
  onTrialsChange,
  onRandomPassProbabilityChange,
  onMinObservedPassAt1Change,
  onMaxControlPassAt1Change,
  onMinLiftOverControlChange,
  onRun,
}: {
  environments: TerminalEnvironmentSpec[]
  attemptText: string
  trials: string
  randomPassProbability: string
  minObservedPassAt1: string
  maxControlPassAt1: string
  minLiftOverControl: string
  result: EnvironmentSpuriousRewardControlResponse | null
  loading: boolean
  onTrialsChange: (value: string) => void
  onRandomPassProbabilityChange: (value: string) => void
  onMinObservedPassAt1Change: (value: string) => void
  onMaxControlPassAt1Change: (value: string) => void
  onMinLiftOverControlChange: (value: string) => void
  onRun: () => void
}) {
  const gate = result?.result.gate || null
  const controlSummary = result?.result.control.pass_at_k_summary['pass@1'] || null
  const reasons = gate?.reasons || []
  const canRun = environments.length > 0 && attemptText.trim().length > 0 && !loading

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Shuffle className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Spurious Reward Control</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Random-label negative control for the current holdout split
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <Shuffle className="w-4 h-4" />
          Audit
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Trials</span>
          <input
            type="number"
            min={1}
            max={5000}
            value={trials}
            onChange={(event) => onTrialsChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Random p</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={randomPassProbability}
            onChange={(event) => onRandomPassProbabilityChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Min obs p@1</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minObservedPassAt1}
            onChange={(event) => onMinObservedPassAt1Change(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Max ctl p@1</span>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={maxControlPassAt1}
            onChange={(event) => onMaxControlPassAt1Change(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Min lift</span>
          <input
            type="number"
            min={-1}
            max={1}
            step={0.05}
            value={minLiftOverControl}
            onChange={(event) => onMinLiftOverControlChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="Verdict"
            value={gate?.ship ? 'ship' : 'hold'}
            hint={reasons.length > 0 ? reasons[0] : 'negative control clear'}
          />
          <MetricTile
            label="Observed p@1"
            value={formatPercent(gate?.observed.observed_pass_at_1)}
            hint={`${result.result.observed_report.n_attempts} holdout attempts`}
          />
          <MetricTile
            label={`Control ${gate?.observed.control_stat || 'p95'}`}
            value={formatPercent(gate?.observed.control_pass_at_1)}
            hint={`${result.result.control.n_trials || 1} trial${result.result.control.n_trials === 1 ? '' : 's'}`}
          />
          <MetricTile
            label="Lift"
            value={formatPercent(gate?.observed.lift_over_control)}
            hint={`mean ${formatPercent(controlSummary?.mean)}`}
          />
        </div>
      ) : null}

      {result ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Control pass@1</p>
            <div className="grid grid-cols-2 gap-2 font-mono text-xs">
              <span className="text-text-muted">p05</span>
              <span className="text-text-primary text-right">{formatPercent(controlSummary?.p05)}</span>
              <span className="text-text-muted">p50</span>
              <span className="text-text-primary text-right">{formatPercent(controlSummary?.p50)}</span>
              <span className="text-text-muted">p95</span>
              <span className="text-text-primary text-right">{formatPercent(controlSummary?.p95)}</span>
            </div>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Gate reasons</p>
            {reasons.length > 0 ? (
              <div className="space-y-1">
                {reasons.map((reason) => (
                  <p key={reason} className="font-mono text-xs text-text-primary break-words">
                    {reason}
                  </p>
                ))}
              </div>
            ) : (
              <p className="font-mono text-xs text-text-primary">clear</p>
            )}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function LocalRolloutPanel({
  environments,
  selectedEnvironment,
  commandAttemptText,
  kValuesText,
  result,
  loading,
  onCommandAttemptTextChange,
  onKValuesTextChange,
  onSeed,
  onRun,
}: {
  environments: TerminalEnvironmentSpec[]
  selectedEnvironment: TerminalEnvironmentSpec | null
  commandAttemptText: string
  kValuesText: string
  result: EnvironmentRolloutPassKResponse | null
  loading: boolean
  onCommandAttemptTextChange: (value: string) => void
  onKValuesTextChange: (value: string) => void
  onSeed: () => void
  onRun: () => void
}) {
  const passEntries = useMemo(
    () => Object.entries(result?.report.pass_at_k || {}),
    [result]
  )
  const latestRollout = result?.rollouts[0] || null
  const verifierObservation = latestRollout?.verifier_observation || null
  const latestMetadata = latestRollout?.attempt.metadata || {}
  const tamperDetected = latestMetadata.tamper_detected === true
  const tamperHint = metadataTamperHint(latestMetadata)

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Local Rollout</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Persistent workspace run for {selectedEnvironment?.id || 'selected environment'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            value={kValuesText}
            onChange={(event) => onKValuesTextChange(event.target.value)}
            className="input w-28 font-mono text-xs"
            aria-label="Local rollout k values"
          />
          <button
            type="button"
            onClick={onSeed}
            disabled={loading || !selectedEnvironment}
            className="btn-icon flex items-center justify-center"
            title="Seed command attempt"
          >
            <ClipboardPaste className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={onRun}
            disabled={loading || environments.length === 0 || commandAttemptText.trim().length === 0}
            className="btn-primary flex items-center gap-2"
          >
            <PlayCircle className="w-4 h-4" />
            Run
          </button>
        </div>
      </div>

      <textarea
        value={commandAttemptText}
        onChange={(event) => onCommandAttemptTextChange(event.target.value)}
        placeholder='[{"environment_id":"env_a","attempt_index":0,"commands":["python -c \"print(42)\""]}]'
        className="input w-full min-h-28 font-mono text-xs resize-y"
      />

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {passEntries.map(([label, value]) => (
            <MetricTile key={label} label={label} value={formatPercent(value)} hint="verifier pass" />
          ))}
          <MetricTile label="Attempts" value={String(result.report.n_attempts)} hint="local runs" />
          <MetricTile
            label="Tool calls"
            value={String(result.report.attempt_summary.mean_tool_calls ?? 0)}
            hint="mean commands"
          />
          {tamperDetected ? (
            <MetricTile label="Guard" value="tampered" hint={tamperHint} />
          ) : null}
        </div>
      ) : null}

      {latestRollout ? (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_220px] gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Observation log</p>
            <div className="space-y-2">
              {latestRollout.observations.slice(0, 4).map((observation, index) => (
                <div key={`${observation.command}:${index}`} className="font-mono text-xs">
                  <div className="flex items-center justify-between gap-3 text-text-muted">
                    <span className="truncate">{observation.command}</span>
                    <span>exit {observation.exit_code}</span>
                  </div>
                  <pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap break-words text-text-primary">
                    {[observation.stdout, observation.stderr].filter(Boolean).join('\n') || '(no output)'}
                  </pre>
                </div>
              ))}
              {verifierObservation ? (
                <div className="font-mono text-xs border-t-2 border-border pt-2">
                  <div className="flex items-center justify-between gap-3 text-text-muted">
                    <span className="truncate">verifier: {verifierObservation.command}</span>
                    <span>exit {verifierObservation.exit_code}</span>
                  </div>
                  <pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap break-words text-text-primary">
                    {[verifierObservation.stdout, verifierObservation.stderr].filter(Boolean).join('\n') ||
                      '(no output)'}
                  </pre>
                </div>
              ) : null}
            </div>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted">Workspace</p>
            <p className="font-mono text-xs text-text-primary mt-2 break-all">{latestRollout.workspace}</p>
            <p className="font-mono text-xs uppercase text-text-muted mt-4">Status</p>
            <p className="font-mono text-xs text-text-primary mt-2">
              {latestRollout.attempt.verifier_status || 'unknown'}
            </p>
          </div>
        </div>
      ) : null}
    </section>
  )
}

function RewardHackingCanaryPanel({
  categoryText,
  result,
  loading,
  onCategoryTextChange,
  onRun,
}: {
  categoryText: string
  result: EnvironmentCanarySuiteResponse | null
  loading: boolean
  onCategoryTextChange: (value: string) => void
  onRun: () => void
}) {
  const canRun = !loading
  const latestResults = result?.summary.results || []
  const categoryLabels = useMemo(
    () => Object.keys(result?.summary.categories || {}),
    [result]
  )

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Guardrail Canaries</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Local exploit attempts for verifier, tests, private fixtures, and task manifests
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <PlayCircle className="w-4 h-4" />
          Run canaries
        </button>
      </div>

      <label className="block">
        <span className="block font-mono text-xs uppercase text-text-muted mb-1">Categories</span>
        <input
          type="text"
          value={categoryText}
          onChange={(event) => onCategoryTextChange(event.target.value)}
          placeholder="verifier_tamper, tests_tamper"
          className="input w-full font-mono text-xs"
        />
      </label>

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile label="Canaries" value={String(result.summary.total)} hint="exploit attempts" />
          <MetricTile
            label="Guarded"
            value={String(result.summary.guarded)}
            hint={`${result.summary.failed} failed`}
          />
          <MetricTile label="Guard rate" value={formatPercent(result.summary.guard_rate)} />
          <MetricTile
            label="Categories"
            value={String(categoryLabels.length)}
            hint={categoryLabels.join(', ') || 'none'}
          />
        </div>
      ) : null}

      {latestResults.length > 0 ? (
        <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
          <p className="font-mono text-xs uppercase text-text-muted mb-2">Latest canaries</p>
          <div className="space-y-2">
            {latestResults.map((item) => (
              <div
                key={item.canary_id}
                className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_120px_120px] gap-2 border-t-2 border-border-subtle pt-2 first:border-t-0 first:pt-0"
              >
                <div className="min-w-0">
                  <p className="font-mono text-xs text-text-primary truncate">{item.name}</p>
                  <p className="font-mono text-xs text-text-muted truncate">{item.category}</p>
                </div>
                <p className="font-mono text-xs text-text-primary">
                  {item.guarded ? 'guarded' : 'needs fix'}
                </p>
                <p className="font-mono text-xs text-text-muted">{item.verifier_status}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function ModelRolloutPanel({
  selectedEnvironment,
  endpointBaseUrl,
  endpointModel,
  endpointApiKey,
  attemptsPerEnvironment,
  maxToolCalls,
  maxObservationChars,
  temperature,
  kValuesText,
  useToolCalling,
  captureLogprobs,
  topLogprobs,
  dppoReplayOutputPath,
  includeWorldModelReplay,
  rwmlHistoryWindow,
  filterZeroStdGroups,
  activeSampling,
  targetPromptGroups,
  result,
  loading,
  onEndpointBaseUrlChange,
  onEndpointModelChange,
  onEndpointApiKeyChange,
  onAttemptsPerEnvironmentChange,
  onMaxToolCallsChange,
  onMaxObservationCharsChange,
  onTemperatureChange,
  onKValuesTextChange,
  onUseToolCallingChange,
  onCaptureLogprobsChange,
  onTopLogprobsChange,
  onDppoReplayOutputPathChange,
  onIncludeWorldModelReplayChange,
  onRwmlHistoryWindowChange,
  onFilterZeroStdGroupsChange,
  onActiveSamplingChange,
  onTargetPromptGroupsChange,
  onRun,
}: {
  selectedEnvironment: TerminalEnvironmentSpec | null
  endpointBaseUrl: string
  endpointModel: string
  endpointApiKey: string
  attemptsPerEnvironment: string
  maxToolCalls: string
  maxObservationChars: string
  temperature: string
  kValuesText: string
  useToolCalling: boolean
  captureLogprobs: boolean
  topLogprobs: string
  dppoReplayOutputPath: string
  includeWorldModelReplay: boolean
  rwmlHistoryWindow: string
  filterZeroStdGroups: boolean
  activeSampling: boolean
  targetPromptGroups: string
  result: EnvironmentRolloutPassKResponse | null
  loading: boolean
  onEndpointBaseUrlChange: (value: string) => void
  onEndpointModelChange: (value: string) => void
  onEndpointApiKeyChange: (value: string) => void
  onAttemptsPerEnvironmentChange: (value: string) => void
  onMaxToolCallsChange: (value: string) => void
  onMaxObservationCharsChange: (value: string) => void
  onTemperatureChange: (value: string) => void
  onKValuesTextChange: (value: string) => void
  onUseToolCallingChange: (value: boolean) => void
  onCaptureLogprobsChange: (value: boolean) => void
  onTopLogprobsChange: (value: string) => void
  onDppoReplayOutputPathChange: (value: string) => void
  onIncludeWorldModelReplayChange: (value: boolean) => void
  onRwmlHistoryWindowChange: (value: string) => void
  onFilterZeroStdGroupsChange: (value: boolean) => void
  onActiveSamplingChange: (value: boolean) => void
  onTargetPromptGroupsChange: (value: string) => void
  onRun: () => void
}) {
  const passEntries = useMemo(
    () => Object.entries(result?.report.pass_at_k || {}),
    [result]
  )
  const latestRollout = result?.rollouts[0] || null
  const verifierObservation = latestRollout?.verifier_observation || null
  const samplingReport = result?.sampling_report || null
  const dppoReport = result?.dppo_report || null
  const dppoReplay = result?.dppo_replay || null
  const latestMetadata = latestRollout?.attempt.metadata || {}
  const behaviorLogprobTokens = Number(latestMetadata.behavior_logprob_tokens ?? 0)
  const behaviorMeanLogprob =
    typeof latestMetadata.behavior_mean_logprob === 'number'
      ? latestMetadata.behavior_mean_logprob
      : null
  const observationBudget = Number(latestMetadata.max_observation_chars ?? 0)
  const tamperDetected = latestMetadata.tamper_detected === true
  const tamperHint = metadataTamperHint(latestMetadata)
  const canRun =
    Boolean(selectedEnvironment) &&
    endpointBaseUrl.trim().length > 0 &&
    endpointModel.trim().length > 0 &&
    !loading

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">Model Rollout</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Served policy attempt for {selectedEnvironment?.id || 'selected environment'}
          </p>
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={!canRun}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <PlayCircle className="w-4 h-4" />
          Run model
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Base URL</span>
          <input
            type="text"
            value={endpointBaseUrl}
            onChange={(event) => onEndpointBaseUrlChange(event.target.value)}
            placeholder="http://localhost:8000/v1"
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Model</span>
          <ModelSelect
            value={endpointModel}
            onChange={onEndpointModelChange}
            placeholder="Select endpoint model..."
            className="input w-full font-mono text-xs"
            catalogOnly
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">API key</span>
          <input
            type="password"
            value={endpointApiKey}
            onChange={(event) => onEndpointApiKeyChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <label className="block">
            <span className="block font-mono text-xs uppercase text-text-muted mb-1">Attempts</span>
            <input
              type="number"
              min={1}
              max={32}
              value={attemptsPerEnvironment}
              onChange={(event) => onAttemptsPerEnvironmentChange(event.target.value)}
              className="input w-full font-mono text-xs"
            />
          </label>
          <label className="block">
            <span className="block font-mono text-xs uppercase text-text-muted mb-1">Calls</span>
            <input
              type="number"
              min={1}
              value={maxToolCalls}
              onChange={(event) => onMaxToolCallsChange(event.target.value)}
              className="input w-full font-mono text-xs"
            />
          </label>
          <label className="block">
            <span className="block font-mono text-xs uppercase text-text-muted mb-1">Obs chars</span>
            <input
              type="number"
              min={500}
              max={50000}
              value={maxObservationChars}
              onChange={(event) => onMaxObservationCharsChange(event.target.value)}
              className="input w-full font-mono text-xs"
            />
          </label>
          <label className="block">
            <span className="block font-mono text-xs uppercase text-text-muted mb-1">Temp</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.1}
              value={temperature}
              onChange={(event) => onTemperatureChange(event.target.value)}
              className="input w-full font-mono text-xs"
            />
          </label>
        </div>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <label className="block md:w-32">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Pass k</span>
          <input
            value={kValuesText}
            onChange={(event) => onKValuesTextChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={useToolCalling}
            onChange={(event) => onUseToolCallingChange(event.target.checked)}
            className="rounded-brutal"
          />
          Tool calling
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_140px_minmax(0,1.4fr)] gap-3">
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={captureLogprobs}
            onChange={(event) => onCaptureLogprobsChange(event.target.checked)}
            className="rounded-brutal"
          />
          Capture logprobs
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Top logprobs</span>
          <input
            type="number"
            min={0}
            max={20}
            value={topLogprobs}
            onChange={(event) => onTopLogprobsChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">DPPO JSONL</span>
          <input
            type="text"
            value={dppoReplayOutputPath}
            onChange={(event) => onDppoReplayOutputPathChange(event.target.value)}
            placeholder="data/dppo_replay/latest.jsonl"
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_140px] gap-3">
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={includeWorldModelReplay}
            onChange={(event) => onIncludeWorldModelReplayChange(event.target.checked)}
            className="rounded-brutal"
          />
          Include ECHO/RWML replay data
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">RWML history</span>
          <input
            type="number"
            min={0}
            max={64}
            value={rwmlHistoryWindow}
            onChange={(event) => onRwmlHistoryWindowChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_140px] gap-3">
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={filterZeroStdGroups}
            onChange={(event) => onFilterZeroStdGroupsChange(event.target.checked)}
            className="rounded-brutal"
          />
          Filter zero-std groups
        </label>
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input
            type="checkbox"
            checked={activeSampling}
            onChange={(event) => onActiveSamplingChange(event.target.checked)}
            className="rounded-brutal"
          />
          Active sampling
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Target groups</span>
          <input
            type="number"
            min={1}
            value={targetPromptGroups}
            onChange={(event) => onTargetPromptGroupsChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      {result ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {passEntries.map(([label, value]) => (
            <MetricTile key={label} label={label} value={formatPercent(value)} hint="model policy" />
          ))}
          <MetricTile label="Attempts" value={String(result.report.n_attempts)} hint="served model" />
          <MetricTile
            label="Tool calls"
            value={String(result.report.attempt_summary.mean_tool_calls ?? 0)}
            hint="mean commands"
          />
          {observationBudget > 0 ? (
            <MetricTile label="Obs budget" value={String(observationBudget)} hint="prompt chars" />
          ) : null}
          {tamperDetected ? (
            <MetricTile label="Guard" value="tampered" hint={tamperHint} />
          ) : null}
        </div>
      ) : null}

      {samplingReport ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="Kept groups"
            value={String(samplingReport.effective_prompt_groups)}
            hint="non-zero std"
          />
          <MetricTile
            label="Dropped"
            value={String(samplingReport.zero_std_groups_dropped)}
            hint="zero std"
          />
          <MetricTile
            label="Refills"
            value={String(samplingReport.active_sampling_refills)}
            hint="extra candidates"
          />
          <MetricTile
            label="Batch"
            value={samplingReport.maintained_batch ? 'kept' : 'short'}
            hint={`${samplingReport.candidate_prompt_groups} candidates`}
          />
        </div>
      ) : null}

      {behaviorLogprobTokens > 0 ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="Logprob tokens"
            value={String(behaviorLogprobTokens)}
            hint="behavior policy"
          />
          <MetricTile
            label="Mean logprob"
            value={behaviorMeanLogprob !== null ? behaviorMeanLogprob.toFixed(3) : '—'}
            hint="action response"
          />
        </div>
      ) : null}

      {dppoReport ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <MetricTile
            label="DPPO rollout"
            value={dppoReport.rollout_logprobs_ready ? 'ready' : 'missing'}
            hint={`${dppoReport.attempts_with_behavior_logprobs}/${dppoReport.attempts} attempts`}
          />
          <MetricTile
            label="DPPO replay"
            value={dppoReport.needs_train_logprob_replay ? 'needed' : 'clear'}
            hint={`${dppoReport.behavior_logprob_tokens} behavior tokens`}
          />
          {dppoReplay ? (
            <MetricTile
              label="Replay file"
              value={String(dppoReplay.records)}
              hint={dppoReplay.path}
            />
          ) : null}
          {dppoReplay?.world_model && dppoReplay.world_model.records > 0 ? (
            <MetricTile
              label="World model"
              value={String(dppoReplay.world_model.records)}
              hint={`${dppoReplay.world_model.rwml_transitions} RWML transitions`}
            />
          ) : null}
        </div>
      ) : null}

      {latestRollout ? (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_220px] gap-3">
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted mb-2">Latest attempt</p>
            <div className="space-y-2">
              {latestRollout.observations.slice(0, 4).map((observation, index) => (
                <div key={`${observation.command}:${index}`} className="font-mono text-xs">
                  <div className="flex items-center justify-between gap-3 text-text-muted">
                    <span className="truncate">{observation.command || '(format)'}</span>
                    <span>exit {observation.exit_code}</span>
                  </div>
                  <pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap break-words text-text-primary">
                    {[observation.stdout, observation.stderr].filter(Boolean).join('\n') || '(no output)'}
                  </pre>
                </div>
              ))}
              {verifierObservation ? (
                <div className="font-mono text-xs border-t-2 border-border pt-2">
                  <div className="flex items-center justify-between gap-3 text-text-muted">
                    <span className="truncate">verifier: {verifierObservation.command}</span>
                    <span>exit {verifierObservation.exit_code}</span>
                  </div>
                  <pre className="mt-1 max-h-20 overflow-auto whitespace-pre-wrap break-words text-text-primary">
                    {[verifierObservation.stdout, verifierObservation.stderr].filter(Boolean).join('\n') ||
                      '(no output)'}
                  </pre>
                </div>
              ) : null}
            </div>
          </div>
          <div className="border-brutal border-border-subtle rounded-brutal bg-background-secondary p-3 min-w-0">
            <p className="font-mono text-xs uppercase text-text-muted">Workspace</p>
            <p className="font-mono text-xs text-text-primary mt-2 break-all">{latestRollout.workspace}</p>
            <p className="font-mono text-xs uppercase text-text-muted mt-4">Status</p>
            <p className="font-mono text-xs text-text-primary mt-2">
              {latestRollout.attempt.verifier_status || 'unknown'}
            </p>
          </div>
        </div>
      ) : null}
    </section>
  )
}

function DppoReplayPanel({
  inputPath,
  outputPath,
  trainLogprobsText,
  divergence,
  threshold,
  result,
  loading,
  onInputPathChange,
  onOutputPathChange,
  onTrainLogprobsTextChange,
  onDivergenceChange,
  onThresholdChange,
  onEnrich,
}: {
  inputPath: string
  outputPath: string
  trainLogprobsText: string
  divergence: 'binary_tv' | 'binary_kl'
  threshold: string
  result: EnvironmentDppoReplaySummary | null
  loading: boolean
  onInputPathChange: (value: string) => void
  onOutputPathChange: (value: string) => void
  onTrainLogprobsTextChange: (value: string) => void
  onDivergenceChange: (value: 'binary_tv' | 'binary_kl') => void
  onThresholdChange: (value: string) => void
  onEnrich: () => void
}) {
  const canEnrich =
    inputPath.trim().length > 0 &&
    outputPath.trim().length > 0 &&
    trainLogprobsText.trim().length > 0 &&
    !loading
  const maskedFraction = result?.dppo ? formatPercent(result.dppo.masked_fraction) : '0%'

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Gauge className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">DPPO Replay</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Attach train-policy logprobs to sampled rollout JSONL
          </p>
        </div>
        <button
          type="button"
          onClick={onEnrich}
          disabled={!canEnrich}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <PlayCircle className="w-4 h-4" />
          Score replay
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Input JSONL</span>
          <input
            type="text"
            value={inputPath}
            onChange={(event) => onInputPathChange(event.target.value)}
            placeholder="data/dppo_replay/latest.jsonl"
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Scored JSONL</span>
          <input
            type="text"
            value={outputPath}
            onChange={(event) => onOutputPathChange(event.target.value)}
            placeholder="data/dppo_replay/latest.scored.jsonl"
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[160px_160px_minmax(0,1fr)] gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Divergence</span>
          <select
            value={divergence}
            onChange={(event) => onDivergenceChange(event.target.value as 'binary_tv' | 'binary_kl')}
            className="input w-full font-mono text-xs"
          >
            <option value="binary_tv">Binary-TV</option>
            <option value="binary_kl">Binary-KL</option>
          </select>
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Threshold</span>
          <input
            type="number"
            min={0}
            step={0.01}
            value={threshold}
            onChange={(event) => onThresholdChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">
            Train logprobs
          </span>
          <textarea
            value={trainLogprobsText}
            onChange={(event) => onTrainLogprobsTextChange(event.target.value)}
            placeholder='[{"environment_id":"env_a","attempt_index":0,"token_logprobs":[-0.1,-0.2]}]'
            className="input w-full min-h-24 font-mono text-xs resize-y"
          />
        </label>
      </div>

      {result ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
            <MetricTile label="Records" value={String(result.records)} hint={result.path} />
            <MetricTile
              label="Train ready"
              value={String(result.train_logprobs_ready_records)}
              hint={`${result.train_logprob_replay_required_records} pending`}
            />
            <MetricTile
              label="Masked"
              value={maskedFraction}
              hint={`${result.dppo?.masked_updates ?? 0}/${result.dppo?.n_tokens ?? 0} tokens`}
            />
            <MetricTile
              label="Max TV"
              value={(result.dppo?.max_binary_tv ?? 0).toFixed(3)}
              hint={result.dppo?.collapse_warning ? 'collapse warning' : result.dppo?.divergence}
            />
          </div>
          {result.world_model && result.world_model.records > 0 ? (
            <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
              <MetricTile
                label="World records"
                value={String(result.world_model.records)}
                hint={`${result.world_model.records_missing_world_model} missing`}
              />
              <MetricTile
                label="RWML transitions"
                value={String(result.world_model.rwml_transitions)}
                hint={`${result.world_model.rwml_mean_transitions_per_record.toFixed(1)} per record`}
              />
              <MetricTile
                label="RWML history"
                value={result.world_model.rwml_mean_prior_pairs.toFixed(2)}
                hint={`max ${result.world_model.rwml_max_prior_pairs} prior pairs`}
              />
              <MetricTile
                label="ECHO obs chars"
                value={String(result.world_model.echo_observation_chars)}
                hint={formatPercent(result.world_model.echo_observation_char_fraction)}
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

function DppoSmokePanel({
  replayPath,
  outputDir,
  baseModel,
  backend,
  maxSteps,
  gpus,
  result,
  loading,
  onReplayPathChange,
  onOutputDirChange,
  onBaseModelChange,
  onBackendChange,
  onMaxStepsChange,
  onGpusChange,
  onPlan,
}: {
  replayPath: string
  outputDir: string
  baseModel: string
  backend: 'auto' | 'verl' | 'skyrl' | 'tmax_open_instruct' | 'grpo_fallback'
  maxSteps: string
  gpus: string
  result: DPPOSmokeLaunchPlan | null
  loading: boolean
  onReplayPathChange: (value: string) => void
  onOutputDirChange: (value: string) => void
  onBaseModelChange: (value: string) => void
  onBackendChange: (value: 'auto' | 'verl' | 'skyrl' | 'tmax_open_instruct' | 'grpo_fallback') => void
  onMaxStepsChange: (value: string) => void
  onGpusChange: (value: string) => void
  onPlan: () => void
}) {
  const commandText = result?.command.join(' ') || ''
  const canPlan =
    replayPath.trim().length > 0 &&
    outputDir.trim().length > 0 &&
    baseModel.trim().length > 0 &&
    !loading

  return (
    <section className="card p-4 space-y-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-accent" />
            <h3 className="font-brand text-lg text-text-primary">DPPO Smoke Launch</h3>
          </div>
          <p className="font-mono text-xs text-text-muted mt-1">
            Build a backend-specific smoke command for verl, SkyRL, or TMax open-instruct
          </p>
        </div>
        <button
          type="button"
          onClick={onPlan}
          disabled={!canPlan}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <PlayCircle className="w-4 h-4" />
          Plan smoke
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Replay JSONL</span>
          <input
            type="text"
            value={replayPath}
            onChange={(event) => onReplayPathChange(event.target.value)}
            placeholder="data/dppo_replay/latest.scored.jsonl"
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Output dir</span>
          <input
            type="text"
            value={outputDir}
            onChange={(event) => onOutputDirChange(event.target.value)}
            placeholder="data/dppo_smoke/latest"
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Base model</span>
          <BaseModelSelect
            value={baseModel}
            onChange={onBaseModelChange}
            className="input w-full font-mono text-xs"
            catalogOnly
          />
          <p className="font-mono text-xs text-text-muted mt-1">
            Select a catalog model or enter an explicit model ID.
          </p>
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Backend</span>
          <select
            value={backend}
            onChange={(event) =>
              onBackendChange(
                event.target.value as 'auto' | 'verl' | 'skyrl' | 'tmax_open_instruct' | 'grpo_fallback'
              )
            }
            className="input w-full font-mono text-xs"
          >
            <option value="auto">Auto detect</option>
            <option value="verl">verl</option>
            <option value="skyrl">SkyRL</option>
            <option value="tmax_open_instruct">TMax open-instruct</option>
            <option value="grpo_fallback">GRPO fallback</option>
          </select>
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">Max steps</span>
          <input
            type="number"
            min={1}
            max={100}
            value={maxSteps}
            onChange={(event) => onMaxStepsChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
        <label className="block">
          <span className="block font-mono text-xs uppercase text-text-muted mb-1">GPUs</span>
          <input
            type="number"
            min={1}
            max={64}
            value={gpus}
            onChange={(event) => onGpusChange(event.target.value)}
            className="input w-full font-mono text-xs"
          />
        </label>
      </div>

      {result ? (
        <div className="space-y-3">
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
            <MetricTile label="Backend" value={result.backend} hint={result.reason} />
            <MetricTile label="Runnable" value={result.runnable ? 'yes' : 'no'} hint={result.available ? 'detected' : 'missing'} />
            <MetricTile label="Steps" value={String(result.max_steps)} hint={`${result.n_gpus_per_node} GPU`} />
            <MetricTile label="Script" value={result.script_path ? 'written' : 'none'} hint={result.script_path || 'dry plan'} />
          </div>
          {commandText.length > 0 ? (
            <pre className="bg-[#1E1E1E] text-white border-brutal border-text-primary rounded-brutal p-3 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">
              {commandText}
            </pre>
          ) : null}
          {result.warnings.length > 0 ? (
            <div className="border-brutal border-status-warning rounded-brutal bg-status-warning/10 p-3 space-y-1">
              {result.warnings.map((warning) => (
                <p key={warning} className="font-mono text-xs text-text-secondary">
                  {warning}
                </p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}

export function EnvironmentLab() {
  const [pipelines, setPipelines] = useState<EnvironmentPipelinesResponse | null>(null)
  const [result, setResult] = useState<EnvironmentNormalizeResponse | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [sourcePath, setSourcePath] = useState('tests/fixtures/tmax_envs/tmax_demo_001.jsonl')
  const [preserveRaw, setPreserveRaw] = useState(true)
  const [benchmarkText, setBenchmarkText] = useState('')
  const [outputDir, setOutputDir] = useState('data/environments/imported')
  const [overwrite, setOverwrite] = useState(false)
  const [materializeResult, setMaterializeResult] = useState('')
  const [attemptText, setAttemptText] = useState('')
  const [baseAttemptText, setBaseAttemptText] = useState('')
  const [holdoutSplitBy, setHoldoutSplitBy] = useState('task_family')
  const [holdoutFraction, setHoldoutFraction] = useState('0.2')
  const [holdoutSeed, setHoldoutSeed] = useState('0')
  const [holdoutMinPassAt1, setHoldoutMinPassAt1] = useState('0')
  const [holdoutMaxTimeoutRate, setHoldoutMaxTimeoutRate] = useState('0.25')
  const [holdoutMaxTamperRate, setHoldoutMaxTamperRate] = useState('0')
  const [holdoutRequireNoContamination, setHoldoutRequireNoContamination] = useState(true)
  const [holdoutModelId, setHoldoutModelId] = useState('')
  const [holdoutRecordToRegistry, setHoldoutRecordToRegistry] = useState(false)
  const [comparisonClusterBy, setComparisonClusterBy] = useState('task_family')
  const [comparisonCompareK, setComparisonCompareK] = useState('1')
  const [comparisonMinDelta, setComparisonMinDelta] = useState('0')
  const [comparisonMinCandidatePassAt1, setComparisonMinCandidatePassAt1] = useState('0')
  const [comparisonNResamples, setComparisonNResamples] = useState('1000')
  const [comparisonRequireCi, setComparisonRequireCi] = useState(true)
  const [spuriousTrials, setSpuriousTrials] = useState('200')
  const [spuriousRandomPassProbability, setSpuriousRandomPassProbability] = useState('0.05')
  const [spuriousMinObservedPassAt1, setSpuriousMinObservedPassAt1] = useState('0')
  const [spuriousMaxControlPassAt1, setSpuriousMaxControlPassAt1] = useState('0.25')
  const [spuriousMinLiftOverControl, setSpuriousMinLiftOverControl] = useState('0')
  const [commandAttemptText, setCommandAttemptText] = useState('')
  const [canaryCategoryText, setCanaryCategoryText] = useState('')
  const [modelEndpointBaseUrl, setModelEndpointBaseUrl] = useState('http://localhost:8000/v1')
  const [modelEndpointModel, setModelEndpointModel] = useState('')
  const [modelEndpointApiKey, setModelEndpointApiKey] = useState('')
  const [modelAttemptsPerEnvironment, setModelAttemptsPerEnvironment] = useState('1')
  const [modelMaxToolCalls, setModelMaxToolCalls] = useState('8')
  const [modelMaxObservationChars, setModelMaxObservationChars] = useState('6000')
  const [modelTemperature, setModelTemperature] = useState('0')
  const [modelUseToolCalling, setModelUseToolCalling] = useState(true)
  const [modelCaptureLogprobs, setModelCaptureLogprobs] = useState(false)
  const [modelTopLogprobs, setModelTopLogprobs] = useState('0')
  const [modelDppoReplayOutputPath, setModelDppoReplayOutputPath] = useState('')
  const [modelIncludeWorldModelReplay, setModelIncludeWorldModelReplay] = useState(false)
  const [modelRwmlHistoryWindow, setModelRwmlHistoryWindow] = useState('4')
  const [modelFilterZeroStdGroups, setModelFilterZeroStdGroups] = useState(false)
  const [modelActiveSampling, setModelActiveSampling] = useState(false)
  const [modelTargetPromptGroups, setModelTargetPromptGroups] = useState('1')
  const [dppoReplayInputPath, setDppoReplayInputPath] = useState('')
  const [dppoReplayOutputPath, setDppoReplayOutputPath] = useState('')
  const [dppoTrainLogprobsText, setDppoTrainLogprobsText] = useState('')
  const [dppoReplayDivergence, setDppoReplayDivergence] =
    useState<'binary_tv' | 'binary_kl'>('binary_tv')
  const [dppoReplayThreshold, setDppoReplayThreshold] = useState('')
  const [dppoSmokeReplayPath, setDppoSmokeReplayPath] = useState('')
  const [dppoSmokeOutputDir, setDppoSmokeOutputDir] = useState('data/dppo_smoke/latest')
  const [dppoSmokeBaseModel, setDppoSmokeBaseModel] = useState('')
  const [dppoSmokeBackend, setDppoSmokeBackend] =
    useState<'auto' | 'verl' | 'skyrl' | 'tmax_open_instruct' | 'grpo_fallback'>('auto')
  const [dppoSmokeMaxSteps, setDppoSmokeMaxSteps] = useState('1')
  const [dppoSmokeGpus, setDppoSmokeGpus] = useState('1')
  const [kValuesText, setKValuesText] = useState('1,4,8')
  const [passKResult, setPassKResult] = useState<EnvironmentPassKResponse | null>(null)
  const [holdoutGateResult, setHoldoutGateResult] = useState<EnvironmentHoldoutGateResponse | null>(null)
  const [holdoutComparisonResult, setHoldoutComparisonResult] =
    useState<EnvironmentHoldoutComparisonResponse | null>(null)
  const [spuriousControlResult, setSpuriousControlResult] =
    useState<EnvironmentSpuriousRewardControlResponse | null>(null)
  const [rolloutResult, setRolloutResult] = useState<EnvironmentRolloutPassKResponse | null>(null)
  const [canaryResult, setCanaryResult] = useState<EnvironmentCanarySuiteResponse | null>(null)
  const [modelRolloutResult, setModelRolloutResult] = useState<EnvironmentRolloutPassKResponse | null>(null)
  const [dppoReplayResult, setDppoReplayResult] = useState<EnvironmentDppoReplaySummary | null>(null)
  const [dppoSmokePlan, setDppoSmokePlan] = useState<DPPOSmokeLaunchPlan | null>(null)
  const [status, setStatus] = useState<LabStatus>({
    kind: 'idle',
    message: 'Ready',
  })

  useEffect(() => {
    environmentApi.pipelines().then((response) => {
      if (response.ok && response.data) setPipelines(response.data)
    })
  }, [])

  const environments = result?.environments ?? EMPTY_ENVIRONMENTS
  const selectedEnvironment = useMemo(
    () => environments.find((env) => env.id === selectedId) || environments[0] || null,
    [environments, selectedId]
  )
  const summary = reportSummary(result?.report || null)
  const loading = status.kind === 'loading'
  const validationCount = result?.errors.length || 0

  const acceptResult = useCallback((data: EnvironmentNormalizeResponse, message: string) => {
    setResult(data)
    setSelectedId(data.environments[0]?.id || null)
    setMaterializeResult('')
    setPassKResult(null)
    setHoldoutGateResult(null)
    setHoldoutComparisonResult(null)
    setSpuriousControlResult(null)
    setRolloutResult(null)
    setModelRolloutResult(null)
    setStatus({ kind: 'ready', message })
  }, [])

  const handleFileChange = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const input = event.currentTarget
      const file = input.files?.[0]
      if (!file) return
      setStatus({ kind: 'loading', message: `Parsing ${file.name}` })
      try {
        const records = parseEnvironmentRecords(await file.text(), file.name)
        if (records.length === 0) throw new Error('No records found')
        const response = await environmentApi.normalize({
          records,
          source: 'tmax',
          source_uri: file.name,
          preserve_raw: preserveRaw,
        })
        if (response.ok && response.data) {
          acceptResult(response.data, `Imported ${response.data.environments.length} environments`)
        } else {
          setStatus({ kind: 'error', message: response.error || 'Import failed' })
        }
      } catch (error) {
        setStatus({ kind: 'error', message: String(error) })
      } finally {
        input.value = ''
      }
    },
    [acceptResult, preserveRaw]
  )

  const handleImportPath = useCallback(async () => {
    const path = sourcePath.trim()
    if (!path) return
    setStatus({ kind: 'loading', message: 'Importing server path' })
    const response = await environmentApi.importJsonl({
      path,
      source: 'tmax',
      preserve_raw: preserveRaw,
    })
    if (response.ok && response.data) {
      acceptResult(response.data, `Imported ${response.data.environments.length} environments`)
    } else {
      setStatus({ kind: 'error', message: response.error || 'Path import failed' })
    }
  }, [acceptResult, preserveRaw, sourcePath])

  const handleDecontaminate = useCallback(async () => {
    if (environments.length === 0) return
    const benchmarkTexts = benchmarkText
      .split(/\n\s*\n/)
      .map((text) => text.trim())
      .filter(Boolean)
    if (benchmarkTexts.length === 0) {
      setStatus({ kind: 'error', message: 'Add benchmark text first' })
      return
    }
    setStatus({ kind: 'loading', message: 'Filtering benchmark overlap' })
    const response = await environmentApi.decontaminate({
      environments,
      benchmark_texts: benchmarkTexts,
    })
    if (response.ok && response.data) {
      acceptResult(
        {
          environments: response.data.environments,
          report: response.data.mix_report,
          errors: [],
        },
        `Kept ${response.data.report.kept}, dropped ${response.data.report.dropped}`
      )
    } else {
      setStatus({ kind: 'error', message: response.error || 'Decontamination failed' })
    }
  }, [acceptResult, benchmarkText, environments])

  const handleMaterialize = useCallback(async () => {
    if (!selectedEnvironment) return
    setStatus({ kind: 'loading', message: 'Materializing environment' })
    setMaterializeResult('')
    const response = await environmentApi.materialize({
      environment: selectedEnvironment,
      output_dir: outputDir,
      overwrite,
    })
    if (response.ok && response.data) {
      setMaterializeResult(response.data.build.path)
      setStatus({ kind: 'ready', message: `Wrote ${response.data.build.files_written.length} files` })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Materialize failed' })
    }
  }, [outputDir, overwrite, selectedEnvironment])

  const handleSeedBaseline = useCallback(() => {
    const maxK = Math.max(...parseKValues(kValuesText))
    const seededAttempts = JSON.stringify(baselineAttempts(environments, maxK), null, 2)
    setAttemptText(seededAttempts)
    setBaseAttemptText(seededAttempts)
  }, [environments, kValuesText])

  const handlePassK = useCallback(async () => {
    if (environments.length === 0) return
    let attempts: EnvironmentAttemptPayload[]
    let kValues: number[]
    try {
      attempts = parseAttemptRecords(attemptText)
      kValues = parseKValues(kValuesText)
      if (attempts.length === 0) throw new Error('No attempt records found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Computing pass@k' })
    const response = await evalAdvancedApi.environmentPassk({
      environments,
      attempts,
      k_values: kValues,
      record_to_registry: false,
    })
    if (response.ok && response.data) {
      setPassKResult(response.data)
      setStatus({ kind: 'ready', message: 'Pass@k report ready' })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Pass@k failed' })
    }
  }, [attemptText, environments, kValuesText])

  const handleHoldoutGate = useCallback(async () => {
    if (environments.length === 0) return
    const fraction = Number(holdoutFraction)
    const seed = Number(holdoutSeed)
    const minPassAt1 = Number(holdoutMinPassAt1)
    const maxTimeoutRate = Number(holdoutMaxTimeoutRate)
    const maxTamperRate = Number(holdoutMaxTamperRate)
    let attempts: EnvironmentAttemptPayload[]
    let kValues: number[]
    if (Number.isNaN(fraction) || fraction <= 0 || fraction >= 1) {
      setStatus({ kind: 'error', message: 'Holdout fraction must be between 0 and 1' })
      return
    }
    if (!Number.isInteger(seed)) {
      setStatus({ kind: 'error', message: 'Holdout seed must be an integer' })
      return
    }
    if (Number.isNaN(minPassAt1) || minPassAt1 < 0 || minPassAt1 > 1) {
      setStatus({ kind: 'error', message: 'Min pass@1 must be between 0 and 1' })
      return
    }
    if (Number.isNaN(maxTimeoutRate) || maxTimeoutRate < 0 || maxTimeoutRate > 1) {
      setStatus({ kind: 'error', message: 'Timeout max must be between 0 and 1' })
      return
    }
    if (Number.isNaN(maxTamperRate) || maxTamperRate < 0 || maxTamperRate > 1) {
      setStatus({ kind: 'error', message: 'Tamper max must be between 0 and 1' })
      return
    }
    if (holdoutRecordToRegistry && holdoutModelId.trim().length === 0) {
      setStatus({ kind: 'error', message: 'Add a registry model id or disable recording' })
      return
    }
    try {
      attempts = parseAttemptRecords(attemptText)
      kValues = parseKValues(kValuesText)
      if (attempts.length === 0) throw new Error('No attempt records found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Running holdout gate' })
    const response = await evalAdvancedApi.environmentHoldoutGate({
      model_id: holdoutModelId.trim() || undefined,
      environments,
      attempts,
      split_by: holdoutSplitBy,
      holdout_fraction: fraction,
      seed,
      k_values: kValues,
      min_pass_at_1: minPassAt1,
      max_timeout_rate: maxTimeoutRate,
      max_tamper_rate: maxTamperRate,
      require_no_contamination: holdoutRequireNoContamination,
      record_to_registry: holdoutRecordToRegistry,
    })
    if (response.ok && response.data) {
      setHoldoutGateResult(response.data)
      setStatus({
        kind: response.data.result.gate.ship ? 'ready' : 'error',
        message: response.data.result.gate.ship ? 'Holdout gate clear' : 'Holdout gate blocked',
      })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Holdout gate failed' })
    }
  }, [
    attemptText,
    environments,
    holdoutFraction,
    holdoutMaxTamperRate,
    holdoutMaxTimeoutRate,
    holdoutMinPassAt1,
    holdoutModelId,
    holdoutRecordToRegistry,
    holdoutRequireNoContamination,
    holdoutSeed,
    holdoutSplitBy,
    kValuesText,
  ])

  const handleHoldoutComparison = useCallback(async () => {
    if (environments.length === 0) return
    const fraction = Number(holdoutFraction)
    const seed = Number(holdoutSeed)
    const compareK = Number(comparisonCompareK)
    const minDelta = Number(comparisonMinDelta)
    const minCandidatePassAt1 = Number(comparisonMinCandidatePassAt1)
    const nResamples = Number(comparisonNResamples)
    const maxTimeoutRate = Number(holdoutMaxTimeoutRate)
    const maxTamperRate = Number(holdoutMaxTamperRate)
    let baseAttempts: EnvironmentAttemptPayload[]
    let candidateAttempts: EnvironmentAttemptPayload[]
    let kValues: number[]
    if (Number.isNaN(fraction) || fraction <= 0 || fraction >= 1) {
      setStatus({ kind: 'error', message: 'Holdout fraction must be between 0 and 1' })
      return
    }
    if (!Number.isInteger(seed)) {
      setStatus({ kind: 'error', message: 'Holdout seed must be an integer' })
      return
    }
    if (!Number.isInteger(compareK) || compareK <= 0) {
      setStatus({ kind: 'error', message: 'Compare k must be a positive integer' })
      return
    }
    if (Number.isNaN(minDelta) || minDelta < -1 || minDelta > 1) {
      setStatus({ kind: 'error', message: 'Min delta must be between -1 and 1' })
      return
    }
    if (Number.isNaN(minCandidatePassAt1) || minCandidatePassAt1 < 0 || minCandidatePassAt1 > 1) {
      setStatus({ kind: 'error', message: 'Min candidate pass@1 must be between 0 and 1' })
      return
    }
    if (!Number.isInteger(nResamples) || nResamples <= 0 || nResamples > 10000) {
      setStatus({ kind: 'error', message: 'Resamples must be an integer from 1 to 10000' })
      return
    }
    if (Number.isNaN(maxTimeoutRate) || maxTimeoutRate < 0 || maxTimeoutRate > 1) {
      setStatus({ kind: 'error', message: 'Timeout max must be between 0 and 1' })
      return
    }
    if (Number.isNaN(maxTamperRate) || maxTamperRate < 0 || maxTamperRate > 1) {
      setStatus({ kind: 'error', message: 'Tamper max must be between 0 and 1' })
      return
    }
    try {
      baseAttempts = parseAttemptRecords(baseAttemptText)
      candidateAttempts = parseAttemptRecords(attemptText)
      kValues = parseKValues(kValuesText)
      if (baseAttempts.length === 0) throw new Error('No base attempt records found')
      if (candidateAttempts.length === 0) throw new Error('No candidate attempt records found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Running holdout comparison' })
    const response = await evalAdvancedApi.environmentHoldoutComparison({
      environments,
      base_attempts: baseAttempts,
      candidate_attempts: candidateAttempts,
      split_by: holdoutSplitBy,
      cluster_by: comparisonClusterBy,
      holdout_fraction: fraction,
      seed,
      k_values: kValues,
      compare_k: compareK,
      min_delta: minDelta,
      min_candidate_pass_at_1: minCandidatePassAt1,
      require_ci_excludes_zero: comparisonRequireCi,
      max_candidate_timeout_rate: maxTimeoutRate,
      max_candidate_tamper_rate: maxTamperRate,
      require_no_contamination: holdoutRequireNoContamination,
      n_resamples: nResamples,
    })
    if (response.ok && response.data) {
      setHoldoutComparisonResult(response.data)
      setStatus({
        kind: response.data.result.gate.ship ? 'ready' : 'error',
        message: response.data.result.gate.ship
          ? 'Holdout comparison clear'
          : 'Holdout comparison blocked',
      })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Holdout comparison failed' })
    }
  }, [
    attemptText,
    baseAttemptText,
    comparisonClusterBy,
    comparisonCompareK,
    comparisonMinCandidatePassAt1,
    comparisonMinDelta,
    comparisonNResamples,
    comparisonRequireCi,
    environments,
    holdoutFraction,
    holdoutMaxTamperRate,
    holdoutMaxTimeoutRate,
    holdoutRequireNoContamination,
    holdoutSeed,
    holdoutSplitBy,
    kValuesText,
  ])

  const handleSpuriousRewardControl = useCallback(async () => {
    if (environments.length === 0) return
    const fraction = Number(holdoutFraction)
    const seed = Number(holdoutSeed)
    const trials = Number(spuriousTrials)
    const randomPassProbability = Number(spuriousRandomPassProbability)
    const minObservedPassAt1 = Number(spuriousMinObservedPassAt1)
    const maxControlPassAt1 = Number(spuriousMaxControlPassAt1)
    const minLiftOverControl = Number(spuriousMinLiftOverControl)
    let attempts: EnvironmentAttemptPayload[]
    let kValues: number[]
    if (Number.isNaN(fraction) || fraction <= 0 || fraction >= 1) {
      setStatus({ kind: 'error', message: 'Holdout fraction must be between 0 and 1' })
      return
    }
    if (!Number.isInteger(seed)) {
      setStatus({ kind: 'error', message: 'Holdout seed must be an integer' })
      return
    }
    if (!Number.isInteger(trials) || trials <= 0 || trials > 5000) {
      setStatus({ kind: 'error', message: 'Spurious trials must be an integer from 1 to 5000' })
      return
    }
    if (Number.isNaN(randomPassProbability) || randomPassProbability < 0 || randomPassProbability > 1) {
      setStatus({ kind: 'error', message: 'Random probability must be between 0 and 1' })
      return
    }
    if (Number.isNaN(minObservedPassAt1) || minObservedPassAt1 < 0 || minObservedPassAt1 > 1) {
      setStatus({ kind: 'error', message: 'Min observed pass@1 must be between 0 and 1' })
      return
    }
    if (Number.isNaN(maxControlPassAt1) || maxControlPassAt1 < 0 || maxControlPassAt1 > 1) {
      setStatus({ kind: 'error', message: 'Max control pass@1 must be between 0 and 1' })
      return
    }
    if (Number.isNaN(minLiftOverControl) || minLiftOverControl < -1 || minLiftOverControl > 1) {
      setStatus({ kind: 'error', message: 'Min lift must be between -1 and 1' })
      return
    }
    try {
      attempts = parseAttemptRecords(attemptText)
      kValues = parseKValues(kValuesText)
      if (attempts.length === 0) throw new Error('No attempt records found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Running spurious reward control' })
    const response = await evalAdvancedApi.environmentSpuriousRewardControl({
      environments,
      attempts,
      split_by: holdoutSplitBy,
      holdout_fraction: fraction,
      seed,
      k_values: kValues,
      n_trials: trials,
      random_pass_probability: randomPassProbability,
      min_observed_pass_at_1: minObservedPassAt1,
      max_control_pass_at_1: maxControlPassAt1,
      min_lift_over_control: minLiftOverControl,
      require_no_contamination: holdoutRequireNoContamination,
    })
    if (response.ok && response.data) {
      setSpuriousControlResult(response.data)
      setStatus({
        kind: response.data.result.gate.ship ? 'ready' : 'error',
        message: response.data.result.gate.ship
          ? 'Spurious control clear'
          : 'Spurious control blocked',
      })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Spurious control failed' })
    }
  }, [
    attemptText,
    environments,
    holdoutFraction,
    holdoutRequireNoContamination,
    holdoutSeed,
    holdoutSplitBy,
    kValuesText,
    spuriousMaxControlPassAt1,
    spuriousMinLiftOverControl,
    spuriousMinObservedPassAt1,
    spuriousRandomPassProbability,
    spuriousTrials,
  ])

  const handleSeedLocalRollout = useCallback(() => {
    setCommandAttemptText(JSON.stringify(seededCommandAttempts(selectedEnvironment), null, 2))
  }, [selectedEnvironment])

  const handleLocalRollout = useCallback(async () => {
    if (environments.length === 0) return
    let commandAttempts: EnvironmentCommandAttemptPayload[]
    let kValues: number[]
    try {
      commandAttempts = parseCommandAttemptRecords(commandAttemptText)
      kValues = parseKValues(kValuesText)
      if (commandAttempts.length === 0) throw new Error('No command attempts found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Running local rollout' })
    const response = await evalAdvancedApi.environmentLocalRolloutPassk({
      environments,
      command_attempts: commandAttempts,
      k_values: kValues,
      record_to_registry: false,
    })
    if (response.ok && response.data) {
      setRolloutResult(response.data)
      setStatus({ kind: 'ready', message: 'Local rollout report ready' })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Local rollout failed' })
    }
  }, [commandAttemptText, environments, kValuesText])

  const handleRunCanaries = useCallback(async () => {
    const categories = parseCommaValues(canaryCategoryText)
    setStatus({ kind: 'loading', message: 'Running guardrail canaries' })
    const response = await evalAdvancedApi.environmentRewardHackingCanaries({
      categories,
      keep_workspace: true,
    })
    if (response.ok && response.data) {
      setCanaryResult(response.data)
      setStatus({
        kind: response.data.summary.failed === 0 ? 'ready' : 'error',
        message: `${response.data.summary.guarded}/${response.data.summary.total} canaries guarded`,
      })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Canary suite failed' })
    }
  }, [canaryCategoryText])

  const handleModelRollout = useCallback(async () => {
    if (!selectedEnvironment) return
    const attempts = Number(modelAttemptsPerEnvironment)
    const maxCalls = Number(modelMaxToolCalls)
    const maxObservationChars = Number(modelMaxObservationChars)
    const temperature = Number(modelTemperature)
    const topLogprobs = Number(modelTopLogprobs)
    const targetPromptGroups = Number(modelTargetPromptGroups)
    const rwmlHistoryWindow = Number(modelRwmlHistoryWindow)
    let kValues: number[]
    if (!Number.isInteger(attempts) || attempts <= 0) {
      setStatus({ kind: 'error', message: 'Attempts must be a positive integer' })
      return
    }
    if (!Number.isInteger(maxCalls) || maxCalls <= 0) {
      setStatus({ kind: 'error', message: 'Calls must be a positive integer' })
      return
    }
    if (
      !Number.isInteger(maxObservationChars) ||
      maxObservationChars < 500 ||
      maxObservationChars > 50000
    ) {
      setStatus({ kind: 'error', message: 'Obs chars must be an integer from 500 to 50000' })
      return
    }
    if (Number.isNaN(temperature) || temperature < 0 || temperature > 2) {
      setStatus({ kind: 'error', message: 'Temperature must be between 0 and 2' })
      return
    }
    if (modelCaptureLogprobs && (!Number.isInteger(topLogprobs) || topLogprobs < 0 || topLogprobs > 20)) {
      setStatus({ kind: 'error', message: 'Top logprobs must be an integer from 0 to 20' })
      return
    }
    if (
      modelIncludeWorldModelReplay &&
      (!Number.isInteger(rwmlHistoryWindow) || rwmlHistoryWindow < 0 || rwmlHistoryWindow > 64)
    ) {
      setStatus({ kind: 'error', message: 'RWML history must be an integer from 0 to 64' })
      return
    }
    if (
      (modelFilterZeroStdGroups || modelActiveSampling) &&
      (!Number.isInteger(targetPromptGroups) || targetPromptGroups <= 0)
    ) {
      setStatus({ kind: 'error', message: 'Target groups must be a positive integer' })
      return
    }
    try {
      kValues = parseKValues(kValuesText)
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Running model rollout' })
    const response = await evalAdvancedApi.environmentModelRolloutPassk({
      endpoint: {
        base_url: modelEndpointBaseUrl.trim(),
        model: modelEndpointModel.trim(),
        api_key: modelEndpointApiKey.trim() || undefined,
      },
      environments: [selectedEnvironment],
      attempts_per_environment: attempts,
      k_values: kValues,
      max_tool_calls: maxCalls,
      max_observation_chars: maxObservationChars,
      temperature,
      use_tool_calling: modelUseToolCalling,
      capture_logprobs: modelCaptureLogprobs,
      top_logprobs: modelCaptureLogprobs && topLogprobs > 0 ? topLogprobs : undefined,
      dppo_replay_output_path: modelDppoReplayOutputPath.trim() || undefined,
      include_world_model_replay: modelIncludeWorldModelReplay,
      rwml_history_window: rwmlHistoryWindow,
      filter_zero_std_groups: modelFilterZeroStdGroups,
      active_sampling: modelActiveSampling,
      target_prompt_groups:
        modelFilterZeroStdGroups || modelActiveSampling ? targetPromptGroups : undefined,
      record_to_registry: false,
    })
    if (response.ok && response.data) {
      setModelRolloutResult(response.data)
      const replayPath = response.data.dppo_replay?.path
      if (replayPath) {
        setDppoReplayInputPath(replayPath)
        setDppoReplayOutputPath(replayPath.replace(/\.jsonl$/, '') + '.scored.jsonl')
      }
      setStatus({ kind: 'ready', message: 'Model rollout report ready' })
    } else {
      setStatus({ kind: 'error', message: response.error || 'Model rollout failed' })
    }
  }, [
    kValuesText,
    modelAttemptsPerEnvironment,
    modelEndpointApiKey,
    modelEndpointBaseUrl,
    modelEndpointModel,
    modelMaxObservationChars,
    modelMaxToolCalls,
    modelTemperature,
    modelCaptureLogprobs,
    modelDppoReplayOutputPath,
    modelIncludeWorldModelReplay,
    modelRwmlHistoryWindow,
    modelTopLogprobs,
    modelFilterZeroStdGroups,
    modelActiveSampling,
    modelTargetPromptGroups,
    modelUseToolCalling,
    selectedEnvironment,
  ])

  const handleDppoReplayEnrich = useCallback(async () => {
    const inputPath = dppoReplayInputPath.trim()
    const outputPath = dppoReplayOutputPath.trim()
    const rawThreshold = dppoReplayThreshold.trim()
    if (!inputPath || !outputPath) {
      setStatus({ kind: 'error', message: 'DPPO replay paths are required' })
      return
    }
    const threshold = rawThreshold.length > 0 ? Number(rawThreshold) : undefined
    if (threshold !== undefined && (Number.isNaN(threshold) || threshold < 0)) {
      setStatus({ kind: 'error', message: 'DPPO threshold must be non-negative' })
      return
    }
    let trainLogprobs: DPPOTrainLogprobsSpec[]
    try {
      trainLogprobs = parseDppoTrainLogprobs(dppoTrainLogprobsText)
      if (trainLogprobs.length === 0) throw new Error('No train logprob records found')
    } catch (error) {
      setStatus({ kind: 'error', message: String(error) })
      return
    }
    setStatus({ kind: 'loading', message: 'Scoring DPPO replay' })
    const response = await evalAdvancedApi.enrichDppoReplay({
      input_path: inputPath,
      output_path: outputPath,
      train_logprobs: trainLogprobs,
      divergence: dppoReplayDivergence,
      threshold,
    })
    if (response.ok && response.data) {
      setDppoReplayResult(response.data.dppo_replay)
      setDppoSmokeReplayPath(response.data.dppo_replay.path)
      setStatus({ kind: 'ready', message: 'DPPO replay scored' })
    } else {
      setStatus({ kind: 'error', message: response.error || 'DPPO replay scoring failed' })
    }
  }, [
    dppoReplayDivergence,
    dppoReplayInputPath,
    dppoReplayOutputPath,
    dppoReplayThreshold,
    dppoTrainLogprobsText,
  ])

  const handleDppoSmokePlan = useCallback(async () => {
    const replayPath = dppoSmokeReplayPath.trim()
    const outputDir = dppoSmokeOutputDir.trim()
    const baseModel = dppoSmokeBaseModel.trim()
    const maxSteps = Number(dppoSmokeMaxSteps)
    const gpus = Number(dppoSmokeGpus)
    if (!replayPath || !outputDir || !baseModel) {
      setStatus({ kind: 'error', message: 'DPPO smoke replay path, output dir, and model are required' })
      return
    }
    if (!Number.isInteger(maxSteps) || maxSteps <= 0 || maxSteps > 100) {
      setStatus({ kind: 'error', message: 'DPPO smoke max steps must be 1-100' })
      return
    }
    if (!Number.isInteger(gpus) || gpus <= 0 || gpus > 64) {
      setStatus({ kind: 'error', message: 'DPPO smoke GPU count must be 1-64' })
      return
    }
    setStatus({ kind: 'loading', message: 'Planning DPPO smoke launch' })
    const response = await evalAdvancedApi.planDppoSmoke({
      replay_path: replayPath,
      output_dir: outputDir,
      base_model: baseModel,
      backend: dppoSmokeBackend,
      max_steps: maxSteps,
      n_gpus_per_node: gpus,
      write_script: true,
    })
    if (response.ok && response.data) {
      setDppoSmokePlan(response.data.plan)
      setStatus({ kind: 'ready', message: 'DPPO smoke plan ready' })
    } else {
      setStatus({ kind: 'error', message: response.error || 'DPPO smoke planning failed' })
    }
  }, [
    dppoSmokeBackend,
    dppoSmokeBaseModel,
    dppoSmokeGpus,
    dppoSmokeMaxSteps,
    dppoSmokeOutputDir,
    dppoSmokeReplayPath,
  ])

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="font-brand text-3xl text-text-primary flex items-center gap-2">
            <Terminal className="w-6 h-6 text-accent" />
            Environment Lab
          </h2>
          <p className="font-mono text-xs text-text-muted mt-1">
            TMax-compatible terminal environments
          </p>
        </div>
        <div
          className={clsx(
            'flex items-center gap-2 border-brutal rounded-brutal px-3 py-2 bg-background-card font-mono text-xs',
            status.kind === 'error' ? 'border-status-error text-status-error' : 'border-border text-text-muted'
          )}
        >
          {status.kind === 'loading' && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {status.kind === 'error' && <AlertCircle className="w-3.5 h-3.5" />}
          {status.kind === 'ready' && <CheckCircle2 className="w-3.5 h-3.5 text-status-success" />}
          {status.kind === 'idle' && <Database className="w-3.5 h-3.5" />}
          <span>{status.message}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-5">
        <aside className="space-y-4">
          <SourcePanel
            sourcePath={sourcePath}
            preserveRaw={preserveRaw}
            onPathChange={setSourcePath}
            onPreserveRawChange={setPreserveRaw}
            onFileChange={handleFileChange}
            onImportPath={handleImportPath}
            loading={loading}
          />

          <section className="card p-4 space-y-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-accent" />
              <h3 className="font-brand text-lg text-text-primary">Decontaminate</h3>
            </div>
            <textarea
              value={benchmarkText}
              onChange={(event) => setBenchmarkText(event.target.value)}
              placeholder="Benchmark text snippets, separated by blank lines"
              className="input w-full min-h-28 font-mono text-xs resize-y"
            />
            <button
              type="button"
              onClick={handleDecontaminate}
              disabled={loading || environments.length === 0}
              className="btn-secondary flex items-center gap-2 w-full justify-center"
            >
              <ShieldCheck className="w-4 h-4" />
              Filter overlap
            </button>
          </section>

          <section className="card p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-accent" />
              <h3 className="font-brand text-lg text-text-primary">Sources</h3>
            </div>
            <div className="space-y-2">
              {Object.entries(pipelines?.external_sources || {}).map(([name, dataset]) => (
                <div key={name} className="flex items-center justify-between gap-3 text-xs">
                  <span className="font-mono text-text-primary">{name}</span>
                  <span className="text-text-muted truncate">{dataset}</span>
                </div>
              ))}
              {!pipelines && <p className="font-mono text-xs text-text-muted">Loading sources</p>}
            </div>
            <div className="border-t-2 border-border pt-3 font-mono text-xs text-text-muted">
              terminal_env_generation:{' '}
              {pipelines?.pipelines[0]?.available ? 'registered' : 'unavailable'}
            </div>
          </section>
        </aside>

        <main className="space-y-4 min-w-0">
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
            <MetricTile label="Environments" value={summary.total} hint={`${validationCount} warnings`} />
            <MetricTile label="Domains" value={summary.domains} hint="observed buckets" />
            <MetricTile label="Skills" value={summary.skills} hint="observed buckets" />
            <MetricTile label="Domain balance" value={summary.balance} hint="entropy score" />
          </div>

          {result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <DistributionList title="Domains" values={result.report.domain_distribution} />
              <DistributionList title="Skills" values={result.report.skill_distribution} />
            </div>
          )}

          <PassKPanel
            environments={environments}
            attemptText={attemptText}
            kValuesText={kValuesText}
            result={passKResult}
            loading={loading}
            onAttemptTextChange={setAttemptText}
            onKValuesTextChange={setKValuesText}
            onSeedBaseline={handleSeedBaseline}
            onEvaluate={handlePassK}
          />

          <HoldoutGatePanel
            environments={environments}
            attemptText={attemptText}
            splitBy={holdoutSplitBy}
            fraction={holdoutFraction}
            seed={holdoutSeed}
            minPassAt1={holdoutMinPassAt1}
            maxTimeoutRate={holdoutMaxTimeoutRate}
            maxTamperRate={holdoutMaxTamperRate}
            requireNoContamination={holdoutRequireNoContamination}
            modelId={holdoutModelId}
            recordToRegistry={holdoutRecordToRegistry}
            result={holdoutGateResult}
            loading={loading}
            onSplitByChange={setHoldoutSplitBy}
            onFractionChange={setHoldoutFraction}
            onSeedChange={setHoldoutSeed}
            onMinPassAt1Change={setHoldoutMinPassAt1}
            onMaxTimeoutRateChange={setHoldoutMaxTimeoutRate}
            onMaxTamperRateChange={setHoldoutMaxTamperRate}
            onRequireNoContaminationChange={setHoldoutRequireNoContamination}
            onModelIdChange={setHoldoutModelId}
            onRecordToRegistryChange={setHoldoutRecordToRegistry}
            onRun={handleHoldoutGate}
          />

          <HoldoutComparisonPanel
            environments={environments}
            baseAttemptText={baseAttemptText}
            candidateAttemptText={attemptText}
            clusterBy={comparisonClusterBy}
            compareK={comparisonCompareK}
            minDelta={comparisonMinDelta}
            minCandidatePassAt1={comparisonMinCandidatePassAt1}
            nResamples={comparisonNResamples}
            requireCiExcludesZero={comparisonRequireCi}
            result={holdoutComparisonResult}
            loading={loading}
            onBaseAttemptTextChange={setBaseAttemptText}
            onClusterByChange={setComparisonClusterBy}
            onCompareKChange={setComparisonCompareK}
            onMinDeltaChange={setComparisonMinDelta}
            onMinCandidatePassAt1Change={setComparisonMinCandidatePassAt1}
            onNResamplesChange={setComparisonNResamples}
            onRequireCiExcludesZeroChange={setComparisonRequireCi}
            onRun={handleHoldoutComparison}
          />

          <SpuriousRewardControlPanel
            environments={environments}
            attemptText={attemptText}
            trials={spuriousTrials}
            randomPassProbability={spuriousRandomPassProbability}
            minObservedPassAt1={spuriousMinObservedPassAt1}
            maxControlPassAt1={spuriousMaxControlPassAt1}
            minLiftOverControl={spuriousMinLiftOverControl}
            result={spuriousControlResult}
            loading={loading}
            onTrialsChange={setSpuriousTrials}
            onRandomPassProbabilityChange={setSpuriousRandomPassProbability}
            onMinObservedPassAt1Change={setSpuriousMinObservedPassAt1}
            onMaxControlPassAt1Change={setSpuriousMaxControlPassAt1}
            onMinLiftOverControlChange={setSpuriousMinLiftOverControl}
            onRun={handleSpuriousRewardControl}
          />

          <LocalRolloutPanel
            environments={environments}
            selectedEnvironment={selectedEnvironment}
            commandAttemptText={commandAttemptText}
            kValuesText={kValuesText}
            result={rolloutResult}
            loading={loading}
            onCommandAttemptTextChange={setCommandAttemptText}
            onKValuesTextChange={setKValuesText}
            onSeed={handleSeedLocalRollout}
            onRun={handleLocalRollout}
          />

          <RewardHackingCanaryPanel
            categoryText={canaryCategoryText}
            result={canaryResult}
            loading={loading}
            onCategoryTextChange={setCanaryCategoryText}
            onRun={handleRunCanaries}
          />

          <ModelRolloutPanel
            selectedEnvironment={selectedEnvironment}
            endpointBaseUrl={modelEndpointBaseUrl}
            endpointModel={modelEndpointModel}
            endpointApiKey={modelEndpointApiKey}
            attemptsPerEnvironment={modelAttemptsPerEnvironment}
            maxToolCalls={modelMaxToolCalls}
            maxObservationChars={modelMaxObservationChars}
            temperature={modelTemperature}
            kValuesText={kValuesText}
            useToolCalling={modelUseToolCalling}
            captureLogprobs={modelCaptureLogprobs}
            topLogprobs={modelTopLogprobs}
            dppoReplayOutputPath={modelDppoReplayOutputPath}
            includeWorldModelReplay={modelIncludeWorldModelReplay}
            rwmlHistoryWindow={modelRwmlHistoryWindow}
            filterZeroStdGroups={modelFilterZeroStdGroups}
            activeSampling={modelActiveSampling}
            targetPromptGroups={modelTargetPromptGroups}
            result={modelRolloutResult}
            loading={loading}
            onEndpointBaseUrlChange={setModelEndpointBaseUrl}
            onEndpointModelChange={setModelEndpointModel}
            onEndpointApiKeyChange={setModelEndpointApiKey}
            onAttemptsPerEnvironmentChange={setModelAttemptsPerEnvironment}
            onMaxToolCallsChange={setModelMaxToolCalls}
            onMaxObservationCharsChange={setModelMaxObservationChars}
            onTemperatureChange={setModelTemperature}
            onKValuesTextChange={setKValuesText}
            onUseToolCallingChange={setModelUseToolCalling}
            onCaptureLogprobsChange={setModelCaptureLogprobs}
            onTopLogprobsChange={setModelTopLogprobs}
            onDppoReplayOutputPathChange={setModelDppoReplayOutputPath}
            onIncludeWorldModelReplayChange={setModelIncludeWorldModelReplay}
            onRwmlHistoryWindowChange={setModelRwmlHistoryWindow}
            onFilterZeroStdGroupsChange={setModelFilterZeroStdGroups}
            onActiveSamplingChange={setModelActiveSampling}
            onTargetPromptGroupsChange={setModelTargetPromptGroups}
            onRun={handleModelRollout}
          />

          <DppoReplayPanel
            inputPath={dppoReplayInputPath}
            outputPath={dppoReplayOutputPath}
            trainLogprobsText={dppoTrainLogprobsText}
            divergence={dppoReplayDivergence}
            threshold={dppoReplayThreshold}
            result={dppoReplayResult}
            loading={loading}
            onInputPathChange={setDppoReplayInputPath}
            onOutputPathChange={setDppoReplayOutputPath}
            onTrainLogprobsTextChange={setDppoTrainLogprobsText}
            onDivergenceChange={setDppoReplayDivergence}
            onThresholdChange={setDppoReplayThreshold}
            onEnrich={handleDppoReplayEnrich}
          />

          <DppoSmokePanel
            replayPath={dppoSmokeReplayPath}
            outputDir={dppoSmokeOutputDir}
            baseModel={dppoSmokeBaseModel}
            backend={dppoSmokeBackend}
            maxSteps={dppoSmokeMaxSteps}
            gpus={dppoSmokeGpus}
            result={dppoSmokePlan}
            loading={loading}
            onReplayPathChange={setDppoSmokeReplayPath}
            onOutputDirChange={setDppoSmokeOutputDir}
            onBaseModelChange={setDppoSmokeBaseModel}
            onBackendChange={setDppoSmokeBackend}
            onMaxStepsChange={setDppoSmokeMaxSteps}
            onGpusChange={setDppoSmokeGpus}
            onPlan={handleDppoSmokePlan}
          />

          {result && result.errors.length > 0 && (
            <div className="card p-4 border-l-4 border-l-status-warning bg-status-warning/10">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-4 h-4 text-status-warning" />
                <h3 className="font-brand text-lg text-text-primary">Validation warnings</h3>
              </div>
              <div className="space-y-1">
                {result.errors.slice(0, 5).map((error) => (
                  <p key={`${error.index}:${error.id || error.error}`} className="font-mono text-xs text-text-secondary">
                    #{error.index} {error.id ? `${error.id}: ` : ''}
                    {(error.validation_errors || [error.error]).filter(Boolean).join(', ')}
                  </p>
                ))}
              </div>
            </div>
          )}

          <EnvironmentList
            environments={environments}
            selectedId={selectedEnvironment?.id || selectedId}
            onSelect={setSelectedId}
          />

          <EnvironmentDetail
            environment={selectedEnvironment}
            outputDir={outputDir}
            overwrite={overwrite}
            materializeResult={materializeResult}
            onOutputDirChange={setOutputDir}
            onOverwriteChange={setOverwrite}
            onMaterialize={handleMaterialize}
            loading={loading}
          />

          <div className="card p-4">
            <div className="flex items-center gap-2 mb-2">
              <Upload className="w-4 h-4 text-accent" />
              <h3 className="font-brand text-lg text-text-primary">Next pipeline link</h3>
            </div>
            <p className="text-sm text-text-secondary">
              Generated bundles are ready for rollout collection, held-out eval assembly, and GRPO
              curriculum sampling once those harness hooks are wired.
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}
