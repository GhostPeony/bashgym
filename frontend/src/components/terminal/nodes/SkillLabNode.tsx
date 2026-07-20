import { memo, useCallback, useEffect, useRef, useState } from 'react'
import type { Node, NodeProps } from '@xyflow/react'
import {
  AlertCircle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  Code2,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Settings2,
  SquareTerminal,
  Trash2,
  WandSparkles,
  XCircle
} from 'lucide-react'
import { clsx } from 'clsx'
import {
  skillLabApi,
  toolkitApi,
  type SkillLabCase,
  type SkillLabContract,
  type SkillLabRun,
  type ToolkitInventoryResponse,
  type ToolkitSkill
} from '../../../services/api'
import { useSkillLabStore, useTerminalStore, useWorkspaceStore } from '../../../stores'
import { DataNodeShell } from './DataNodeShell'
import { hueFor } from './dataPanels'
import {
  ConfigPill,
  ConfigRow,
  ConfigRows,
  ConfigSection,
  NodeConfigModal
} from './NodeConfigModal'
import {
  buildSkillLabTerminalPrompt,
  defaultSkillContract,
  emptySkillCase,
  formatSkillPercent,
  isCatalogSkillActive,
  kpiTone,
  normalizePatternList,
  skillLabPlanScope,
  skillIdFor,
  skillSourceLabel,
  validateSkillContract
} from './skillLabModel'
import type { DataNodeData } from './types'

export type SkillLabNodeType = Node<DataNodeData, 'skilllab'>

const ACTIVE_POLL_MS = 3_000
const IDLE_POLL_MS = 15_000

function RunStatus({ run, onClick }: { run: SkillLabRun; onClick?: () => void }) {
  const tone =
    run.status === 'failed'
      ? 'text-status-error'
      : run.status === 'completed'
        ? run.kpis?.verdict === 'effective'
          ? 'text-status-success'
          : 'text-status-warning'
        : 'text-accent'
  const content = (
    <>
      {run.status === 'running' || run.status === 'queued' ? (
        <Loader2 className="h-3 w-3 flex-shrink-0 animate-spin text-accent" />
      ) : run.status === 'completed' ? (
        <CheckCircle2 className={clsx('h-3 w-3 flex-shrink-0', tone)} />
      ) : (
        <XCircle className="h-3 w-3 flex-shrink-0 text-status-error" />
      )}
      <span className="min-w-0 flex-1 truncate text-text-secondary">{run.skill_name}</span>
      <span className={clsx('flex-shrink-0 uppercase', tone)}>
        {run.kpis?.verdict || run.status}
      </span>
    </>
  )
  return onClick ? (
    <button
      type="button"
      className="flex w-full min-w-0 items-center gap-2 rounded-brutal px-1 py-0.5 text-left font-mono text-[10px] hover:bg-background-secondary"
      onClick={onClick}
    >
      {content}
    </button>
  ) : (
    <div className="flex min-w-0 items-center gap-2 font-mono text-[10px]">{content}</div>
  )
}

function KpiTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-brutal border-brutal border-border-subtle bg-background-card px-2 py-1.5">
      <div className="truncate font-mono text-[8px] uppercase text-text-muted">{label}</div>
      <div className="mt-0.5 truncate font-mono text-sm font-bold text-text-primary">{value}</div>
    </div>
  )
}

function CaseEditor({
  item,
  index,
  onChange,
  onRemove
}: {
  item: SkillLabCase
  index: number
  onChange: (next: SkillLabCase) => void
  onRemove: () => void
}) {
  return (
    <div className="border-b border-border-subtle pb-3 last:border-b-0 last:pb-0">
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] font-bold uppercase text-text-secondary">
          Case {index + 1}
        </span>
        <label className="ml-auto flex items-center gap-2 font-mono text-[9px] uppercase text-text-muted">
          <input
            type="checkbox"
            checked={item.should_invoke}
            onChange={(event) => onChange({ ...item, should_invoke: event.target.checked })}
          />
          Should invoke
        </label>
        <button
          type="button"
          className="node-btn node-btn-danger"
          onClick={onRemove}
          title="Remove case"
        >
          <Trash2 className="h-3 w-3" />
        </button>
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <label className="node-field">
          <span className="node-field-label">Name</span>
          <input
            className="input-brutal min-h-9 font-mono text-[11px]"
            value={item.name}
            onChange={(event) => onChange({ ...item, name: event.target.value })}
          />
        </label>
        <label className="node-field md:col-span-2">
          <span className="node-field-label">Agent task</span>
          <textarea
            className="input-brutal min-h-20 font-mono text-[11px]"
            value={item.prompt}
            onChange={(event) => onChange({ ...item, prompt: event.target.value })}
            placeholder="A held-out task that should exercise this skill"
          />
        </label>
        <label className="node-field">
          <span className="node-field-label">Required output patterns</span>
          <textarea
            className="input-brutal min-h-16 font-mono text-[11px]"
            value={item.expected_patterns.join('\n')}
            onChange={(event) =>
              onChange({ ...item, expected_patterns: normalizePatternList(event.target.value) })
            }
            placeholder="One deterministic pattern per line"
          />
        </label>
        <label className="node-field">
          <span className="node-field-label">Forbidden output patterns</span>
          <textarea
            className="input-brutal min-h-16 font-mono text-[11px]"
            value={item.forbidden_patterns.join('\n')}
            onChange={(event) =>
              onChange({ ...item, forbidden_patterns: normalizePatternList(event.target.value) })
            }
            placeholder="Patterns that make the attempt fail"
          />
        </label>
      </div>
    </div>
  )
}

export const SkillLabNode = memo(function SkillLabNode({
  data,
  selected
}: NodeProps<SkillLabNodeType>) {
  const workspaceId = useWorkspaceStore((state) => state.activeWorkspaceId)
  const runMap = useSkillLabStore((state) => state.runsByWorkspace)
  const runErrors = useSkillLabStore((state) => state.errorByWorkspace)
  const terminalSessions = useTerminalStore((state) => state.sessions)
  const terminalPanels = useTerminalStore((state) => state.panels)
  const refreshRuns = useSkillLabStore((state) => state.refresh)
  const launchRun = useSkillLabStore((state) => state.launch)
  const runs = runMap[workspaceId] || []
  const [inventory, setInventory] = useState<ToolkitInventoryResponse | null>(null)
  const [contract, setContract] = useState<SkillLabContract | null>(null)
  const [open, setOpen] = useState(false)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [loadingInventory, setLoadingInventory] = useState(false)
  const [planning, setPlanning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [launching, setLaunching] = useState(false)
  const [goal, setGoal] = useState('')
  const [showGoal, setShowGoal] = useState(false)
  const [depth, setDepth] = useState<'quick' | 'thorough'>('quick')
  const [advanced, setAdvanced] = useState(false)
  const [runnerTarget, setRunnerTarget] = useState('')
  const [error, setError] = useState<string | null>(null)
  const selectedSkillId = String(data.adapterConfig?.selectedSkillId || contract?.skill_id || '')
  const latestRun = runs.find((run) => run.skill_id === selectedSkillId) || runs[0]
  const selectedRun = runs.find((run) => run.run_id === selectedRunId) || null
  const mountedRef = useRef(true)
  const openRequestRef = useRef<unknown>(undefined)

  const loadInventory = useCallback(async (refresh = false) => {
    setLoadingInventory(true)
    const response = await toolkitApi.inventory({ includeRemote: false, refresh })
    if (!mountedRef.current) return
    setLoadingInventory(false)
    if (response.ok && response.data) {
      setInventory(response.data)
      setError(null)
    } else {
      setError(response.error || 'Unable to load skills')
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    void loadInventory(false)
    return () => {
      mountedRef.current = false
    }
  }, [loadInventory])

  useEffect(() => {
    const requestedAt = data.adapterConfig?.openRequestedAt
    if (!requestedAt || requestedAt === openRequestRef.current) return
    openRequestRef.current = requestedAt
    setOpen(true)
  }, [data.adapterConfig?.openRequestedAt])

  useEffect(() => {
    if (!selectedSkillId) return
    let cancelled = false
    const load = async () => {
      const response = await skillLabApi.contract(workspaceId, selectedSkillId)
      if (cancelled) return
      const endpoint = inventory?.endpoint_capabilities.find((item) => item.ok)?.endpoint_id || ''
      if (response.ok && response.data) {
        setContract({
          ...response.data,
          endpoint_id: response.data.endpoint_id || endpoint
        })
      } else setContract(defaultSkillContract(workspaceId, selectedSkillId, endpoint))
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [inventory?.endpoint_capabilities, selectedSkillId, workspaceId])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      if (cancelled || document.hidden) return
      const next = await refreshRuns(workspaceId)
      if (cancelled) return
      const active = next.some((run) => run.status === 'queued' || run.status === 'running')
      timer = window.setTimeout(() => void poll(), active ? ACTIVE_POLL_MS : IDLE_POLL_MS)
    }
    void poll()
    const onVisibility = () => {
      if (!document.hidden) {
        if (timer !== undefined) window.clearTimeout(timer)
        void poll()
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [refreshRuns, workspaceId])

  const skills = (inventory?.skills || []).filter(isCatalogSkillActive)
  const selectedSkill = skills.find((skill) => skillIdFor(skill) === selectedSkillId)
  const endpoints = inventory?.endpoint_capabilities.filter((item) => item.enabled) || []
  const preferredEndpointId = endpoints.find((item) => item.ok)?.endpoint_id || ''
  const agentTerminals = Array.from(terminalSessions.values()).filter(
    (session) => session.agentKind === 'claude' || session.agentKind === 'codex'
  )
  const hermesSkillCount = skills.filter((skill) =>
    [skill.source, ...(skill.available_sources || [])].includes('hermes')
  ).length
  const isTerminalTarget = runnerTarget.startsWith('terminal:') || runnerTarget.startsWith('new:')
  const validationErrors = contract ? validateSkillContract(contract) : ['Choose a skill']

  useEffect(() => {
    if (runnerTarget) return
    if (preferredEndpointId) {
      setRunnerTarget('hermes')
      return
    }
    if (agentTerminals[0]) setRunnerTarget(`terminal:${agentTerminals[0].id}`)
  }, [agentTerminals, preferredEndpointId, runnerTarget])

  const selectSkill = useCallback(
    (skill: ToolkitSkill) => {
      const skillId = skillIdFor(skill)
      setContract(defaultSkillContract(workspaceId, skillId, preferredEndpointId))
      useTerminalStore.getState().updatePanelConfig(data.panelId, {
        ...(data.adapterConfig || {}),
        selectedSkillId: skillId,
        selectedSkillName: skill.name,
        selectedSkillRevision: skill.revision
      })
    },
    [data.adapterConfig, data.panelId, preferredEndpointId, workspaceId]
  )

  const saveContract = async (nextContract = contract) => {
    if (!nextContract) return false
    setSaving(true)
    const response = await skillLabApi.saveContract(nextContract)
    setSaving(false)
    if (!response.ok || !response.data) {
      setError(response.error || 'Unable to save skill criteria')
      return false
    }
    setContract(response.data)
    setError(null)
    return true
  }

  const buildPlan = async () => {
    if (!selectedSkillId) {
      setError('Choose a skill')
      return null
    }
    const endpointId = advanced ? contract?.endpoint_id || preferredEndpointId : preferredEndpointId
    if (!endpointId) {
      setError('Connect a healthy agent in Settings before evaluating')
      return null
    }
    setPlanning(true)
    setError(null)
    const response = await skillLabApi.plan({
      workspace_id: workspaceId,
      skill_id: selectedSkillId,
      endpoint_id: endpointId,
      goal: goal.trim() || undefined,
      depth
    })
    setPlanning(false)
    if (!response.ok || !response.data) {
      setError(response.error || 'Unable to build an evaluation plan')
      return null
    }
    setContract(response.data)
    return response.data
  }

  const startRun = async () => {
    if (!selectedSkillId || (advanced && validationErrors.length)) return
    setLaunching(true)
    setError(null)
    try {
      const runnableContract = advanced ? contract : await buildPlan()
      if (!runnableContract) return
      const saved = await saveContract(runnableContract)
      if (!saved) return
      const run = await launchRun({
        workspace_id: workspaceId,
        skill_id: runnableContract.skill_id,
        endpoint_id: runnableContract.endpoint_id || '',
        cases: runnableContract.cases,
        thresholds: runnableContract.thresholds
      })
      useTerminalStore.getState().updatePanelConfig(data.panelId, {
        ...(data.adapterConfig || {}),
        selectedSkillId: runnableContract.skill_id,
        latestRunId: run.run_id
      })
      setOpen(false)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLaunching(false)
    }
  }

  const handoffToTerminal = async () => {
    if (!selectedSkill) {
      setError('Choose a skill')
      return
    }
    if (!window.bashgym?.terminal) {
      setError('Claude and Codex handoff is available in the desktop app')
      return
    }

    setLaunching(true)
    setError(null)
    try {
      let terminalId = runnerTarget.startsWith('terminal:')
        ? runnerTarget.slice('terminal:'.length)
        : ''
      const requestedKind = runnerTarget === 'new:codex' ? 'codex' : 'claude'
      if (!terminalId) {
        terminalId = useTerminalStore
          .getState()
          .createTerminal(
            undefined,
            requestedKind === 'claude' ? 'Claude Code' : 'Codex',
            requestedKind
          )
        const deadline = Date.now() + 30_000
        while (Date.now() < deadline) {
          const session = useTerminalStore.getState().sessions.get(terminalId)
          if (session?.agentKind === requestedKind) break
          await new Promise((resolve) => window.setTimeout(resolve, 400))
        }
        if (useTerminalStore.getState().sessions.get(terminalId)?.agentKind !== requestedKind) {
          throw new Error(
            `${requestedKind === 'claude' ? 'Claude' : 'Codex'} did not become ready in time`
          )
        }
      }

      const prompt = buildSkillLabTerminalPrompt({
        skill: selectedSkill,
        workspaceId,
        goal,
        depth
      })
      await window.bashgym.terminal.write(terminalId, `${prompt}\r`)
      const terminalStore = useTerminalStore.getState()
      terminalStore.setActiveTerminal(terminalId)
      const panel =
        terminalPanels.find((item) => item.terminalId === terminalId) ||
        terminalStore.panels.find((item) => item.terminalId === terminalId)
      if (panel) terminalStore.setActivePanel(panel.id)
      setOpen(false)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setLaunching(false)
    }
  }

  const startEvaluation = () => (isTerminalTarget ? handoffToTerminal() : startRun())

  const buildContext = () =>
    [
      '## Skill Lab',
      `- selected skill: ${selectedSkill?.name || 'none'}`,
      `- skill id: ${selectedSkillId || 'none'}`,
      `- revision: ${selectedSkill?.revision || 'unknown'}`,
      `- linked endpoint: ${contract?.endpoint_id || 'none'}`,
      `- latest verdict: ${latestRun?.kpis?.verdict || latestRun?.status || 'untested'}`,
      `- success uplift: ${formatSkillPercent(latestRun?.kpis?.success_uplift, true)}`,
      `- routing F1: ${formatSkillPercent(latestRun?.kpis?.routing_f1)}`
    ].join('\n')

  const statusBarClass =
    latestRun?.status === 'failed'
      ? 'bg-status-error'
      : latestRun?.status === 'running' || latestRun?.status === 'queued'
        ? 'bg-accent animate-pulse'
        : latestRun?.kpis?.verdict === 'effective'
          ? 'bg-status-success'
          : latestRun?.kpis
            ? 'bg-status-warning'
            : 'bg-background-tertiary'
  const planScope = skillLabPlanScope(depth)
  const footerIssue = isTerminalTarget
    ? !selectedSkillId
      ? 'Choose a skill'
      : !window.bashgym?.terminal
        ? 'Terminal handoff requires the desktop app'
        : error
    : advanced
      ? validationErrors[0] || error
      : !selectedSkillId
        ? 'Choose a skill'
        : !preferredEndpointId
          ? 'Connect a healthy agent in Settings'
          : error

  return (
    <>
      <DataNodeShell
        panelId={data.panelId}
        title={data.title}
        flowerVariant="skilllab"
        selected={selected}
        hasConnections={data.hasConnections}
        buildContext={data.hasTerminalConnections ? buildContext : undefined}
        statusBarClass={statusBarClass}
        hue={hueFor('skilllab')}
        onFocus={data.onFocus}
        onClose={data.onClose}
        headerRight={
          <>
            <button
              type="button"
              className="node-btn"
              onClick={() => void loadInventory(true)}
              title="Refresh skills"
            >
              {loadingInventory ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <RefreshCw className="h-3 w-3" />
              )}
            </button>
            <button
              type="button"
              className="node-btn node-btn-accent"
              onClick={() => setOpen(true)}
              title="Open Skill Lab"
            >
              <Settings2 className="h-3 w-3" />
            </button>
          </>
        }
      >
        <div className="space-y-2">
          {selectedSkill ? (
            <div className="node-section !p-2">
              <div className="flex min-w-0 items-center gap-2">
                <BrainCircuit className="h-4 w-4 flex-shrink-0 text-accent" />
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-xs font-bold text-text-primary">
                    {selectedSkill.name}
                  </div>
                  <div className="truncate font-mono text-[9px] text-text-muted">
                    {skillSourceLabel(selectedSkill)} ·{' '}
                    {selectedSkill.revision?.slice(0, 8) || 'unversioned'}
                  </div>
                </div>
                <ConfigPill tone={kpiTone(latestRun?.kpis)}>
                  {latestRun?.kpis?.verdict || 'untested'}
                </ConfigPill>
              </div>
            </div>
          ) : (
            <div className="space-y-1">
              <div className="font-mono text-[9px] uppercase text-text-muted">Loaded skills</div>
              {skills.slice(0, 4).map((skill) => (
                <button
                  key={skillIdFor(skill)}
                  type="button"
                  className="node-config-toggle justify-start"
                  onClick={() => {
                    selectSkill(skill)
                    setOpen(true)
                  }}
                >
                  <BrainCircuit className="h-3 w-3" />
                  <span className="truncate">{skill.name}</span>
                </button>
              ))}
            </div>
          )}

          <div className="grid grid-cols-3 gap-1.5">
            <KpiTile
              label="Uplift"
              value={formatSkillPercent(latestRun?.kpis?.success_uplift, true)}
            />
            <KpiTile label="Route F1" value={formatSkillPercent(latestRun?.kpis?.routing_f1)} />
            <KpiTile label="Forced" value={formatSkillPercent(latestRun?.kpis?.forced_pass_rate)} />
          </div>

          {latestRun ? (
            <RunStatus run={latestRun} onClick={() => setSelectedRunId(latestRun.run_id)} />
          ) : (
            <div className="py-1 text-center font-mono text-[10px] text-text-muted">
              No skill evals yet
            </div>
          )}
          {error || runErrors[workspaceId] ? (
            <div className="flex items-start gap-1.5 font-mono text-[9px] text-status-error">
              <AlertCircle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span className="line-clamp-2">{error || runErrors[workspaceId]}</span>
            </div>
          ) : null}
        </div>
      </DataNodeShell>

      <NodeConfigModal
        isOpen={open}
        onClose={() => setOpen(false)}
        title="Skill Lab"
        description="Evaluate a loaded skill"
        size="xl"
        footer={
          <div className="flex w-full items-center gap-3">
            <span className="min-w-0 flex-1 truncate font-mono text-[10px] text-status-error">
              {footerIssue}
            </span>
            {advanced ? (
              <button
                type="button"
                className="btn-secondary"
                disabled={!contract || saving}
                onClick={() => void saveContract()}
              >
                {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
                Save criteria
              </button>
            ) : null}
            <button
              type="button"
              className="btn-primary"
              disabled={
                !selectedSkillId ||
                (!isTerminalTarget && !advanced && !preferredEndpointId) ||
                (!isTerminalTarget && advanced && (!contract || validationErrors.length > 0)) ||
                (isTerminalTarget && !window.bashgym?.terminal) ||
                launching ||
                planning
              }
              onClick={() => void startEvaluation()}
            >
              {launching || planning ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Play className="h-3 w-3" />
              )}
              {planning ? 'Building plan' : isTerminalTarget ? 'Send to agent' : 'Run evaluation'}
            </button>
          </div>
        }
      >
        <ConfigSection title="Evaluation">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <label className="node-field">
              <span className="node-field-label">Loaded skill</span>
              <select
                className="input-brutal min-h-9 font-mono text-[11px]"
                value={selectedSkillId}
                onChange={(event) => {
                  const skill = skills.find((item) => skillIdFor(item) === event.target.value)
                  if (skill) selectSkill(skill)
                }}
              >
                <option value="">Choose skill</option>
                {skills.map((skill) => (
                  <option key={skillIdFor(skill)} value={skillIdFor(skill)}>
                    {skill.name} · {skillSourceLabel(skill)} ·{' '}
                    {skill.revision?.slice(0, 7) || 'unversioned'}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {selectedSkill ? (
            <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-text-secondary">
              {selectedSkill.description || 'No skill description provided.'}
            </p>
          ) : null}
          <div className="flex flex-wrap items-center gap-2 font-mono text-[9px] text-text-muted">
            <ConfigPill tone="neutral">{skills.length} loaded</ConfigPill>
            <ConfigPill tone={hermesSkillCount ? 'accent' : 'neutral'}>
              {hermesSkillCount} in Hermes
            </ConfigPill>
          </div>
          <label className="node-field mt-3">
            <span className="node-field-label">Test with</span>
            <div className="relative">
              <select
                className="input-brutal min-h-10 w-full font-mono text-[11px]"
                value={runnerTarget}
                onChange={(event) => setRunnerTarget(event.target.value)}
              >
                <option value="">Choose an agent</option>
                <option value="hermes" disabled={!preferredEndpointId}>
                  Hermes{preferredEndpointId ? ' · recorded comparison' : ' · unavailable'}
                </option>
                {agentTerminals.map((session) => (
                  <option key={session.id} value={`terminal:${session.id}`}>
                    {session.agentKind === 'claude' ? 'Claude' : 'Codex'} · {session.title}
                  </option>
                ))}
                <option value="new:claude">New Claude terminal</option>
                <option value="new:codex">New Codex terminal</option>
              </select>
            </div>
            <span className="flex items-center gap-1.5 font-mono text-[9px] leading-relaxed text-text-muted">
              {isTerminalTarget ? (
                <>
                  <SquareTerminal className="h-3 w-3 flex-shrink-0" /> The agent receives this skill
                  through BashGym's Skill Lab tools.
                </>
              ) : (
                <>
                  <Bot className="h-3 w-3 flex-shrink-0" /> Hermes records baseline, available, and
                  forced-skill results.
                </>
              )}
            </span>
          </label>
          {showGoal ? (
            <label className="node-field mt-3">
              <span className="node-field-label">Specific focus</span>
              <textarea
                autoFocus
                className="input-brutal min-h-20 font-mono text-[11px]"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                placeholder="Example: avoid activating for general browser questions"
              />
            </label>
          ) : (
            <button
              type="button"
              className="node-config-toggle mt-3 w-full justify-start"
              onClick={() => setShowGoal(true)}
            >
              <Plus className="h-3.5 w-3.5" />
              Add a specific focus
            </button>
          )}
          <div className="mt-3 grid grid-cols-2 gap-2" role="group" aria-label="Evaluation depth">
            {(['quick', 'thorough'] as const).map((option) => {
              const cases = option === 'quick' ? 4 : 8
              return (
                <button
                  key={option}
                  type="button"
                  className={clsx(
                    'node-config-toggle min-h-14 items-start justify-center px-3 text-left',
                    depth === option && '!border-accent !bg-accent/10'
                  )}
                  aria-pressed={depth === option}
                  onClick={() => setDepth(option)}
                >
                  <span className="min-w-0">
                    <span className="block font-mono text-[10px] font-bold uppercase text-text-primary">
                      {option}
                    </span>
                    <span className="mt-0.5 block font-mono text-[9px] text-text-muted">
                      {cases} examples · {cases * 3 + 1} calls
                    </span>
                  </span>
                </button>
              )
            })}
          </div>
          <div className="mt-3 flex items-center gap-3 border-t border-border-subtle pt-3">
            <WandSparkles className="h-4 w-4 flex-shrink-0 text-accent" />
            <div className="min-w-0 flex-1">
              <div className="font-mono text-[10px] font-bold uppercase text-text-primary">
                Generated at launch
              </div>
              <div className="font-mono text-[9px] text-text-muted">
                {planScope.cases} held-out examples · {planScope.totalCalls} total model calls
              </div>
            </div>
            {isTerminalTarget ? (
              <ConfigPill tone="accent">
                <Code2 className="mr-1 inline h-3 w-3" />
                Terminal
              </ConfigPill>
            ) : preferredEndpointId ? (
              <ConfigPill tone="success">Hermes ready</ConfigPill>
            ) : (
              <ConfigPill tone="error">No agent</ConfigPill>
            )}
          </div>
        </ConfigSection>

        <button
          type="button"
          className="node-config-toggle w-full justify-between"
          aria-expanded={advanced}
          onClick={() => setAdvanced((value) => !value)}
        >
          <span className="flex items-center gap-2">
            <Settings2 className="h-3.5 w-3.5" />
            Advanced
          </span>
          <ChevronDown
            className={clsx('h-3.5 w-3.5 transition-transform', advanced && 'rotate-180')}
          />
        </button>

        {advanced ? (
          <>
            <ConfigSection title="Agent and skill details">
              <label className="node-field">
                <span className="node-field-label">Agent endpoint</span>
                <select
                  className="input-brutal min-h-9 font-mono text-[11px]"
                  value={contract?.endpoint_id || ''}
                  disabled={!contract}
                  onChange={(event) =>
                    contract && setContract({ ...contract, endpoint_id: event.target.value })
                  }
                >
                  <option value="">Choose endpoint</option>
                  {endpoints.map((endpoint) => (
                    <option key={endpoint.endpoint_id} value={endpoint.endpoint_id}>
                      {endpoint.label}
                      {endpoint.ok ? '' : ' · unavailable'}
                    </option>
                  ))}
                </select>
              </label>
              {selectedSkill ? (
                <ConfigRows>
                  <ConfigRow label="Revision" value={selectedSkill.revision?.slice(0, 12)} />
                  <ConfigRow
                    label="Allowed tools"
                    value={selectedSkill.allowed_tools?.join(', ') || 'not declared'}
                  />
                  <ConfigRow
                    label="Shadowed copies"
                    value={selectedSkill.shadowed_paths?.length || 0}
                  />
                </ConfigRows>
              ) : null}
            </ConfigSection>

            <ConfigSection title={`Held-out cases (${contract?.cases.length || 0})`}>
              <div className="space-y-3">
                {contract?.cases.map((item, index) => (
                  <CaseEditor
                    key={item.case_id}
                    item={item}
                    index={index}
                    onChange={(next) =>
                      setContract({
                        ...contract,
                        cases: contract.cases.map((candidate) =>
                          candidate.case_id === item.case_id ? next : candidate
                        )
                      })
                    }
                    onRemove={() =>
                      setContract({
                        ...contract,
                        cases: contract.cases.filter(
                          (candidate) => candidate.case_id !== item.case_id
                        )
                      })
                    }
                  />
                ))}
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="node-btn node-btn-wide node-btn-accent"
                    disabled={!selectedSkillId || planning}
                    onClick={() => void buildPlan()}
                  >
                    {planning ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <WandSparkles className="h-3 w-3" />
                    )}
                    Generate cases
                  </button>
                  {contract ? (
                    <button
                      type="button"
                      className="node-btn node-btn-wide"
                      onClick={() =>
                        setContract({
                          ...contract,
                          cases: [...contract.cases, emptySkillCase(contract.cases.length + 1)]
                        })
                      }
                    >
                      <Plus className="h-3 w-3" />
                      Add case
                    </button>
                  ) : null}
                </div>
              </div>
            </ConfigSection>

            <ConfigSection title="Release gate">
              {contract ? (
                <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
                  {(
                    [
                      ['min_uplift', 'Min uplift'],
                      ['min_forced_pass_rate', 'Forced pass'],
                      ['min_routing_precision', 'Route precision'],
                      ['min_routing_recall', 'Route recall'],
                      ['max_false_activation_rate', 'Max false activation']
                    ] as const
                  ).map(([key, label]) => (
                    <label className="node-field" key={key}>
                      <span className="node-field-label">{label}</span>
                      <input
                        className="input-brutal min-h-9 font-mono text-[11px]"
                        type="number"
                        min="0"
                        max="1"
                        step="0.05"
                        value={contract.thresholds[key]}
                        onChange={(event) =>
                          setContract({
                            ...contract,
                            thresholds: {
                              ...contract.thresholds,
                              [key]: Number(event.target.value)
                            }
                          })
                        }
                      />
                    </label>
                  ))}
                </div>
              ) : null}
            </ConfigSection>
          </>
        ) : null}

        <ConfigSection title={`Recent runs (${runs.length})`}>
          <div className="space-y-2">
            {runs.slice(0, 8).map((run) => (
              <RunStatus key={run.run_id} run={run} onClick={() => setSelectedRunId(run.run_id)} />
            ))}
            {!runs.length && (
              <p className="py-3 text-center font-mono text-xs text-text-muted">
                No runs in this workspace
              </p>
            )}
          </div>
        </ConfigSection>
      </NodeConfigModal>

      <NodeConfigModal
        isOpen={Boolean(selectedRun)}
        onClose={() => setSelectedRunId(null)}
        title={selectedRun?.skill_name || 'Skill eval'}
        description={selectedRun ? `${selectedRun.endpoint_id} · ${selectedRun.status}` : undefined}
        size="lg"
        footer={
          <button type="button" className="btn-secondary" onClick={() => setSelectedRunId(null)}>
            Close
          </button>
        }
      >
        {selectedRun ? (
          <>
            <ConfigSection title="KPI summary">
              <div className="mb-2 flex flex-wrap gap-1.5">
                <ConfigPill tone={kpiTone(selectedRun.kpis)}>
                  {selectedRun.kpis?.verdict || selectedRun.status}
                </ConfigPill>
                <ConfigPill tone="neutral">
                  {selectedRun.kpis?.evaluated_cases || 0} cases
                </ConfigPill>
              </div>
              <ConfigRows>
                <ConfigRow
                  label="Success uplift"
                  value={formatSkillPercent(selectedRun.kpis?.success_uplift, true)}
                />
                <ConfigRow
                  label="Baseline pass@1"
                  value={formatSkillPercent(selectedRun.kpis?.baseline_pass_rate)}
                />
                <ConfigRow
                  label="Available pass@1"
                  value={formatSkillPercent(selectedRun.kpis?.available_pass_rate)}
                />
                <ConfigRow
                  label="Forced pass@1"
                  value={formatSkillPercent(selectedRun.kpis?.forced_pass_rate)}
                />
                <ConfigRow
                  label="Routing precision"
                  value={formatSkillPercent(selectedRun.kpis?.routing_precision)}
                />
                <ConfigRow
                  label="Routing recall"
                  value={formatSkillPercent(selectedRun.kpis?.routing_recall)}
                />
                <ConfigRow
                  label="False activation"
                  value={formatSkillPercent(selectedRun.kpis?.false_activation_rate)}
                />
                <ConfigRow
                  label="Average latency"
                  value={
                    selectedRun.kpis?.average_duration_ms != null
                      ? `${Math.round(selectedRun.kpis.average_duration_ms)} ms`
                      : undefined
                  }
                />
              </ConfigRows>
            </ConfigSection>
            <ConfigSection title={`Attempts (${selectedRun.attempts?.length || 0})`}>
              <div className="space-y-2">
                {selectedRun.attempts?.map((attempt, index) => (
                  <div
                    key={`${attempt.case_id}-${attempt.arm}-${index}`}
                    className="border-b border-border-subtle pb-2 last:border-b-0"
                  >
                    <div className="flex items-center gap-2 font-mono text-[10px]">
                      {attempt.passed ? (
                        <CheckCircle2 className="h-3 w-3 text-status-success" />
                      ) : (
                        <XCircle className="h-3 w-3 text-status-error" />
                      )}
                      <span className="min-w-0 flex-1 truncate text-text-primary">
                        {attempt.case_name || attempt.case_id}
                      </span>
                      <span className="uppercase text-text-muted">{attempt.arm}</span>
                    </div>
                    {attempt.error ? (
                      <div className="mt-1 font-mono text-[9px] text-status-error">
                        {attempt.error}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            </ConfigSection>
          </>
        ) : null}
      </NodeConfigModal>
    </>
  )
})
