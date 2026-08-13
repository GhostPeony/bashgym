/**
 * WebSocket service for real-time updates
 *
 * Message types from backend (bashgym/api/websocket.py):
 * - training:progress - Training metrics update
 * - training:complete - Training finished
 * - training:failed - Training failed
 * - task:status - Task status changed
 * - task:complete - Task completed
 * - trace:added - New trace added
 * - trace:promoted - Trace promoted to gold
 * - trace:demoted - Trace demoted from gold
 * - router:stats - Router statistics update
 * - router:decision - Routing decision made
 * - verification:result - Verification completed
 * - system:status - System status update
 * - error - Error message
 * - connected - Connection established
 * - subscribed - Topic subscription confirmed
 * - unsubscribed - Topic unsubscription confirmed
 * - pong - Heartbeat response
 */

import { useEffect } from 'react'
import { useTrainingStore } from '../stores/trainingStore'
import { useRouterStore } from '../stores/routerStore'
import { useTracesStore } from '../stores/tracesStore'
import { useOrchestratorStore } from '../stores/orchestratorStore'
import { useActivityStore } from '../stores/activityStore'
import { useCascadeStore } from '../stores/cascadeStore'
import { useCanvasOrchestratorStore } from '../stores/canvasOrchestratorStore'
import { useWorkspaceStore } from '../stores/workspaceStore'
import { useHFContextStore } from '../stores/hfContextStore'
import { useCampaignStore } from '../stores/campaignStore'
import { parseCampaignHint, type CampaignHintV1 } from '../stores/campaignFreshness'
import { campaignApi } from './api'

type MessageHandler = (data: any) => void

interface WebSocketMessage {
  type: string
  payload: Record<string, any>
  timestamp?: string
}

// Message types matching backend MessageType enum
export const MessageTypes = {
  // Training events
  TRAINING_QUEUED: 'training:queued',
  TRAINING_PROGRESS: 'training:progress',
  TRAINING_COMPLETE: 'training:complete',
  TRAINING_FAILED: 'training:failed',
  TRAINING_LOG: 'training:log',
  // Task events
  TASK_STATUS: 'task:status',
  TASK_COMPLETE: 'task:complete',
  // Trace events
  TRACE_ADDED: 'trace:added',
  TRACE_PROMOTED: 'trace:promoted',
  TRACE_DEMOTED: 'trace:demoted',
  // Router events
  ROUTER_STATS: 'router:stats',
  ROUTER_DECISION: 'router:decision',
  // Verification events
  VERIFICATION_RESULT: 'verification:result',
  // Guardrail events
  GUARDRAIL_BLOCKED: 'guardrail:blocked',
  GUARDRAIL_WARN: 'guardrail:warn',
  GUARDRAIL_PII_REDACTED: 'guardrail:pii_redacted',
  // System events
  SYSTEM_STATUS: 'system:status',
  ERROR: 'error',
  // Workspace canvas events
  WORKSPACE_CANVAS_INTENT: 'workspace:canvas:intent',
  WORKSPACE_CONTEXT_UPDATED: 'workspace:context:updated',
  CAMPAIGN_HINT: 'campaign:hint',
  CAMPAIGN_SUBSCRIBED: 'campaign:subscribed',
  CAMPAIGN_SUBSCRIPTION_ERROR: 'campaign:subscription-error',
  // Connection events
  CONNECTED: 'connected',
  SUBSCRIBED: 'subscribed',
  UNSUBSCRIBED: 'unsubscribed',
  PONG: 'pong',
  // HuggingFace events
  HF_JOB_STARTED: 'hf:job:started',
  HF_JOB_LOG: 'hf:job:log',
  HF_JOB_COMPLETED: 'hf:job:completed',
  HF_JOB_FAILED: 'hf:job:failed',
  HF_JOB_METRICS: 'hf:job:metrics',
  HF_SPACE_READY: 'hf:space:ready',
  HF_SPACE_ERROR: 'hf:space:error',
  // Orchestration events
  ORCHESTRATION_DECOMPOSING: 'orchestration:decomposing',
  ORCHESTRATION_READY: 'orchestration:ready',
  ORCHESTRATION_TASK_STARTED: 'orchestration:task:started',
  ORCHESTRATION_TASK_COMPLETED: 'orchestration:task:completed',
  ORCHESTRATION_TASK_FAILED: 'orchestration:task:failed',
  ORCHESTRATION_BUDGET_UPDATE: 'orchestration:budget:update',
  ORCHESTRATION_COMPLETE: 'orchestration:complete',
  ORCHESTRATION_CANCELLED: 'orchestration:cancelled',
  ORCHESTRATION_TASK_RETRYING: 'orchestration:task:retrying',
  ORCHESTRATION_MERGE_RESULT: 'orchestration:merge:result',
  // AutoResearch events
  AUTORESEARCH_EXPERIMENT: 'autoresearch:experiment',
  AUTORESEARCH_STATUS: 'autoresearch:status',
  AUTORESEARCH_TRACE_EXPERIMENT: 'autoresearch:trace-experiment',
  AUTORESEARCH_TRACE_COMPLETE: 'autoresearch:trace-research-complete',
  // Schema research events
  AUTORESEARCH_SCHEMA_EXPERIMENT: 'schema-research:experiment',
  AUTORESEARCH_SCHEMA_STATUS: 'schema-research:status',
  // Cascade RL events
  CASCADE_STAGE_STARTED: 'cascade:stage-started',
  CASCADE_STAGE_COMPLETED: 'cascade:stage-completed',
  CASCADE_STAGE_FAILED: 'cascade:stage-failed',
  CASCADE_STAGE_SKIPPED: 'cascade:stage-skipped',
  CASCADE_COMPLETED: 'cascade:completed',
  CASCADE_PROGRESS: 'cascade:progress'
} as const

type LiveTicketResult = { ok: boolean; data?: { ticket?: string } }

interface WebSocketServiceOptions {
  url?: string
  socketFactory?: (url: string) => WebSocket
  liveTicket?: (workspaceId: string) => Promise<LiveTicketResult>
  scheduleReconnect?: (callback: () => void, delay: number) => ReturnType<typeof setTimeout>
  clearScheduledReconnect?: (timer: ReturnType<typeof setTimeout>) => void
  handleCampaignHint?: (hint: CampaignHintV1) => void | Promise<void>
  handleConnection?: (connected: boolean, generation: number) => void | Promise<void>
  handleSubscription?: (
    workspaceId: string,
    subscribed: boolean,
    generation: number
  ) => void | Promise<void>
  addActivity?: (type: string, payload: Record<string, any>) => void
}

const SOCKET_CONNECTING = 0
const SOCKET_OPEN = 1

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

export class WebSocketService {
  private ws: WebSocket | null = null
  private url: string
  private handlers: Map<string, Set<MessageHandler>> = new Map()
  private reconnectAttempts = 0
  private reconnectDelay = 1000
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private manuallyDisconnected = false
  private connectionGeneration = 0
  private campaignWorkspaceRefs = new Map<string, number>()
  private pendingCampaignWorkspaces = new Set<string>()
  private subscribedCampaignWorkspaces = new Set<string>()
  private campaignSubscriptionRetryAttempts = new Map<string, number>()
  private campaignSubscriptionRetryTimers = new Map<string, ReturnType<typeof setTimeout>>()
  private socketFactory: (url: string) => WebSocket
  private liveTicket: (workspaceId: string) => Promise<LiveTicketResult>
  private scheduleReconnect: WebSocketServiceOptions['scheduleReconnect']
  private clearScheduledReconnect: NonNullable<WebSocketServiceOptions['clearScheduledReconnect']>
  private handleCampaignHint: NonNullable<WebSocketServiceOptions['handleCampaignHint']>
  private handleConnection: NonNullable<WebSocketServiceOptions['handleConnection']>
  private handleSubscription: NonNullable<WebSocketServiceOptions['handleSubscription']>
  private addActivity: NonNullable<WebSocketServiceOptions['addActivity']>

  constructor(options: WebSocketServiceOptions = {}) {
    const env =
      (
        import.meta as ImportMeta & {
          env?: { VITE_WS_URL?: string; VITE_MODE?: string }
        }
      ).env ?? {}
    if (options.url) {
      this.url = options.url
    } else if (typeof window !== 'undefined' && window.bashgym?.runtime?.webSocketUrl) {
      this.url = window.bashgym.runtime.webSocketUrl
    } else if (env.VITE_WS_URL) {
      this.url = env.VITE_WS_URL
    } else if (typeof window !== 'undefined' && env.VITE_MODE === 'web') {
      // Web mode: derive WebSocket URL from current host (same-origin)
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      this.url = `${proto}//${window.location.host}/ws`
    } else {
      this.url = 'ws://127.0.0.1:8003/ws'
    }
    this.socketFactory = options.socketFactory ?? ((url) => new WebSocket(url))
    this.liveTicket = options.liveTicket ?? ((workspaceId) => campaignApi.liveTicket(workspaceId))
    this.scheduleReconnect =
      options.scheduleReconnect ?? ((callback, delay) => setTimeout(callback, delay))
    this.clearScheduledReconnect =
      options.clearScheduledReconnect ?? ((timer) => clearTimeout(timer))
    this.handleCampaignHint =
      options.handleCampaignHint ?? ((hint) => useCampaignStore.getState().handleHint(hint))
    this.handleConnection =
      options.handleConnection ??
      ((connected, generation) =>
        useCampaignStore.getState().handleConnection(connected, generation))
    this.handleSubscription =
      options.handleSubscription ??
      ((workspaceId, subscribed, generation) =>
        useCampaignStore.getState().handleSubscription(workspaceId, subscribed, generation))
    this.addActivity =
      options.addActivity ??
      ((type, payload) => useActivityStore.getState().addEvent(type, payload))
  }

  retainCampaignWorkspace(workspaceId: string): () => void {
    const normalized = workspaceId.trim()
    if (!normalized) return () => {}
    this.campaignWorkspaceRefs.set(
      normalized,
      (this.campaignWorkspaceRefs.get(normalized) ?? 0) + 1
    )
    void this.syncCampaignSubscriptions()
    let released = false
    return () => {
      if (released) return
      released = true
      const remaining = (this.campaignWorkspaceRefs.get(normalized) ?? 1) - 1
      if (remaining > 0) {
        this.campaignWorkspaceRefs.set(normalized, remaining)
        return
      }
      this.campaignWorkspaceRefs.delete(normalized)
      this.pendingCampaignWorkspaces.delete(normalized)
      this.subscribedCampaignWorkspaces.delete(normalized)
      this.clearCampaignSubscriptionRetry(normalized)
      this.send('campaign:unsubscribe', { workspace_id: normalized })
    }
  }

  private async syncCampaignSubscriptions(
    socket = this.ws,
    generation = this.connectionGeneration
  ) {
    if (!socket || socket !== this.ws || socket.readyState !== SOCKET_OPEN) return
    for (const workspaceId of this.campaignWorkspaceRefs.keys()) {
      if (
        this.pendingCampaignWorkspaces.has(workspaceId) ||
        this.subscribedCampaignWorkspaces.has(workspaceId)
      )
        continue
      this.pendingCampaignWorkspaces.add(workspaceId)
      let sent = false
      try {
        const response = await this.liveTicket(workspaceId)
        const ticket = response.ok ? response.data?.ticket : undefined
        if (
          ticket &&
          typeof ticket === 'string' &&
          socket === this.ws &&
          generation === this.connectionGeneration &&
          socket.readyState === SOCKET_OPEN &&
          this.campaignWorkspaceRefs.has(workspaceId)
        ) {
          socket.send(JSON.stringify({ type: 'campaign:subscribe', payload: { ticket } }))
          sent = true
        }
      } catch {
        sent = false
      } finally {
        // Keep the workspace pending until the server acknowledges. This avoids
        // minting another single-use ticket when a second consumer mounts.
        if (!sent) {
          this.pendingCampaignWorkspaces.delete(workspaceId)
          this.scheduleCampaignSubscriptionRetry(workspaceId, socket, generation)
        }
      }
    }
  }

  private clearCampaignSubscriptionRetry(workspaceId: string) {
    const timer = this.campaignSubscriptionRetryTimers.get(workspaceId)
    if (timer) this.clearScheduledReconnect(timer)
    this.campaignSubscriptionRetryTimers.delete(workspaceId)
    this.campaignSubscriptionRetryAttempts.delete(workspaceId)
  }

  private clearAllCampaignSubscriptionRetries() {
    for (const workspaceId of this.campaignSubscriptionRetryTimers.keys()) {
      this.clearCampaignSubscriptionRetry(workspaceId)
    }
  }

  private scheduleCampaignSubscriptionRetry(
    workspaceId: string,
    socket: WebSocket,
    generation: number
  ) {
    if (
      this.campaignSubscriptionRetryTimers.has(workspaceId) ||
      !this.campaignWorkspaceRefs.has(workspaceId) ||
      socket !== this.ws ||
      generation !== this.connectionGeneration
    )
      return
    const attempt = (this.campaignSubscriptionRetryAttempts.get(workspaceId) ?? 0) + 1
    this.campaignSubscriptionRetryAttempts.set(workspaceId, attempt)
    const delay = Math.min(250 * 2 ** Math.max(0, attempt - 1), 8_000)
    const timer = this.scheduleReconnect!(() => {
      this.campaignSubscriptionRetryTimers.delete(workspaceId)
      void this.syncCampaignSubscriptions(socket, generation)
    }, delay)
    this.campaignSubscriptionRetryTimers.set(workspaceId, timer)
  }

  connect() {
    if (this.ws?.readyState === SOCKET_OPEN || this.ws?.readyState === SOCKET_CONNECTING) return
    this.manuallyDisconnected = false

    console.log('WebSocket: Attempting connection to:', this.url)
    try {
      const socket = this.socketFactory(this.url)
      const generation = ++this.connectionGeneration
      this.ws = socket
      console.log('WebSocket: Created, readyState:', socket.readyState)

      socket.onopen = () => {
        if (this.ws !== socket || generation !== this.connectionGeneration) return
        console.log('WebSocket connected successfully to', this.url)
        this.reconnectAttempts = 0
        useTrainingStore.getState().setConnected(true)
        this.pendingCampaignWorkspaces.clear()
        this.subscribedCampaignWorkspaces.clear()
        void this.handleConnection(true, generation)
        void this.syncCampaignSubscriptions(socket, generation)
      }

      socket.onclose = (event) => {
        if (this.ws !== socket || generation !== this.connectionGeneration) return
        console.log('WebSocket disconnected, code:', event.code, 'reason:', event.reason)
        this.ws = null
        useTrainingStore.getState().setConnected(false)
        this.pendingCampaignWorkspaces.clear()
        this.subscribedCampaignWorkspaces.clear()
        this.clearAllCampaignSubscriptionRetries()
        void this.handleConnection(false, generation)
        if (!this.manuallyDisconnected) this.attemptReconnect()
      }

      socket.onerror = (error) => {
        if (this.ws !== socket || generation !== this.connectionGeneration) return
        console.error('WebSocket error:', error)
        console.error('WebSocket URL was:', this.url)
        console.error('WebSocket readyState:', socket.readyState)
      }

      socket.onmessage = (event) => {
        if (this.ws !== socket || generation !== this.connectionGeneration) return
        try {
          const parsed: unknown = JSON.parse(event.data)
          if (
            !isPlainRecord(parsed) ||
            typeof parsed.type !== 'string' ||
            !isPlainRecord(parsed.payload)
          )
            return
          const message = parsed as unknown as WebSocketMessage
          this.handleMessage(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      this.attemptReconnect()
    }
  }

  disconnect() {
    this.manuallyDisconnected = true
    if (this.reconnectTimer) this.clearScheduledReconnect(this.reconnectTimer)
    this.reconnectTimer = null
    const socket = this.ws
    this.ws = null
    this.connectionGeneration++
    this.pendingCampaignWorkspaces.clear()
    this.subscribedCampaignWorkspaces.clear()
    this.clearAllCampaignSubscriptionRetries()
    socket?.close()
    useTrainingStore.getState().setConnected(false)
    void this.handleConnection(false, this.connectionGeneration)
  }

  private attemptReconnect() {
    this.reconnectAttempts++
    const delay = Math.min(30_000, this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1))

    console.log(`Attempting reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)

    this.reconnectTimer = this.scheduleReconnect!(() => {
      this.reconnectTimer = null
      this.connect()
    }, delay)
  }

  private handleMessage(message: WebSocketMessage) {
    const { type } = message
    let payload = message.payload

    if (type === MessageTypes.CAMPAIGN_HINT) {
      const hint = parseCampaignHint(payload)
      if (!hint) return
      payload = hint
      void this.handleCampaignHint(hint)
    }

    // Campaign frames are reconciliation transport, not user Activity.
    const campaignControlFrame =
      type === MessageTypes.CAMPAIGN_HINT ||
      type === MessageTypes.CAMPAIGN_SUBSCRIBED ||
      type === MessageTypes.CAMPAIGN_SUBSCRIPTION_ERROR
    if (!campaignControlFrame) this.addActivity(type, payload ?? {})

    // Handle built-in message types
    switch (type) {
      // Training events
      case MessageTypes.TRAINING_QUEUED:
        useCanvasOrchestratorStore.getState().handleTrainingQueued(payload as any)
        break

      case MessageTypes.TRAINING_PROGRESS: {
        useCanvasOrchestratorStore.getState().handleTrainingProgress(payload as any)
        const metrics = {
          loss: payload.loss,
          learningRate: payload.learning_rate,
          gradNorm: payload.grad_norm || 0,
          epoch: payload.epoch,
          step: payload.step || 0,
          totalSteps: payload.total_steps || 0,
          eta: payload.eta,
          simulation: payload.simulation || false,
          timestamp: Date.now(),
          // Richer metrics forwarded from the backend (present only when emitted).
          evalLoss: payload.eval_loss,
          samplesProcessed: payload.samples_processed,
          tokensPerSecond: payload.tokens_per_second,
          gpuMemoryGb: payload.gpu_memory_gb,
          gpuUtilization: payload.gpu_utilization,
          computeTarget: payload.compute_target,
          sessionDistillationLoss: payload.session_distillation_loss,
          sessionDistillationKl: payload.session_distillation_kl,
          sessionDistillationCe: payload.session_distillation_ce,
          sessionDistillationMaskedTokens: payload.session_distillation_masked_tokens
        }
        const store = useTrainingStore.getState()
        if (!store.currentRun && payload.run_id) {
          // Reconnected to an orphaned training run — hydrate the store
          console.log('WebSocket: Reconnected to training run', payload.run_id)
          store.hydrateFromReconnect(payload.run_id, metrics)
        } else if (store.currentRun && payload.run_id && payload.run_id !== store.currentRun.id) {
          // A different run started streaming — switch to it (resets loss history)
          console.log('WebSocket: New training run detected', payload.run_id)
          store.hydrateFromReconnect(payload.run_id, metrics)
        } else {
          store.updateMetrics(metrics)
        }
        break
      }

      case MessageTypes.TRAINING_COMPLETE:
        if (payload.run_id) {
          useCanvasOrchestratorStore
            .getState()
            .handleTrainingTerminalStatus(payload.run_id, 'completed')
        }
        useTrainingStore.getState().setStatus('completed')
        break

      case MessageTypes.TRAINING_FAILED:
        if (payload.run_id) {
          useCanvasOrchestratorStore
            .getState()
            .handleTrainingTerminalStatus(payload.run_id, 'failed')
        }
        useTrainingStore.getState().setStatus('failed')
        break

      case MessageTypes.TRAINING_LOG:
        useTrainingStore.getState().addLog({
          timestamp: Date.now(),
          message: payload.message,
          level: payload.level || 'info'
        })
        break

      // Cloud (HuggingFace/managed) job metrics — map the provider's fields into
      // the same live metrics stream as local runs so the loss curve populates.
      case MessageTypes.HF_JOB_METRICS: {
        const m = payload.metrics || {}
        const store = useTrainingStore.getState()
        if (store.currentRun && m.loss !== undefined) {
          store.updateMetrics({
            loss: m.loss,
            learningRate: m.learning_rate ?? 0,
            gradNorm: m.grad_norm ?? 0,
            epoch: m.epoch ?? 0,
            step: m.step ?? 0,
            totalSteps: m.total_steps ?? 0,
            evalLoss: m.eval_loss,
            timestamp: Date.now()
          })
        }
        break
      }

      // Task events
      case MessageTypes.TASK_STATUS:
      case MessageTypes.TASK_COMPLETE:
        // Task status updates can be handled by custom subscribers
        break

      // Router events
      case MessageTypes.ROUTER_STATS:
        useRouterStore.getState().updateStats({
          totalRequests: payload.total_requests,
          teacherRequests: payload.teacher_requests,
          studentRequests: payload.student_requests,
          teacherSuccessRate: payload.teacher_success_rate,
          studentSuccessRate: payload.student_success_rate,
          avgTeacherLatency: payload.avg_teacher_latency,
          avgStudentLatency: payload.avg_student_latency,
          currentStudentRate: payload.current_student_rate
        })
        break

      // Trace events
      case MessageTypes.TRACE_PROMOTED:
        useTracesStore.getState().promoteTrace(payload.trace_id)
        break

      case MessageTypes.TRACE_DEMOTED:
        useTracesStore.getState().demoteTrace(payload.trace_id)
        break

      case MessageTypes.TRACE_ADDED:
        if (payload.data) {
          useTracesStore.getState().addTrace(payload.data)
        }
        break

      // Verification events
      case MessageTypes.VERIFICATION_RESULT:
        // Verification results handled by custom subscribers
        break

      // Orchestration events
      case MessageTypes.ORCHESTRATION_DECOMPOSING:
        useOrchestratorStore.getState().handleDecomposing(payload)
        break

      case MessageTypes.ORCHESTRATION_READY:
        useOrchestratorStore.getState().handleReady(payload)
        break

      case MessageTypes.ORCHESTRATION_TASK_STARTED:
        useOrchestratorStore.getState().handleTaskStarted(payload)
        break

      case MessageTypes.ORCHESTRATION_TASK_COMPLETED:
        useOrchestratorStore.getState().handleTaskCompleted(payload)
        break

      case MessageTypes.ORCHESTRATION_TASK_FAILED:
        useOrchestratorStore.getState().handleTaskFailed(payload)
        break

      case MessageTypes.ORCHESTRATION_BUDGET_UPDATE:
        useOrchestratorStore.getState().handleBudgetUpdate(payload)
        break

      case MessageTypes.ORCHESTRATION_COMPLETE:
        useOrchestratorStore.getState().handleComplete(payload)
        break

      case MessageTypes.ORCHESTRATION_CANCELLED:
        useOrchestratorStore.getState().handleCancelled(payload)
        break

      case MessageTypes.ORCHESTRATION_TASK_RETRYING:
        useOrchestratorStore.getState().handleTaskRetrying(payload)
        break

      case MessageTypes.ORCHESTRATION_MERGE_RESULT:
        useOrchestratorStore.getState().handleMergeResult(payload)
        break

      // Cascade RL events
      case MessageTypes.CASCADE_STAGE_STARTED:
        useCascadeStore.getState().handleStageStarted(payload)
        break
      case MessageTypes.CASCADE_STAGE_COMPLETED:
        useCascadeStore.getState().handleStageCompleted(payload)
        break
      case MessageTypes.CASCADE_STAGE_FAILED:
        useCascadeStore.getState().handleStageFailed(payload)
        break
      case MessageTypes.CASCADE_STAGE_SKIPPED:
        useCascadeStore.getState().handleStageSkipped(payload)
        break
      case MessageTypes.CASCADE_COMPLETED:
        useCascadeStore.getState().handleCompleted(payload)
        break
      case MessageTypes.CASCADE_PROGRESS:
        useCascadeStore.getState().handleProgress(payload)
        break

      // Connection events
      case MessageTypes.CONNECTED:
        console.log('WebSocket: Connected -', payload.message)
        break

      case MessageTypes.SUBSCRIBED:
        console.log('WebSocket: Subscribed to', payload.topic)
        break

      case MessageTypes.UNSUBSCRIBED:
        console.log('WebSocket: Unsubscribed from', payload.topic)
        break

      case MessageTypes.ERROR:
        console.error('WebSocket: Server error -', payload.message)
        break

      case MessageTypes.PONG:
        // Heartbeat response, connection is alive
        break

      case MessageTypes.WORKSPACE_CANVAS_INTENT: {
        // Intents addressed to a specific workspace only materialize on that
        // workspace's canvas
        const intentWs = (payload as { workspace_id?: string }).workspace_id
        if (intentWs && intentWs !== useWorkspaceStore.getState().activeWorkspaceId) {
          console.info('[ws] workspace intent for inactive workspace skipped', intentWs)
          break
        }
        useCanvasOrchestratorStore.getState().handleWorkspaceIntent(payload as any)
        const semanticType = String((payload as { type?: string }).type || '')
        if (semanticType === 'skill_lab.prepared' || semanticType === 'skill_lab.inspected') {
          useActivityStore.getState().addEvent('skill-eval:prepared', {
            skill_name: (payload as any).entity?.skill_name,
            skill_id: (payload as any).entity?.skill_id
          })
        } else if (semanticType === 'skill_lab.skill.saved') {
          useActivityStore.getState().addEvent('skill-eval:skill-saved', {
            skill_name: (payload as any).entity?.skill_name,
            skill_id: (payload as any).entity?.skill_id
          })
        } else if (semanticType.startsWith('hf-context:')) {
          useActivityStore.getState().addEvent(semanticType, {
            ...(payload as any).entity,
            ...(payload as any).payload
          })
          if (semanticType !== 'hf-context:discovery-started') {
            const workspaceId = intentWs || useWorkspaceStore.getState().activeWorkspaceId
            if (workspaceId) void useHFContextStore.getState().load(workspaceId)
          }
        }
        break
      }

      case MessageTypes.WORKSPACE_CONTEXT_UPDATED:
        // Lightweight notification only; Activity feed already records it.
        break

      case MessageTypes.CAMPAIGN_HINT:
        // Parsed above; the store schedules authoritative snapshot reconciliation.
        break

      case MessageTypes.CAMPAIGN_SUBSCRIBED: {
        const workspaceId = typeof payload.workspace_id === 'string' ? payload.workspace_id : ''
        if (workspaceId && this.campaignWorkspaceRefs.has(workspaceId)) {
          this.pendingCampaignWorkspaces.delete(workspaceId)
          this.subscribedCampaignWorkspaces.add(workspaceId)
          this.clearCampaignSubscriptionRetry(workspaceId)
          void this.handleSubscription(workspaceId, true, this.connectionGeneration)
        }
        break
      }

      case MessageTypes.CAMPAIGN_SUBSCRIPTION_ERROR: {
        // The denial is intentionally non-disclosing, so invalidate all pending
        // attempts and replay every desired workspace on a fresh connection.
        for (const workspaceId of this.campaignWorkspaceRefs.keys()) {
          void this.handleSubscription(workspaceId, false, this.connectionGeneration)
        }
        const socket = this.ws
        if (socket) socket.close()
        break
      }
    }

    // Notify custom handlers
    const handlers = this.handlers.get(type)
    if (handlers) {
      handlers.forEach((handler) => handler(payload))
    }

    // Notify wildcard handlers
    const wildcardHandlers = this.handlers.get('*')
    if (wildcardHandlers) {
      wildcardHandlers.forEach((handler) => handler(message))
    }
  }

  subscribe(type: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set())
    }
    this.handlers.get(type)!.add(handler)

    // Return unsubscribe function
    return () => {
      this.handlers.get(type)?.delete(handler)
    }
  }

  send(type: string, payload: Record<string, any>) {
    if (this.ws?.readyState === SOCKET_OPEN) {
      this.ws.send(JSON.stringify({ type, payload }))
    } else {
      console.warn('WebSocket not connected, cannot send message')
    }
  }

  /**
   * Subscribe to a topic on the server.
   * The server will send messages for this topic to this connection.
   */
  subscribeTopic(topic: string) {
    this.send('subscribe', { topic })
  }

  /**
   * Unsubscribe from a topic on the server.
   */
  unsubscribeTopic(topic: string) {
    this.send('unsubscribe', { topic })
  }

  /**
   * Send a ping to check connection health.
   * Server will respond with a pong message.
   */
  ping() {
    this.send('ping', {})
  }

  /**
   * Check if the WebSocket is connected.
   */
  isConnected(): boolean {
    return this.ws?.readyState === SOCKET_OPEN
  }
}

// Singleton instance
export const wsService = new WebSocketService()

// React hook for WebSocket subscriptions
export function useWebSocket(type: string, handler: MessageHandler) {
  useEffect(() => {
    const unsubscribe = wsService.subscribe(type, handler)
    return unsubscribe
  }, [type, handler])
}
