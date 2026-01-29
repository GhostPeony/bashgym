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

import { useTrainingStore, useRouterStore, useTracesStore } from '../stores'

type MessageHandler = (data: any) => void

interface WebSocketMessage {
  type: string
  payload: Record<string, any>
  timestamp?: string
}

// Message types matching backend MessageType enum
export const MessageTypes = {
  // Training events
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
  HF_SPACE_READY: 'hf:space:ready',
  HF_SPACE_ERROR: 'hf:space:error',
} as const

class WebSocketService {
  private ws: WebSocket | null = null
  private url: string
  private handlers: Map<string, Set<MessageHandler>> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  constructor() {
    this.url = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    console.log('WebSocket: Attempting connection to:', this.url)
    try {
      this.ws = new WebSocket(this.url)
      console.log('WebSocket: Created, readyState:', this.ws.readyState)

      this.ws.onopen = () => {
        console.log('WebSocket connected successfully to', this.url)
        this.reconnectAttempts = 0
        useTrainingStore.getState().setConnected(true)
      }

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected, code:', event.code, 'reason:', event.reason)
        useTrainingStore.getState().setConnected(false)
        this.attemptReconnect()
      }

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        console.error('WebSocket URL was:', this.url)
        console.error('WebSocket readyState:', this.ws?.readyState)
      }

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
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
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnect attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)

    console.log(`Attempting reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect()
    }, delay)
  }

  private handleMessage(message: WebSocketMessage) {
    const { type, payload } = message

    // Handle built-in message types
    switch (type) {
      // Training events
      case MessageTypes.TRAINING_PROGRESS:
        useTrainingStore.getState().updateMetrics({
          loss: payload.loss,
          learningRate: payload.learning_rate,
          gradNorm: payload.grad_norm || 0,
          epoch: payload.epoch,
          step: payload.step || 0,
          totalSteps: payload.total_steps || 0,
          eta: payload.eta,
          simulation: payload.simulation || false,
          timestamp: Date.now()
        })
        break

      case MessageTypes.TRAINING_COMPLETE:
        useTrainingStore.getState().setStatus('completed')
        break

      case MessageTypes.TRAINING_FAILED:
        useTrainingStore.getState().setStatus('failed')
        break

      case MessageTypes.TRAINING_LOG:
        useTrainingStore.getState().addLog({
          timestamp: Date.now(),
          message: payload.message,
          level: payload.level || 'info'
        })
        break

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
    if (this.ws?.readyState === WebSocket.OPEN) {
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
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Singleton instance
export const wsService = new WebSocketService()

// React hook for WebSocket subscriptions
export function useWebSocket(type: string, handler: MessageHandler) {
  const { useEffect } = require('react')

  useEffect(() => {
    const unsubscribe = wsService.subscribe(type, handler)
    return unsubscribe
  }, [type, handler])
}
