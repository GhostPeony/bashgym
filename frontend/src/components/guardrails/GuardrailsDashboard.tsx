import { useState, useEffect, useCallback } from 'react'
import {
  Shield,
  ShieldAlert,
  ShieldCheck,
  ShieldX,
  AlertTriangle,
  Plus,
  Trash2,
  Play,
  Save,
  RefreshCw,
  Loader2,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronRight,
  FileText,
  Code,
  MessageSquare,
  Zap,
  Eye,
  Lock,
  Unlock,
  Settings
} from 'lucide-react'
import { clsx } from 'clsx'
import { observabilityApi, GuardrailEvent, GuardrailStats } from '../../services/api'
import { wsService, MessageTypes } from '../../services/websocket'

interface GuardrailRule {
  id: string
  name: string
  type: 'input' | 'output' | 'both'
  category: 'content' | 'injection' | 'topic' | 'pii' | 'custom'
  enabled: boolean
  config: {
    action: 'block' | 'warn' | 'log'
    patterns?: string[]
    topics?: string[]
    threshold?: number
    colang_code?: string
  }
}

interface TestResult {
  rule_id: string
  rule_name: string
  triggered: boolean
  action: string
  details: string
}

const DEFAULT_RULES: GuardrailRule[] = [
  {
    id: 'content_mod',
    name: 'Content Moderation',
    type: 'both',
    category: 'content',
    enabled: true,
    config: {
      action: 'block',
      threshold: 0.8
    }
  },
  {
    id: 'injection_detect',
    name: 'Prompt Injection Detection',
    type: 'input',
    category: 'injection',
    enabled: true,
    config: {
      action: 'block',
      patterns: ['ignore previous', 'disregard instructions', 'you are now']
    }
  },
  {
    id: 'topic_control',
    name: 'Topic Control',
    type: 'both',
    category: 'topic',
    enabled: false,
    config: {
      action: 'warn',
      topics: ['coding', 'software', 'development']
    }
  },
  {
    id: 'pii_filter',
    name: 'PII Filter',
    type: 'output',
    category: 'pii',
    enabled: true,
    config: {
      action: 'block',
      patterns: ['ssn', 'credit_card', 'password']
    }
  },
  {
    id: 'dangerous_cmd',
    name: 'Dangerous Command Blocker',
    type: 'output',
    category: 'custom',
    enabled: true,
    config: {
      action: 'block',
      patterns: ['rm -rf /', 'sudo', 'chmod 777', ':(){:|:&};:', 'mkfs', 'dd if=']
    }
  }
]

const CATEGORY_INFO = {
  content: { label: 'Content Safety', icon: ShieldAlert, color: 'text-status-error' },
  injection: { label: 'Injection Prevention', icon: Lock, color: 'text-status-warning' },
  topic: { label: 'Topic Control', icon: MessageSquare, color: 'text-accent' },
  pii: { label: 'PII Protection', icon: Eye, color: 'text-status-info' },
  custom: { label: 'Custom Rules', icon: Code, color: 'text-text-secondary' }
}

const ACTION_COLORS = {
  block: 'bg-status-error/20 text-status-error border-brutal border-status-error',
  warn: 'bg-status-warning/20 text-status-warning border-brutal border-status-warning',
  log: 'bg-status-info/20 text-status-info border-brutal border-status-info',
  modify: 'bg-accent-light text-accent-dark border-brutal border-accent',
  allow: 'bg-status-success/20 text-status-success border-brutal border-status-success'
}

export function GuardrailsDashboard() {
  const [rules, setRules] = useState<GuardrailRule[]>(DEFAULT_RULES)
  const [isDefaultConfig, setIsDefaultConfig] = useState(true)
  const [expandedRule, setExpandedRule] = useState<string | null>(null)
  const [testInput, setTestInput] = useState('')
  const [testResults, setTestResults] = useState<TestResult[] | null>(null)
  const [isTesting, setIsTesting] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<'rules' | 'colang' | 'logs'>('rules')

  // API data
  const [events, setEvents] = useState<GuardrailEvent[]>([])
  const [stats, setStats] = useState<GuardrailStats | null>(null)
  const [isLoadingEvents, setIsLoadingEvents] = useState(false)
  const [isLoadingStats, setIsLoadingStats] = useState(false)
  const [eventFilter, setEventFilter] = useState<{ action?: string; type?: string }>({})

  // Fetch guardrail events
  const fetchEvents = useCallback(async () => {
    setIsLoadingEvents(true)
    const result = await observabilityApi.listGuardrailEvents({
      action: eventFilter.action,
      check_type: eventFilter.type,
      limit: 100
    })
    if (result.ok && result.data) {
      setEvents(result.data.events)
    }
    setIsLoadingEvents(false)
  }, [eventFilter])

  // Fetch guardrail stats
  const fetchStats = useCallback(async () => {
    setIsLoadingStats(true)
    const result = await observabilityApi.getGuardrailStats()
    if (result.ok && result.data) {
      setStats(result.data)
    }
    setIsLoadingStats(false)
  }, [])

  // Fetch settings on mount
  useEffect(() => {
    const fetchSettings = async () => {
      const result = await observabilityApi.getSettings()
      if (result.ok && result.data) {
        // Update rules based on server settings
        setRules(prev => prev.map(rule => {
          if (rule.category === 'pii' && result.data!.guardrails.pii_filtering !== undefined) {
            return { ...rule, enabled: result.data!.guardrails.pii_filtering }
          }
          if (rule.category === 'injection' && result.data!.guardrails.injection_detection !== undefined) {
            return { ...rule, enabled: result.data!.guardrails.injection_detection }
          }
          if (rule.category === 'custom' && result.data!.guardrails.code_safety !== undefined) {
            return { ...rule, enabled: result.data!.guardrails.code_safety }
          }
          return rule
        }))
      }
    }
    fetchSettings()
    fetchStats()
  }, [fetchStats])

  // Fetch events when tab changes to logs
  useEffect(() => {
    if (activeTab === 'logs') {
      fetchEvents()
    }
  }, [activeTab, fetchEvents])

  // Subscribe to WebSocket events
  useEffect(() => {
    const handleGuardrailBlocked = (payload: any) => {
      const newEvent: GuardrailEvent = {
        timestamp: new Date().toISOString(),
        check_type: payload.check_type,
        location: payload.location,
        action_taken: 'block',
        confidence: payload.confidence,
        original_content: payload.content_preview || '',
        details: payload.details || {}
      }
      setEvents(prev => [newEvent, ...prev].slice(0, 100))
      // Update stats
      setStats(prev => prev ? {
        ...prev,
        total_events: prev.total_events + 1,
        by_action: { ...prev.by_action, block: (prev.by_action.block || 0) + 1 }
      } : null)
    }

    const handleGuardrailWarn = (payload: any) => {
      const newEvent: GuardrailEvent = {
        timestamp: new Date().toISOString(),
        check_type: payload.check_type,
        location: payload.location,
        action_taken: 'warn',
        confidence: payload.confidence,
        original_content: payload.content_preview || '',
        details: payload.details || {}
      }
      setEvents(prev => [newEvent, ...prev].slice(0, 100))
      setStats(prev => prev ? {
        ...prev,
        total_events: prev.total_events + 1,
        by_action: { ...prev.by_action, warn: (prev.by_action.warn || 0) + 1 }
      } : null)
    }

    const handlePiiRedacted = (payload: any) => {
      const newEvent: GuardrailEvent = {
        timestamp: new Date().toISOString(),
        check_type: 'pii_filter',
        location: payload.location,
        action_taken: 'modify',
        confidence: 1.0,
        original_content: `${payload.redaction_count} items redacted`,
        details: { pii_types: payload.pii_types, ...payload.details }
      }
      setEvents(prev => [newEvent, ...prev].slice(0, 100))
      setStats(prev => prev ? {
        ...prev,
        total_events: prev.total_events + 1,
        by_action: { ...prev.by_action, modify: (prev.by_action.modify || 0) + 1 }
      } : null)
    }

    const unsubBlocked = wsService.subscribe(MessageTypes.GUARDRAIL_BLOCKED, handleGuardrailBlocked)
    const unsubWarn = wsService.subscribe(MessageTypes.GUARDRAIL_WARN, handleGuardrailWarn)
    const unsubPii = wsService.subscribe(MessageTypes.GUARDRAIL_PII_REDACTED, handlePiiRedacted)

    return () => {
      unsubBlocked()
      unsubWarn()
      unsubPii()
    }
  }, [])

  const toggleRule = (id: string) => {
    setRules(prev => prev.map(r =>
      r.id === id ? { ...r, enabled: !r.enabled } : r
    ))
    setIsDefaultConfig(false)
  }

  const updateRule = (id: string, updates: Partial<GuardrailRule>) => {
    setRules(prev => prev.map(r =>
      r.id === id ? { ...r, ...updates } : r
    ))
    setIsDefaultConfig(false)
  }

  const addRule = () => {
    const newRule: GuardrailRule = {
      id: `rule_${Date.now()}`,
      name: 'New Rule',
      type: 'both',
      category: 'custom',
      enabled: false,
      config: {
        action: 'warn',
        patterns: []
      }
    }
    setRules(prev => [...prev, newRule])
    setExpandedRule(newRule.id)
    setIsDefaultConfig(false)
  }

  const deleteRule = (id: string) => {
    setRules(prev => prev.filter(r => r.id !== id))
    setIsDefaultConfig(false)
  }

  const runTest = async () => {
    if (!testInput.trim()) return

    setIsTesting(true)
    setTestResults(null)

    // Local pattern matching for testing (basic functionality)
    // Full NeMo guardrails evaluation requires backend connection
    const results: TestResult[] = rules
      .filter(r => r.enabled)
      .map(rule => {
        let triggered = false
        let details = ''

        // Only pattern-based rules can be tested locally
        if (rule.category === 'injection' || rule.category === 'custom' || rule.category === 'pii') {
          const patterns = rule.config.patterns || []
          const matched = patterns.find(p => testInput.toLowerCase().includes(p.toLowerCase()))
          if (matched) {
            triggered = true
            details = `Matched pattern: "${matched}"`
          }
        } else {
          // Content moderation and topic control require NeMo backend
          details = 'Requires NeMo guardrails service for full evaluation'
        }

        return {
          rule_id: rule.id,
          rule_name: rule.name,
          triggered,
          action: triggered ? rule.config.action : 'pass',
          details: triggered ? details : (details || 'No pattern matches')
        }
      })

    setTestResults(results)
    setIsTesting(false)
  }

  const handleSave = async () => {
    setIsSaving(true)

    // Build settings from rules
    const piiRule = rules.find(r => r.category === 'pii')
    const injectionRule = rules.find(r => r.category === 'injection')
    const customRule = rules.find(r => r.category === 'custom')

    const result = await observabilityApi.updateGuardrailSettings({
      pii_filtering: piiRule?.enabled,
      injection_detection: injectionRule?.enabled,
      code_safety: customRule?.enabled
    })

    if (result.ok) {
      setIsDefaultConfig(false)
    }

    setIsSaving(false)
  }

  const enabledCount = rules.filter(r => r.enabled).length

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-6 border-b-brutal border-border">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="font-brand text-3xl text-text-primary">Guardrails</h1>
              {isDefaultConfig && (
                <span className="tag">
                  <span>Default Configuration</span>
                </span>
              )}
            </div>
            <p className="text-sm text-text-secondary mt-1">
              Configure input/output safety filters powered by NemoGuard
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs uppercase tracking-widest text-text-muted mr-2">
              {enabledCount} of {rules.length} active
            </span>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="btn-primary flex items-center gap-2"
            >
              {isSaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              Save Configuration
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mt-6">
          {[
            { id: 'rules' as const, label: 'Rules', icon: Shield },
            { id: 'colang' as const, label: 'Colang Config', icon: Code },
            { id: 'logs' as const, label: 'Activity Log', icon: FileText, badge: stats?.total_events },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 font-mono text-xs uppercase tracking-widest border-brutal rounded-brutal transition-press',
                activeTab === tab.id
                  ? 'bg-accent text-white border-border shadow-brutal-sm'
                  : 'bg-background-card text-text-secondary border-border-subtle hover:border-border hover-press'
              )}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
              {tab.badge !== undefined && tab.badge > 0 && (
                <span className="ml-1 px-1.5 py-0.5 text-xs border-brutal rounded-brutal bg-background-card text-text-primary">
                  {tab.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6">
        {activeTab === 'rules' && (
          <div className="grid grid-cols-3 gap-6">
            {/* Rules List */}
            <div className="col-span-2 space-y-4">
              <div className="card border-brutal shadow-brutal-sm rounded-brutal bg-background-card">
                <div className="flex items-center justify-between p-4 border-b-brutal border-border">
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5 text-accent" />
                    <h3 className="font-brand text-xl text-text-primary">Guardrail Rules</h3>
                  </div>
                  <button onClick={addRule} className="btn-secondary text-sm flex items-center gap-1">
                    <Plus className="w-4 h-4" />
                    Add Rule
                  </button>
                </div>

                <div className="divide-y divide-border">
                  {rules.map(rule => {
                    const CategoryIcon = CATEGORY_INFO[rule.category].icon
                    const isExpanded = expandedRule === rule.id

                    return (
                      <div key={rule.id}>
                        <div className="flex items-center gap-3 p-4">
                          <button
                            onClick={() => setExpandedRule(isExpanded ? null : rule.id)}
                            className="p-1 hover:bg-background-secondary rounded-brutal border-brutal border-transparent hover:border-border transition-press"
                          >
                            {isExpanded ? (
                              <ChevronDown className="w-4 h-4 text-text-muted" />
                            ) : (
                              <ChevronRight className="w-4 h-4 text-text-muted" />
                            )}
                          </button>
                          <CategoryIcon className={clsx('w-5 h-5', CATEGORY_INFO[rule.category].color)} />
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-text-primary">{rule.name}</span>
                              <span className="tag">
                                <span>{rule.type}</span>
                              </span>
                              <span className={clsx(
                                'font-mono text-xs uppercase tracking-widest px-2 py-0.5 border-brutal rounded-brutal',
                                rule.config.action === 'block' ? 'bg-status-error/20 text-status-error border-status-error' :
                                rule.config.action === 'warn' ? 'bg-status-warning/20 text-status-warning border-status-warning' :
                                'bg-status-info/20 text-status-info border-status-info'
                              )}>
                                {rule.config.action}
                              </span>
                            </div>
                            <p className="text-xs text-text-muted font-mono">{CATEGORY_INFO[rule.category].label}</p>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={rule.enabled}
                              onChange={() => toggleRule(rule.id)}
                              className="sr-only peer"
                            />
                            <div className={clsx(
                              'w-10 h-5 border-brutal rounded-brutal transition-colors',
                              rule.enabled ? 'bg-accent border-accent' : 'bg-background-tertiary border-border'
                            )}>
                              <div className={clsx(
                                'absolute top-[3px] w-3.5 h-3.5 bg-white border-brutal border-border rounded-brutal transition-transform',
                                rule.enabled ? 'translate-x-[22px]' : 'translate-x-[3px]'
                              )} />
                            </div>
                          </label>
                          <button
                            onClick={() => deleteRule(rule.id)}
                            className="btn-icon w-8 h-8 flex items-center justify-center text-text-muted hover:text-status-error hover:border-status-error"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>

                        {isExpanded && (
                          <div className="px-4 pb-4 ml-12 space-y-4">
                            <div className="p-4 bg-background-secondary border-brutal border-border rounded-brutal space-y-4">
                              {/* Rule Name */}
                              <div>
                                <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">Rule Name</label>
                                <input
                                  type="text"
                                  value={rule.name}
                                  onChange={(e) => updateRule(rule.id, { name: e.target.value })}
                                  className="input text-sm w-full"
                                />
                              </div>

                              {/* Type and Action */}
                              <div className="grid grid-cols-3 gap-3">
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">Apply To</label>
                                  <select
                                    value={rule.type}
                                    onChange={(e) => updateRule(rule.id, { type: e.target.value as 'input' | 'output' | 'both' })}
                                    className="input text-sm w-full"
                                  >
                                    <option value="input">Input only</option>
                                    <option value="output">Output only</option>
                                    <option value="both">Both</option>
                                  </select>
                                </div>
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">Category</label>
                                  <select
                                    value={rule.category}
                                    onChange={(e) => updateRule(rule.id, { category: e.target.value as GuardrailRule['category'] })}
                                    className="input text-sm w-full"
                                  >
                                    {Object.entries(CATEGORY_INFO).map(([key, info]) => (
                                      <option key={key} value={key}>{info.label}</option>
                                    ))}
                                  </select>
                                </div>
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">Action</label>
                                  <select
                                    value={rule.config.action}
                                    onChange={(e) => updateRule(rule.id, {
                                      config: { ...rule.config, action: e.target.value as 'block' | 'warn' | 'log' }
                                    })}
                                    className="input text-sm w-full"
                                  >
                                    <option value="block">Block</option>
                                    <option value="warn">Warn</option>
                                    <option value="log">Log only</option>
                                  </select>
                                </div>
                              </div>

                              {/* Patterns */}
                              {(rule.category === 'injection' || rule.category === 'custom' || rule.category === 'pii') && (
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                                    Patterns (one per line)
                                  </label>
                                  <textarea
                                    value={rule.config.patterns?.join('\n') || ''}
                                    onChange={(e) => updateRule(rule.id, {
                                      config: { ...rule.config, patterns: e.target.value.split('\n').filter(Boolean) }
                                    })}
                                    rows={4}
                                    className="input text-sm w-full font-mono"
                                    placeholder="Enter patterns to match..."
                                  />
                                </div>
                              )}

                              {/* Topics */}
                              {rule.category === 'topic' && (
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                                    Allowed Topics (comma-separated)
                                  </label>
                                  <input
                                    type="text"
                                    value={rule.config.topics?.join(', ') || ''}
                                    onChange={(e) => updateRule(rule.id, {
                                      config: { ...rule.config, topics: e.target.value.split(',').map(t => t.trim()).filter(Boolean) }
                                    })}
                                    className="input text-sm w-full"
                                    placeholder="coding, development, software"
                                  />
                                </div>
                              )}

                              {/* Threshold */}
                              {rule.category === 'content' && (
                                <div>
                                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1">
                                    Confidence Threshold
                                  </label>
                                  <div className="flex items-center gap-3">
                                    <input
                                      type="range"
                                      min="0"
                                      max="1"
                                      step="0.05"
                                      value={rule.config.threshold || 0.8}
                                      onChange={(e) => updateRule(rule.id, {
                                        config: { ...rule.config, threshold: parseFloat(e.target.value) }
                                      })}
                                      className="flex-1"
                                    />
                                    <span className="font-brand text-xl text-text-primary w-14 text-right">
                                      {((rule.config.threshold || 0.8) * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Test Panel */}
            <div className="space-y-4">
              <div className="card border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
                <h3 className="font-brand text-xl text-text-primary mb-4">Test Guardrails</h3>
                <textarea
                  value={testInput}
                  onChange={(e) => setTestInput(e.target.value)}
                  placeholder="Enter text to test against guardrails..."
                  rows={6}
                  className="input w-full text-sm mb-4"
                />
                <button
                  onClick={runTest}
                  disabled={isTesting || !testInput.trim()}
                  className="btn-primary w-full flex items-center justify-center gap-2"
                >
                  {isTesting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Testing...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Run Test
                    </>
                  )}
                </button>
              </div>

              {testResults && (
                <div className="card border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
                  <h3 className="font-brand text-xl text-text-primary mb-3">Test Results</h3>
                  <div className="space-y-2">
                    {testResults.map(result => (
                      <div
                        key={result.rule_id}
                        className={clsx(
                          'p-3 border-brutal rounded-brutal',
                          result.triggered
                            ? result.action === 'block' ? 'bg-status-error/10 border-status-error' :
                              result.action === 'warn' ? 'bg-status-warning/10 border-status-warning' :
                              'bg-status-info/10 border-status-info'
                            : 'bg-status-success/10 border-status-success'
                        )}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          {result.triggered ? (
                            result.action === 'block' ? <ShieldX className="w-4 h-4 text-status-error" /> :
                            result.action === 'warn' ? <ShieldAlert className="w-4 h-4 text-status-warning" /> :
                            <Shield className="w-4 h-4 text-status-info" />
                          ) : (
                            <ShieldCheck className="w-4 h-4 text-status-success" />
                          )}
                          <span className="text-sm font-medium text-text-primary">{result.rule_name}</span>
                        </div>
                        <p className="text-xs text-text-muted ml-6 font-mono">{result.details}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Quick Stats */}
              <div className="card card-accent border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-brand text-xl text-text-primary">Statistics</h3>
                  <button
                    onClick={fetchStats}
                    disabled={isLoadingStats}
                    className="btn-icon w-8 h-8 flex items-center justify-center"
                  >
                    <RefreshCw className={clsx('w-4 h-4 text-text-muted', isLoadingStats && 'animate-spin')} />
                  </button>
                </div>
                {stats ? (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">Total Events</span>
                      <span className="font-brand text-2xl text-text-primary">{stats.total_events}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                      <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">Block Rate</span>
                      <span className="font-brand text-2xl text-status-error">
                        {(stats.block_rate * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="section-divider" />

                    {Object.entries(stats.by_action).map(([action, count]) => (
                      <div key={action} className="flex items-center justify-between p-3 bg-background-secondary border-brutal border-border rounded-brutal">
                        <div className="flex items-center gap-2">
                          <span className={clsx(
                            'font-mono text-xs uppercase tracking-widest px-2 py-0.5 border-brutal rounded-brutal',
                            ACTION_COLORS[action as keyof typeof ACTION_COLORS] || 'bg-background-tertiary border-border'
                          )}>
                            {action}
                          </span>
                        </div>
                        <span className="font-brand text-xl text-text-primary">{count}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-text-muted font-mono">No statistics available yet</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'colang' && (
          <div className="card border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-6">
            <div className="flex items-center gap-3 mb-4">
              <Code className="w-5 h-5 text-accent" />
              <div>
                <h3 className="font-brand text-xl text-text-primary">Colang Configuration</h3>
                <p className="text-sm text-text-muted">Advanced guardrail rules using NeMo Colang syntax</p>
              </div>
            </div>
            <div className="terminal-chrome">
              <div className="terminal-header">
                <div className="terminal-dot terminal-dot-red" />
                <div className="terminal-dot terminal-dot-yellow" />
                <div className="terminal-dot terminal-dot-green" />
                <span className="font-mono text-xs text-text-muted ml-2">colang.config</span>
              </div>
              <textarea
                className="w-full font-mono text-sm bg-background-terminal text-green-400 p-4 border-none outline-none resize-y"
                rows={18}
                placeholder={`# Define user intents
define user ask about coding
  "how do I write code"
  "help me program"
  "coding question"

# Define bot responses
define bot refuse off topic
  "I can only help with coding-related questions."

# Define flows
define flow
  user ask about coding
  bot respond helpfully

define flow
  user ask off topic
  bot refuse off topic`}
              />
            </div>
            <div className="flex justify-end mt-4">
              <button className="btn-primary flex items-center gap-2">
                <Save className="w-4 h-4" />
                Save Colang Config
              </button>
            </div>
          </div>
        )}

        {activeTab === 'logs' && (
          <div className="card border-brutal shadow-brutal-sm rounded-brutal bg-background-card">
            <div className="p-4 border-b-brutal border-border">
              <div className="flex items-center justify-between">
                <h3 className="font-brand text-xl text-text-primary">Activity Log</h3>
                <div className="flex items-center gap-2">
                  <select
                    value={eventFilter.action || ''}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, action: e.target.value || undefined }))}
                    className="input text-sm"
                  >
                    <option value="">All Actions</option>
                    <option value="block">Blocked</option>
                    <option value="warn">Warned</option>
                    <option value="modify">Modified</option>
                    <option value="allow">Allowed</option>
                  </select>
                  <select
                    value={eventFilter.type || ''}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, type: e.target.value || undefined }))}
                    className="input text-sm"
                  >
                    <option value="">All Types</option>
                    <option value="injection_detection">Injection</option>
                    <option value="code_safety">Code Safety</option>
                    <option value="pii_filter">PII Filter</option>
                    <option value="content_moderation">Content</option>
                  </select>
                  <button
                    onClick={fetchEvents}
                    disabled={isLoadingEvents}
                    className="btn-secondary text-sm flex items-center gap-1"
                  >
                    <RefreshCw className={clsx('w-4 h-4', isLoadingEvents && 'animate-spin')} />
                    Refresh
                  </button>
                </div>
              </div>
            </div>

            {events.length === 0 ? (
              <div className="p-12 text-center">
                <FileText className="w-12 h-12 text-text-muted mx-auto mb-4" />
                <h3 className="font-brand text-xl text-text-primary mb-2">No activity recorded yet</h3>
                <p className="text-text-muted text-sm">Guardrail events will appear here in real-time</p>
              </div>
            ) : (
              <div className="divide-y divide-border max-h-[600px] overflow-y-auto">
                {events.map((event, idx) => (
                  <div key={`${event.timestamp}-${idx}`} className="p-4 hover:bg-background-secondary transition-press">
                    <div className="flex items-start gap-3">
                      {event.action_taken === 'block' ? (
                        <ShieldX className="w-5 h-5 text-status-error mt-0.5" />
                      ) : event.action_taken === 'warn' ? (
                        <ShieldAlert className="w-5 h-5 text-status-warning mt-0.5" />
                      ) : event.action_taken === 'modify' ? (
                        <Eye className="w-5 h-5 text-accent mt-0.5" />
                      ) : (
                        <ShieldCheck className="w-5 h-5 text-status-success mt-0.5" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className={clsx(
                            'font-mono text-xs uppercase tracking-widest px-2 py-0.5 border-brutal rounded-brutal',
                            ACTION_COLORS[event.action_taken as keyof typeof ACTION_COLORS] || 'bg-background-tertiary border-border'
                          )}>
                            {event.action_taken}
                          </span>
                          <span className="tag">
                            <span>{event.check_type.replace(/_/g, ' ')}</span>
                          </span>
                          {event.model_source && (
                            <span className="tag">
                              <span>{event.model_source}</span>
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-text-primary mb-1 truncate">
                          {event.original_content}
                        </p>
                        <div className="flex items-center gap-4 font-mono text-xs text-text-muted">
                          <span>{event.location}</span>
                          <span>Confidence: {(event.confidence * 100).toFixed(0)}%</span>
                          <span>{new Date(event.timestamp).toLocaleString()}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
