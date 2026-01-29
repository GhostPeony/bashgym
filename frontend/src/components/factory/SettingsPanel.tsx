import { useState } from 'react'
import {
  Shield,
  ShieldAlert,
  Sparkles,
  Code,
  AlertTriangle,
  Copy,
  Check,
  X,
} from 'lucide-react'
import { clsx } from 'clsx'
import type { FactoryConfig } from '../../services/api'
import { PII_TYPES, RISK_LEVELS } from './types'

interface SettingsPanelProps {
  config: FactoryConfig
  onConfigChange: (config: FactoryConfig) => void
}

export function SettingsPanel({ config, onConfigChange }: SettingsPanelProps) {
  const togglePiiType = (type: string) => {
    const current = config.privacy.pii_types
    const updated = current.includes(type)
      ? current.filter(t => t !== type)
      : [...current, type]
    onConfigChange({
      ...config,
      privacy: { ...config.privacy, pii_types: updated }
    })
  }

  const addBlockedPattern = () => {
    onConfigChange({
      ...config,
      safety: {
        ...config.safety,
        blocked_patterns: [...config.safety.blocked_patterns, '']
      }
    })
  }

  return (
    <div className="p-6 space-y-6">
      {/* 2x2 Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Privacy Card - top-left */}
        <div className="card-elevated p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary" />
              <div>
                <h3 className="text-sm font-medium text-text-primary">Privacy (Safe Synthesizer)</h3>
                <p className="text-xs text-text-muted">Differential privacy for PII protection</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={config.privacy.enabled}
                onChange={(e) => onConfigChange({
                  ...config,
                  privacy: { ...config.privacy, enabled: e.target.checked }
                })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-background-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>

          <div className={clsx('space-y-3', !config.privacy.enabled && 'opacity-50 pointer-events-none')}>
            <div>
              <label className="block text-xs font-medium text-text-primary mb-1.5">Privacy Budget (Epsilon)</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="1"
                  max="16"
                  step="0.5"
                  value={config.privacy.epsilon}
                  onChange={(e) => onConfigChange({
                    ...config,
                    privacy: { ...config.privacy, epsilon: parseFloat(e.target.value) }
                  })}
                  className="flex-1"
                />
                <span className="text-xs font-mono text-text-primary w-10 text-center">{config.privacy.epsilon}</span>
              </div>
              <p className="text-[10px] text-text-muted mt-0.5">Lower = more privacy. Recommended: 4-12</p>
            </div>

            <div>
              <label className="block text-xs font-medium text-text-primary mb-1.5">PII Types to Detect</label>
              <div className="flex flex-wrap gap-1.5">
                {PII_TYPES.map(type => (
                  <button
                    key={type}
                    onClick={() => togglePiiType(type)}
                    className={clsx(
                      'px-2 py-1 rounded-full text-xs transition-colors',
                      config.privacy.pii_types.includes(type)
                        ? 'bg-primary text-white'
                        : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
                    )}
                  >
                    {type.replace('_', ' ')}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Safety Card - top-right */}
        <div className="card-elevated p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-primary" />
              <div>
                <h3 className="text-sm font-medium text-text-primary">Safety Configuration</h3>
                <p className="text-xs text-text-muted">Control risk levels and blocked patterns</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={config.safety.enabled}
                onChange={(e) => onConfigChange({
                  ...config,
                  safety: { ...config.safety, enabled: e.target.checked }
                })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-background-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>

          <div className={clsx('space-y-3', !config.safety.enabled && 'opacity-50 pointer-events-none')}>
            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.safety.block_dangerous_commands}
                  onChange={(e) => onConfigChange({
                    ...config,
                    safety: { ...config.safety, block_dangerous_commands: e.target.checked }
                  })}
                  className="rounded"
                />
                <div>
                  <span className="text-xs font-medium text-text-primary">Block Dangerous Commands</span>
                  <p className="text-[10px] text-text-muted">Prevent rm -rf, sudo, etc.</p>
                </div>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={config.safety.require_confirmation_for_high_risk}
                  onChange={(e) => onConfigChange({
                    ...config,
                    safety: { ...config.safety, require_confirmation_for_high_risk: e.target.checked }
                  })}
                  className="rounded"
                />
                <div>
                  <span className="text-xs font-medium text-text-primary">Require High-Risk Confirmation</span>
                  <p className="text-[10px] text-text-muted">Flag high-risk outputs for review</p>
                </div>
              </label>
            </div>

            <div>
              <label className="block text-xs font-medium text-text-primary mb-1.5">Max Risk Level</label>
              <div className="flex gap-2">
                {RISK_LEVELS.map(level => (
                  <button
                    key={level.id}
                    onClick={() => onConfigChange({
                      ...config,
                      safety: { ...config.safety, max_risk_level: level.id as 'normal' | 'elevated' | 'high' }
                    })}
                    className={clsx(
                      'flex-1 py-1.5 px-3 rounded-lg text-xs font-medium transition-colors',
                      config.safety.max_risk_level === level.id
                        ? level.id === 'normal' ? 'bg-status-success text-white' :
                          level.id === 'elevated' ? 'bg-status-warning text-white' :
                          'bg-status-error text-white'
                        : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
                    )}
                  >
                    {level.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-text-primary mb-1.5">Blocked Patterns</label>
              <div className="space-y-1.5">
                {config.safety.blocked_patterns.map((pattern, idx) => (
                  <div key={idx} className="flex items-center gap-1.5">
                    <input
                      type="text"
                      value={pattern}
                      onChange={(e) => {
                        const updated = [...config.safety.blocked_patterns]
                        updated[idx] = e.target.value
                        onConfigChange({
                          ...config,
                          safety: { ...config.safety, blocked_patterns: updated }
                        })
                      }}
                      placeholder="rm -rf|sudo"
                      className="input text-xs flex-1 font-mono"
                    />
                    <button
                      onClick={() => {
                        const updated = config.safety.blocked_patterns.filter((_, i) => i !== idx)
                        onConfigChange({
                          ...config,
                          safety: { ...config.safety, blocked_patterns: updated }
                        })
                      }}
                      className="p-1 text-text-muted hover:text-status-error"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                <button
                  onClick={addBlockedPattern}
                  className="text-xs text-primary hover:underline"
                >
                  + Add pattern
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Prompt Optimization Card - bottom-left */}
        <div className="card-elevated p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              <div>
                <h3 className="text-sm font-medium text-text-primary">Prompt Optimization (MIPROv2)</h3>
                <p className="text-xs text-text-muted">Auto-optimize prompts before synthesis</p>
              </div>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={config.prompt_optimization.enabled}
                onChange={(e) => onConfigChange({
                  ...config,
                  prompt_optimization: { ...config.prompt_optimization, enabled: e.target.checked }
                })}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-background-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary"></div>
            </label>
          </div>

          <div className={clsx('space-y-3', !config.prompt_optimization.enabled && 'opacity-50 pointer-events-none')}>
            <div>
              <label className="block text-xs font-medium text-text-primary mb-1.5">Intensity</label>
              <div className="flex gap-2">
                {['light', 'medium', 'heavy'].map(intensity => (
                  <button
                    key={intensity}
                    onClick={() => onConfigChange({
                      ...config,
                      prompt_optimization: { ...config.prompt_optimization, intensity }
                    })}
                    className={clsx(
                      'flex-1 py-1.5 px-3 rounded-lg text-xs font-medium transition-colors',
                      config.prompt_optimization.intensity === intensity
                        ? 'bg-primary text-white'
                        : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
                    )}
                  >
                    {intensity.charAt(0).toUpperCase() + intensity.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-text-primary">Max Demos</label>
              <input
                type="number"
                min="1"
                max="10"
                value={config.prompt_optimization.max_demos}
                onChange={(e) => onConfigChange({
                  ...config,
                  prompt_optimization: { ...config.prompt_optimization, max_demos: parseInt(e.target.value) }
                })}
                className="input w-20 text-xs text-center"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-text-primary">Stop Threshold</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={config.prompt_optimization.metric_threshold}
                onChange={(e) => onConfigChange({
                  ...config,
                  prompt_optimization: { ...config.prompt_optimization, metric_threshold: parseFloat(e.target.value) }
                })}
                className="input w-20 text-xs text-center"
              />
            </div>
          </div>
        </div>

        {/* CLI Command Builder - bottom-right */}
        <CLICommandCard config={config} />
      </div>

      {/* Safety Best Practices - full width below */}
      <div className="card-elevated p-4 bg-status-warning/5 border-status-warning/20">
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-status-warning flex-shrink-0" />
          <div className="text-sm">
            <p className="font-medium text-text-primary mb-1">Safety Best Practices</p>
            <ul className="text-text-secondary space-y-1 list-disc list-inside text-xs">
              <li>Mark columns that generate commands as "elevated" or "high" risk</li>
              <li>Include negative examples to train verifiers</li>
              <li>Use blocked patterns to prevent dangerous command synthesis</li>
              <li>Always pair synthetic data with human-in-the-loop review</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function CLICommandCard({ config }: { config: FactoryConfig }) {
  const [copied, setCopied] = useState(false)

  const generateCommand = () => {
    const parts = ['python', '-m', 'bashgym.data_factory']

    if (config.columns.length > 0) {
      parts.push('--schema', `"${config.columns.map(c => `${c.name}:${c.type}`).join(',')}"`)
    }

    if (config.seeds.length > 0) {
      parts.push('--seeds', 'seeds.json')
    }

    parts.push('--output-format', config.output.format)
    parts.push('--row-count', config.output.row_count.toString())

    if (config.output.task_name) {
      parts.push('--task-name', `"${config.output.task_name}"`)
    }

    parts.push('--model', config.default_model.model_id)
    parts.push('--temperature', config.default_model.temperature.toString())

    if (config.privacy.enabled) {
      parts.push('--enable-privacy')
      parts.push('--epsilon', config.privacy.epsilon.toString())
    }

    if (config.safety.enabled) {
      parts.push('--enable-safety')
      if (config.safety.blocked_patterns.length > 0) {
        parts.push('--blocked-patterns', `"${config.safety.blocked_patterns.join('|')}"`)
      }
    }

    if (config.prompt_optimization.enabled) {
      parts.push('--optimize-prompts')
      parts.push('--opt-intensity', config.prompt_optimization.intensity)
    }

    return parts.join(' \\\n  ')
  }

  const command = generateCommand()

  const handleCopy = () => {
    navigator.clipboard.writeText(command.replace(/\\\n\s*/g, ' '))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="card-elevated p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Code className="w-5 h-5 text-primary" />
          <div>
            <h3 className="text-sm font-medium text-text-primary">CLI Command</h3>
            <p className="text-xs text-text-muted">Run from command line</p>
          </div>
        </div>
        <button
          onClick={handleCopy}
          className="btn-secondary text-xs flex items-center gap-1.5 py-1 px-2"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-status-success" />
              Copied
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              Copy
            </>
          )}
        </button>
      </div>
      <pre className="bg-background-tertiary p-3 rounded-lg overflow-x-auto text-xs text-text-primary font-mono">
        {command}
      </pre>
      <p className="text-[10px] text-text-muted mt-2">
        Export seeds to JSON first. Use <code className="px-1 py-0.5 bg-background-tertiary rounded">--help</code> for all options.
      </p>
    </div>
  )
}
