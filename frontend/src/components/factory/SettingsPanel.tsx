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
        <div className="card p-4">
          <div className="card-accent" />
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Shield className="w-5 h-5 text-accent" />
              <div>
                <h3 className="font-brand text-lg text-text-primary">Privacy (Safe Synthesizer)</h3>
                <p className="text-xs text-text-muted font-mono">Differential privacy for PII protection</p>
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
              <div className="w-11 h-6 bg-background-secondary border-brutal border-border rounded-brutal peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-brutal after:border-border after:rounded-brutal after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
            </label>
          </div>

          <div className={clsx('space-y-3', !config.privacy.enabled && 'opacity-50 pointer-events-none')}>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-primary mb-1.5">Privacy Budget (Epsilon)</label>
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
                  className="flex-1 accent-accent"
                />
                <span className="text-xs font-mono text-text-primary w-10 text-center border-brutal border-border rounded-brutal px-1 py-0.5 bg-background-secondary">{config.privacy.epsilon}</span>
              </div>
              <p className="text-[10px] text-text-muted mt-0.5 font-mono">Lower = more privacy. Recommended: 4-12</p>
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-primary mb-1.5">PII Types to Detect</label>
              <div className="flex flex-wrap gap-1.5">
                {PII_TYPES.map(type => (
                  <button
                    key={type}
                    onClick={() => togglePiiType(type)}
                    className={clsx(
                      'tag transition-colors cursor-pointer',
                      config.privacy.pii_types.includes(type)
                        ? 'bg-accent text-white'
                        : ''
                    )}
                  >
                    <span>{type.replace('_', ' ')}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Safety Card - top-right */}
        <div className="card p-4">
          <div className="card-accent" />
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-accent" />
              <div>
                <h3 className="font-brand text-lg text-text-primary">Safety Configuration</h3>
                <p className="text-xs text-text-muted font-mono">Control risk levels and blocked patterns</p>
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
              <div className="w-11 h-6 bg-background-secondary border-brutal border-border rounded-brutal peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-brutal after:border-border after:rounded-brutal after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
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
                  className="rounded-brutal"
                />
                <div>
                  <span className="font-mono text-xs font-semibold text-text-primary">Block Dangerous Commands</span>
                  <p className="text-[10px] text-text-muted font-mono">Prevent rm -rf, sudo, etc.</p>
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
                  className="rounded-brutal"
                />
                <div>
                  <span className="font-mono text-xs font-semibold text-text-primary">Require High-Risk Confirmation</span>
                  <p className="text-[10px] text-text-muted font-mono">Flag high-risk outputs for review</p>
                </div>
              </label>
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-primary mb-1.5">Max Risk Level</label>
              <div className="flex gap-2">
                {RISK_LEVELS.map(level => (
                  <button
                    key={level.id}
                    onClick={() => onConfigChange({
                      ...config,
                      safety: { ...config.safety, max_risk_level: level.id as 'normal' | 'elevated' | 'high' }
                    })}
                    className={clsx(
                      'flex-1 py-1.5 px-3 rounded-brutal text-xs font-mono font-semibold uppercase tracking-widest transition-all border-brutal',
                      config.safety.max_risk_level === level.id
                        ? level.id === 'normal' ? 'bg-status-success text-white border-status-success shadow-brutal-sm' :
                          level.id === 'elevated' ? 'bg-status-warning text-white border-status-warning shadow-brutal-sm' :
                          'bg-status-error text-white border-status-error shadow-brutal-sm'
                        : 'bg-background-secondary text-text-secondary border-border hover:text-text-primary'
                    )}
                  >
                    {level.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-primary mb-1.5">Blocked Patterns</label>
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
                      className="btn-icon w-7 h-7 flex items-center justify-center text-text-muted hover:text-status-error"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
                <button
                  onClick={addBlockedPattern}
                  className="font-mono text-xs uppercase tracking-widest text-accent-dark hover:underline"
                >
                  + Add pattern
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Prompt Optimization Card - bottom-left */}
        <div className="card p-4">
          <div className="card-accent" />
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-accent" />
              <div>
                <h3 className="font-brand text-lg text-text-primary">Prompt Optimization (MIPROv2)</h3>
                <p className="text-xs text-text-muted font-mono">Auto-optimize prompts before synthesis</p>
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
              <div className="w-11 h-6 bg-background-secondary border-brutal border-border rounded-brutal peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-brutal after:border-border after:rounded-brutal after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
            </label>
          </div>

          <div className={clsx('space-y-3', !config.prompt_optimization.enabled && 'opacity-50 pointer-events-none')}>
            <div>
              <label className="block font-mono text-xs uppercase tracking-widest text-text-primary mb-1.5">Intensity</label>
              <div className="flex gap-2">
                {['light', 'medium', 'heavy'].map(intensity => (
                  <button
                    key={intensity}
                    onClick={() => onConfigChange({
                      ...config,
                      prompt_optimization: { ...config.prompt_optimization, intensity }
                    })}
                    className={clsx(
                      'flex-1 py-1.5 px-3 rounded-brutal text-xs font-mono font-semibold uppercase tracking-widest transition-all border-brutal',
                      config.prompt_optimization.intensity === intensity
                        ? 'bg-accent-light text-accent-dark border-border shadow-brutal-sm'
                        : 'bg-background-secondary text-text-secondary border-border hover:text-text-primary'
                    )}
                  >
                    {intensity.charAt(0).toUpperCase() + intensity.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="font-mono text-xs uppercase tracking-widest text-text-primary">Max Demos</label>
              <input
                type="number"
                min="1"
                max="10"
                value={config.prompt_optimization.max_demos}
                onChange={(e) => onConfigChange({
                  ...config,
                  prompt_optimization: { ...config.prompt_optimization, max_demos: parseInt(e.target.value) }
                })}
                className="input w-20 text-xs text-center font-mono"
              />
            </div>

            <div className="flex items-center justify-between">
              <label className="font-mono text-xs uppercase tracking-widest text-text-primary">Stop Threshold</label>
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
                className="input w-20 text-xs text-center font-mono"
              />
            </div>
          </div>
        </div>

        {/* CLI Command Builder - bottom-right */}
        <CLICommandCard config={config} />
      </div>

      {/* Safety Best Practices - full width below */}
      <div className="card p-4 border-status-warning">
        <div className="flex gap-3">
          <AlertTriangle className="w-5 h-5 text-status-warning flex-shrink-0" />
          <div>
            <p className="font-brand text-lg text-text-primary mb-1">Safety Best Practices</p>
            <ul className="text-text-secondary space-y-1 list-disc list-inside text-xs font-mono">
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
    <div className="card p-4">
      <div className="card-accent" />
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Code className="w-5 h-5 text-accent" />
          <div>
            <h3 className="font-brand text-lg text-text-primary">CLI Command</h3>
            <p className="text-xs text-text-muted font-mono">Run from command line</p>
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
      <div className="terminal-chrome">
        <div className="terminal-header">
          <div className="terminal-dot terminal-dot-red" />
          <div className="terminal-dot terminal-dot-yellow" />
          <div className="terminal-dot terminal-dot-green" />
          <span className="font-mono text-xs text-text-muted ml-2">command</span>
        </div>
        <pre className="text-xs text-text-primary font-mono p-3 overflow-x-auto">
          <span className="terminal-prompt">$</span> {command}
        </pre>
      </div>
      <p className="text-[10px] text-text-muted mt-2 font-mono">
        Export seeds to JSON first. Use <code className="px-1 py-0.5 border-brutal border-border rounded-brutal bg-background-secondary">--help</code> for all options.
      </p>
    </div>
  )
}
