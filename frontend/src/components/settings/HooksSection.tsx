import { useState, useEffect, useCallback } from 'react'
import {
  CheckCircle,
  XCircle,
  Copy,
  Download,
  Info,
  Loader2,
  FolderOpen,
  AlertTriangle,
  Terminal,
  Sparkles,
  RefreshCw,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Settings
} from 'lucide-react'
import { hooksApi, ToolStatus } from '../../services/api'
import { useTutorialComplete } from '../../hooks'
import { clsx } from 'clsx'

// Tool configuration
const TOOL_CONFIG: Record<string, {
  icon: React.ElementType
  color: string
  description: string
  installable: boolean
  pluginPath: string
}> = {
  'Claude Code': {
    icon: Terminal,
    color: 'text-orange-400',
    description: 'Anthropic\'s AI coding assistant',
    installable: true,
    pluginPath: '~/.claude/hooks/'
  },
  'OpenCode': {
    icon: Sparkles,
    color: 'text-blue-400',
    description: 'Open-source AI coding agent',
    installable: true,
    pluginPath: '~/.config/opencode/plugins/'
  },
  'Aider': {
    icon: Terminal,
    color: 'text-green-400',
    description: 'AI pair programming',
    installable: false,
    pluginPath: ''
  },
  'Continue': {
    icon: Terminal,
    color: 'text-purple-400',
    description: 'VS Code AI assistant',
    installable: false,
    pluginPath: ''
  },
  'Cursor': {
    icon: Terminal,
    color: 'text-cyan-400',
    description: 'AI-first code editor',
    installable: false,
    pluginPath: ''
  }
}

// OpenCode config template for Ollama
const OPENCODE_OLLAMA_CONFIG = `{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen2.5-coder:7b": { "name": "Qwen 2.5 Coder 7B" }
      }
    }
  },
  "model": "ollama/qwen2.5-coder:7b"
}`

interface ToolCardProps {
  tool: ToolStatus
  onInstall: (toolType: string) => Promise<void>
  isInstalling: boolean
}

function ToolCard({ tool, onInstall, isInstalling }: ToolCardProps) {
  const config = TOOL_CONFIG[tool.name] || {
    icon: Terminal,
    color: 'text-text-muted',
    description: '',
    installable: false,
    pluginPath: ''
  }
  const Icon = config.icon

  // Can install if it's a supported tool (Claude Code or OpenCode)
  const canInstall = config.installable

  return (
    <div className="p-3 rounded-lg border bg-background-secondary border-border-color">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={clsx('p-2 rounded-lg bg-background-tertiary', config.color)}>
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-text-primary">{tool.name}</span>
              {tool.installed && (
                <span className="text-xs px-1.5 py-0.5 rounded bg-status-success/20 text-status-success">
                  detected
                </span>
              )}
            </div>
            <p className="text-xs text-text-muted">{config.description}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {tool.hooks_installed ? (
            <div className="flex items-center gap-1 text-status-success">
              <CheckCircle className="w-4 h-4" />
              <span className="text-xs font-medium">Active</span>
            </div>
          ) : canInstall ? (
            <button
              onClick={() => onInstall(tool.adapter_type)}
              disabled={isInstalling}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-primary text-white hover:bg-primary/90 disabled:opacity-50 transition-colors"
            >
              {isInstalling ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Download className="w-3.5 h-3.5" />
              )}
              Install Plugin
            </button>
          ) : (
            <span className="text-xs text-text-muted">Coming soon</span>
          )}
        </div>
      </div>

      {/* Show plugin path when installed */}
      {tool.hooks_installed && config.pluginPath && (
        <div className="mt-2 flex items-center gap-1 text-xs text-text-muted">
          <FolderOpen className="w-3 h-3" />
          <span className="font-mono">{config.pluginPath}</span>
        </div>
      )}
    </div>
  )
}

function OpenCodeLocalModelGuide() {
  const [expanded, setExpanded] = useState(false)
  const [copiedConfig, setCopiedConfig] = useState(false)

  const copyConfig = async () => {
    await navigator.clipboard.writeText(OPENCODE_OLLAMA_CONFIG)
    setCopiedConfig(true)
    setTimeout(() => setCopiedConfig(false), 2000)
  }

  return (
    <div className="rounded-lg border border-border-subtle overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-background-tertiary transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-text-muted" />
          <span className="text-sm font-medium text-text-primary">Configure OpenCode for Local Models</span>
          <span className="text-xs text-text-muted">(optional)</span>
        </div>
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-text-muted" />
        ) : (
          <ChevronRight className="w-4 h-4 text-text-muted" />
        )}
      </button>

      {expanded && (
        <div className="p-4 border-t border-border-subtle bg-background-tertiary/50 space-y-4">
          <p className="text-xs text-text-secondary">
            To use OpenCode with local Ollama models instead of cloud APIs, create this config file:
          </p>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <code className="text-xs text-text-muted">~/.config/opencode/opencode.json</code>
              <button
                onClick={copyConfig}
                className="flex items-center gap-1 text-xs text-primary hover:underline"
              >
                {copiedConfig ? (
                  <>
                    <CheckCircle className="w-3 h-3" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="w-3 h-3" />
                    Copy config
                  </>
                )}
              </button>
            </div>
            <pre className="p-3 text-xs font-mono bg-background-secondary rounded-lg overflow-x-auto max-h-40 text-text-secondary">
              {OPENCODE_OLLAMA_CONFIG}
            </pre>
          </div>

          <div className="text-xs text-text-muted space-y-1">
            <p><strong>Prerequisites:</strong></p>
            <p>1. Install Ollama: <a href="https://ollama.com/download" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">ollama.com/download</a></p>
            <p>2. Pull a model: <code className="px-1 bg-background-secondary rounded">ollama pull qwen2.5-coder:7b</code></p>
            <p>3. Start Ollama: <code className="px-1 bg-background-secondary rounded">ollama serve</code></p>
          </div>

          <a
            href="https://opencode.ai/docs/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1 text-xs text-primary hover:underline"
          >
            OpenCode documentation
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      )}
    </div>
  )
}

export function HooksSection() {
  const [tools, setTools] = useState<ToolStatus[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [installingTool, setInstallingTool] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [installResult, setInstallResult] = useState<{ success: boolean; message: string } | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const { complete: completeTutorialStep } = useTutorialComplete()

  const fetchStatus = useCallback(async () => {
    setIsLoading(true)
    setApiError(null)

    // Default tools that should always be shown
    const defaultTools: ToolStatus[] = [
      { name: 'Claude Code', installed: false, hooks_installed: false, hooks_path: null, adapter_type: 'claude_code' },
      { name: 'OpenCode', installed: false, hooks_installed: false, hooks_path: null, adapter_type: 'opencode' }
    ]

    try {
      const result = await hooksApi.getStatus()
      if (result.ok && result.data) {
        if (result.data.tools && Array.isArray(result.data.tools)) {
          // Merge API tools with defaults to ensure both Claude Code and OpenCode always appear
          const apiToolsMap = new Map(result.data.tools.map((t: ToolStatus) => [t.name, t]))
          const mergedTools = defaultTools.map(dt => {
            const apiTool = apiToolsMap.get(dt.name)
            return apiTool ? { ...dt, ...apiTool } : dt
          })
          setTools(mergedTools)
        } else if (result.data.all_installed !== undefined) {
          // Legacy format - Claude Code only
          setTools([
            {
              name: 'Claude Code',
              installed: true,
              hooks_installed: result.data.all_installed || false,
              hooks_path: result.data.hooks_dir || null,
              adapter_type: 'claude_code'
            },
            {
              name: 'OpenCode',
              installed: false,
              hooks_installed: false,
              hooks_path: null,
              adapter_type: 'opencode'
            }
          ])
        } else {
          setTools(defaultTools)
        }
      } else {
        setApiError(result.error || 'Failed to connect to API')
        setTools(defaultTools)
      }
    } catch (err) {
      setApiError('API server not running')
      setTools(defaultTools)
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  const handleInstall = async (toolType: string) => {
    if (installingTool) return
    setInstallingTool(toolType)
    setInstallResult(null)

    try {
      const result = await hooksApi.install({ tools: [toolType] })
      if (result.ok && result.data) {
        if (result.data.success) {
          const toolName = toolType === 'claude_code' ? 'Claude Code' : 'OpenCode'
          setInstallResult({ success: true, message: `${toolName} trace capture plugin installed!` })
          completeTutorialStep('install_hooks')
        } else {
          setInstallResult({ success: false, message: result.data.errors?.join('; ') || 'Installation failed' })
        }
        await fetchStatus()
      } else {
        setInstallResult({ success: false, message: result.error || 'Installation failed' })
      }
    } catch (err) {
      setInstallResult({ success: false, message: 'API server not reachable. Try the CLI command below.' })
    } finally {
      setInstallingTool(null)
    }
  }

  const handleCopyCommand = async () => {
    try {
      await navigator.clipboard.writeText('python -m bashgym.trace_capture.setup')
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-5 h-5 animate-spin text-text-muted" />
      </div>
    )
  }

  return (
    <div className="space-y-4" data-tutorial="hooks-section">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-text-primary">Trace Capture Plugins</h3>
          <p className="text-xs text-text-muted mt-0.5">
            Capture tool calls from AI coding assistants for training
          </p>
        </div>
        <button
          onClick={fetchStatus}
          className="p-1.5 rounded-lg hover:bg-background-tertiary text-text-muted"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* API Warning */}
      {apiError && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-status-warning/10 border border-status-warning/20">
          <AlertTriangle className="w-4 h-4 text-status-warning flex-shrink-0 mt-0.5" />
          <div className="text-xs text-text-secondary">
            <p className="font-medium text-text-primary">API not available</p>
            <p>Start the backend to enable one-click install, or use CLI below.</p>
          </div>
        </div>
      )}


      {/* Tool Cards - Only show installable tools */}
      <div className="space-y-2">
        {tools
          .filter(t => TOOL_CONFIG[t.name]?.installable)
          .map(tool => (
            <ToolCard
              key={tool.name}
              tool={tool}
              onInstall={handleInstall}
              isInstalling={installingTool === tool.adapter_type}
            />
          ))}
      </div>

      {/* Install Result */}
      {installResult && (
        <div className={clsx(
          'flex items-center gap-2 p-3 rounded-lg text-sm',
          installResult.success
            ? 'bg-status-success/10 text-status-success'
            : 'bg-status-error/10 text-status-error'
        )}>
          {installResult.success ? (
            <CheckCircle className="w-4 h-4" />
          ) : (
            <XCircle className="w-4 h-4" />
          )}
          {installResult.message}
        </div>
      )}

      {/* OpenCode Local Model Guide */}
      <OpenCodeLocalModelGuide />

      {/* CLI Alternative */}
      <div className="space-y-2 pt-2 border-t border-border-subtle">
        <p className="text-xs text-text-muted">Install via CLI (works without API):</p>
        <div className="flex items-center gap-2">
          <code className="flex-1 px-3 py-2 text-xs font-mono bg-background-tertiary rounded-lg text-text-secondary">
            python -m bashgym.trace_capture.setup
          </code>
          <button
            onClick={handleCopyCommand}
            className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
          >
            {copied ? (
              <CheckCircle className="w-4 h-4 text-status-success" />
            ) : (
              <Copy className="w-4 h-4 text-text-muted" />
            )}
          </button>
        </div>
      </div>

      {/* How it works */}
      <div className="flex items-start gap-2 p-3 rounded-lg bg-status-info/10 border border-status-info/20">
        <Info className="w-4 h-4 text-status-info flex-shrink-0 mt-0.5" />
        <div className="text-xs text-text-secondary">
          <p className="font-medium text-text-primary mb-1">How Trace Capture Works</p>
          <p>
            Plugins hook into your AI coding tool and record tool calls (bash, edit, write, read) to
            <code className="px-1 mx-1 bg-background-tertiary rounded">~/.bashgym/traces/</code>.
            These traces can be used to fine-tune local models. Works with <strong>any model</strong> -
            cloud APIs or local Ollama.
          </p>
        </div>
      </div>
    </div>
  )
}
