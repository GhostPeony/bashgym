import { useState, useEffect, useCallback } from 'react'
import {
  Key,
  Eye,
  EyeOff,
  CheckCircle,
  XCircle,
  Loader2,
  RefreshCw,
  AlertTriangle,
  Info,
  ExternalLink,
} from 'lucide-react'
import { settingsApi, EnvKeyStatus } from '../../services/api'
import { clsx } from 'clsx'

// Provider configuration
const PROVIDERS: {
  envKey: string
  name: string
  initial: string
  placeholder: string
  helpUrl: string
  color: string
  bgConfigured: string
}[] = [
  {
    envKey: 'ANTHROPIC_API_KEY',
    name: 'Anthropic',
    initial: 'A',
    placeholder: 'sk-ant-...',
    helpUrl: 'https://console.anthropic.com/',
    color: 'text-status-warning',
    bgConfigured: 'bg-status-warning/10',
  },
  {
    envKey: 'OPENAI_API_KEY',
    name: 'OpenAI',
    initial: 'O',
    placeholder: 'sk-...',
    helpUrl: 'https://platform.openai.com/api-keys',
    color: 'text-status-success',
    bgConfigured: 'bg-status-success/10',
  },
  {
    envKey: 'GOOGLE_API_KEY',
    name: 'Google (Gemini)',
    initial: 'G',
    placeholder: 'AI...',
    helpUrl: 'https://aistudio.google.com/apikey',
    color: 'text-status-info',
    bgConfigured: 'bg-status-info/10',
  },
  {
    envKey: 'NVIDIA_API_KEY',
    name: 'NVIDIA',
    initial: 'N',
    placeholder: 'nvapi-...',
    helpUrl: 'https://build.nvidia.com/',
    color: 'text-accent',
    bgConfigured: 'bg-accent-light',
  },
  {
    envKey: 'HF_TOKEN',
    name: 'HuggingFace',
    initial: 'H',
    placeholder: 'hf_...',
    helpUrl: 'https://huggingface.co/settings/tokens',
    color: 'text-accent-dark',
    bgConfigured: 'bg-accent-dark/10',
  },
]

interface TestResult {
  valid: boolean
  error?: string
}

interface ProviderCardProps {
  provider: typeof PROVIDERS[number]
  status: EnvKeyStatus | undefined
  onSave: (envKey: string, value: string) => Promise<void>
  onTest: (envKey: string) => Promise<void>
  testResult: TestResult | null
  isTesting: boolean
  isSaving: boolean
}

function ProviderCard({
  provider,
  status,
  onSave,
  onTest,
  testResult,
  isTesting,
  isSaving,
}: ProviderCardProps) {
  const [editing, setEditing] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [showValue, setShowValue] = useState(false)

  const isConfigured = status?.is_set ?? false

  const handleSave = async () => {
    if (!inputValue.trim()) return
    await onSave(provider.envKey, inputValue.trim())
    setInputValue('')
    setEditing(false)
    setShowValue(false)
  }

  const handleCancel = () => {
    setInputValue('')
    setEditing(false)
    setShowValue(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSave()
    if (e.key === 'Escape') handleCancel()
  }

  return (
    <div className="p-3 border-brutal border-border rounded-brutal bg-background-card shadow-brutal-sm">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          {/* Icon square */}
          <div
            className={clsx(
              'w-10 h-10 border-brutal border-border rounded-brutal flex items-center justify-center font-mono font-semibold text-sm',
              isConfigured
                ? [provider.bgConfigured, provider.color]
                : 'bg-background-secondary text-text-muted'
            )}
          >
            {provider.initial}
          </div>

          {/* Provider info */}
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-mono font-semibold text-text-primary">
                {provider.name}
              </span>
              {isConfigured && (
                <span className="tag text-[10px] py-0 px-1.5">
                  <span>configured</span>
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <code className="text-xs text-text-muted font-mono">{provider.envKey}</code>
              <a
                href={provider.helpUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="text-text-muted hover:text-accent transition-press"
              >
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>

        {/* Right side actions */}
        {!editing && (
          <div className="flex items-center gap-2">
            {isConfigured && (
              <button
                onClick={() => onTest(provider.envKey)}
                disabled={isTesting}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold border-brutal border-border rounded-brutal text-text-secondary hover:border-border hover:text-text-primary transition-press disabled:opacity-50"
              >
                {isTesting ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : testResult ? (
                  testResult.valid ? (
                    <CheckCircle className="w-3.5 h-3.5 text-status-success" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-status-error" />
                  )
                ) : (
                  <Key className="w-3.5 h-3.5" />
                )}
                {isTesting
                  ? 'Testing...'
                  : testResult
                    ? testResult.valid
                      ? 'Valid'
                      : 'Invalid'
                    : 'Test'}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Bottom area: masked value / edit input */}
      <div className="mt-2 ml-[52px]">
        {editing ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <input
                  type={showValue ? 'text' : 'password'}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={provider.placeholder}
                  className="input w-full text-sm pr-9"
                  autoFocus
                />
                <button
                  type="button"
                  onClick={() => setShowValue(!showValue)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary transition-press"
                >
                  {showValue ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={handleSave}
                disabled={!inputValue.trim() || isSaving}
                className="btn-primary flex items-center gap-1.5 px-3 py-1.5 text-xs disabled:opacity-50"
              >
                {isSaving ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <CheckCircle className="w-3.5 h-3.5" />
                )}
                Save
              </button>
              <button
                onClick={handleCancel}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold border-brutal border-border rounded-brutal text-text-secondary hover:text-text-primary transition-press"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : isConfigured ? (
          <div className="flex items-center gap-2">
            <code className="text-xs font-mono text-text-muted">{status?.masked_value}</code>
            {status?.source && (
              <span className="text-[10px] font-mono text-text-muted">({status.source})</span>
            )}
            <button
              onClick={() => setEditing(true)}
              className="text-xs font-mono font-semibold text-accent-dark hover:text-accent transition-press"
            >
              Change
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted">Not configured</span>
            <button
              onClick={() => setEditing(true)}
              className="text-xs font-mono font-semibold text-accent-dark hover:text-accent transition-press"
            >
              Add key
            </button>
          </div>
        )}

        {/* Test error message */}
        {testResult && !testResult.valid && testResult.error && !editing && (
          <div className="flex items-center gap-1.5 mt-1.5 text-status-error">
            <AlertTriangle className="w-3 h-3 flex-shrink-0" />
            <span className="text-xs font-mono">{testResult.error}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export function ApiKeysSection() {
  const [keys, setKeys] = useState<EnvKeyStatus[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [apiError, setApiError] = useState<string | null>(null)
  const [testingKey, setTestingKey] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, TestResult>>({})
  const [savingKey, setSavingKey] = useState<string | null>(null)

  const fetchKeys = useCallback(async () => {
    setIsLoading(true)
    setApiError(null)
    try {
      const result = await settingsApi.getEnvKeys()
      if (result.ok && result.data) {
        setKeys(result.data.keys)
      } else {
        setApiError(result.error || 'Failed to fetch API keys')
      }
    } catch {
      setApiError('API server not running')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchKeys()
  }, [fetchKeys])

  const handleSave = async (envKey: string, value: string) => {
    setSavingKey(envKey)
    // Clear any previous test result for this key
    setTestResults((prev) => {
      const next = { ...prev }
      delete next[envKey]
      return next
    })
    try {
      const result = await settingsApi.updateEnvKeys({ [envKey]: value })
      if (result.ok) {
        await fetchKeys()
      } else {
        setApiError(result.error || 'Failed to save key')
      }
    } catch {
      setApiError('API server not reachable')
    } finally {
      setSavingKey(null)
    }
  }

  const handleTest = async (envKey: string) => {
    setTestingKey(envKey)
    // Clear previous result
    setTestResults((prev) => {
      const next = { ...prev }
      delete next[envKey]
      return next
    })
    try {
      const result = await settingsApi.testEnvKey(envKey)
      if (result.ok && result.data) {
        setTestResults((prev) => ({
          ...prev,
          [envKey]: { valid: result.data!.valid, error: result.data!.message || undefined },
        }))
      } else {
        setTestResults((prev) => ({
          ...prev,
          [envKey]: { valid: false, error: result.error || 'Test failed' },
        }))
      }
    } catch {
      setTestResults((prev) => ({
        ...prev,
        [envKey]: { valid: false, error: 'API server not reachable' },
      }))
    } finally {
      setTestingKey(null)
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
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-brand text-lg text-text-primary">API Keys</h3>
          <p className="text-xs text-text-muted mt-0.5">
            Provider credentials saved to your local .env file
          </p>
        </div>
        <button
          onClick={fetchKeys}
          className="btn-icon w-8 h-8 text-text-muted"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* API Warning */}
      {apiError && (
        <div className="flex items-start gap-2 p-3 border-brutal border-status-warning rounded-brutal bg-background-card">
          <AlertTriangle className="w-4 h-4 text-status-warning flex-shrink-0 mt-0.5" />
          <div className="text-xs text-text-secondary">
            <p className="font-mono font-semibold text-text-primary">API not available</p>
            <p>{apiError}</p>
          </div>
        </div>
      )}

      {/* Provider Cards */}
      <div className="space-y-2">
        {PROVIDERS.map((provider) => {
          const status = keys.find((k) => k.key === provider.envKey)
          return (
            <ProviderCard
              key={provider.envKey}
              provider={provider}
              status={status}
              onSave={handleSave}
              onTest={handleTest}
              testResult={testResults[provider.envKey] ?? null}
              isTesting={testingKey === provider.envKey}
              isSaving={savingKey === provider.envKey}
            />
          )
        })}
      </div>

      {/* Footer info box */}
      <div className="flex items-start gap-2 p-3 border-brutal border-accent rounded-brutal bg-background-card">
        <Info className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
        <p className="text-xs text-text-secondary">
          Keys are saved to your local <code className="px-1 mx-0.5 border border-border rounded-brutal bg-background-secondary font-mono">.env</code> file and never leave your machine.
        </p>
      </div>
    </div>
  )
}
