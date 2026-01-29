import { useState, useEffect } from 'react'
import {
  Cloud,
  Server,
  Layers,
  Database,
  Settings,
  Sparkles,
  AlertCircle,
  ExternalLink,
  Loader2,
  Key,
  CheckCircle2,
  XCircle,
  Eye,
  EyeOff
} from 'lucide-react'
import { hfApi, HFStatus } from '../../services/api'
import { HFStatus as HFStatusBadge } from './HFStatus'
import { CloudTraining } from './CloudTraining'
import { SpaceManager } from './SpaceManager'
import { DatasetBrowser } from './DatasetBrowser'
import { clsx } from 'clsx'

type Tab = 'training' | 'spaces' | 'datasets'

export function HFDashboard() {
  const [status, setStatus] = useState<HFStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<Tab>('training')

  // Token configuration state
  const [tokenInput, setTokenInput] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [configuring, setConfiguring] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const [configSuccess, setConfigSuccess] = useState(false)

  const fetchStatus = async () => {
    const result = await hfApi.getStatus()
    if (result.ok && result.data) {
      setStatus(result.data)
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  const handleConfigureToken = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!tokenInput.trim()) return

    setConfiguring(true)
    setConfigError(null)
    setConfigSuccess(false)

    const result = await hfApi.configureToken(tokenInput.trim())

    if (result.ok && result.data?.success) {
      setConfigSuccess(true)
      setTokenInput('')
      // Refresh status
      await fetchStatus()
    } else {
      setConfigError(result.error || 'Failed to configure token')
    }
    setConfiguring(false)
  }

  const handleRemoveToken = async () => {
    if (!confirm('Remove HuggingFace token? You will need to reconfigure to use HF features.')) return

    const result = await hfApi.removeToken()
    if (result.ok) {
      await fetchStatus()
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-accent-primary" />
      </div>
    )
  }

  // Not configured state
  if (!status?.enabled) {
    return (
      <div className="h-full p-6">
        <div className="max-w-xl mx-auto mt-12">
          <div className="text-center mb-8">
            <Cloud className="w-16 h-16 mx-auto text-text-secondary opacity-50 mb-4" />
            <h1 className="text-2xl font-semibold text-text-primary">HuggingFace Integration</h1>
            <p className="text-text-secondary mt-2">
              Connect your HuggingFace account to access cloud training, ZeroGPU Spaces, and more.
            </p>
          </div>

          <div className="bg-background-secondary rounded-lg border border-border p-6">
            <h2 className="text-lg font-medium text-text-primary mb-4 flex items-center gap-2">
              <Key className="w-5 h-5" />
              Configure Token
            </h2>

            {configSuccess && (
              <div className="mb-4 p-3 bg-status-success/10 border border-status-success/30 rounded-lg flex items-center gap-2 text-status-success">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm">Token configured successfully!</span>
              </div>
            )}

            {configError && (
              <div className="mb-4 p-3 bg-status-error/10 border border-status-error/30 rounded-lg flex items-center gap-2 text-status-error">
                <XCircle className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm">{configError}</span>
              </div>
            )}

            <form onSubmit={handleConfigureToken} className="space-y-4">
              <div>
                <label className="block text-sm text-text-secondary mb-2">
                  HuggingFace API Token
                </label>
                <div className="relative">
                  <input
                    type={showToken ? 'text' : 'password'}
                    value={tokenInput}
                    onChange={(e) => setTokenInput(e.target.value)}
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                    className="w-full px-3 py-2 pr-10 bg-background-primary border border-border rounded-lg text-text-primary text-sm font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowToken(!showToken)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-text-secondary hover:text-text-primary"
                  >
                    {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-text-muted mt-1">
                  Get your token from{' '}
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent-primary hover:underline"
                  >
                    huggingface.co/settings/tokens
                  </a>
                </p>
              </div>

              <button
                type="submit"
                disabled={configuring || !tokenInput.trim()}
                className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-accent-primary text-white rounded-lg hover:bg-accent-primary/90 disabled:opacity-50 transition-colors"
              >
                {configuring ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Verifying...
                  </>
                ) : (
                  <>
                    <Key className="w-4 h-4" />
                    Connect HuggingFace
                  </>
                )}
              </button>
            </form>

            <div className="mt-6 pt-6 border-t border-border">
              <p className="text-xs text-text-muted">
                Alternatively, set <code className="px-1 py-0.5 bg-background-tertiary rounded">HF_TOKEN</code> environment variable and restart the server.
              </p>
            </div>

            <div className="mt-6 p-4 bg-accent-primary/10 rounded-lg border border-accent-primary/30">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-4 h-4 text-accent-primary" />
                <span className="text-sm font-medium text-text-primary">HuggingFace Pro</span>
              </div>
              <p className="text-xs text-text-secondary">
                Pro subscribers ($9/month) get access to cloud training on GPUs, ZeroGPU Spaces, 1TB storage, and $2/month inference credits.
              </p>
              <a
                href="https://huggingface.co/subscribe/pro"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 mt-2 text-xs text-accent-primary hover:underline"
              >
                Learn more about Pro
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Non-Pro state (limited features)
  const isPro = status.pro_enabled

  const tabs = [
    {
      id: 'training' as Tab,
      label: 'Cloud Training',
      icon: Server,
      requiresPro: true,
    },
    {
      id: 'spaces' as Tab,
      label: 'ZeroGPU Spaces',
      icon: Layers,
      requiresPro: true,
    },
    {
      id: 'datasets' as Tab,
      label: 'Datasets',
      icon: Database,
      requiresPro: false,
    },
  ]

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 p-6 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <Cloud className="w-8 h-8 text-accent-primary" />
              <div>
                <h1 className="text-xl font-semibold text-text-primary">HuggingFace</h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-sm text-text-secondary">{status.username}</span>
                  {isPro && (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 text-xs font-medium bg-accent-primary/20 text-accent-primary rounded">
                      <Sparkles className="w-3 h-3" />
                      Pro
                    </span>
                  )}
                  {status.token_source && (
                    <span className="text-xs text-text-muted">
                      (token: {status.token_source === 'env' ? 'env var' : 'stored'})
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {status.token_source === 'stored' && (
              <button
                onClick={handleRemoveToken}
                className="px-3 py-1.5 text-sm text-status-error hover:bg-status-error/10 rounded-lg transition-colors"
                title="Disconnect HuggingFace"
              >
                Disconnect
              </button>
            )}
            <a
              href="https://huggingface.co/settings/profile"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 rounded-lg hover:bg-background-tertiary transition-colors"
              title="HuggingFace Settings"
            >
              <Settings className="w-5 h-5 text-text-secondary" />
            </a>
          </div>
        </div>

        {/* Pro Banner for non-Pro users */}
        {!isPro && (
          <div className="mt-4 p-3 bg-accent-primary/10 rounded-lg border border-accent-primary/30 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-accent-primary" />
              <span className="text-sm text-text-primary">
                Upgrade to Pro for cloud training and ZeroGPU Spaces
              </span>
            </div>
            <a
              href="https://huggingface.co/subscribe/pro"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 px-3 py-1.5 bg-accent-primary text-white text-sm rounded-lg hover:bg-accent-primary/90 transition-colors"
            >
              <Sparkles className="w-4 h-4" />
              Upgrade
            </a>
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mt-4">
          {tabs.map((tab) => {
            const isDisabled = tab.requiresPro && !isPro
            return (
              <button
                key={tab.id}
                onClick={() => !isDisabled && setActiveTab(tab.id)}
                disabled={isDisabled}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  activeTab === tab.id
                    ? 'bg-accent-primary text-white'
                    : isDisabled
                    ? 'text-text-secondary opacity-50 cursor-not-allowed'
                    : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {tab.requiresPro && !isPro && (
                  <Sparkles className="w-3 h-3 text-accent-primary" />
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'training' && <CloudTraining />}
        {activeTab === 'spaces' && <SpaceManager />}
        {activeTab === 'datasets' && <DatasetBrowser />}
      </div>
    </div>
  )
}
