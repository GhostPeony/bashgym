import { useState } from 'react'
import { Settings, Sparkles, Loader2, Key, CheckCircle2, XCircle, Eye, EyeOff } from 'lucide-react'
import { hfApi } from '../../services/api'
import { CloudTraining } from './CloudTraining'
import { SpaceManager } from './SpaceManager'
import { DatasetBrowser } from './DatasetBrowser'
import { MyModels } from './MyModels'
import { BucketsTab } from './BucketsTab'
import { ResearchTab } from './ResearchTab'
import { TracesTab } from './TracesTab'
import { clsx } from 'clsx'
import { GhostPeonyIcon } from '../common/GhostPeonyIcon'
import { hfStatusResource } from '../../stores/hfResources'
import { useSessionResource } from '../../stores/sessionResource'
import { hfDashboardTabs, type HFDashboardTab } from './hfDashboardTabs'

export function HFDashboard() {
  const { data: status, loading, refresh: fetchStatus } = useSessionResource(hfStatusResource)
  const [activeTab, setActiveTab] = useState<HFDashboardTab>('training')

  // Token configuration state
  const [tokenInput, setTokenInput] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [configuring, setConfiguring] = useState(false)
  const [configError, setConfigError] = useState<string | null>(null)
  const [configSuccess, setConfigSuccess] = useState(false)

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
    if (!confirm('Remove HuggingFace token? You will need to reconfigure to use HF features.'))
      return

    const result = await hfApi.removeToken()
    if (result.ok) {
      await fetchStatus()
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-accent" />
      </div>
    )
  }

  // Not configured state
  if (!status?.enabled) {
    return (
      <div className="h-full p-6">
        <div className="max-w-xl mx-auto mt-12">
          <div className="text-center mb-8">
            <GhostPeonyIcon
              name="huggingface"
              size="3xl"
              tone="neutral"
              muted
              className="mx-auto mb-4"
            />
            <h1 className="text-2xl font-brand text-text-primary">HuggingFace Integration</h1>
            <p className="text-text-secondary mt-2">
              Connect your Hugging Face account to browse models, datasets, Spaces, and provider
              jobs.
            </p>
          </div>

          <div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-6">
            <h2 className="text-lg font-brand text-text-primary mb-4 flex items-center gap-2">
              <Key className="w-5 h-5" />
              Configure Token
            </h2>

            {configSuccess && (
              <div className="mb-4 p-3 bg-background-card border-2 border-status-success rounded-brutal flex items-center gap-2 text-status-success">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm font-mono">Token configured successfully!</span>
              </div>
            )}

            {configError && (
              <div className="mb-4 p-3 bg-background-card border-2 border-status-error rounded-brutal flex items-center gap-2 text-status-error">
                <XCircle className="w-4 h-4 flex-shrink-0" />
                <span className="text-sm font-mono">{configError}</span>
              </div>
            )}

            <form onSubmit={handleConfigureToken} className="space-y-4">
              <div>
                <label className="block text-sm font-mono text-text-secondary mb-2 uppercase tracking-widest">
                  HuggingFace API Token
                </label>
                <div className="relative">
                  <input
                    type={showToken ? 'text' : 'password'}
                    value={tokenInput}
                    onChange={(e) => setTokenInput(e.target.value)}
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                    className="input w-full pr-10 font-mono text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShowToken(!showToken)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-text-secondary hover:text-text-primary"
                  >
                    {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-text-muted mt-1 font-mono">
                  Get your token from{' '}
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent hover:underline"
                  >
                    huggingface.co/settings/tokens
                  </a>
                </p>
              </div>

              <button
                type="submit"
                disabled={configuring || !tokenInput.trim()}
                className="btn-primary w-full flex items-center justify-center gap-2"
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

            <div className="mt-6 pt-6 border-t-2 border-border">
              <p className="text-xs text-text-muted font-mono">
                Alternatively, set{' '}
                <code className="px-1 py-0.5 bg-background-secondary border border-border-subtle rounded-brutal font-mono">
                  HF_TOKEN
                </code>{' '}
                environment variable and restart the server.
              </p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Non-Pro state (limited features)
  const isPro = status.pro_enabled

  const tabs = hfDashboardTabs()

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 p-6 border-b-2 border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <GhostPeonyIcon name="huggingface" size="xl" tone="accent" hue={52} active />
              <div>
                <h1 className="text-xl font-brand text-text-primary">HuggingFace</h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-sm text-text-secondary font-mono">{status.username}</span>
                  {isPro && (
                    <span className="tag">
                      <span className="flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        Pro
                      </span>
                    </span>
                  )}
                  {status.token_source && (
                    <span className="text-xs text-text-muted font-mono">
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
                className="btn-ghost text-sm text-status-error hover:bg-background-secondary"
                title="Disconnect HuggingFace"
              >
                Disconnect
              </button>
            )}
            <a
              href="https://huggingface.co/settings/profile"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-icon"
              title="HuggingFace Settings"
            >
              <Settings className="w-5 h-5 text-text-secondary" />
            </a>
          </div>
        </div>

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
                  'flex items-center gap-2 px-4 py-2 text-sm font-mono border-2 rounded-brutal transition-press',
                  activeTab === tab.id
                    ? 'bg-accent text-white border-accent-dark shadow-brutal-sm'
                    : isDisabled
                      ? 'text-text-secondary border-border-subtle cursor-not-allowed bg-background-secondary'
                      : 'text-text-secondary border-border hover:text-text-primary hover:bg-background-secondary hover-press'
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {tab.requiresPro && !isPro && <Sparkles className="w-3 h-3 text-accent" />}
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
        {activeTab === 'models' && <MyModels />}
        {activeTab === 'buckets' && <BucketsTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'traces' && <TracesTab />}
      </div>
    </div>
  )
}
