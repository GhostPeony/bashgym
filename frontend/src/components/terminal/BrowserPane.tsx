import { useState } from 'react'
import { Globe, X, RefreshCw, ArrowLeft, ArrowRight } from 'lucide-react'
import { useTerminalStore } from '../../stores'

interface BrowserPaneProps {
  id: string
  title: string
  url?: string
  isActive: boolean
}

export function BrowserPane({ id, title, url: initialUrl, isActive }: BrowserPaneProps) {
  const { removePanel } = useTerminalStore()
  const [url, setUrl] = useState(initialUrl || 'http://localhost:3000')
  const [inputUrl, setInputUrl] = useState(url)

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation()
    removePanel(id)
  }

  const handleNavigate = () => {
    let newUrl = inputUrl
    if (!newUrl.startsWith('http://') && !newUrl.startsWith('https://')) {
      newUrl = 'http://' + newUrl
    }
    setUrl(newUrl)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleNavigate()
    }
  }

  return (
    <div className="h-full flex flex-col bg-background-secondary">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-background-secondary border-b border-border-subtle">
        <div className="flex items-center gap-1">
          <button className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <button className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary">
            <ArrowRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => setUrl(url + '?refresh=' + Date.now())}
            className="p-1 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* URL Bar */}
        <div className="flex-1 flex items-center gap-2 px-3 py-1 bg-background-tertiary rounded-lg">
          <Globe className="w-4 h-4 text-text-muted flex-shrink-0" />
          <input
            type="text"
            value={inputUrl}
            onChange={(e) => setInputUrl(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
            placeholder="Enter URL..."
          />
        </div>

        <button
          onClick={handleClose}
          className="p-1 rounded hover:bg-status-error/20 text-text-muted hover:text-status-error"
          title="Close"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Browser Content */}
      <div className="flex-1 bg-white">
        <iframe
          src={url}
          className="w-full h-full border-0"
          title={title}
          sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        />
      </div>
    </div>
  )
}
