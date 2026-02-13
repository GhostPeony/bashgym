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
    <div className="terminal-chrome h-full flex flex-col">
      {/* Header — terminal-header with macOS dots and nav buttons */}
      <div className="terminal-header">
        <div className="flex items-center gap-1.5">
          <span className="terminal-dot terminal-dot-red" />
          <span className="terminal-dot terminal-dot-yellow" />
          <span className="terminal-dot terminal-dot-green" />
        </div>

        <div className="flex items-center gap-1 ml-3">
          <button className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary">
            <ArrowLeft className="w-3.5 h-3.5 text-text-muted" />
          </button>
          <button className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary">
            <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
          </button>
          <button
            onClick={() => setUrl(url + '?refresh=' + Date.now())}
            className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary"
          >
            <RefreshCw className="w-3.5 h-3.5 text-text-muted" />
          </button>
        </div>

        {/* URL Bar — hard bordered input */}
        <div className="flex-1 flex items-center gap-2 mx-2">
          <div className="flex-1 flex items-center gap-2 px-3 py-1 border-brutal border-border rounded-brutal bg-background-secondary">
            <Globe className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />
            <input
              type="text"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 bg-transparent text-sm font-mono text-text-primary placeholder:text-text-muted focus:outline-none"
              placeholder="Enter URL..."
            />
          </div>
        </div>

        <button
          onClick={handleClose}
          className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-status-error/20"
          title="Close"
        >
          <X className="w-3.5 h-3.5 text-text-muted hover:text-status-error" />
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
