import { useState } from 'react'
import { Send, Terminal, Wifi, WifiOff } from 'lucide-react'
import { useTerminalStore, useTrainingStore } from '../../stores'
import { clsx } from 'clsx'

export function StatusBar() {
  const { sessions, broadcastCommand, setBroadcastCommand, executeBroadcast } = useTerminalStore()
  const { currentRun, isConnected } = useTrainingStore()

  const [isFocused, setIsFocused] = useState(false)

  const terminalCount = sessions.size
  const hasTerminals = terminalCount > 0

  const handleBroadcast = () => {
    if (broadcastCommand.trim() && hasTerminals) {
      executeBroadcast()
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleBroadcast()
    }
  }

  return (
    <footer className="h-10 flex items-center justify-between px-4 border-t border-border bg-background-card">
      {/* Left Section - Status Info */}
      <div className="flex items-center gap-4">
        {/* Terminal Count */}
        <div className="flex items-center gap-2 text-text-secondary font-mono text-xs">
          <Terminal className="w-4 h-4" />
          <span>
            {terminalCount} Terminal{terminalCount !== 1 ? 's' : ''}
          </span>
        </div>

        {/* Training Status */}
        {currentRun && (
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                'status-dot',
                currentRun.status === 'running' && 'status-success',
                currentRun.status === 'paused' && 'status-warning',
                currentRun.status === 'failed' && 'status-error'
              )}
            />
            <span className="text-text-secondary font-mono text-xs">
              Training: {currentRun.currentMetrics?.step || 0}/
              {currentRun.currentMetrics?.totalSteps || '?'}
            </span>
          </div>
        )}

        {/* Connection Status */}
        <div className="flex items-center gap-2">
          {isConnected ? (
            <>
              <Wifi className="w-4 h-4 text-status-success" />
              <span className="font-mono text-xs border-brutal border-border rounded-brutal px-1.5 py-0.5 text-status-success">
                Connected
              </span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-text-muted" />
              <span className="font-mono text-xs border-brutal border-border rounded-brutal px-1.5 py-0.5 text-text-muted">
                Disconnected
              </span>
            </>
          )}
        </div>
      </div>

      {/* Center/Right Section - Broadcast Input */}
      <div className="flex items-center gap-2 flex-1 max-w-xl mx-4">
        <div
          className={clsx(
            'flex-1 flex items-center gap-2 px-3 py-1.5 border-brutal border-border rounded-brutal transition-colors',
            isFocused
              ? 'border-accent bg-background-primary'
              : 'bg-background-secondary',
            !hasTerminals && 'opacity-50'
          )}
        >
          <span className="text-accent font-mono text-sm">$</span>
          <input
            type="text"
            value={broadcastCommand}
            onChange={(e) => setBroadcastCommand(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder={
              hasTerminals
                ? `Broadcast to ${terminalCount} terminal${terminalCount !== 1 ? 's' : ''}...`
                : 'No terminals open'
            }
            disabled={!hasTerminals}
            className="flex-1 bg-transparent text-text-primary placeholder:text-text-muted text-sm font-mono focus:outline-none"
          />
        </div>
        <button
          onClick={handleBroadcast}
          disabled={!hasTerminals || !broadcastCommand.trim()}
          className={clsx(
            'px-3 py-1.5 rounded-brutal flex items-center gap-2 text-sm font-medium transition-colors',
            hasTerminals && broadcastCommand.trim()
              ? 'btn-primary'
              : 'bg-background-secondary text-text-muted cursor-not-allowed'
          )}
        >
          <Send className="w-4 h-4" />
          <span>Send</span>
        </button>
      </div>

      {/* Right Section - Hotkey Hints */}
      <div className="flex items-center gap-3 font-mono text-xs text-text-muted">
        <kbd className="px-1.5 py-0.5 border-brutal border-border bg-background-secondary rounded-brutal">
          Ctrl+N
        </kbd>
        <span>New</span>
        <kbd className="px-1.5 py-0.5 border-brutal border-border bg-background-secondary rounded-brutal">
          Ctrl+D
        </kbd>
        <span>Theme</span>
      </div>
    </footer>
  )
}
