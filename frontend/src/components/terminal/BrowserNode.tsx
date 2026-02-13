import { memo, useCallback } from 'react'
import { Handle, Position, NodeProps } from '@xyflow/react'
import {
  Globe,
  Maximize2,
  X,
  Copy,
  ExternalLink,
  Loader2,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  Lock,
  Unlock
} from 'lucide-react'
import { clsx } from 'clsx'

export interface BrowserNodeData {
  panelId: string
  url: string
  pageTitle?: string
  favicon?: string
  isLoading?: boolean
  statusCode?: number
  errorMessage?: string
  thumbnailUrl?: string
  onFocus?: (panelId: string) => void
  onClose?: (panelId: string) => void
  onCopyUrl?: (url: string) => void
  onOpenExternal?: (url: string) => void
  onRefresh?: (panelId: string) => void
}

export type BrowserNodeType = NodeProps<BrowserNodeData>

// Get status color based on HTTP status code
function getStatusColor(statusCode?: number): string {
  if (!statusCode) return 'text-text-muted'
  if (statusCode >= 200 && statusCode < 300) return 'text-status-success'
  if (statusCode >= 300 && statusCode < 400) return 'text-accent'
  if (statusCode >= 400 && statusCode < 500) return 'text-status-warning'
  if (statusCode >= 500) return 'text-status-error'
  return 'text-text-muted'
}

// Get status badge color — solid backgrounds, hard borders
function getStatusBadgeColor(statusCode?: number): string {
  if (!statusCode) return 'bg-background-tertiary text-text-muted border-border-subtle'
  if (statusCode >= 200 && statusCode < 300) return 'bg-status-success text-white border-status-success'
  if (statusCode >= 300 && statusCode < 400) return 'bg-accent text-white border-accent'
  if (statusCode >= 400 && statusCode < 500) return 'bg-status-warning text-white border-status-warning'
  if (statusCode >= 500) return 'bg-status-error text-white border-status-error'
  return 'bg-background-tertiary text-text-muted border-border-subtle'
}

// Get status text
function getStatusText(statusCode?: number): string {
  const texts: Record<number, string> = {
    200: 'OK',
    201: 'Created',
    204: 'No Content',
    301: 'Moved',
    302: 'Found',
    304: 'Not Modified',
    400: 'Bad Request',
    401: 'Unauthorized',
    403: 'Forbidden',
    404: 'Not Found',
    500: 'Server Error',
    502: 'Bad Gateway',
    503: 'Unavailable'
  }
  return statusCode ? texts[statusCode] || String(statusCode) : ''
}

// Extract domain from URL
function getDomain(url: string): string {
  try {
    const parsed = new URL(url)
    return parsed.hostname
  } catch {
    return url
  }
}

// Check if URL is HTTPS
function isSecure(url: string): boolean {
  return url.startsWith('https://')
}

// Truncate URL for display
function truncateUrl(url: string, maxLength: number = 40): string {
  if (url.length <= maxLength) return url
  return url.slice(0, maxLength - 3) + '...'
}

export const BrowserNode = memo(function BrowserNode({ data, selected }: BrowserNodeType) {
  const {
    panelId,
    url,
    pageTitle,
    favicon,
    isLoading,
    statusCode,
    errorMessage,
    thumbnailUrl,
    onFocus,
    onClose,
    onCopyUrl,
    onOpenExternal,
    onRefresh
  } = data

  const handleFocus = useCallback(() => {
    onFocus?.(panelId)
  }, [panelId, onFocus])

  const handleClose = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onClose?.(panelId)
  }, [panelId, onClose])

  const handleCopyUrl = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onCopyUrl?.(url)
  }, [url, onCopyUrl])

  const handleOpenExternal = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onOpenExternal?.(url)
  }, [url, onOpenExternal])

  const handleRefresh = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    onRefresh?.(panelId)
  }, [panelId, onRefresh])

  const domain = getDomain(url)
  const secure = isSecure(url)
  const hasError = !!errorMessage || (statusCode && statusCode >= 400)

  return (
    <div
      className={clsx(
        'w-[300px] card !rounded-brutal border-brutal cursor-pointer',
        hasError ? 'border-status-error' : 'border-border hover:border-border',
        selected && 'border-accent shadow-brutal'
      )}
      onClick={handleFocus}
    >
      {/* Connection handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!bg-accent !w-2 !h-2 !border-brutal !border-border"
      />

      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 bg-background-secondary border-b border-brutal border-border rounded-t-brutal">
        <div className="p-1.5 border-brutal border-border-subtle rounded-brutal bg-background-tertiary">
          {favicon ? (
            <img src={favicon} alt="" className="w-4 h-4" onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none'
            }} />
          ) : (
            <Globe className="w-4 h-4 text-text-muted" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-mono font-semibold text-text-primary truncate block" title={pageTitle || url}>
            {pageTitle || domain}
          </span>
          <div className="flex items-center gap-1 text-[10px] text-text-muted font-mono">
            {secure ? (
              <Lock className="w-2.5 h-2.5 text-status-success" />
            ) : (
              <Unlock className="w-2.5 h-2.5 text-status-warning" />
            )}
            <span className="truncate">{domain}</span>
          </div>
        </div>
        <div className="flex items-center gap-0.5">
          {onRefresh && (
            <button
              onClick={handleRefresh}
              className={clsx(
                'p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press',
                isLoading && 'animate-spin'
              )}
              title="Refresh"
              disabled={isLoading}
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          )}
          {onCopyUrl && (
            <button
              onClick={handleCopyUrl}
              className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
              title="Copy URL"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
          )}
          {onOpenExternal && (
            <button
              onClick={handleOpenExternal}
              className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
              title="Open in browser"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          )}
          <button
            onClick={handleFocus}
            className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
            title="Focus panel"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-status-error/20 text-text-muted hover:text-status-error transition-press"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 text-xs border-b border-brutal border-border font-mono">
        {isLoading ? (
          <>
            <Loader2 className="w-3 h-3 animate-spin text-accent" />
            <span className="text-text-muted">Loading...</span>
          </>
        ) : hasError ? (
          <>
            <AlertCircle className="w-3 h-3 text-status-error" />
            <span className="text-status-error truncate flex-1">
              {errorMessage || `Error ${statusCode}`}
            </span>
          </>
        ) : statusCode ? (
          <>
            <CheckCircle2 className={clsx('w-3 h-3', getStatusColor(statusCode))} />
            <span className={clsx(
              'px-1.5 py-0.5 text-[10px] font-mono font-semibold uppercase',
              'border-brutal rounded-brutal',
              getStatusBadgeColor(statusCode)
            )}>
              {statusCode} {getStatusText(statusCode)}
            </span>
          </>
        ) : (
          <span className="text-text-muted">Ready</span>
        )}
      </div>

      {/* Thumbnail preview area */}
      <div className="h-[120px] bg-background-tertiary flex items-center justify-center border-b border-brutal border-border-subtle">
        {thumbnailUrl ? (
          <img
            src={thumbnailUrl}
            alt="Page preview"
            className="w-full h-full object-cover object-top"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none'
            }}
          />
        ) : isLoading ? (
          <div className="flex flex-col items-center gap-2 text-text-muted">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span className="text-[10px] font-mono">Loading preview...</span>
          </div>
        ) : hasError ? (
          <div className="flex flex-col items-center gap-2 text-status-error">
            <AlertCircle className="w-6 h-6" />
            <span className="text-[10px] font-mono">Failed to load</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 text-text-muted">
            <Globe className="w-8 h-8" />
            <span className="text-[10px] font-mono">No preview</span>
          </div>
        )}
      </div>

      {/* URL display */}
      <div className="px-3 py-1.5 bg-background-secondary rounded-b-brutal">
        <div className="text-[9px] text-text-muted truncate font-mono" title={url}>
          {truncateUrl(url, 50)}
        </div>
      </div>
    </div>
  )
})
