import { useEffect, useCallback } from 'react'
import {
  Home,
  ArrowUp,
  RefreshCw,
  FolderTree,
  Loader2,
  AlertCircle
} from 'lucide-react'
import { clsx } from 'clsx'
import { useFileStore, initializeFileStore } from '../../stores/fileStore'
import { useTerminalStore } from '../../stores/terminalStore'
import { FileTreeItem } from './FileTreeItem'

interface FileBrowserProps {
  id: string
  title: string
  isActive: boolean
}

export function FileBrowser({ id, title, isActive }: FileBrowserProps) {
  const {
    rootPath,
    tree,
    isLoading,
    error,
    refresh,
    goUp,
    setRootPath,
    selectFile
  } = useFileStore()

  const { addPanel } = useTerminalStore()

  // Initialize file store on mount
  useEffect(() => {
    initializeFileStore()
  }, [])

  // Handle file selection
  const handleSelect = useCallback((path: string) => {
    selectFile(path)
  }, [selectFile])

  // Handle file open (double-click)
  const handleOpen = useCallback((path: string) => {
    // Open in preview pane
    addPanel({
      type: 'preview',
      title: path.split(/[/\\]/).pop() || 'Preview',
      filePath: path
    })
  }, [addPanel])

  // Go to home directory
  const handleGoHome = useCallback(async () => {
    const homePath = await window.bashgym?.files?.getHomeDirectory()
    if (homePath) {
      setRootPath(homePath)
    }
  }, [setRootPath])

  // Navigate to a path from breadcrumbs
  const handleBreadcrumbClick = useCallback((path: string) => {
    setRootPath(path)
  }, [setRootPath])

  // Parse path into breadcrumb segments
  const getBreadcrumbs = () => {
    if (!rootPath) return []

    // Handle Windows paths (C:\foo\bar) and Unix paths (/foo/bar)
    const isWindows = rootPath.includes(':\\')
    const separator = isWindows ? '\\' : '/'
    const parts = rootPath.split(separator).filter(Boolean)

    const breadcrumbs: { name: string; path: string }[] = []
    let currentPath = isWindows ? '' : '/'

    for (const part of parts) {
      if (isWindows && !currentPath) {
        // First part is drive letter on Windows
        currentPath = part + ':\\'
        breadcrumbs.push({ name: part + ':', path: currentPath })
      } else {
        currentPath = isWindows
          ? `${currentPath}${part}\\`
          : `${currentPath}${part}/`
        breadcrumbs.push({ name: part, path: currentPath.slice(0, -1) })
      }
    }

    return breadcrumbs
  }

  const breadcrumbs = getBreadcrumbs()

  return (
    <div className="h-full flex flex-col bg-background-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-background-secondary border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <FolderTree className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-text-primary">{title}</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleGoHome}
            className="p-1.5 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
            title="Go to home directory"
          >
            <Home className="w-4 h-4" />
          </button>
          <button
            onClick={goUp}
            className="p-1.5 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary"
            title="Go up one level"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
          <button
            onClick={refresh}
            className={clsx(
              'p-1.5 rounded hover:bg-background-tertiary text-text-muted hover:text-text-secondary',
              isLoading && 'animate-spin'
            )}
            title="Refresh"
            disabled={isLoading}
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Breadcrumbs */}
      <div className="flex items-center gap-1 px-3 py-1.5 bg-background-tertiary border-b border-border-subtle overflow-x-auto">
        <button
          onClick={handleGoHome}
          className="text-xs text-text-muted hover:text-text-primary"
        >
          ~
        </button>
        {breadcrumbs.map((crumb, index) => (
          <span key={crumb.path} className="flex items-center gap-1">
            <span className="text-text-muted">/</span>
            <button
              onClick={() => handleBreadcrumbClick(crumb.path)}
              className={clsx(
                'text-xs hover:text-text-primary transition-colors',
                index === breadcrumbs.length - 1
                  ? 'text-text-primary font-medium'
                  : 'text-text-muted'
              )}
            >
              {crumb.name}
            </button>
          </span>
        ))}
      </div>

      {/* File tree */}
      <div className="flex-1 overflow-auto py-1">
        {isLoading && tree.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-6 h-6 animate-spin text-text-muted" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-4">
            <AlertCircle className="w-8 h-8 text-status-error mb-2" />
            <p className="text-sm text-status-error">{error}</p>
            <button
              onClick={refresh}
              className="mt-2 text-xs text-primary hover:underline"
            >
              Try again
            </button>
          </div>
        ) : tree.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-4">
            <FolderTree className="w-8 h-8 text-text-muted mb-2" />
            <p className="text-sm text-text-muted">Empty directory</p>
          </div>
        ) : (
          tree.map((node) => (
            <FileTreeItem
              key={node.path}
              node={node}
              depth={0}
              onSelect={handleSelect}
              onOpen={handleOpen}
            />
          ))
        )}
      </div>

      {/* Footer with path */}
      <div className="px-3 py-1.5 bg-background-secondary border-t border-border-subtle">
        <p className="text-xs text-text-muted truncate" title={rootPath}>
          {rootPath}
        </p>
      </div>
    </div>
  )
}
