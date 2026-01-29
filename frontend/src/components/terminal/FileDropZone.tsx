import { ReactNode, useCallback } from 'react'
import { FileDown, FolderInput } from 'lucide-react'
import { useFileDrop, formatPathForTerminal } from '../../hooks/useFileDrop'
import { clsx } from 'clsx'

interface FileDropZoneProps {
  children: ReactNode
  terminalId: string
  className?: string
  disabled?: boolean
}

/**
 * Wraps terminal content to handle file drops
 * When files are dropped, their paths are inserted at the cursor position
 */
export function FileDropZone({
  children,
  terminalId,
  className,
  disabled = false
}: FileDropZoneProps) {

  const handleDropPath = useCallback((path: string) => {
    if (disabled) return

    // Format the path for terminal (handle spaces, etc.)
    const formattedPath = formatPathForTerminal(path)

    // Send the path to the terminal
    window.bashgym?.terminal.write(terminalId, formattedPath)
  }, [terminalId, disabled])

  const { isDragOver, isValidDrag, dropProps } = useFileDrop({
    onDropPath: handleDropPath,
    multiple: true
  })

  if (disabled) {
    return <div className={className}>{children}</div>
  }

  return (
    <div
      className={clsx('relative', className)}
      {...dropProps}
    >
      {children}

      {/* Drop overlay */}
      {isDragOver && (
        <div
          className={clsx(
            'absolute inset-0 z-50 flex items-center justify-center transition-all',
            'bg-background-primary/90 backdrop-blur-sm',
            'border-2 border-dashed rounded-lg',
            isValidDrag
              ? 'border-primary bg-primary/5'
              : 'border-status-error bg-status-error/5'
          )}
        >
          <div className="text-center">
            <div
              className={clsx(
                'w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center',
                isValidDrag ? 'bg-primary/10' : 'bg-status-error/10'
              )}
            >
              {isValidDrag ? (
                <FolderInput className="w-8 h-8 text-primary" />
              ) : (
                <FileDown className="w-8 h-8 text-status-error" />
              )}
            </div>
            <p
              className={clsx(
                'text-sm font-medium',
                isValidDrag ? 'text-primary' : 'text-status-error'
              )}
            >
              {isValidDrag ? 'Drop files here' : 'Invalid file type'}
            </p>
            <p className="text-xs text-text-muted mt-1">
              {isValidDrag
                ? 'File paths will be inserted at cursor'
                : 'This file type is not supported'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Simpler drop zone for generic drop targets (not terminals)
 */
interface GenericDropZoneProps {
  children: ReactNode
  onDrop: (paths: string[]) => void
  className?: string
  hint?: string
}

export function GenericDropZone({
  children,
  onDrop,
  className,
  hint = 'Drop files here'
}: GenericDropZoneProps) {

  const handleDrop = useCallback(({ paths }: { paths: string[] }) => {
    if (paths.length > 0) {
      onDrop(paths)
    }
  }, [onDrop])

  const { isDragOver, isValidDrag, dropProps } = useFileDrop({
    onDrop: handleDrop,
    multiple: true
  })

  return (
    <div
      className={clsx('relative', className)}
      {...dropProps}
    >
      {children}

      {isDragOver && (
        <div
          className={clsx(
            'absolute inset-0 z-50 flex items-center justify-center',
            'bg-background-primary/80 backdrop-blur-sm',
            'border-2 border-dashed rounded-lg',
            isValidDrag ? 'border-primary' : 'border-status-error'
          )}
        >
          <div className="text-center">
            <FolderInput
              className={clsx(
                'w-8 h-8 mx-auto mb-2',
                isValidDrag ? 'text-primary' : 'text-status-error'
              )}
            />
            <p
              className={clsx(
                'text-sm font-medium',
                isValidDrag ? 'text-text-primary' : 'text-status-error'
              )}
            >
              {isValidDrag ? hint : 'Invalid file type'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
