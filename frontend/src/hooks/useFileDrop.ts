import { useState, useCallback, useRef, DragEvent } from 'react'

export interface FileDropResult {
  files: File[]
  paths: string[]
}

export interface UseFileDropOptions {
  onDrop?: (result: FileDropResult) => void
  onDropPath?: (path: string) => void
  acceptTypes?: string[]  // e.g., ['.ts', '.tsx', '.js']
  multiple?: boolean
}

export interface UseFileDropReturn {
  isDragOver: boolean
  isValidDrag: boolean
  handleDragEnter: (e: DragEvent) => void
  handleDragOver: (e: DragEvent) => void
  handleDragLeave: (e: DragEvent) => void
  handleDrop: (e: DragEvent) => void
  dropProps: {
    onDragEnter: (e: DragEvent) => void
    onDragOver: (e: DragEvent) => void
    onDragLeave: (e: DragEvent) => void
    onDrop: (e: DragEvent) => void
  }
}

/**
 * Hook for handling file drag-and-drop
 * Supports both native file drops and internal file browser drags
 */
export function useFileDrop(options: UseFileDropOptions = {}): UseFileDropReturn {
  const { onDrop, onDropPath, acceptTypes, multiple = true } = options

  const [isDragOver, setIsDragOver] = useState(false)
  const [isValidDrag, setIsValidDrag] = useState(false)
  const dragCounter = useRef(0)

  const isValidFile = useCallback((file: File | DataTransferItem): boolean => {
    if (!acceptTypes || acceptTypes.length === 0) return true

    const name = 'name' in file ? file.name : ''
    if (!name) return true  // Can't validate without name

    return acceptTypes.some(ext => name.endsWith(ext))
  }, [acceptTypes])

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    dragCounter.current++

    // Check if this is a file drag
    const hasFiles = e.dataTransfer?.types?.includes('Files')
    const hasInternalPath = e.dataTransfer?.types?.includes('text/filepath')

    if (hasFiles || hasInternalPath) {
      setIsDragOver(true)

      // Validate file types if possible
      if (e.dataTransfer?.items) {
        const items = Array.from(e.dataTransfer.items)
        const valid = items.some(item => {
          if (item.kind === 'file') {
            return isValidFile(item)
          }
          return true
        })
        setIsValidDrag(valid)
      } else {
        setIsValidDrag(true)
      }
    }
  }, [isValidFile])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    // Set drop effect based on validity
    if (e.dataTransfer) {
      e.dataTransfer.dropEffect = isValidDrag ? 'copy' : 'none'
    }
  }, [isValidDrag])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    dragCounter.current--

    if (dragCounter.current === 0) {
      setIsDragOver(false)
      setIsValidDrag(false)
    }
  }, [])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()

    dragCounter.current = 0
    setIsDragOver(false)
    setIsValidDrag(false)

    // Check for internal file path first
    const internalPath = e.dataTransfer?.getData('text/filepath')
    if (internalPath) {
      onDropPath?.(internalPath)
      onDrop?.({ files: [], paths: [internalPath] })
      return
    }

    // Handle native file drops
    const files = e.dataTransfer?.files
    if (!files || files.length === 0) return

    const fileList = Array.from(files)
    const filteredFiles = acceptTypes
      ? fileList.filter(f => isValidFile(f))
      : fileList

    const finalFiles = multiple ? filteredFiles : filteredFiles.slice(0, 1)

    if (finalFiles.length === 0) return

    // Extract paths - in Electron, files have a path property
    const paths = finalFiles.map(file => (file as any).path || file.name)

    if (onDropPath && paths.length > 0) {
      // Send each path
      paths.forEach(path => onDropPath(path))
    }

    onDrop?.({ files: finalFiles, paths })
  }, [onDrop, onDropPath, acceptTypes, multiple, isValidFile])

  return {
    isDragOver,
    isValidDrag,
    handleDragEnter,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    dropProps: {
      onDragEnter: handleDragEnter,
      onDragOver: handleDragOver,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop
    }
  }
}

/**
 * Get the file path from a native file drop
 * Works in Electron where File objects have a path property
 */
export function getFilePath(file: File): string {
  return (file as any).path || file.name
}

/**
 * Format file paths for terminal insertion
 * Handles spaces and special characters
 */
export function formatPathForTerminal(path: string): string {
  // If path contains spaces or special chars, wrap in quotes
  if (/[\s&|<>()$`\\!]/.test(path)) {
    // Escape any existing quotes
    const escaped = path.replace(/"/g, '\\"')
    return `"${escaped}"`
  }
  return path
}

/**
 * Format multiple paths for terminal insertion
 */
export function formatPathsForTerminal(paths: string[]): string {
  return paths.map(formatPathForTerminal).join(' ')
}
