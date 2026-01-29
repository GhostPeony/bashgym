import { useState, useEffect, memo, DragEvent } from 'react'
import {
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  File,
  FileText,
  FileCode,
  FileJson,
  Image,
  FileType,
  Loader2
} from 'lucide-react'
import { clsx } from 'clsx'
import type { FileNode } from '../../stores/fileStore'
import { useFileStore } from '../../stores/fileStore'

interface FileTreeItemProps {
  node: FileNode
  depth: number
  onSelect?: (path: string) => void
  onOpen?: (path: string) => void
}

// Get icon based on file extension
function getFileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase()

  switch (ext) {
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
    case 'py':
    case 'rs':
    case 'go':
    case 'java':
    case 'c':
    case 'cpp':
    case 'h':
    case 'hpp':
    case 'rb':
    case 'php':
    case 'swift':
    case 'kt':
      return <FileCode className="w-4 h-4 text-blue-400" />
    case 'json':
    case 'yaml':
    case 'yml':
    case 'toml':
      return <FileJson className="w-4 h-4 text-yellow-400" />
    case 'md':
    case 'mdx':
    case 'txt':
    case 'rst':
      return <FileText className="w-4 h-4 text-text-muted" />
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'svg':
    case 'webp':
    case 'ico':
      return <Image className="w-4 h-4 text-green-400" />
    case 'html':
    case 'css':
    case 'scss':
    case 'less':
      return <FileType className="w-4 h-4 text-orange-400" />
    default:
      return <File className="w-4 h-4 text-text-muted" />
  }
}

export const FileTreeItem = memo(function FileTreeItem({
  node,
  depth,
  onSelect,
  onOpen
}: FileTreeItemProps) {
  const { expandedPaths, toggleExpand, selectedPath, loadDirectory } = useFileStore()
  const isExpanded = expandedPaths.has(node.path)
  const isSelected = selectedPath === node.path
  const [children, setChildren] = useState<FileNode[]>(node.children || [])
  const [isLoading, setIsLoading] = useState(false)

  // Load children when expanding a directory
  useEffect(() => {
    if (node.type === 'directory' && isExpanded && children.length === 0) {
      setIsLoading(true)
      loadDirectory(node.path)
        .then((files) => {
          setChildren(files)
        })
        .finally(() => {
          setIsLoading(false)
        })
    }
  }, [isExpanded, node.path, node.type, loadDirectory, children.length])

  const handleClick = () => {
    if (node.type === 'directory') {
      toggleExpand(node.path)
    }
    onSelect?.(node.path)
  }

  const handleDoubleClick = () => {
    if (node.type === 'file') {
      onOpen?.(node.path)
    }
  }

  // Drag handlers for dragging files to terminal
  const handleDragStart = (e: DragEvent) => {
    e.dataTransfer.effectAllowed = 'copy'
    e.dataTransfer.setData('text/plain', node.path)
    e.dataTransfer.setData('text/filepath', node.path)
  }

  return (
    <div>
      <div
        className={clsx(
          'flex items-center gap-1 py-1 px-2 cursor-pointer rounded-md transition-colors group',
          'hover:bg-background-tertiary',
          isSelected && 'bg-primary/10 text-primary'
        )}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        draggable
        onDragStart={handleDragStart}
      >
        {/* Expand/collapse icon for directories */}
        {node.type === 'directory' ? (
          <button
            className="p-0.5 rounded hover:bg-background-secondary"
            onClick={(e) => {
              e.stopPropagation()
              toggleExpand(node.path)
            }}
          >
            {isLoading ? (
              <Loader2 className="w-3 h-3 animate-spin text-text-muted" />
            ) : isExpanded ? (
              <ChevronDown className="w-3 h-3 text-text-muted" />
            ) : (
              <ChevronRight className="w-3 h-3 text-text-muted" />
            )}
          </button>
        ) : (
          <span className="w-4" /> // Spacer for files
        )}

        {/* File/folder icon */}
        {node.type === 'directory' ? (
          isExpanded ? (
            <FolderOpen className="w-4 h-4 text-yellow-500 flex-shrink-0" />
          ) : (
            <Folder className="w-4 h-4 text-yellow-500 flex-shrink-0" />
          )
        ) : (
          getFileIcon(node.name)
        )}

        {/* File name */}
        <span className="text-sm truncate flex-1">{node.name}</span>

        {/* File size for files (optional) */}
        {node.type === 'file' && node.size !== undefined && (
          <span className="text-xs text-text-muted opacity-0 group-hover:opacity-100">
            {formatFileSize(node.size)}
          </span>
        )}
      </div>

      {/* Children */}
      {node.type === 'directory' && isExpanded && (
        <div>
          {children.map((child) => (
            <FileTreeItem
              key={child.path}
              node={child}
              depth={depth + 1}
              onSelect={onSelect}
              onOpen={onOpen}
            />
          ))}
        </div>
      )}
    </div>
  )
})

// Format file size for display
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}
