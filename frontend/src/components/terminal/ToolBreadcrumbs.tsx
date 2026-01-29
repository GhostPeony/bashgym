import { memo } from 'react'
import { clsx } from 'clsx'
import {
  FileText,
  Edit3,
  Terminal,
  Search,
  Globe,
  FolderSearch,
  ListTodo,
  HelpCircle,
  ChevronRight
} from 'lucide-react'

export interface ToolHistoryItem {
  tool: string
  target?: string
  timestamp: number
}

export interface ToolBreadcrumbsProps {
  history: ToolHistoryItem[]
  maxItems?: number
  className?: string
  showTimestamps?: boolean
}

// Get icon for tool type
function getToolIcon(tool: string) {
  const toolLower = tool.toLowerCase()
  if (toolLower.includes('read')) return <FileText className="w-2.5 h-2.5" />
  if (toolLower.includes('edit') || toolLower.includes('write')) return <Edit3 className="w-2.5 h-2.5" />
  if (toolLower.includes('bash') || toolLower.includes('terminal')) return <Terminal className="w-2.5 h-2.5" />
  if (toolLower.includes('grep') || toolLower.includes('search')) return <Search className="w-2.5 h-2.5" />
  if (toolLower.includes('glob') || toolLower.includes('find')) return <FolderSearch className="w-2.5 h-2.5" />
  if (toolLower.includes('web') || toolLower.includes('fetch')) return <Globe className="w-2.5 h-2.5" />
  if (toolLower.includes('todo')) return <ListTodo className="w-2.5 h-2.5" />
  if (toolLower.includes('ask')) return <HelpCircle className="w-2.5 h-2.5" />
  return <Terminal className="w-2.5 h-2.5" />
}

// Get short tool name
function getShortToolName(tool: string): string {
  const names: Record<string, string> = {
    read: 'Read',
    edit: 'Edit',
    write: 'Write',
    bash: 'Bash',
    grep: 'Grep',
    glob: 'Glob',
    webfetch: 'Web',
    websearch: 'Search',
    todowrite: 'Todo',
    askuserquestion: 'Ask',
    task: 'Task'
  }

  const toolLower = tool.toLowerCase()
  for (const [key, name] of Object.entries(names)) {
    if (toolLower.includes(key)) return name
  }
  // Return first word, capitalized
  return tool.split(/[_\s]/)[0].slice(0, 6)
}

// Get color class based on tool type
function getToolColor(tool: string): string {
  const toolLower = tool.toLowerCase()
  if (toolLower.includes('read')) return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
  if (toolLower.includes('edit') || toolLower.includes('write')) return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
  if (toolLower.includes('bash') || toolLower.includes('terminal')) return 'bg-green-500/20 text-green-400 border-green-500/30'
  if (toolLower.includes('grep') || toolLower.includes('search') || toolLower.includes('glob')) return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
  if (toolLower.includes('web')) return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
  if (toolLower.includes('todo')) return 'bg-pink-500/20 text-pink-400 border-pink-500/30'
  if (toolLower.includes('ask')) return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
  return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
}

export const ToolBreadcrumbs = memo(function ToolBreadcrumbs({
  history,
  maxItems = 5,
  className,
  showTimestamps = false
}: ToolBreadcrumbsProps) {
  if (!history || history.length === 0) {
    return null
  }

  // Get last N items
  const displayItems = history.slice(-maxItems)
  const hasMore = history.length > maxItems

  return (
    <div className={clsx('flex items-center gap-1 overflow-hidden', className)}>
      {hasMore && (
        <span className="text-[10px] text-text-muted">...</span>
      )}
      {displayItems.map((item, index) => (
        <div key={`${item.tool}-${item.timestamp}`} className="flex items-center gap-1">
          {index > 0 && (
            <ChevronRight className="w-2.5 h-2.5 text-text-muted/50 flex-shrink-0" />
          )}
          <div
            className={clsx(
              'flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] border',
              getToolColor(item.tool)
            )}
            title={item.target ? `${item.tool}: ${item.target}` : item.tool}
          >
            {getToolIcon(item.tool)}
            <span className="font-medium">{getShortToolName(item.tool)}</span>
          </div>
        </div>
      ))}
    </div>
  )
})

// Compact version showing just icons
export const ToolBreadcrumbsCompact = memo(function ToolBreadcrumbsCompact({
  history,
  maxItems = 6,
  className
}: Omit<ToolBreadcrumbsProps, 'showTimestamps'>) {
  if (!history || history.length === 0) {
    return null
  }

  const displayItems = history.slice(-maxItems)
  const hasMore = history.length > maxItems

  return (
    <div className={clsx('flex items-center gap-0.5', className)}>
      {hasMore && (
        <span className="text-[9px] text-text-muted mr-0.5">+{history.length - maxItems}</span>
      )}
      {displayItems.map((item, index) => (
        <div
          key={`${item.tool}-${item.timestamp}`}
          className={clsx(
            'flex items-center justify-center w-4 h-4 rounded',
            getToolColor(item.tool)
          )}
          title={item.target ? `${item.tool}: ${item.target}` : item.tool}
        >
          {getToolIcon(item.tool)}
        </div>
      ))}
    </div>
  )
})
