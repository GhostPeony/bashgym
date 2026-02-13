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

// Get color class based on tool type — solid bg, hard border
function getToolColor(tool: string): string {
  const toolLower = tool.toLowerCase()
  if (toolLower.includes('read')) return 'bg-blue-100 text-blue-700 border-blue-700 dark:bg-blue-900 dark:text-blue-300 dark:border-blue-400'
  if (toolLower.includes('edit') || toolLower.includes('write')) return 'bg-amber-100 text-amber-700 border-amber-700 dark:bg-amber-900 dark:text-amber-300 dark:border-amber-400'
  if (toolLower.includes('bash') || toolLower.includes('terminal')) return 'bg-green-100 text-green-700 border-green-700 dark:bg-green-900 dark:text-green-300 dark:border-green-400'
  if (toolLower.includes('grep') || toolLower.includes('search') || toolLower.includes('glob')) return 'bg-purple-100 text-purple-700 border-purple-700 dark:bg-purple-900 dark:text-purple-300 dark:border-purple-400'
  if (toolLower.includes('web')) return 'bg-cyan-100 text-cyan-700 border-cyan-700 dark:bg-cyan-900 dark:text-cyan-300 dark:border-cyan-400'
  if (toolLower.includes('todo')) return 'bg-pink-100 text-pink-700 border-pink-700 dark:bg-pink-900 dark:text-pink-300 dark:border-pink-400'
  if (toolLower.includes('ask')) return 'bg-orange-100 text-orange-700 border-orange-700 dark:bg-orange-900 dark:text-orange-300 dark:border-orange-400'
  return 'bg-background-secondary text-text-secondary border-border'
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
    <div className={clsx('flex items-center gap-1 overflow-hidden font-mono', className)}>
      {hasMore && (
        <span className="text-[10px] text-text-muted font-mono">...</span>
      )}
      {displayItems.map((item, index) => (
        <div key={`${item.tool}-${item.timestamp}`} className="flex items-center gap-1">
          {index > 0 && (
            <ChevronRight className="w-2.5 h-2.5 text-text-muted flex-shrink-0" />
          )}
          <div
            className={clsx(
              'flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-mono',
              'border-brutal rounded-brutal',
              getToolColor(item.tool)
            )}
            title={item.target ? `${item.tool}: ${item.target}` : item.tool}
          >
            {getToolIcon(item.tool)}
            <span className="font-semibold uppercase tracking-wider">{getShortToolName(item.tool)}</span>
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
        <span className="text-[9px] text-text-muted mr-0.5 font-mono">+{history.length - maxItems}</span>
      )}
      {displayItems.map((item, index) => (
        <div
          key={`${item.tool}-${item.timestamp}`}
          className={clsx(
            'flex items-center justify-center w-4 h-4',
            'border-brutal rounded-brutal',
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
