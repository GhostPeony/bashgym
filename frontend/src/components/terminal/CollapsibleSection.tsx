import { memo, useState, useCallback, type ReactNode } from 'react'
import { clsx } from 'clsx'

export interface CollapsibleSectionProps {
  title: string
  icon?: ReactNode
  defaultExpanded?: boolean
  isExpanded?: boolean
  onToggle?: (expanded: boolean) => void
  children: ReactNode
  className?: string
  headerClassName?: string
  contentClassName?: string
  badge?: ReactNode
}

export const CollapsibleSection = memo(function CollapsibleSection({
  title,
  icon,
  defaultExpanded = false,
  isExpanded: controlledExpanded,
  onToggle,
  children,
  className,
  headerClassName,
  contentClassName,
  badge
}: CollapsibleSectionProps) {
  const [internalExpanded, setInternalExpanded] = useState(defaultExpanded)

  // Support both controlled and uncontrolled modes
  const isExpanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded

  const handleToggle = useCallback(() => {
    const newExpanded = !isExpanded
    setInternalExpanded(newExpanded)
    onToggle?.(newExpanded)
  }, [isExpanded, onToggle])

  return (
    <div className={clsx('border-t border-brutal border-border', className)}>
      <button
        type="button"
        onClick={handleToggle}
        className={clsx(
          'w-full flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono',
          'hover:bg-background-secondary transition-press',
          'text-text-muted hover:text-text-secondary',
          headerClassName
        )}
      >
        {/* Triangular chevron for expand/collapse */}
        <span className="flex-shrink-0 w-3 h-3 flex items-center justify-center">
          {isExpanded ? (
            <span
              className="block w-0 h-0"
              style={{
                borderLeft: '4px solid transparent',
                borderRight: '4px solid transparent',
                borderTop: '5px solid currentColor'
              }}
            />
          ) : (
            <span
              className="block w-0 h-0"
              style={{
                borderTop: '4px solid transparent',
                borderBottom: '4px solid transparent',
                borderLeft: '5px solid currentColor'
              }}
            />
          )}
        </span>
        {icon && <span className="flex-shrink-0">{icon}</span>}
        <span className="flex-1 text-left font-semibold uppercase tracking-wider text-[10px]">{title}</span>
        {badge && <span className="flex-shrink-0">{badge}</span>}
      </button>
      {isExpanded && (
        <div className={clsx('px-3 pb-2', contentClassName)}>
          {children}
        </div>
      )}
    </div>
  )
})
