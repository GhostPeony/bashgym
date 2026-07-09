import { useState } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import { clsx } from 'clsx'

interface MaskedValueProps {
  value: string
  className?: string
}

/**
 * Sensitive value hidden behind an eye toggle. Fixed-width mask so the
 * value's length doesn't leak on screen shares. Generalization of MaskedHost.
 */
export function MaskedValue({ value, className }: MaskedValueProps) {
  const [visible, setVisible] = useState(false)

  return (
    <span className={clsx('inline-flex items-center gap-1.5 min-w-0', className)}>
      <span className="truncate">{visible ? value : '••••••••'}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          setVisible((v) => !v)
        }}
        className="text-text-muted hover:text-text-primary transition-press flex-shrink-0"
        title={visible ? 'Hide' : 'Show'}
      >
        {visible ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
      </button>
    </span>
  )
}
