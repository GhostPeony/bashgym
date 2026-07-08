import { useState } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import { clsx } from 'clsx'

interface MaskedHostProps {
  /** Optional user shown before the @ */
  username?: string
  host: string
  port?: number | string
  className?: string
}

/**
 * Connection label with the host/IP hidden behind an eye toggle.
 * Fixed-width mask so the address length doesn't leak on screen shares.
 */
export function MaskedHost({ username, host, port, className }: MaskedHostProps) {
  const [visible, setVisible] = useState(false)

  const full = `${username ? `${username}@` : ''}${host}${port != null ? `:${port}` : ''}`
  const masked = `${username ? `${username}@` : ''}${'•'.repeat(8)}`

  return (
    <span className={clsx('inline-flex items-center gap-1.5', className)}>
      <span>{visible ? full : masked}</span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          setVisible((v) => !v)
        }}
        className="text-text-muted hover:text-text-primary transition-press"
        title={visible ? 'Hide address' : 'Show address'}
      >
        {visible ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
      </button>
    </span>
  )
}
