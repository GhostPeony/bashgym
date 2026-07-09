import type { ReactNode } from 'react'
import { clsx } from 'clsx'
import { Modal } from '../../common/Modal'

type ModalSize = 'sm' | 'md' | 'lg' | 'xl'

interface NodeConfigModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  description?: string
  children: ReactNode
  footer?: ReactNode
  size?: ModalSize
}

export function NodeConfigModal({
  isOpen,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'lg'
}: NodeConfigModalProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      description={description}
      footer={footer}
      size={size}
    >
      <div className="space-y-3">{children}</div>
    </Modal>
  )
}

export function ConfigSection({
  title,
  children,
  className
}: {
  title?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={clsx('node-section space-y-2', className)}>
      {title ? <div className="node-section-title">{title}</div> : null}
      {children}
    </section>
  )
}

export function ConfigRows({ children }: { children: ReactNode }) {
  return (
    <div className="divide-y divide-border-subtle border-brutal border-border-subtle rounded-brutal bg-background-card">
      {children}
    </div>
  )
}

export function ConfigRow({
  label,
  value,
  title
}: {
  label: ReactNode
  value?: ReactNode
  title?: string
}) {
  return (
    <div className="grid grid-cols-[140px_minmax(0,1fr)] gap-3 px-3 py-2 text-xs font-mono">
      <span className="text-text-muted uppercase tracking-wider">{label}</span>
      <span className="min-w-0 break-words text-text-primary" title={title}>
        {value == null || value === '' ? <span className="text-text-muted">-</span> : value}
      </span>
    </div>
  )
}

export function ConfigPill({
  children,
  tone = 'neutral'
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'success' | 'warning' | 'error'
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 border-brutal rounded-brutal px-2 py-1 text-[10px] font-mono font-bold uppercase tracking-wider',
        tone === 'neutral' && 'border-border-subtle bg-background-card text-text-secondary',
        tone === 'accent' && 'border-accent/60 bg-accent/10 text-accent',
        tone === 'success' && 'border-status-success/60 bg-status-success/10 text-status-success',
        tone === 'warning' && 'border-status-warning/60 bg-status-warning/10 text-status-warning',
        tone === 'error' && 'border-status-error/60 bg-status-error/10 text-status-error'
      )}
    >
      {children}
    </span>
  )
}
