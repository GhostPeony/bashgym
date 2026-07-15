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
  layout?: 'document' | 'workspace'
}

export function NodeConfigModal({
  isOpen,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'lg',
  layout = 'document'
}: NodeConfigModalProps) {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={title}
      description={description}
      footer={footer}
      size={size}
      variant="canvas"
      bodyClassName={layout === 'workspace' ? 'modal-canvas-body-workspace' : undefined}
    >
      <div
        className={clsx(
          'node-config-modal-content',
          layout === 'workspace' && 'node-config-modal-workspace'
        )}
      >
        {children}
      </div>
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
    <section className={clsx('node-config-section', className)}>
      {title ? <div className="node-config-section-title">{title}</div> : null}
      <div className="node-config-section-content">{children}</div>
    </section>
  )
}

export function ConfigRows({ children }: { children: ReactNode }) {
  return (
    <div className="node-config-rows">
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
    <div className="node-config-row">
      <span className="node-config-row-label">{label}</span>
      <span className="node-config-row-value" title={title}>
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
        'node-config-pill',
        tone === 'neutral' && 'node-config-pill-neutral',
        tone === 'accent' && 'node-config-pill-accent',
        tone === 'success' && 'node-config-pill-success',
        tone === 'warning' && 'node-config-pill-warning',
        tone === 'error' && 'node-config-pill-error'
      )}
    >
      {children}
    </span>
  )
}
