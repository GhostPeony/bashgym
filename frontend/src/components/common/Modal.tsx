import { useEffect, useCallback, useId } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { clsx } from 'clsx'

interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  description?: string
  children: React.ReactNode
  footer?: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'canvas'
  bodyClassName?: string
}

export function Modal({
  isOpen,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'md',
  variant = 'default',
  bodyClassName
}: ModalProps) {
  const titleId = useId()
  const descriptionId = useId()
  const isCanvas = variant === 'canvas'

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose()
      }
    },
    [onClose]
  )

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown)
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [isOpen, handleKeyDown])

  if (!isOpen) return null

  const sizes = {
    sm: 'max-w-sm',
    md: 'max-w-xl',
    lg: 'max-w-3xl',
    xl: 'max-w-6xl'
  }

  return createPortal(
    <div className="modal-root fixed inset-0 z-50 flex items-center justify-center overflow-hidden">
      <div
        className={clsx('absolute inset-0', isCanvas && 'modal-canvas-backdrop')}
        style={isCanvas ? undefined : { backgroundColor: 'rgba(27, 32, 64, 0.5)' }}
        onClick={onClose}
      />

      <div
        className={clsx(
          'relative w-full max-h-full overflow-hidden',
          isCanvas
            ? 'modal-canvas-shell'
            : 'bg-background-card border-brutal border-border shadow-brutal rounded-brutal',
          sizes[size]
        )}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
      >
        <div
          className={clsx(
            isCanvas
              ? 'modal-canvas-header'
              : 'flex items-start justify-between px-6 py-4 border-b border-border'
          )}
        >
          <div className="min-w-0">
            <h2
              id={titleId}
              className={
                isCanvas ? 'modal-canvas-title' : 'text-lg font-brand font-normal text-text-primary'
              }
            >
              {title}
            </h2>
            {description && (
              <p
                id={descriptionId}
                className={
                  isCanvas ? 'modal-canvas-description' : 'text-sm text-text-secondary mt-1'
                }
              >
                {description}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            className={
              isCanvas
                ? 'node-btn node-btn-danger h-8 w-8'
                : 'btn-icon w-8 h-8 text-text-muted hover:text-text-primary'
            }
            title="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div
          className={clsx(
            isCanvas ? 'modal-canvas-body' : 'px-6 py-4 max-h-[70vh] overflow-y-auto',
            bodyClassName
          )}
        >
          {children}
        </div>

        {footer && (
          <div
            className={
              isCanvas
                ? 'modal-canvas-footer'
                : 'flex items-center justify-end gap-3 px-6 py-4 border-t border-border'
            }
          >
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  )
}
