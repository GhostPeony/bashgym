import { forwardRef } from 'react'
import { clsx } from 'clsx'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'icon' | 'cta'
  size?: 'sm' | 'md' | 'lg'
  isLoading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      leftIcon,
      rightIcon,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      'inline-flex items-center justify-center font-mono font-semibold tracking-wide transition-press disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none'

    const variants = {
      primary:
        'btn-primary text-white',
      secondary:
        'btn-secondary',
      ghost:
        'btn-ghost',
      danger:
        'bg-status-error text-white border-brutal border-border box-shadow-brutal-sm hover:translate-x-0.5 hover:translate-y-0.5',
      icon:
        'btn-icon',
      cta:
        'btn-cta',
    }

    const sizes = {
      sm: 'px-3 py-1.5 text-xs gap-1.5',
      md: 'px-5 py-2 text-sm gap-2',
      lg: 'px-7 py-3 text-sm gap-2'
    }

    // Icon variant ignores size padding
    const sizeStyles = variant === 'icon' ? '' : sizes[size]

    return (
      <button
        ref={ref}
        className={clsx(baseStyles, variants[variant], sizeStyles, className)}
        disabled={disabled || isLoading}
        {...props}
      >
        {isLoading ? (
          <svg
            className="animate-spin h-4 w-4"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : (
          leftIcon
        )}
        {children}
        {!isLoading && rightIcon}
      </button>
    )
  }
)

Button.displayName = 'Button'
