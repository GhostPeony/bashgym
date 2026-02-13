/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: 'var(--accent)',
          light: 'var(--accent-light)',
          dark: 'var(--accent-dark)'
        },
        accent: {
          DEFAULT: 'var(--accent)',
          light: 'var(--accent-light)',
          dark: 'var(--accent-dark)'
        },
        background: {
          primary: 'var(--bg-primary)',
          secondary: 'var(--bg-secondary)',
          tertiary: 'var(--bg-tertiary)',
          card: 'var(--bg-card)',
          terminal: 'var(--bg-terminal)'
        },
        text: {
          primary: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted: 'var(--text-muted)'
        },
        border: {
          DEFAULT: 'var(--border-color)',
          subtle: 'var(--border-subtle)'
        },
        status: {
          success: 'var(--status-success)',
          warning: 'var(--status-warning)',
          error: 'var(--status-error)',
          info: 'var(--status-info)'
        }
      },
      fontFamily: {
        brand: [
          'Playfair Display',
          'Georgia',
          'serif'
        ],
        sans: [
          'Inter',
          '-apple-system',
          'BlinkMacSystemFont',
          'system-ui',
          'sans-serif'
        ],
        mono: [
          'JetBrains Mono',
          'Consolas',
          'monospace'
        ]
      },
      fontSize: {
        'xs': ['12px', '16px'],
        'sm': ['13px', '18px'],
        'base': ['14px', '20px'],
        'lg': ['16px', '24px'],
        'xl': ['18px', '28px'],
        '2xl': ['24px', '32px'],
        '3xl': ['30px', '36px'],
        'display': ['36px', '44px']
      },
      boxShadow: {
        'brutal': 'var(--shadow)',
        'brutal-sm': 'var(--shadow-sm)',
        'brutal-hover': '2px 2px 0px var(--shadow-color)',
        'brutal-pressed': '1px 1px 0px var(--shadow-color)',
        'none': 'none'
      },
      borderWidth: {
        'brutal': 'var(--border-weight)',
        '2': '2px',
        '3': '3px'
      },
      borderRadius: {
        'brutal': 'var(--radius)',
        'none': '0px',
        'sm': '2px',
        'DEFAULT': '4px',
        'lg': '8px',
        'xl': '12px',
        'full': '9999px'
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem'
      },
      transitionDuration: {
        '150': '150ms',
        '250': '250ms',
        '350': '350ms'
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'spin-slow': 'spin 20s linear infinite',
        'attention-pulse': 'attention-pulse 2s ease-in-out infinite'
      }
    }
  },
  plugins: []
}
