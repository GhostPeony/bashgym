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
        // Light mode (Apple-inspired)
        primary: {
          light: '#0066CC',
          DEFAULT: 'var(--color-primary)',
          dark: '#76B900'
        },
        accent: {
          light: '#76B900',
          DEFAULT: 'var(--color-accent)',
          dark: '#00A6FF'
        },
        background: {
          primary: 'var(--bg-primary)',
          secondary: 'var(--bg-secondary)',
          tertiary: 'var(--bg-tertiary)',
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
        // NVIDIA green for accents
        nvidia: {
          green: '#76B900',
          dark: '#5A8F00'
        },
        // Status colors
        status: {
          success: '#34C759',
          warning: '#FF9500',
          error: '#FF3B30',
          info: '#007AFF'
        }
      },
      fontFamily: {
        sans: [
          'SF Pro Display',
          '-apple-system',
          'BlinkMacSystemFont',
          'system-ui',
          'Segoe UI',
          'Roboto',
          'sans-serif'
        ],
        mono: [
          'SF Mono',
          'JetBrains Mono',
          'Menlo',
          'Monaco',
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
        'glow-green': '0 0 20px rgba(118, 185, 0, 0.3)',
        'glow-blue': '0 0 20px rgba(0, 166, 255, 0.3)',
        'glow-red': '0 0 20px rgba(255, 59, 48, 0.3)',
        'subtle': '0 1px 2px rgba(0, 0, 0, 0.05)',
        'card': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)',
        'elevated': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)'
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 8s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate'
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(118, 185, 0, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(118, 185, 0, 0.4)' }
        }
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem'
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
        '3xl': '24px'
      },
      transitionDuration: {
        '250': '250ms',
        '350': '350ms'
      }
    }
  },
  plugins: []
}
