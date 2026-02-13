import { useEffect, useRef, useCallback, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebglAddon } from '@xterm/addon-webgl'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { Square, X, MoreHorizontal, Copy, ClipboardPaste, Loader2, MessageSquare, Wrench, Coffee } from 'lucide-react'
import { useTerminalStore, useThemeStore, useAccentStore } from '../../stores'
import { FileDropZone } from './FileDropZone'
import { clsx } from 'clsx'
import '@xterm/xterm/css/xterm.css'

// Strip ANSI escape sequences from raw PTY data so regex patterns can match cleanly
const ANSI_RE = /[\x1b\x9b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]/g

function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, '')
}

interface TerminalPaneProps {
  id: string
  title: string
  isActive: boolean
  /** Optional callback to close a popup instead of closing the terminal */
  onPopupClose?: () => void
}

export function TerminalPane({ id, title, isActive, onPopupClose }: TerminalPaneProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const terminalRef = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const lastScrollPositionRef = useRef<number | null>(null)
  const outputBufferRef = useRef('')
  const flushTimerRef = useRef<ReturnType<typeof setTimeout>>()
  const idleTimerRef = useRef<ReturnType<typeof setTimeout>>()

  const { closeTerminal, updateSession, sessions, setViewMode, setActiveTerminal, panels, viewMode, renamePanel } = useTerminalStore()

  // Helper to fit terminal while preserving scroll position
  const fitWithScrollPreservation = useCallback(() => {
    if (!fitAddonRef.current || !terminalRef.current) return

    // Check if terminal is fully initialized (has element and dimensions)
    const terminal = terminalRef.current
    if (!terminal.element || !terminal.element.offsetParent) return

    try {
      // Save current scroll position (distance from bottom)
      const terminal = terminalRef.current
      const buffer = terminal.buffer.active
      const currentLine = buffer.viewportY
      const totalLines = buffer.length
      const viewportRows = terminal.rows
      const distanceFromBottom = totalLines - currentLine - viewportRows

      // Perform fit
      fitAddonRef.current.fit()

      // Restore scroll position (from bottom, to handle new content)
      // If user was at the bottom, stay at the bottom
      // Otherwise, try to maintain the same relative position
      if (distanceFromBottom <= 1) {
        // User was at/near bottom, scroll to bottom
        terminal.scrollToBottom()
      } else {
        // User was scrolled up, restore approximate position
        const newTotalLines = terminal.buffer.active.length
        const newViewportRows = terminal.rows
        const targetLine = Math.max(0, newTotalLines - newViewportRows - distanceFromBottom)
        terminal.scrollToLine(targetLine)
      }
    } catch (e) {
      // Ignore fit errors
    }
  }, [])
  const { theme } = useThemeStore()
  const { accentHue } = useAccentStore()

  const session = sessions.get(id)

  // State for inline title editing
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [editTitle, setEditTitle] = useState(title)
  const titleInputRef = useRef<HTMLInputElement>(null)

  // Find the panel ID for this terminal
  const panel = panels.find(p => p.terminalId === id)

  // Start editing title
  const startTitleEdit = useCallback(() => {
    setEditTitle(title)
    setIsEditingTitle(true)
    setTimeout(() => titleInputRef.current?.focus(), 0)
  }, [title])

  // Save title
  const saveTitleEdit = useCallback(() => {
    if (panel && editTitle.trim()) {
      renamePanel(panel.id, editTitle.trim())
    }
    setIsEditingTitle(false)
  }, [panel, editTitle, renamePanel])

  // Cancel editing
  const cancelTitleEdit = useCallback(() => {
    setIsEditingTitle(false)
    setEditTitle(title)
  }, [title])

  // HSL to hex conversion for xterm theme colors
  const hslToHex = useCallback((h: number, s: number, l: number): string => {
    s /= 100
    l /= 100
    const a = s * Math.min(l, 1 - l)
    const f = (n: number) => {
      const k = (n + h / 30) % 12
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
      return Math.round(255 * color).toString(16).padStart(2, '0')
    }
    return `#${f(0)}${f(8)}${f(4)}`
  }, [])

  // HSL to rgba hex string (xterm needs #RRGGBBAA for selection)
  const hslToHexAlpha = useCallback((h: number, s: number, l: number, alpha: number): string => {
    const hex = hslToHex(h, s, l)
    const a = Math.round(alpha * 255).toString(16).padStart(2, '0')
    return `${hex}${a}`
  }, [hslToHex])

  // Accent-influenced terminal theme colors
  const getThemeColors = useCallback(() => {
    const h = accentHue

    if (theme === 'dark') {
      // Dark: rich accent-tinted background — nighttime garden, not void
      const bg = hslToHex(h, 35, 8)
      const bgLight = hslToHex(h, 25, 15)
      const accent = hslToHex(h, 50, 68)

      return {
        background: bg,
        foreground: '#E8E4E0',
        cursor: accent,
        cursorAccent: bg,
        selectionBackground: hslToHexAlpha(h, 40, 55, 0.35),
        black: bg,
        red: '#FF453A',
        green: accent,
        yellow: '#FFD60A',
        blue: '#0A84FF',
        magenta: hslToHex((h + 60) % 360, 45, 65),
        cyan: hslToHex((h + 180) % 360, 40, 60),
        white: '#E8E4E0',
        brightBlack: bgLight,
        brightRed: '#FF6961',
        brightGreen: hslToHex(h, 50, 72),
        brightYellow: '#FFE066',
        brightBlue: '#409CFF',
        brightMagenta: hslToHex((h + 60) % 360, 50, 75),
        brightCyan: hslToHex((h + 180) % 360, 45, 70),
        brightWhite: '#FFFFFF'
      }
    }

    // Light: colored accent wash — the terminal lives inside the gradient
    const bg = hslToHex(h, 45, 85)
    const fg = '#000000'
    const fgMid = hslToHex(h, 20, 35)
    const accent = hslToHex(h, 45, 38)
    const accentVivid = hslToHex(h, 55, 32)

    return {
      background: bg,
      foreground: fg,
      cursor: fg,
      cursorAccent: bg,
      selectionBackground: hslToHexAlpha(h, 25, 40, 0.15),
      black: fg,
      red: '#B84A4A',
      green: accent,
      yellow: '#C4923A',
      blue: '#2472C8',
      magenta: hslToHex((h + 60) % 360, 30, 45),
      cyan: hslToHex((h + 180) % 360, 35, 40),
      white: bg,
      brightBlack: fgMid,
      brightRed: '#CD3131',
      brightGreen: accentVivid,
      brightYellow: '#E5C07B',
      brightBlue: '#3B8EEA',
      brightMagenta: hslToHex((h + 60) % 360, 40, 55),
      brightCyan: hslToHex((h + 180) % 360, 40, 50),
      brightWhite: '#FFFFFF'
    }
  }, [theme, accentHue, hslToHex, hslToHexAlpha])

  // Initialize terminal (only runs once per terminal ID)
  useEffect(() => {
    if (!containerRef.current || terminalRef.current) return

    // Use accent-aware theme colors for initialization
    const initialColors = getThemeColors()

    const terminal = new Terminal({
      cursorBlink: true,
      cursorStyle: 'bar',
      fontSize: 13,
      fontFamily: '"SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas, monospace',
      lineHeight: 1.2,
      letterSpacing: 0,
      theme: initialColors,
      allowTransparency: false,
      scrollback: 10000,
      rightClickSelectsWord: true,
      allowProposedApi: true
    })

    const fitAddon = new FitAddon()
    terminal.loadAddon(fitAddon)

    // Try WebGL, fall back to canvas
    try {
      const webglAddon = new WebglAddon()
      webglAddon.onContextLoss(() => {
        webglAddon.dispose()
      })
      terminal.loadAddon(webglAddon)
    } catch {
      console.log('WebGL not available, using canvas renderer')
    }

    // Web links addon
    terminal.loadAddon(new WebLinksAddon())

    terminal.open(containerRef.current)

    // Wait for terminal to be fully ready before fitting
    // The terminal needs a frame to initialize its render service
    requestAnimationFrame(() => {
      if (terminalRef.current && fitAddonRef.current) {
        try {
          fitAddonRef.current.fit()
        } catch (e) {
          // Ignore - terminal may not be fully ready
        }
      }
    })

    // Handle keyboard shortcuts for copy
    terminal.attachCustomKeyEventHandler((event) => {
      // Only handle keydown events
      if (event.type !== 'keydown') return true

      // Ctrl+Shift+C for copy
      if (event.ctrlKey && event.shiftKey && event.code === 'KeyC') {
        const selection = terminal.getSelection()
        if (selection) {
          navigator.clipboard.writeText(selection)
        }
        return false
      }

      // Ctrl+C - copy if selection exists, otherwise let SIGINT through
      if (event.ctrlKey && !event.shiftKey && event.code === 'KeyC') {
        const selection = terminal.getSelection()
        if (selection) {
          navigator.clipboard.writeText(selection)
          terminal.clearSelection()
          return false
        }
        return true // Let SIGINT through
      }

      // For Ctrl+V and Ctrl+Shift+V, just block xterm handling
      // The paste event listener will handle the actual paste
      if (event.ctrlKey && event.code === 'KeyV') {
        return false // Block xterm, let browser paste event fire
      }

      return true
    })

    // Single paste handler - catches Ctrl+V, right-click paste, and menu paste
    const handlePasteEvent = (e: ClipboardEvent) => {
      e.preventDefault()
      e.stopPropagation()
      const text = e.clipboardData?.getData('text')
      if (text) {
        window.bashgym?.terminal.write(id, text)
      }
    }
    containerRef.current.addEventListener('paste', handlePasteEvent)

    terminalRef.current = terminal
    fitAddonRef.current = fitAddon

    // Banner colors
    const green = '\x1b[38;2;118;185;0m'  // NVIDIA green
    const dim = '\x1b[2m'
    const reset = '\x1b[0m'
    const clear = '\x1b[2J\x1b[H'  // Clear screen and move cursor to top

    // Function to write the Bash Gym banner
    const writeBanner = () => {
      const line = '\x1b[38;2;60;60;60m'       // Dark gray for separator

      // Reset terminal state and clear
      terminal.reset()
      terminal.writeln('')
      terminal.writeln(`${green}  ██████╗  █████╗ ███████╗██╗  ██╗     ██████╗ ██╗   ██╗███╗   ███╗${reset}`)
      terminal.writeln(`${green}  ██╔══██╗██╔══██╗██╔════╝██║  ██║    ██╔════╝ ╚██╗ ██╔╝████╗ ████║${reset}`)
      terminal.writeln(`${green}  ██████╔╝███████║███████╗███████║    ██║  ███╗ ╚████╔╝ ██╔████╔██║${reset}`)
      terminal.writeln(`${green}  ██╔══██╗██╔══██║╚════██║██╔══██║    ██║   ██║  ╚██╔╝  ██║╚██╔╝██║${reset}`)
      terminal.writeln(`${green}  ██████╔╝██║  ██║███████║██║  ██║    ╚██████╔╝   ██║   ██║ ╚═╝ ██║${reset}`)
      terminal.writeln(`${green}  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝     ╚═════╝    ╚═╝   ╚═╝     ╚═╝${reset}`)
      terminal.writeln('')
      terminal.writeln(`${dim}                    Self-Improving Agentic Development Gym${reset}`)
      terminal.writeln('')
      terminal.writeln(`${line}  ═══════════════════════════════════════════════════════════════════════════${reset}`)
      terminal.writeln('')

      // Force resize sync after banner to update PTY cursor position
      setTimeout(() => {
        fitAddon.fit()
        const { cols, rows } = terminal
        window.bashgym?.terminal.resize(id, cols, rows)
      }, 50)
    }

    // Check if Electron terminal API is available (browser mode has no PTY)
    if (!window.bashgym?.terminal) {
      const yellow = '\x1b[33m'
      const dim = '\x1b[2m'
      const reset = '\x1b[0m'
      terminal.writeln('')
      terminal.writeln(`${yellow}  Terminal requires the Bash Gym desktop app.${reset}`)
      terminal.writeln('')
      terminal.writeln(`${dim}  The browser preview does not support terminal sessions.${reset}`)
      terminal.writeln(`${dim}  Please launch the Electron app to use terminals.${reset}`)
      terminal.writeln('')

      return () => {
        terminal.dispose()
        terminalRef.current = null
        fitAddonRef.current = null
      }
    }

    // Create PTY process
    window.bashgym?.terminal.create(id).then((result) => {
      if (!result.success) {
        terminal.writeln(`\x1b[31mFailed to create terminal: ${result.error}\x1b[0m`)
        return
      }

      // Handle input
      terminal.onData((data) => {
        window.bashgym?.terminal.write(id, data)
      })

      // Handle resize
      terminal.onResize(({ cols, rows }) => {
        window.bashgym?.terminal.resize(id, cols, rows)
      })
    })

    // Track if banner has been shown (only for first terminal)
    let bannerShown = false
    const shouldShowBanner = session?.showBanner ?? false

    // Listen for output
    const removeDataListener = window.bashgym?.terminal.onData(id, (data) => {
      // Show banner after first output (shell is ready) - only for first terminal
      if (shouldShowBanner && !bannerShown) {
        bannerShown = true
        // Small delay to let shell fully initialize
        setTimeout(() => {
          writeBanner()
        }, 100)
      }

      terminal.write(data)

      // Parse output for session state detection
      parseTerminalOutput(data)
    })

    // Listen for exit
    const removeExitListener = window.bashgym?.terminal.onExit(id, (exitCode) => {
      terminal.writeln(`\r\n\x1b[33mProcess exited with code ${exitCode}\x1b[0m`)
    })

    // Handle resize with debouncing
    let resizeTimeout: ReturnType<typeof setTimeout> | null = null
    const handleResize = () => {
      if (resizeTimeout) {
        clearTimeout(resizeTimeout)
      }
      resizeTimeout = setTimeout(() => {
        try {
          // Preserve scroll position during resize
          const buffer = terminal.buffer.active
          const currentLine = buffer.viewportY
          const totalLines = buffer.length
          const viewportRows = terminal.rows
          const distanceFromBottom = totalLines - currentLine - viewportRows

          fitAddon.fit()

          // Restore scroll position
          if (distanceFromBottom <= 1) {
            terminal.scrollToBottom()
          } else {
            const newTotalLines = terminal.buffer.active.length
            const newViewportRows = terminal.rows
            const targetLine = Math.max(0, newTotalLines - newViewportRows - distanceFromBottom)
            terminal.scrollToLine(targetLine)
          }
        } catch (e) {
          // Ignore fit errors (can happen during unmount)
        }
      }, 50)
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.current)

    // Also listen for window resize
    window.addEventListener('resize', handleResize)

    // Initial fit after a short delay to ensure container has dimensions
    setTimeout(() => {
      try {
        fitAddon.fit()
      } catch (e) {
        // Ignore
      }
    }, 100)

    // Right-click context menu
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault()
      setContextMenu({ x: e.clientX, y: e.clientY })
    }
    containerRef.current.addEventListener('contextmenu', handleContextMenu)

    return () => {
      removeDataListener?.()
      removeExitListener?.()
      resizeObserver.disconnect()
      window.removeEventListener('resize', handleResize)
      if (resizeTimeout) {
        clearTimeout(resizeTimeout)
      }
      containerRef.current?.removeEventListener('contextmenu', handleContextMenu)
      containerRef.current?.removeEventListener('paste', handlePasteEvent)
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current)
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
      terminal.dispose()
      terminalRef.current = null
      fitAddonRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]) // Only re-initialize when terminal ID changes, not on theme change

  // Update theme when theme or accent hue changes
  useEffect(() => {
    if (terminalRef.current) {
      const colors = getThemeColors()
      terminalRef.current.options.theme = colors
      // Force xterm to repaint (WebGL renderer caches colors)
      terminalRef.current.refresh(0, terminalRef.current.rows - 1)
    }
  }, [theme, accentHue, getThemeColors])

  // Focus terminal and refit when active
  useEffect(() => {
    if (isActive && terminalRef.current && fitAddonRef.current) {
      terminalRef.current.focus()
      // Refit when becoming active (handles tab switches and visibility changes)
      // Use multiple delays to handle different timing scenarios
      const timeouts = [10, 50, 150].map(delay =>
        setTimeout(() => {
          fitWithScrollPreservation()
        }, delay)
      )
      return () => timeouts.forEach(t => clearTimeout(t))
    }
  }, [isActive, fitWithScrollPreservation])

  // Refit on view mode or panel count changes
  useEffect(() => {
    if (fitAddonRef.current) {
      const timeoutId = setTimeout(() => {
        fitWithScrollPreservation()
      }, 100)
      return () => clearTimeout(timeoutId)
    }
  }, [viewMode, panels.length, fitWithScrollPreservation])

  // Use IntersectionObserver to detect when terminal becomes visible
  // This handles cases like canvas popup where isActive might not change
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && fitAddonRef.current) {
            // Terminal became visible, refit after a short delay
            setTimeout(() => {
              fitWithScrollPreservation()
              if (isActive) {
                terminalRef.current?.focus()
              }
            }, 50)
          }
        })
      },
      { threshold: 0.1 }
    )

    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [isActive, fitWithScrollPreservation])

  // Parse terminal output for state detection
  // Buffers chunks and strips ANSI codes before matching patterns
  const parseTerminalOutput = (data: string) => {
    outputBufferRef.current += data

    // Debounce: flush after 80ms of no new data (handles split PTY chunks)
    if (flushTimerRef.current) clearTimeout(flushTimerRef.current)
    flushTimerRef.current = setTimeout(() => {
      const raw = outputBufferRef.current
      outputBufferRef.current = ''
      const clean = stripAnsi(raw)

      // --- Pattern matching in priority order ---

      // 1. Spinner + tool name → tool_calling
      const toolMatch = clean.match(/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s*(Read|Edit|Write|Bash|Glob|Grep|Task|WebFetch|WebSearch|NotebookEdit)/i)
      if (toolMatch) {
        updateSession(id, {
          status: 'tool_calling',
          currentTool: toolMatch[1],
          lastActivity: Date.now()
        })
        resetIdleTimer()
        return
      }

      // 2. Spinner + Thinking/Planning → running
      if (/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]\s*(Thinking|Planning)/i.test(clean)) {
        updateSession(id, {
          status: 'running',
          lastActivity: Date.now()
        })
        resetIdleTimer()
        return
      }

      // 3. Spinner without recognized label → running
      if (/[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]/.test(clean)) {
        updateSession(id, {
          status: 'running',
          lastActivity: Date.now()
        })
        resetIdleTimer()
        return
      }

      // 4. Tool completion markers (checkmark/cross) → still running (processing result)
      if (/[✓✗✔✘]/.test(clean)) {
        updateSession(id, {
          status: 'running',
          currentTool: undefined,
          lastActivity: Date.now()
        })
        resetIdleTimer()
        return
      }

      // 5. Claude prompt (waiting for user input)
      //    - ">" on its own line (Claude's input prompt)
      //    - Box drawing chars from Claude's UI (╭, ╰)
      if (/^\s*>\s*$/m.test(clean) || /[╭╰]/.test(clean)) {
        updateSession(id, {
          status: 'waiting_input',
          currentTool: undefined,
          lastActivity: Date.now()
        })
        // Clear idle timer - we've reached a definitive state
        if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
        return
      }

      // 6. Shell prompt → idle (back to shell)
      if (/PS\s+[A-Z]:\\[^>]*>\s*$/m.test(clean) || /[$#%]\s*$/m.test(clean)) {
        updateSession(id, {
          status: 'idle',
          currentTool: undefined,
          lastActivity: Date.now()
        })
        if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
        return
      }

      // Any other output = activity, reset idle timer
      resetIdleTimer()
    }, 80)
  }

  // Reset the idle timeout — if no activity for 8s while running/tool_calling,
  // transition to waiting_input (Claude likely finished and is waiting)
  const resetIdleTimer = () => {
    if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
    idleTimerRef.current = setTimeout(() => {
      const current = useTerminalStore.getState().sessions.get(id)
      if (current && (current.status === 'running' || current.status === 'tool_calling')) {
        updateSession(id, {
          status: 'waiting_input',
          currentTool: undefined,
          lastActivity: Date.now()
        })
      }
    }, 8000)
  }

  const [showMenu, setShowMenu] = useState(false)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null)

  const handleClose = (e: React.MouseEvent) => {
    e.stopPropagation()
    // If in a popup context, close the popup instead of the terminal
    if (onPopupClose) {
      onPopupClose()
    } else {
      closeTerminal(id)
    }
  }

  const handleClear = () => {
    if (terminalRef.current) {
      terminalRef.current.clear()
    }
    setShowMenu(false)
  }

  const handleReset = () => {
    if (terminalRef.current) {
      terminalRef.current.reset()
    }
    setShowMenu(false)
  }

  const handleCopy = useCallback(() => {
    if (terminalRef.current) {
      const selection = terminalRef.current.getSelection()
      if (selection) {
        navigator.clipboard.writeText(selection)
        terminalRef.current.clearSelection()
      }
    }
    setContextMenu(null)
    setShowMenu(false)
  }, [])

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text) {
        window.bashgym?.terminal.write(id, text)
      }
    } catch {
      console.log('Failed to read clipboard')
    }
    setContextMenu(null)
    setShowMenu(false)
  }, [id])

  // Get status icon based on agent status
  const getStatusIcon = () => {
    switch (session?.status) {
      case 'running':
        return <Loader2 className="w-3 h-3 animate-spin text-accent" />
      case 'waiting_input':
        return <MessageSquare className="w-3 h-3 text-status-warning" />
      case 'tool_calling':
        return <Wrench className="w-3 h-3 text-accent animate-pulse" />
      case 'idle':
      default:
        return <Coffee className="w-3 h-3 text-text-muted" />
    }
  }

  // Get status tooltip
  const getStatusTooltip = () => {
    switch (session?.status) {
      case 'running':
        return 'Agent is processing...'
      case 'waiting_input':
        return 'Waiting for input'
      case 'tool_calling':
        return `Using tool: ${session?.currentTool || 'Unknown'}`
      case 'idle':
      default:
        return 'Idle'
    }
  }

  return (
    <FileDropZone terminalId={id} className="h-full">
      <div className="terminal-chrome h-full flex flex-col">
        {/* Terminal Header — macOS dots + title in JetBrains Mono */}
        <div className="terminal-header">
          <div className="flex items-center gap-1.5">
            <span className="terminal-dot terminal-dot-red" />
            <span className="terminal-dot terminal-dot-yellow" />
            <span className="terminal-dot terminal-dot-green" />
          </div>
          <div className="flex items-center gap-2 ml-3 flex-1 min-w-0">
            {/* Status indicator */}
            <div title={getStatusTooltip()} className="flex-shrink-0">
              {getStatusIcon()}
            </div>
            {isEditingTitle ? (
              <input
                ref={titleInputRef}
                type="text"
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onBlur={saveTitleEdit}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    saveTitleEdit()
                  } else if (e.key === 'Escape') {
                    cancelTitleEdit()
                  }
                  e.stopPropagation()
                }}
                className="input !text-sm !py-0.5 !px-1.5 w-32 font-mono"
                autoFocus
              />
            ) : (
              <span
                className="text-sm font-mono font-semibold text-text-primary truncate max-w-[200px] cursor-pointer hover:text-accent"
                onDoubleClick={startTitleEdit}
                title="Double-click to rename"
              >
                {title}
              </span>
            )}
            {session?.gitBranch && (
              <span className="tag !text-[10px] !py-0 !px-1.5">
                <span>{session.gitBranch}</span>
              </span>
            )}
            {session?.currentTool && session?.status === 'tool_calling' && (
              <span className="tag !text-[10px] !py-0 !px-1.5 !bg-status-info !border-status-info">
                <span>{session.currentTool}</span>
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            {/* More options dropdown */}
            <div className="relative">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
                title="More options"
              >
                <MoreHorizontal className="w-4 h-4" />
              </button>
              {showMenu && (
                <>
                  <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
                  <div className="absolute right-0 top-full mt-1 z-20 card !rounded-brutal py-1 min-w-[160px]">
                    <button
                      onClick={handleCopy}
                      className="menu-item w-full !px-3 !py-1.5 !text-sm"
                    >
                      <Copy className="w-3.5 h-3.5" />
                      Copy
                      <span className="ml-auto text-xs text-text-muted font-mono">Ctrl+C</span>
                    </button>
                    <button
                      onClick={handlePaste}
                      className="menu-item w-full !px-3 !py-1.5 !text-sm"
                    >
                      <ClipboardPaste className="w-3.5 h-3.5" />
                      Paste
                      <span className="ml-auto text-xs text-text-muted font-mono">Ctrl+V</span>
                    </button>
                    <div className="section-divider my-1" />
                    <button
                      onClick={handleClear}
                      className="menu-item w-full !px-3 !py-1.5 !text-sm"
                    >
                      Clear Terminal
                    </button>
                    <button
                      onClick={handleReset}
                      className="menu-item w-full !px-3 !py-1.5 !text-sm"
                    >
                      Reset Terminal
                    </button>
                  </div>
                </>
              )}
            </div>
            <button
              onClick={() => {
                setActiveTerminal(id)
                setViewMode('single')
              }}
              className="p-1 hover:bg-background-tertiary text-text-muted hover:text-text-secondary transition-press"
              title="Fullscreen"
            >
              <Square className="w-3 h-3" />
            </button>
            <button
              onClick={handleClose}
              className="p-1 hover:bg-status-error/20 text-text-muted hover:text-status-error transition-press"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Terminal Content */}
        <div
          ref={containerRef}
          className={clsx(
            'flex-1 overflow-hidden',
            session?.attention === 'success' && 'terminal-attention-success',
            session?.attention === 'error' && 'terminal-attention-error',
            session?.attention === 'waiting' && 'terminal-attention-waiting'
          )}
        />

        {/* Right-click Context Menu */}
        {contextMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setContextMenu(null)} />
            <div
              className="fixed z-50 card !rounded-brutal py-1 min-w-[160px]"
              style={{ left: contextMenu.x, top: contextMenu.y }}
            >
              <button
                onClick={handleCopy}
                className="menu-item w-full !px-3 !py-1.5 !text-sm"
              >
                <Copy className="w-3.5 h-3.5" />
                Copy
                <span className="ml-auto text-xs text-text-muted font-mono">Ctrl+C</span>
              </button>
              <button
                onClick={handlePaste}
                className="menu-item w-full !px-3 !py-1.5 !text-sm"
              >
                <ClipboardPaste className="w-3.5 h-3.5" />
                Paste
                <span className="ml-auto text-xs text-text-muted font-mono">Ctrl+V</span>
              </button>
              <div className="section-divider my-1" />
              <button
                onClick={() => {
                  if (terminalRef.current) {
                    terminalRef.current.selectAll()
                  }
                  setContextMenu(null)
                }}
                className="menu-item w-full !px-3 !py-1.5 !text-sm"
              >
                Select All
              </button>
              <button
                onClick={() => {
                  handleClear()
                  setContextMenu(null)
                }}
                className="menu-item w-full !px-3 !py-1.5 !text-sm"
              >
                Clear
              </button>
            </div>
          </>
        )}
      </div>
    </FileDropZone>
  )
}
