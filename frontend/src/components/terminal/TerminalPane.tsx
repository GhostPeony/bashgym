import { useEffect, useRef, useCallback, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebglAddon } from '@xterm/addon-webgl'
import { WebLinksAddon } from '@xterm/addon-web-links'
import {
  Square,
  X,
  MoreHorizontal,
  Copy,
  ClipboardPaste,
  Loader2,
  MessageSquare,
  Wrench,
  Coffee
} from 'lucide-react'
import {
  useTerminalStore,
  useThemeStore,
  useAccentStore,
  useWorkspaceStore,
  getTerminalFgColor
} from '../../stores'
import { FileDropZone } from './FileDropZone'
import { clsx } from 'clsx'
import { stripAnsi } from '../../utils/ansi'
import { maybeAutoSnapshot } from '../../utils/monitorRouting'
import {
  detectTerminalActivity,
  detectTerminalAgentKind,
  TerminalOutputBatcher,
  terminalOutputLines
} from './terminalAgentRuntime'
import '@xterm/xterm/css/xterm.css'

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
  const outputBatcherRef = useRef<TerminalOutputBatcher | null>(null)
  const outputFlowOwnerRef = useRef(crypto.randomUUID())
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const startPtyRef = useRef<(() => void) | null>(null)
  const lastSummaryPublishRef = useRef(0)
  const fitFrameRef = useRef<number | undefined>(undefined)
  const lastSyncedSizeRef = useRef<{ cols: number; rows: number } | undefined>(undefined)

  const requestCloseTerminal = useTerminalStore((state) => state.requestCloseTerminal)
  const updateSession = useTerminalStore((state) => state.updateSession)
  const setViewMode = useTerminalStore((state) => state.setViewMode)
  const setActiveTerminal = useTerminalStore((state) => state.setActiveTerminal)
  const viewMode = useTerminalStore((state) => state.viewMode)
  const renamePanel = useTerminalStore((state) => state.renamePanel)
  const panelCount = useTerminalStore((state) => state.panels.length)
  const panel = useTerminalStore((state) =>
    state.panels.find((candidate) => candidate.terminalId === id)
  )
  const session = useTerminalStore((state) => state.sessions.get(id))

  // When the PTY exits we keep the pane alive and offer a restart instead of a dead view
  const [exitedCode, setExitedCode] = useState<number | null>(null)

  const canFitTerminal = useCallback(() => {
    const surface = containerRef.current
    if (!surface?.isConnected || surface.clientWidth < 1 || surface.clientHeight < 1) return false
    const style = getComputedStyle(surface)
    return style.display !== 'none' && style.visibility !== 'hidden'
  }, [])

  const syncTerminalSize = useCallback(
    async (force = false): Promise<boolean> => {
      const terminal = terminalRef.current
      if (!terminal) return false
      const next = { cols: terminal.cols, rows: terminal.rows }
      const previous = lastSyncedSizeRef.current
      if (!force && previous?.cols === next.cols && previous.rows === next.rows) return true
      const resized = (await window.bashgym?.terminal.resize(id, next.cols, next.rows)) ?? false
      if (resized) lastSyncedSizeRef.current = next
      return resized
    },
    [id]
  )

  // Helper to fit terminal while preserving scroll position
  const fitWithScrollPreservation = useCallback(() => {
    if (!fitAddonRef.current || !terminalRef.current) return

    const terminal = terminalRef.current
    if (!terminal.element || !canFitTerminal()) return

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
      // A fit can preserve the same cols/rows, in which case xterm does not
      // emit onResize. Send the dimensions explicitly so a PTY created after
      // an earlier failed resize still receives the real viewport size.
      void syncTerminalSize()

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
    } catch {
      // Ignore fit errors
    }
  }, [canFitTerminal, syncTerminalSize])

  const scheduleFitWithScrollPreservation = useCallback(() => {
    if (fitFrameRef.current !== undefined) return
    fitFrameRef.current = requestAnimationFrame(() => {
      fitFrameRef.current = undefined
      fitWithScrollPreservation()
    })
  }, [fitWithScrollPreservation])
  const { resolvedTheme } = useThemeStore()
  const { accentHue, terminalFgHue } = useAccentStore()

  // State for inline title editing
  const [isEditingTitle, setIsEditingTitle] = useState(false)
  const [editTitle, setEditTitle] = useState(title)
  const titleInputRef = useRef<HTMLInputElement>(null)

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
      return Math.round(255 * color)
        .toString(16)
        .padStart(2, '0')
    }
    return `#${f(0)}${f(8)}${f(4)}`
  }, [])

  // HSL to rgba hex string (xterm needs #RRGGBBAA for selection)
  const hslToHexAlpha = useCallback(
    (h: number, s: number, l: number, alpha: number): string => {
      const hex = hslToHex(h, s, l)
      const a = Math.round(alpha * 255)
        .toString(16)
        .padStart(2, '0')
      return `${hex}${a}`
    },
    [hslToHex]
  )

  // Accent-influenced terminal theme colors
  const getThemeColors = useCallback(() => {
    const h = accentHue
    const fgValue = typeof terminalFgHue === 'number' ? terminalFgHue : -1

    if (resolvedTheme === 'dark') {
      // Dark: rich accent-tinted background — nighttime garden, not void
      const bg = hslToHex(h, 35, 8)
      const bgLight = hslToHex(h, 25, 15)
      const accent = hslToHex(h, 50, 68)

      const fg = getTerminalFgColor('dark', fgValue)

      return {
        background: bg,
        foreground: fg,
        cursor: accent,
        cursorAccent: bg,
        selectionBackground: hslToHexAlpha(h, 40, 55, 0.35),
        black: bg,
        red: '#FF453A',
        green: accent,
        yellow: fg,
        blue: '#0A84FF',
        magenta: hslToHex((h + 60) % 360, 45, 65),
        cyan: hslToHex((h + 180) % 360, 40, 60),
        white: fg,
        brightBlack: bgLight,
        brightRed: '#FF6961',
        brightGreen: hslToHex(h, 50, 72),
        brightYellow: fg,
        brightBlue: '#409CFF',
        brightMagenta: hslToHex((h + 60) % 360, 50, 75),
        brightCyan: hslToHex((h + 180) % 360, 45, 70),
        brightWhite: '#FFFFFF'
      }
    }

    // Light: colored accent wash — the terminal lives inside the gradient
    const bg = hslToHex(h, 45, 85)

    const fg = getTerminalFgColor('light', fgValue)
    const fgMid = fgValue >= 0 ? hslToHex(fgValue, 30, 45) : hslToHex(h, 20, 35)
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
      yellow: fg,
      blue: '#2472C8',
      magenta: hslToHex((h + 60) % 360, 30, 45),
      cyan: hslToHex((h + 180) % 360, 35, 40),
      white: bg,
      brightBlack: fgMid,
      brightRed: '#CD3131',
      brightGreen: accentVivid,
      brightYellow: fg,
      brightBlue: '#3B8EEA',
      brightMagenta: hslToHex((h + 60) % 360, 40, 55),
      brightCyan: hslToHex((h + 180) % 360, 40, 50),
      brightWhite: '#FFFFFF'
    }
  }, [resolvedTheme, accentHue, terminalFgHue, hslToHex, hslToHexAlpha])

  // Initialize terminal (only runs once per terminal ID)
  useEffect(() => {
    if (!containerRef.current || terminalRef.current) return
    const outputFlowOwner = outputFlowOwnerRef.current

    // Use accent-aware theme colors for initialization
    const initialColors = getThemeColors()

    const terminal = new Terminal({
      cursorBlink: true,
      cursorStyle: 'bar',
      fontSize: 13,
      fontFamily: '"SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas, monospace',
      lineHeight: 1.2,
      letterSpacing: 0,
      minimumContrastRatio: 4.5,
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
    scheduleFitWithScrollPreservation()

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
    const green = '\x1b[38;2;118;185;0m' // NVIDIA green
    const dim = '\x1b[2m'
    const reset = '\x1b[0m'

    // Function to write the Bash Gym banner
    const writeBanner = () => {
      const line = '\x1b[38;2;60;60;60m' // Dark gray for separator

      // Reset terminal state and clear
      terminal.reset()
      terminal.writeln('')
      terminal.writeln(
        `${green}  ██████╗  █████╗ ███████╗██╗  ██╗     ██████╗ ██╗   ██╗███╗   ███╗${reset}`
      )
      terminal.writeln(
        `${green}  ██╔══██╗██╔══██╗██╔════╝██║  ██║    ██╔════╝ ╚██╗ ██╔╝████╗ ████║${reset}`
      )
      terminal.writeln(
        `${green}  ██████╔╝███████║███████╗███████║    ██║  ███╗ ╚████╔╝ ██╔████╔██║${reset}`
      )
      terminal.writeln(
        `${green}  ██╔══██╗██╔══██║╚════██║██╔══██║    ██║   ██║  ╚██╔╝  ██║╚██╔╝██║${reset}`
      )
      terminal.writeln(
        `${green}  ██████╔╝██║  ██║███████║██║  ██║    ╚██████╔╝   ██║   ██║ ╚═╝ ██║${reset}`
      )
      terminal.writeln(
        `${green}  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝     ╚═════╝    ╚═╝   ╚═╝     ╚═╝${reset}`
      )
      terminal.writeln('')
      terminal.writeln(`${dim}                         Self-Improving Training Workspace${reset}`)
      terminal.writeln('')
      terminal.writeln(
        `${line}  ═══════════════════════════════════════════════════════════════════════════${reset}`
      )
      terminal.writeln('')

      // Force resize sync after banner to update PTY cursor position
      scheduleFitWithScrollPreservation()
    }

    // Check if Electron terminal API is available (browser mode has no PTY)
    if (!window.bashgym?.terminal) {
      const dim = '\x1b[2m'
      const reset = '\x1b[0m'
      terminal.writeln('')
      terminal.writeln('  Terminal requires the Bash Gym desktop app.')
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

    outputBatcherRef.current?.dispose()
    outputBatcherRef.current = new TerminalOutputBatcher(processTerminalOutput)

    // Handle input/resize — registered once per xterm instance, valid across PTY restarts
    terminal.onData((data) => {
      window.bashgym?.terminal.write(id, data)
    })
    terminal.onResize(({ cols, rows }) => {
      window.bashgym?.terminal.resize(id, cols, rows)
    })

    // Track if banner has been shown (only for first terminal)
    let bannerShown = false
    const shouldShowBanner = session?.showBanner ?? false

    // Live output listener — registered only AFTER create/attach resolves so
    // the replayed scrollback (snapshotted at create time in main) is written
    // before any live chunks. Registering earlier double-writes bytes that
    // arrive between subscription and the snapshot.
    let disposed = false
    let removeDataListener: (() => void) | undefined
    const ensureDataListener = () => {
      if (!removeDataListener)
        removeDataListener = window.bashgym?.terminal.onData(id, (output) => {
          const data = typeof output === 'string' ? output : output.data
          const frameId = typeof output === 'string' ? undefined : output.frameId
          // Show banner after first output (shell is ready) - only for first terminal
          if (shouldShowBanner && !bannerShown) {
            bannerShown = true
            // Small delay to let shell fully initialize
            setTimeout(() => {
              if (!disposed) writeBanner()
            }, 100)
          }

          // ACK only after xterm's parser finishes this frame. Main keeps one
          // frame in flight and coalesces Claude's later redraws behind it.
          terminal.write(
            data,
            frameId === undefined
              ? undefined
              : () => window.bashgym?.terminal.ackOutput?.(id, outputFlowOwner, frameId)
          )

          // Parse output for session state detection
          parseTerminalOutput(data)
        })
      // Re-acquire on every successful create. A restarted PTY is a new
      // main-process session even when this pane's data listener is unchanged.
      window.bashgym?.terminal.setOutputFlowControl?.(id, outputFlowOwner, true)
    }

    const syncPtySize = async (): Promise<boolean> => {
      if (!terminal.element || !canFitTerminal()) return false
      try {
        fitAddon.fit()
        return await syncTerminalSize(true)
      } catch {
        return false
      }
    }

    // Create or re-attach the PTY process (main process keeps PTYs alive across remounts)
    const startPty = () => {
      const requestedCwd = useTerminalStore.getState().sessions.get(id)?.cwd
      window.bashgym?.terminal
        .create(id, requestedCwd !== '~' ? requestedCwd : undefined)
        .then(async (result) => {
          if (disposed) return
          if (!result.success) {
            terminal.writeln(`\x1b[31mFailed to create terminal: ${result.error}\x1b[0m`)
            return
          }
          setExitedCode(null)
          if (result.cwd) {
            updateSession(id, { cwd: result.cwd })
          }
          // PowerShell startup is slow enough that initial xterm fit events often
          // arrive before the PTY exists. Synchronize once after create succeeds,
          // before a full-screen Claude/Codex TUI is launched into the default 80x24 PTY.
          await syncPtySize()
          if (disposed) return
          // Auto-type the launch command into a fresh PTY (not on re-attach)
          if (!result.attached) {
            const pending = useTerminalStore.getState().sessions.get(id)?.launchCommand
            if (pending) {
              updateSession(id, { launchCommand: undefined })
              let command = pending
              if ((pending === 'claude' || pending === 'codex') && window.bashgym?.agentBridge) {
                const panelId = useTerminalStore
                  .getState()
                  .panels.find((candidate) => candidate.terminalId === id)?.id
                const bridge = await window.bashgym.agentBridge.prepareLaunch({
                  kind: pending,
                  workspaceId: useWorkspaceStore.getState().activeWorkspaceId,
                  terminalId: id,
                  panelId
                })
                if (disposed) return
                if (bridge.success && bridge.command) {
                  command = bridge.command
                } else {
                  terminal.writeln(
                    `\x1b[33mSkill Lab bridge unavailable: ${bridge.error || 'unknown error'}\x1b[0m`
                  )
                }
              }
              setTimeout(() => {
                if (!disposed) window.bashgym?.terminal.write(id, command + '\r')
              }, 500)
            }
          }
          // Re-attached to a live PTY: replay retained scrollback
          if (result.attached && result.buffer) {
            if (disposed) return
            terminal.write(result.buffer)
            parseTerminalOutput(result.buffer)
            // Sync PTY size to this (possibly new) viewport
            scheduleFitWithScrollPreservation()
          }
          ensureDataListener()
        })
    }

    startPty()
    startPtyRef.current = startPty

    // Listen for exit — surface a restart overlay instead of leaving a dead pane
    const removeExitListener = window.bashgym?.terminal.onExit(id, (exitCode) => {
      terminal.writeln(`\r\n\x1b[2mProcess exited with code ${exitCode}\x1b[0m`)
      setExitedCode(exitCode)
      updateSession(id, {
        status: 'idle',
        attention: exitCode === 0 ? 'none' : 'error',
        currentTool: undefined,
        taskSummary: undefined
      })
    })

    // ResizeObserver, view changes, and visibility changes all converge on one
    // animation-frame fit. Hidden canvas terminals do no layout or PTY work.
    const handleResize = () => scheduleFitWithScrollPreservation()

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.current)

    // Also listen for window resize
    window.addEventListener('resize', handleResize)

    scheduleFitWithScrollPreservation()

    // Right-click context menu
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault()
      setContextMenu({ x: e.clientX, y: e.clientY })
    }
    const currentContainer = containerRef.current
    currentContainer.addEventListener('contextmenu', handleContextMenu)

    return () => {
      disposed = true
      window.bashgym?.terminal.setOutputFlowControl?.(id, outputFlowOwner, false)
      removeDataListener?.()
      removeExitListener?.()
      resizeObserver.disconnect()
      window.removeEventListener('resize', handleResize)
      if (fitFrameRef.current !== undefined) cancelAnimationFrame(fitFrameRef.current)
      fitFrameRef.current = undefined
      currentContainer?.removeEventListener('contextmenu', handleContextMenu)
      currentContainer?.removeEventListener('paste', handlePasteEvent)
      outputBatcherRef.current?.dispose()
      outputBatcherRef.current = null
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
      terminal.dispose()
      terminalRef.current = null
      fitAddonRef.current = null
      lastSyncedSizeRef.current = undefined
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]) // Only re-initialize when terminal ID changes, not on theme change

  // Update theme when theme, accent hue, or terminal fg hue changes
  useEffect(() => {
    if (terminalRef.current) {
      const colors = getThemeColors()
      terminalRef.current.options.theme = colors
      // Force xterm to repaint (WebGL renderer caches colors)
      terminalRef.current.refresh(0, terminalRef.current.rows - 1)
    }
  }, [resolvedTheme, accentHue, terminalFgHue, getThemeColors])

  // Focus terminal and refit when active
  useEffect(() => {
    if (isActive && terminalRef.current && fitAddonRef.current) {
      terminalRef.current.focus()
      scheduleFitWithScrollPreservation()
    }
  }, [isActive, scheduleFitWithScrollPreservation])

  // Refit on view mode or panel count changes
  useEffect(() => {
    if (fitAddonRef.current) {
      scheduleFitWithScrollPreservation()
    }
  }, [viewMode, panelCount, scheduleFitWithScrollPreservation])

  // Use IntersectionObserver to detect when terminal becomes visible
  // This handles cases like canvas popup where isActive might not change
  useEffect(() => {
    if (!containerRef.current) return

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && fitAddonRef.current) {
            scheduleFitWithScrollPreservation()
            // Don't yank focus away from an input the user just clicked into
            const active = document.activeElement as HTMLElement | null
            const typingElsewhere =
              active &&
              active !== document.body &&
              !active.closest('.xterm') &&
              (active.tagName === 'INPUT' ||
                active.tagName === 'TEXTAREA' ||
                active.isContentEditable)
            if (isActive && !typingElsewhere) terminalRef.current?.focus()
          }
        })
      },
      { threshold: 0.1 }
    )

    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [isActive, scheduleFitWithScrollPreservation])

  // Raw PTY chunks can arrive faster than a debounce ever settles. The batcher
  // preserves split sequences while forcing a bounded refresh during sustained
  // Claude/Codex output.
  const parseTerminalOutput = (data: string) => {
    outputBatcherRef.current?.push(data)
  }

  const processTerminalOutput = (raw: string) => {
    const clean = stripAnsi(raw)
    const current = useTerminalStore.getState().sessions.get(id)
    if (!current) return

    const now = Date.now()
    const detectedAgentKind = detectTerminalAgentKind(clean)
    const activity = detectTerminalActivity(clean)
    const retainedLines = terminalOutputLines(clean)
    const updates: Partial<typeof current> = {}
    const commitUpdates = () => {
      const changed = Object.entries(updates).some(
        ([key, value]) => current[key as keyof typeof current] !== value
      )
      if (changed) updateSession(id, updates)
    }
    const markActive = () => {
      if (now - current.lastActivity >= 1_000) updates.lastActivity = now
    }

    if (detectedAgentKind) updates.agentKind = detectedAgentKind

    if (retainedLines.length > 0) {
      const previous = current.lastOutput || []
      // Keep enough text for the optional stream overlay without putting raw
      // PTY chunks into durable activity state.
      const next = [...previous, ...retainedLines].slice(-80)
      if (next.join('\n') !== previous.join('\n')) updates.lastOutput = next
    }

    const hasAgent = Boolean(detectedAgentKind || current.agentKind)
    const publishSummary =
      hasAgent &&
      activity.summary &&
      activity.summary !== current.taskSummary &&
      (activity.status === 'tool_calling' || now - lastSummaryPublishRef.current >= 500)
    if (publishSummary) {
      updates.taskSummary = activity.summary
      lastSummaryPublishRef.current = now
    }

    if (activity.status === 'tool_calling') {
      updates.status = 'tool_calling'
      updates.attention = 'none'
      updates.currentTool = activity.currentTool
      markActive()
      if (activity.currentTool) {
        const lastHistory = current.toolHistory?.[current.toolHistory.length - 1]
        if (lastHistory?.tool !== activity.currentTool || lastHistory?.target !== activity.target) {
          updates.toolHistory = [
            ...(current.toolHistory || []),
            { tool: activity.currentTool, target: activity.target, timestamp: now }
          ].slice(-12)
        }
      }
      commitUpdates()
      resetIdleTimer()
      return
    }

    if (activity.status === 'running') {
      updates.status = 'running'
      updates.attention = 'none'
      updates.currentTool = undefined
      markActive()
      commitUpdates()
      resetIdleTimer()
      return
    }

    if (activity.status === 'waiting_input') {
      const wasWorking = current.status === 'running' || current.status === 'tool_calling'
      updates.status = 'waiting_input'
      updates.attention = 'waiting'
      updates.currentTool = undefined
      updates.lastActivity = now
      commitUpdates()
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
      if (wasWorking) maybeAutoSnapshot(id)
      return
    }

    if (activity.status === 'idle') {
      const wasWorking = current.status === 'running' || current.status === 'tool_calling'
      updates.status = 'idle'
      updates.attention = 'none'
      updates.currentTool = undefined
      updates.agentKind = undefined
      updates.taskSummary = undefined
      updates.lastActivity = now
      if (activity.shellCwd) updates.cwd = activity.shellCwd
      commitUpdates()
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
      if (wasWorking) maybeAutoSnapshot(id)
      return
    }

    if (clean.trim().length > 0 && current.status === 'idle') {
      updates.status = 'running'
      updates.attention = 'none'
      markActive()
    }
    commitUpdates()
    resetIdleTimer()
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
          attention: 'waiting',
          currentTool: undefined,
          lastActivity: Date.now()
        })
        maybeAutoSnapshot(id)
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
      requestCloseTerminal(id)
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
        {/* Terminal Header */}
        <div className="terminal-header">
          <div className="flex items-center gap-2 flex-1 min-w-0">
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
        <div className="relative flex-1 overflow-hidden">
          <div
            ref={containerRef}
            className={clsx(
              'h-full w-full overflow-hidden',
              session?.attention === 'success' && 'terminal-attention-success',
              session?.attention === 'error' && 'terminal-attention-error',
              session?.attention === 'waiting' && 'terminal-attention-waiting',
              session?.status === 'running' && 'terminal-status-running',
              session?.status === 'tool_calling' && 'terminal-status-tool-calling',
              session?.status === 'waiting_input' && 'terminal-status-waiting-input'
            )}
          />

          {/* Exited overlay — restart the shell without losing the pane */}
          {exitedCode !== null && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-background-primary/70">
              <div className="card flex flex-col items-center gap-3 px-6 py-5">
                <span className="text-sm font-mono text-text-muted">
                  Process exited with code {exitedCode}
                </span>
                <button
                  onClick={() => startPtyRef.current?.()}
                  className="px-4 py-1.5 text-sm font-mono font-semibold bg-accent text-background-primary hover:opacity-90 transition-press"
                >
                  Restart shell
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right-click Context Menu */}
        {contextMenu && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setContextMenu(null)} />
            <div
              className="fixed z-50 card !rounded-brutal py-1 min-w-[160px]"
              style={{ left: contextMenu.x, top: contextMenu.y }}
            >
              <button onClick={handleCopy} className="menu-item w-full !px-3 !py-1.5 !text-sm">
                <Copy className="w-3.5 h-3.5" />
                Copy
                <span className="ml-auto text-xs text-text-muted font-mono">Ctrl+C</span>
              </button>
              <button onClick={handlePaste} className="menu-item w-full !px-3 !py-1.5 !text-sm">
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
