import { useEffect, useRef, useState } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'

interface UseTerminalOptions {
  id: string
  onData?: (data: string) => void
  onExit?: (exitCode: number) => void
}

interface UseTerminalReturn {
  terminalRef: React.RefObject<HTMLDivElement>
  terminal: Terminal | null
  isReady: boolean
  write: (data: string) => void
  resize: () => void
}

export function useTerminal({
  id,
  onData,
  onExit
}: UseTerminalOptions): UseTerminalReturn {
  const terminalRef = useRef<HTMLDivElement>(null)
  const [terminal, setTerminal] = useState<Terminal | null>(null)
  const [fitAddon, setFitAddon] = useState<FitAddon | null>(null)
  const [isReady, setIsReady] = useState(false)

  // Initialize terminal
  useEffect(() => {
    if (!terminalRef.current || terminal) return

    const term = new Terminal({
      cursorBlink: true,
      cursorStyle: 'bar',
      fontSize: 13,
      fontFamily: '"SF Mono", "JetBrains Mono", Menlo, Monaco, Consolas, monospace',
      lineHeight: 1.2,
      theme: {
        background: '#0D0D0D',
        foreground: '#FFFFFF',
        cursor: '#76B900'
      }
    })

    const fit = new FitAddon()
    term.loadAddon(fit)

    term.open(terminalRef.current)
    fit.fit()

    setTerminal(term)
    setFitAddon(fit)

    // Create PTY process
    window.bashgym?.terminal.create(id).then((result) => {
      if (result.success) {
        setIsReady(true)

        // Handle input
        term.onData((data) => {
          window.bashgym?.terminal.write(id, data)
        })

        // Handle resize
        term.onResize(({ cols, rows }) => {
          window.bashgym?.terminal.resize(id, cols, rows)
        })
      }
    })

    // Listen for output
    const removeDataListener = window.bashgym?.terminal.onData(id, (data) => {
      term.write(data)
      onData?.(data)
    })

    // Listen for exit
    const removeExitListener = window.bashgym?.terminal.onExit(id, (exitCode) => {
      onExit?.(exitCode)
    })

    return () => {
      removeDataListener?.()
      removeExitListener?.()
      term.dispose()
    }
  }, [id])

  // Handle resize
  useEffect(() => {
    if (!fitAddon || !terminalRef.current) return

    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit()
    })

    resizeObserver.observe(terminalRef.current)

    return () => {
      resizeObserver.disconnect()
    }
  }, [fitAddon])

  const write = (data: string) => {
    window.bashgym?.terminal.write(id, data)
  }

  const resize = () => {
    fitAddon?.fit()
  }

  return {
    terminalRef,
    terminal,
    isReady,
    write,
    resize
  }
}
