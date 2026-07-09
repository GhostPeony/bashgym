import { useState, useRef } from 'react'
import { Send, TextCursorInput } from 'lucide-react'
import type { AgentStatus } from '../../stores'

interface QuickPromptProps {
  terminalId: string
  status: AgentStatus
}

/**
 * Send a prompt into a terminal without opening it. Send submits (adds Enter);
 * Prefill only types the text. When the agent is mid-task, sending requires a
 * second confirmation so nothing lands in a busy agent by accident.
 */
export function QuickPrompt({ terminalId, status }: QuickPromptProps) {
  const [text, setText] = useState('')
  const [armed, setArmed] = useState(false)
  const armTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const busy = status === 'running' || status === 'tool_calling'

  const write = (submit: boolean) => {
    if (!text.trim()) return
    window.bashgym?.terminal.write(terminalId, text + (submit ? '\r' : ''))
    setText('')
    setArmed(false)
    if (armTimerRef.current) clearTimeout(armTimerRef.current)
  }

  const handleSend = () => {
    if (busy && !armed) {
      setArmed(true)
      if (armTimerRef.current) clearTimeout(armTimerRef.current)
      armTimerRef.current = setTimeout(() => setArmed(false), 3_000)
      return
    }
    write(true)
  }

  return (
    <div>
      <div className="flex items-center gap-1">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              handleSend()
            }
          }}
          onClick={(e) => e.stopPropagation()}
          placeholder="Quick prompt…"
          className="input flex-1 !py-1 !px-2 !text-xs font-mono min-w-0"
        />
        <button
          onClick={(e) => { e.stopPropagation(); handleSend() }}
          disabled={!text.trim()}
          className="node-btn node-btn-accent"
          title="Send into the terminal (submits with Enter)"
        >
          <Send className="w-3 h-3" />
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); write(false) }}
          disabled={!text.trim()}
          className="node-btn"
          title="Prefill into the terminal input without submitting"
        >
          <TextCursorInput className="w-3 h-3" />
        </button>
      </div>
      {armed && (
        <p className="font-mono text-[10px] text-status-warning mt-1">
          agent is busy — press Enter again to send
        </p>
      )}
    </div>
  )
}
