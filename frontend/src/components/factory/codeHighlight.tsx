import { ReactNode } from 'react'

const CODE_PATTERNS = [
  /^(import|from|export|const|let|var|function|class|def|return)\b/m,
  /[{}();]/,
  /=>|->|::/,
  /^\s*(#|\/\/)/m,
  /\.(py|js|ts|sh|yml|json|toml)$/m,
]

export function looksLikeCode(text: string): boolean {
  if (!text || text.length < 10) return false
  let matches = 0
  for (const pattern of CODE_PATTERNS) {
    if (pattern.test(text)) matches++
  }
  return matches >= 2
}

interface Token {
  text: string
  className?: string
}

const HIGHLIGHT_RULES: [RegExp, string][] = [
  // Comments
  [/(#.*|\/\/.*)$/gm, 'text-text-muted italic'],
  // Strings
  [/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, 'text-status-success'],
  // Keywords
  [/\b(import|from|export|const|let|var|function|class|def|return|if|else|for|while|try|catch|async|await|yield|raise|except|finally|with|as)\b/g, 'text-accent'],
  // Numbers
  [/\b(\d+\.?\d*)\b/g, 'text-status-warning'],
]

export function highlightCode(text: string): ReactNode[] {
  // Split into lines and apply highlighting per line
  const lines = text.split('\n')
  const result: ReactNode[] = []

  lines.forEach((line, lineIdx) => {
    if (lineIdx > 0) result.push('\n')

    // Try to match rules in priority order
    const segments = highlightLine(line)
    segments.forEach((seg, segIdx) => {
      const key = `${lineIdx}-${segIdx}`
      if (seg.className) {
        result.push(<span key={key} className={seg.className}>{seg.text}</span>)
      } else {
        result.push(<span key={key}>{seg.text}</span>)
      }
    })
  })

  return result
}

function highlightLine(line: string): Token[] {
  // Check for full-line comment first
  if (/^\s*(#|\/\/)/.test(line)) {
    return [{ text: line, className: 'text-text-muted italic' }]
  }

  const tokens: Token[] = []
  let remaining = line

  while (remaining.length > 0) {
    let earliest: { index: number; length: number; className: string } | null = null

    for (const [pattern, className] of HIGHLIGHT_RULES) {
      pattern.lastIndex = 0
      const match = pattern.exec(remaining)
      if (match && (!earliest || match.index < earliest.index)) {
        earliest = { index: match.index, length: match[0].length, className }
      }
    }

    if (earliest && earliest.index >= 0) {
      if (earliest.index > 0) {
        tokens.push({ text: remaining.slice(0, earliest.index) })
      }
      tokens.push({ text: remaining.slice(earliest.index, earliest.index + earliest.length), className: earliest.className })
      remaining = remaining.slice(earliest.index + earliest.length)
    } else {
      tokens.push({ text: remaining })
      break
    }
  }

  return tokens.length > 0 ? tokens : [{ text: line }]
}
