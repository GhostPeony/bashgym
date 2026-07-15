export interface StreamGeometry {
  x: number
  y: number
  width: number
  height: number
}

export interface StreamViewport {
  width: number
  height: number
}

const BOX_ONLY_PATTERN = /^[\s┌┐└┘╭╮╰╯├┤┬┴┼│┃─━═╪╫╬╌╍╎╏┄┅┆┇┈┉┊┋█▓▒░▄▀▌▐▛▜▝▘■▪●‼]+$/

export function normalizeTerminalFeed(lines: string[]): string[] {
  const candidates: string[] = []
  for (const rawLine of lines) {
    // TerminalPane has already stripped ANSI control sequences before these
    // retained lines reach the presentation layer.
    const normalizedLine = rawLine.replace(/\r/g, '')
    if (BOX_ONLY_PATTERN.test(normalizedLine)) continue
    const softened = normalizedLine
      .replace(/[┌┐└┘╭╮╰╯├┤┬┴┼│┃]/g, ' ')
      .replace(/[─━═]{6,}/g, ' ')
      .replace(/^[ \t]*(?:[█▓▒░▄▀▌▐▛▜▝▘■▪●‼][ \t]*){2,}/g, '')
      .replace(/^[ \t]*[█▓▒░▄▀▌▐▛▜▝▘■▪●‼]{1,}[ \t]*/g, '')
      .replace(/^[ \t]*!{1,3}[ \t]*/g, '')
      .replace(/[ \t]+$/g, '')
    const meaningful = softened.trim()
    if (!meaningful) continue
    if (/hift\+tab to cycle|for agents/i.test(meaningful)) continue
    const clipped = softened.length > 320 ? `${softened.slice(0, 319)}…` : softened
    candidates.push(clipped)
  }

  // PTYs redraw the same status/banner block many times. Preserve the newest
  // copy of each line so the overlay reads as a feed, not terminal replay.
  const newestIndex = new Map<string, number>()
  candidates.forEach((line, index) => newestIndex.set(line.trim(), index))
  return candidates.filter((line, index) => newestIndex.get(line.trim()) === index)
}

export function clampStreamGeometry(
  geometry: StreamGeometry,
  viewport: StreamViewport
): StreamGeometry {
  const minimumWidth = 300
  const minimumHeight = 260
  const maximumWidth = Math.max(minimumWidth, viewport.width - 16)
  const maximumHeight = Math.max(minimumHeight, viewport.height - 64)
  const width = Math.min(Math.max(geometry.width, minimumWidth), maximumWidth)
  const height = Math.min(Math.max(geometry.height, minimumHeight), maximumHeight)
  return {
    width,
    height,
    x: Math.min(Math.max(geometry.x, 8), Math.max(8, viewport.width - width - 8)),
    y: Math.min(Math.max(geometry.y, 56), Math.max(56, viewport.height - height - 8))
  }
}
