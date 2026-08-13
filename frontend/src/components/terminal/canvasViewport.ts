import type { Viewport } from '@xyflow/react'

export interface ViewportStorage {
  getItem: (key: string) => string | null
  setItem: (key: string, value: string) => void
}

export function loadCanvasViewport(storage: ViewportStorage, key: string): Viewport | null {
  try {
    const stored = storage.getItem(key)
    if (!stored) return null
    const parsed = JSON.parse(stored) as Partial<Viewport>
    if (
      typeof parsed.x !== 'number' ||
      typeof parsed.y !== 'number' ||
      typeof parsed.zoom !== 'number' ||
      !Number.isFinite(parsed.x) ||
      !Number.isFinite(parsed.y) ||
      !Number.isFinite(parsed.zoom) ||
      parsed.zoom <= 0
    )
      return null
    return { x: parsed.x, y: parsed.y, zoom: parsed.zoom }
  } catch {
    return null
  }
}

export function saveCanvasViewport(
  storage: ViewportStorage,
  key: string,
  viewport: Viewport
): void {
  try {
    storage.setItem(key, JSON.stringify(viewport))
  } catch {
    // Ignore unavailable or full storage.
  }
}
