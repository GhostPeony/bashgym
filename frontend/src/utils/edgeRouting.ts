/**
 * Shared edge routing utility.
 *
 * Routes content (text or binary data URLs) from a source panel to all
 * terminal panels connected via canvas edges. Used by BrowserPane for
 * screenshots and by integration nodes (Context, Neon, Vercel) for
 * piping query results into terminals.
 */

import { useTerminalStore } from '../stores'

export interface RouteResult {
  /** Number of terminals the content was routed to */
  routed: number
  /** Error message if something went wrong before routing */
  error?: string
}

/**
 * Route text content to all terminals linked to `sourcePanelId`.
 *
 * Converts the text into a base64 data URL, writes it to a temp file via
 * Electron IPC, then prefills the file path into every connected terminal.
 */
export async function routeToLinkedTerminals(
  sourcePanelId: string,
  content: string,
  fileExtension = 'md'
): Promise<RouteResult> {
  // Encode text content as a base64 data URL so writeTempFile can handle it
  const base64 = btoa(unescape(encodeURIComponent(content)))
  const dataUrl = `data:text/plain;base64,${base64}`

  return routeBinaryToLinkedTerminals(sourcePanelId, dataUrl, fileExtension)
}

/**
 * Route a binary data URL (e.g. a screenshot) to all terminals linked to
 * `sourcePanelId`.
 *
 * Writes the data URL to a temp file via Electron IPC, then prefills the
 * resulting file path into every connected terminal's input.
 */
export async function routeBinaryToLinkedTerminals(
  sourcePanelId: string,
  dataUrl: string,
  fileExtension = 'png'
): Promise<RouteResult> {
  const result = await window.bashgym?.files.writeTempFile(dataUrl, fileExtension)
  if (!result?.success || !result.path) {
    return { routed: 0, error: result?.error ?? 'Failed to write temp file' }
  }

  const { canvasEdges, panels } = useTerminalStore.getState()
  const filePath = result.path

  const connectedEdges = canvasEdges.filter(
    e => e.source === sourcePanelId || e.target === sourcePanelId
  )

  let routed = 0
  for (const edge of connectedEdges) {
    const targetPanelId = edge.source === sourcePanelId ? edge.target : edge.source
    const targetPanel = panels.find(p => p.id === targetPanelId && p.type === 'terminal')
    if (targetPanel?.terminalId) {
      // Prefill the terminal input without submitting (no \r) so the user can review
      window.bashgym?.terminal.write(targetPanel.terminalId, filePath)
      routed++
    }
  }

  return { routed }
}
