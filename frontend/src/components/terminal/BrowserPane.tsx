import { useState, useEffect, useRef, useCallback } from 'react'
import { Globe, X, RefreshCw, ArrowLeft, ArrowRight, Camera, Crosshair } from 'lucide-react'
import { useTerminalStore } from '../../stores'

interface WebviewElement extends HTMLElement {
  src: string
  reload: () => void
  goBack: () => void
  goForward: () => void
  getWebContentsId: () => number
  executeJavaScript: (code: string) => Promise<any>
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      webview: React.HTMLAttributes<HTMLElement> & { src?: string }
    }
  }
}

// Injected into the webview to highlight elements on hover and capture on click
const PICKER_SCRIPT = `(function() {
  if (window.__bgymPickerActive) return;
  window.__bgymPickerActive = true;
  window.__bgymSelectedBounds = null;
  var s = document.createElement('style');
  s.id = '__bgym_style';
  s.textContent = '.__bgym_hover{outline:2px solid #E8868B!important;outline-offset:1px!important;cursor:crosshair!important;}';
  document.head.appendChild(s);
  var last = null;
  function onOver(e) {
    if (last) last.classList.remove('__bgym_hover');
    last = e.target; last.classList.add('__bgym_hover');
    e.stopPropagation();
  }
  function onClick(e) {
    e.preventDefault(); e.stopPropagation();
    var r = e.target.getBoundingClientRect();
    window.__bgymSelectedBounds = {
      x: Math.round(r.left + window.scrollX),
      y: Math.round(r.top + window.scrollY),
      width: Math.round(r.width), height: Math.round(r.height),
      vpW: window.innerWidth, vpH: window.innerHeight
    };
    cleanup();
  }
  function cleanup() {
    document.removeEventListener('mouseover', onOver, true);
    document.removeEventListener('click', onClick, true);
    if (last) last.classList.remove('__bgym_hover');
    var el = document.getElementById('__bgym_style'); if (el) el.remove();
    window.__bgymPickerActive = false;
  }
  document.addEventListener('mouseover', onOver, true);
  document.addEventListener('click', onClick, true);
})();`

async function copyDataUrlToClipboard(dataUrl: string) {
  const blob = await fetch(dataUrl).then(r => r.blob())
  await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })])
}

interface BrowserPaneProps {
  id: string
  title: string
  url?: string
  isActive: boolean
}

export function BrowserPane({ id, title, url: initialUrl, isActive }: BrowserPaneProps) {
  const { removePanel, setPanelUrl, setPanelThumbnail } = useTerminalStore()
  const [url, setUrl] = useState(initialUrl || 'http://localhost:3000')
  const [inputUrl, setInputUrl] = useState(url)
  const [pickerMode, setPickerMode] = useState(false)
  const [actionStatus, setActionStatus] = useState<'idle' | 'working' | 'done'>('idle')
  const webviewRef = useRef<WebviewElement>(null)
  const pickerPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Screenshot via IPC — uses main process webContents.capturePage() which is reliable
  const captureScreenshot = useCallback(async (rect?: { x: number; y: number; width: number; height: number; vpW: number; vpH: number }) => {
    const webview = webviewRef.current
    if (!webview) return null
    try {
      const webContentsId = webview.getWebContentsId()
      const result = await window.bashgym?.browser?.screenshot(webContentsId, rect)
      return result?.success ? result.dataUrl ?? null : null
    } catch {
      return null
    }
  }, [])

  // Store thumbnail after every page load (covers initial load + URL navigation)
  const updateThumbnail = useCallback(async () => {
    const dataUrl = await captureScreenshot()
    if (dataUrl) setPanelThumbnail(id, dataUrl)
  }, [id, captureScreenshot, setPanelThumbnail])

  // Route a screenshot to all terminals connected to this browser node via canvas edges
  const routeToConnectedTerminals = useCallback(async (dataUrl: string) => {
    const result = await window.bashgym?.files.writeTempFile(dataUrl, 'png')
    if (!result?.success || !result.path) return

    const { canvasEdges, panels } = useTerminalStore.getState()
    const filePath = result.path

    const connectedEdges = canvasEdges.filter(e => e.source === id || e.target === id)
    for (const edge of connectedEdges) {
      const targetPanelId = edge.source === id ? edge.target : edge.source
      const targetPanel = panels.find(p => p.id === targetPanelId && p.type === 'terminal')
      if (targetPanel?.terminalId) {
        // Prefill the terminal's input without submitting (no \r) so the user can review before sending
        window.bashgym?.terminal.write(targetPanel.terminalId, filePath)
      }
    }
  }, [id])

  useEffect(() => {
    const webview = webviewRef.current
    if (!webview) return
    webview.addEventListener('did-finish-load', updateThumbnail)
    return () => webview.removeEventListener('did-finish-load', updateThumbnail)
  }, [updateThumbnail])

  // Ctrl+R forwarded from main
  useEffect(() => {
    if (!isActive) return
    const cleanup = window.bashgym?.window.onAppKeydown?.((data) => {
      if (data.ctrlKey && (data.key === 'r' || data.key === 'R')) {
        webviewRef.current?.reload()
      }
    })
    return () => cleanup?.()
  }, [isActive])

  // Cancel picker on Escape
  useEffect(() => {
    if (!pickerMode) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') cancelPicker()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pickerMode])

  // Clean up poll on unmount
  useEffect(() => () => { if (pickerPollRef.current) clearInterval(pickerPollRef.current) }, [])

  const cancelPicker = useCallback(() => {
    if (pickerPollRef.current) { clearInterval(pickerPollRef.current); pickerPollRef.current = null }
    setPickerMode(false)
    webviewRef.current?.executeJavaScript('window.__bgymPickerActive && (window.__bgymPickerActive = false);').catch(() => {})
  }, [])

  const handleClose = (e: React.MouseEvent) => { e.stopPropagation(); removePanel(id) }

  const handleNavigate = () => {
    let newUrl = inputUrl
    if (!newUrl.startsWith('http://') && !newUrl.startsWith('https://')) newUrl = 'http://' + newUrl
    setUrl(newUrl)
    setPanelUrl(id, newUrl)
    // did-finish-load fires after navigation → thumbnail auto-updates
  }

  const handleKeyDown = (e: React.KeyboardEvent) => { if (e.key === 'Enter') handleNavigate() }

  // Camera: full screenshot → clipboard + canvas thumbnail + route to connected terminals
  const handleScreenshot = useCallback(async () => {
    if (actionStatus === 'working') return
    setActionStatus('working')
    try {
      const dataUrl = await captureScreenshot()
      if (!dataUrl) throw new Error('No screenshot')
      setPanelThumbnail(id, dataUrl)
      await copyDataUrlToClipboard(dataUrl)
      await routeToConnectedTerminals(dataUrl)
      setActionStatus('done')
      setTimeout(() => setActionStatus('idle'), 2000)
    } catch {
      setActionStatus('idle')
    }
  }, [id, captureScreenshot, setPanelThumbnail, routeToConnectedTerminals, actionStatus])

  // Crosshair: inject picker, poll for click, capture element region → clipboard + terminal
  const handleStartPicker = useCallback(async () => {
    const webview = webviewRef.current
    if (!webview || pickerMode) return
    setPickerMode(true)
    try {
      await webview.executeJavaScript('window.__bgymSelectedBounds = null;')
      await webview.executeJavaScript(PICKER_SCRIPT)

      pickerPollRef.current = setInterval(async () => {
        try {
          const bounds = await webview.executeJavaScript('window.__bgymSelectedBounds')
          if (!bounds) return
          clearInterval(pickerPollRef.current!); pickerPollRef.current = null
          setPickerMode(false)

          const dataUrl = await captureScreenshot(bounds)
          if (dataUrl) {
            await copyDataUrlToClipboard(dataUrl)
            await routeToConnectedTerminals(dataUrl)
          }
        } catch { /* ignore poll errors */ }
      }, 200)

      // Auto-cancel after 30s
      setTimeout(() => { if (pickerPollRef.current) cancelPicker() }, 30000)
    } catch {
      setPickerMode(false)
    }
  }, [pickerMode, captureScreenshot, cancelPicker, routeToConnectedTerminals])

  return (
    <div className="terminal-chrome h-full flex flex-col">
      <div className="terminal-header">
        <div className="flex items-center gap-1">
          <button onClick={() => webviewRef.current?.goBack()} className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary">
            <ArrowLeft className="w-3.5 h-3.5 text-text-muted" />
          </button>
          <button onClick={() => webviewRef.current?.goForward()} className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary">
            <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
          </button>
          <button onClick={() => webviewRef.current?.reload()} className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-background-tertiary">
            <RefreshCw className="w-3.5 h-3.5 text-text-muted" />
          </button>
        </div>

        <div className="flex-1 flex items-center gap-2 mx-2">
          <div className="flex-1 flex items-center gap-2 px-3 py-1 border-brutal border-border rounded-brutal bg-background-secondary">
            <Globe className="w-3.5 h-3.5 text-text-muted flex-shrink-0" />
            <input
              type="text"
              value={inputUrl}
              onChange={(e) => setInputUrl(e.target.value)}
              onKeyDown={handleKeyDown}
              className="flex-1 bg-transparent text-sm font-mono text-text-primary placeholder:text-text-muted focus:outline-none"
              placeholder="Enter URL..."
            />
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleStartPicker}
            className={`btn-icon !w-6 !h-6 !border-0 !shadow-none ${pickerMode ? 'bg-accent/20' : 'hover:bg-accent/20'}`}
            title={pickerMode ? 'Click an element to capture it' : 'Pick element → clipboard'}
          >
            <Crosshair className={`w-3.5 h-3.5 ${pickerMode ? 'text-accent animate-pulse' : 'text-accent'}`} />
          </button>
          <button
            onClick={handleScreenshot}
            disabled={actionStatus === 'working'}
            className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-accent/20"
            title="Screenshot → clipboard + Claude"
          >
            <Camera className={`w-3.5 h-3.5 ${
              actionStatus === 'working' ? 'text-text-muted animate-pulse' :
              actionStatus === 'done' ? 'text-status-success' : 'text-accent'
            }`} />
          </button>
          <button onClick={handleClose} className="btn-icon !w-6 !h-6 !border-0 !shadow-none hover:bg-status-error/20" title="Close">
            <X className="w-3.5 h-3.5 text-text-muted hover:text-status-error" />
          </button>
        </div>
      </div>

      {pickerMode && (
        <div className="px-3 py-1.5 bg-accent/10 border-b border-brutal border-accent text-[10px] font-mono text-accent text-center">
          Hover to highlight · Click to capture element · Esc to cancel
        </div>
      )}

      <div className="flex-1">
        <webview
          ref={webviewRef as any}
          src={url}
          style={{ width: '100%', height: '100%', border: 'none' }}
        />
      </div>
    </div>
  )
}
