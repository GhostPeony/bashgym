import { useState } from 'react'
import { ChevronRight, ChevronDown, Eye, Link2, FileText } from 'lucide-react'
import type { CanvasEdge, Panel } from '../../stores'

interface ConnectionsTreeProps {
  panelId: string
  panels: Panel[]
  canvasEdges: CanvasEdge[]
  recentFiles: string[]
}

function basename(p: string): string {
  return p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? p
}

/**
 * How this terminal is wired into the canvas (monitor edges, data-node links)
 * plus the files its agent touched most recently.
 */
export function ConnectionsTree({ panelId, panels, canvasEdges, recentFiles }: ConnectionsTreeProps) {
  const [open, setOpen] = useState(false)

  const links: Array<{ key: string; icon: 'eye' | 'link'; label: string }> = []
  for (const edge of canvasEdges) {
    if (edge.source !== panelId && edge.target !== panelId) continue
    const otherId = edge.source === panelId ? edge.target : edge.source
    const other = panels.find((p) => p.id === otherId)
    if (!other) continue
    if (edge.type === 'monitor') {
      const autoTag = edge.data?.auto && edge.data.auto !== 'off' ? ` · auto:${edge.data.auto}` : ''
      links.push({
        key: edge.id,
        icon: 'eye',
        label: edge.source === panelId ? `watched by ${other.title}${autoTag}` : `watching ${other.title}${autoTag}`
      })
    } else {
      links.push({ key: edge.id, icon: 'link', label: `${other.type}: ${other.title}` })
    }
  }

  const count = links.length + recentFiles.length
  if (count === 0) return null

  return (
    <div>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v) }}
        className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-text-muted hover:text-text-primary transition-press"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        Connections · {links.length} link{links.length !== 1 ? 's' : ''}
        {recentFiles.length > 0 && ` · ${recentFiles.length} file${recentFiles.length !== 1 ? 's' : ''}`}
      </button>
      {open && (
        <div className="mt-1 ml-2 pl-2 border-l border-brutal border-border-subtle space-y-0.5">
          {links.map((link) => (
            <div key={link.key} className="flex items-center gap-1.5 font-mono text-[10px] text-text-secondary">
              {link.icon === 'eye'
                ? <Eye className="w-2.5 h-2.5 text-[hsl(270_45%_60%)] flex-shrink-0" />
                : <Link2 className="w-2.5 h-2.5 text-accent flex-shrink-0" />}
              <span className="truncate">{link.label}</span>
            </div>
          ))}
          {recentFiles.slice(-5).reverse().map((file) => (
            <div key={file} className="flex items-center gap-1.5 font-mono text-[10px] text-text-muted" title={file}>
              <FileText className="w-2.5 h-2.5 flex-shrink-0" />
              <span className="truncate">{basename(file)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
