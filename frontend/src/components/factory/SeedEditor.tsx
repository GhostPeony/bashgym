import { useMemo } from 'react'
import { Trash2, CheckCircle, AlertCircle } from 'lucide-react'
import { clsx } from 'clsx'
import type { SeedExample, ColumnConfig } from '../../services/api'
import { TagEditor } from './TagEditor'
import { looksLikeCode, highlightCode } from './codeHighlight'

interface SeedEditorProps {
  seed: SeedExample
  seedIndex: number
  columns: ColumnConfig[]
  onUpdate: (id: string, updates: Partial<SeedExample>) => void
  onDelete: (id: string) => void
  onNavigateToCreate: () => void
}

export function SeedEditor({ seed, seedIndex, columns, onUpdate, onDelete, onNavigateToCreate }: SeedEditorProps) {
  const llmColumns = columns.filter(c => c.type === 'llm')
  const showSideBySide = llmColumns.length === 2 && columns.length === llmColumns.length

  const quality = useMemo(() => {
    const total = columns.length
    if (total === 0) return { filled: 0, total: 0, percent: 0 }
    const filled = columns.filter(col => {
      const val = seed.data[col.name]
      return val && val.trim().length > 0
    }).length
    return { filled, total, percent: Math.round((filled / total) * 100) }
  }, [seed.data, columns])

  const handleDataChange = (colName: string, value: string) => {
    onUpdate(seed.id, { data: { ...seed.data, [colName]: value } })
  }

  const handleTagsChange = (tags: string[]) => {
    onUpdate(seed.id, { tags })
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b-2 border-border flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="font-brand text-lg text-text-primary">Seed #{seedIndex + 1}</span>
          <span className={clsx(
            'tag',
            seed.source === 'gold_trace' ? 'text-status-success' :
            seed.source === 'imported' ? 'text-status-info' :
            'text-text-muted'
          )}>
            <span>
              {seed.source === 'gold_trace' ? 'Gold Trace' : seed.source === 'imported' ? 'Imported' : 'Manual'}
            </span>
          </span>
        </div>
        <button
          onClick={() => onDelete(seed.id)}
          className="btn-icon w-8 h-8 flex items-center justify-center text-text-muted hover:text-status-error"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Tags */}
        <div>
          <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1.5">Tags</label>
          <TagEditor tags={seed.tags || []} onChange={handleTagsChange} />
        </div>

        {/* Column editors */}
        {columns.length > 0 ? (
          <div className={clsx(
            showSideBySide ? 'grid grid-cols-2 gap-4' : 'space-y-4'
          )}>
            {columns.map(col => {
              const value = seed.data[col.name] || ''
              const isCode = looksLikeCode(value)

              return (
                <div key={col.id}>
                  <label className="block font-mono text-xs uppercase tracking-widest text-text-secondary mb-1.5">
                    {col.name}
                    <span className="text-text-muted ml-1.5">({col.type})</span>
                  </label>
                  <textarea
                    value={value}
                    onChange={(e) => handleDataChange(col.name, e.target.value)}
                    rows={6}
                    className="input text-sm w-full font-mono resize-y"
                    placeholder={`Enter ${col.name}...`}
                  />
                  {isCode && value.length > 0 && (
                    <div className="terminal-chrome mt-2">
                      <div className="terminal-header">
                        <div className="terminal-dot terminal-dot-red" />
                        <div className="terminal-dot terminal-dot-yellow" />
                        <div className="terminal-dot terminal-dot-green" />
                        <span className="font-mono text-xs text-text-muted ml-2">preview</span>
                      </div>
                      <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap p-3">
                        {highlightCode(value)}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-sm text-text-muted py-8 text-center font-mono">
            No columns defined.{' '}
            <button onClick={onNavigateToCreate} className="text-accent-dark hover:underline">
              Configure schema
            </button>{' '}
            in Advanced Configuration first.
          </div>
        )}

        {/* Quality assessment */}
        {columns.length > 0 && (
          <div className="card p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-xs uppercase tracking-widest text-text-secondary">Completeness</span>
              <span className="text-xs text-text-muted font-mono">
                {quality.filled}/{quality.total} fields
              </span>
            </div>
            <div className="progress-bar">
              <div
                className={clsx(
                  'progress-fill',
                  quality.percent === 100 ? '!bg-status-success' :
                  quality.percent >= 50 ? '!bg-status-warning' :
                  '!bg-status-error'
                )}
                style={{ width: `${quality.percent}%` }}
              />
            </div>
            <div className="flex items-center gap-1.5 mt-2">
              {quality.percent === 100 ? (
                <>
                  <CheckCircle className="w-3.5 h-3.5 text-status-success" />
                  <span className="text-xs text-status-success font-mono">All fields populated</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-text-muted" />
                  <span className="text-xs text-text-muted font-mono">
                    {quality.total - quality.filled} empty field{quality.total - quality.filled !== 1 ? 's' : ''}
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Metadata */}
        {seed.created_at && (
          <div className="text-xs text-text-muted pt-2 border-t-2 border-border font-mono">
            Created: {new Date(seed.created_at).toLocaleDateString()}
            {seed.trace_id && <span className="ml-3">Trace: {seed.trace_id.slice(0, 8)}...</span>}
          </div>
        )}
      </div>
    </div>
  )
}
