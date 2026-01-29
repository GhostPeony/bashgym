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
      <div className="p-4 border-b border-border-subtle flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-medium text-text-primary">Seed #{seedIndex + 1}</span>
          <span className={clsx(
            'text-xs px-2 py-0.5 rounded-full',
            seed.source === 'gold_trace' ? 'bg-status-success/20 text-status-success' :
            seed.source === 'imported' ? 'bg-status-info/20 text-status-info' :
            'bg-background-tertiary text-text-muted'
          )}>
            {seed.source === 'gold_trace' ? 'Gold Trace' : seed.source === 'imported' ? 'Imported' : 'Manual'}
          </span>
        </div>
        <button
          onClick={() => onDelete(seed.id)}
          className="p-1.5 text-text-muted hover:text-status-error transition-colors rounded hover:bg-background-tertiary"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Tags */}
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1.5">Tags</label>
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
                  <label className="block text-xs font-medium text-text-secondary mb-1.5">
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
                    <div className="mt-2 p-3 bg-background-tertiary rounded-lg overflow-x-auto">
                      <pre className="text-xs font-mono text-text-primary whitespace-pre-wrap">
                        {highlightCode(value)}
                      </pre>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="text-sm text-text-muted py-8 text-center">
            No columns defined.{' '}
            <button onClick={onNavigateToCreate} className="text-primary hover:underline">
              Configure schema
            </button>{' '}
            in Advanced Configuration first.
          </div>
        )}

        {/* Quality assessment */}
        {columns.length > 0 && (
          <div className="card-elevated p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-text-secondary">Completeness</span>
              <span className="text-xs text-text-muted">
                {quality.filled}/{quality.total} fields
              </span>
            </div>
            <div className="h-1.5 bg-background-tertiary rounded-full overflow-hidden">
              <div
                className={clsx(
                  'h-full rounded-full transition-all',
                  quality.percent === 100 ? 'bg-status-success' :
                  quality.percent >= 50 ? 'bg-status-warning' :
                  'bg-status-error'
                )}
                style={{ width: `${quality.percent}%` }}
              />
            </div>
            <div className="flex items-center gap-1.5 mt-2">
              {quality.percent === 100 ? (
                <>
                  <CheckCircle className="w-3.5 h-3.5 text-status-success" />
                  <span className="text-xs text-status-success">All fields populated</span>
                </>
              ) : (
                <>
                  <AlertCircle className="w-3.5 h-3.5 text-text-muted" />
                  <span className="text-xs text-text-muted">
                    {quality.total - quality.filled} empty field{quality.total - quality.filled !== 1 ? 's' : ''}
                  </span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Metadata */}
        {seed.created_at && (
          <div className="text-xs text-text-muted pt-2 border-t border-border-subtle">
            Created: {new Date(seed.created_at).toLocaleDateString()}
            {seed.trace_id && <span className="ml-3">Trace: {seed.trace_id.slice(0, 8)}...</span>}
          </div>
        )}
      </div>
    </div>
  )
}
