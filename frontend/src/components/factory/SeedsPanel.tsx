import { useState, useMemo, useCallback } from 'react'
import {
  Search,
  Plus,
  FolderOpen,
  Layers,
  Trash2,
  Copy,
  ChevronUp,
  ChevronDown,
  CheckCircle,
} from 'lucide-react'
import { clsx } from 'clsx'
import type { FactoryConfig, SeedExample } from '../../services/api'
import { SeedEditor } from './SeedEditor'
import { SEED_CATEGORY_TAGS } from './types'
import type { SeedSourceFilter } from './types'

interface SeedsPanelProps {
  config: FactoryConfig
  onConfigChange: (config: FactoryConfig) => void
  onImportFromTraces: () => void
  onNavigateToCreate: () => void
}

export function SeedsPanel({ config, onConfigChange, onImportFromTraces, onNavigateToCreate }: SeedsPanelProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [sourceFilter, setSourceFilter] = useState<SeedSourceFilter>('all')
  const [tagFilter, setTagFilter] = useState<string | null>(null)
  const [selectedSeedId, setSelectedSeedId] = useState<string | null>(null)
  const [selectedSeedIds, setSelectedSeedIds] = useState<Set<string>>(new Set())

  // Source counts
  const sourceCounts = useMemo(() => {
    const counts = { all: config.seeds.length, gold_trace: 0, imported: 0, manual: 0 }
    for (const seed of config.seeds) {
      if (seed.source in counts) {
        counts[seed.source as keyof typeof counts]++
      }
    }
    return counts
  }, [config.seeds])

  // Active tags across all seeds
  const activeTags = useMemo(() => {
    const tags = new Set<string>()
    for (const seed of config.seeds) {
      for (const tag of seed.tags || []) {
        tags.add(tag)
      }
    }
    return Array.from(tags)
  }, [config.seeds])

  // Filtered seeds
  const filteredSeeds = useMemo(() => {
    return config.seeds.filter(seed => {
      // Source filter
      if (sourceFilter !== 'all' && seed.source !== sourceFilter) return false

      // Tag filter
      if (tagFilter && !(seed.tags || []).includes(tagFilter)) return false

      // Search query
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        const dataMatch = Object.values(seed.data).some(v => v.toLowerCase().includes(q))
        const tagMatch = (seed.tags || []).some(t => t.includes(q))
        if (!dataMatch && !tagMatch) return false
      }

      return true
    })
  }, [config.seeds, sourceFilter, tagFilter, searchQuery])

  const selectedSeed = config.seeds.find(s => s.id === selectedSeedId) || null

  // Seed helpers
  const addSeed = useCallback(() => {
    const newSeed: SeedExample = {
      id: `seed_${Date.now()}`,
      data: config.columns.reduce((acc, col) => ({ ...acc, [col.name]: '' }), {}),
      source: 'manual',
      created_at: new Date().toISOString(),
      tags: [],
    }
    onConfigChange({
      ...config,
      seeds: [...config.seeds, newSeed]
    })
    setSelectedSeedId(newSeed.id)
  }, [config, onConfigChange])

  const removeSeed = useCallback((id: string) => {
    onConfigChange({
      ...config,
      seeds: config.seeds.filter(s => s.id !== id)
    })
    if (selectedSeedId === id) setSelectedSeedId(null)
    setSelectedSeedIds(prev => {
      const next = new Set(prev)
      next.delete(id)
      return next
    })
  }, [config, onConfigChange, selectedSeedId])

  const updateSeed = useCallback((id: string, updates: Partial<SeedExample>) => {
    onConfigChange({
      ...config,
      seeds: config.seeds.map(s => s.id === id ? { ...s, ...updates } : s)
    })
  }, [config, onConfigChange])

  const moveSeed = useCallback((id: string, direction: 'up' | 'down') => {
    const idx = config.seeds.findIndex(s => s.id === id)
    if (idx < 0) return
    const targetIdx = direction === 'up' ? idx - 1 : idx + 1
    if (targetIdx < 0 || targetIdx >= config.seeds.length) return

    const newSeeds = [...config.seeds]
    ;[newSeeds[idx], newSeeds[targetIdx]] = [newSeeds[targetIdx], newSeeds[idx]]
    onConfigChange({ ...config, seeds: newSeeds })
  }, [config, onConfigChange])

  // Bulk operations
  const toggleSelect = useCallback((id: string) => {
    setSelectedSeedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const bulkDelete = useCallback(() => {
    onConfigChange({
      ...config,
      seeds: config.seeds.filter(s => !selectedSeedIds.has(s.id))
    })
    if (selectedSeedId && selectedSeedIds.has(selectedSeedId)) {
      setSelectedSeedId(null)
    }
    setSelectedSeedIds(new Set())
  }, [config, onConfigChange, selectedSeedIds, selectedSeedId])

  const bulkDuplicate = useCallback(() => {
    const dupes: SeedExample[] = []
    for (const id of selectedSeedIds) {
      const seed = config.seeds.find(s => s.id === id)
      if (seed) {
        dupes.push({
          ...seed,
          id: `seed_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
          source: 'manual',
          created_at: new Date().toISOString(),
        })
      }
    }
    onConfigChange({
      ...config,
      seeds: [...config.seeds, ...dupes]
    })
    setSelectedSeedIds(new Set())
  }, [config, onConfigChange, selectedSeedIds])

  // Quality helper
  const seedQuality = (seed: SeedExample) => {
    const total = config.columns.length
    if (total === 0) return 0
    const filled = config.columns.filter(col => {
      const val = seed.data[col.name]
      return val && val.trim().length > 0
    }).length
    return filled / total
  }

  const sourceFilterTabs: { id: SeedSourceFilter; label: string }[] = [
    { id: 'all', label: 'All' },
    { id: 'gold_trace', label: 'Gold' },
    { id: 'imported', label: 'Imported' },
    { id: 'manual', label: 'Manual' },
  ]

  return (
    <div className="h-full flex">
      {/* Left Panel - Seed List */}
      <div className="w-96 border-r border-border-subtle flex flex-col">
        {/* Info bar */}
        <div className="px-4 pt-4 pb-2">
          <div className="flex items-center justify-between text-xs text-text-muted">
            <span>
              <span className="font-medium text-text-primary">{config.seeds.length}</span> seeds
            </span>
            <span>Recommended: 20-100</span>
          </div>
        </div>

        {/* Search */}
        <div className="px-4 pb-3">
          <div className="flex items-center gap-2 px-3 py-2 bg-background-tertiary rounded-lg">
            <Search className="w-4 h-4 text-text-muted flex-shrink-0" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search seeds..."
              className="flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
            />
          </div>
        </div>

        {/* Source filter tabs */}
        <div className="px-4 pb-2 flex gap-1">
          {sourceFilterTabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setSourceFilter(tab.id)}
              className={clsx(
                'px-2.5 py-1 text-xs rounded-md transition-colors',
                sourceFilter === tab.id
                  ? 'bg-primary text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
              )}
            >
              {tab.label}
              {sourceCounts[tab.id] > 0 && (
                <span className="ml-1 opacity-70">{sourceCounts[tab.id]}</span>
              )}
            </button>
          ))}
        </div>

        {/* Category tag filter */}
        {activeTags.length > 0 && (
          <div className="px-4 pb-2 overflow-x-auto">
            <div className="flex gap-1">
              {tagFilter && (
                <button
                  onClick={() => setTagFilter(null)}
                  className="px-2 py-0.5 text-xs rounded-full bg-primary/15 text-primary flex-shrink-0"
                >
                  Clear
                </button>
              )}
              {activeTags.map(tag => (
                <button
                  key={tag}
                  onClick={() => setTagFilter(tagFilter === tag ? null : tag)}
                  className={clsx(
                    'px-2 py-0.5 text-xs rounded-full flex-shrink-0 transition-colors',
                    tagFilter === tag
                      ? 'bg-primary text-white'
                      : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
                  )}
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Bulk action bar */}
        {selectedSeedIds.size > 0 && (
          <div className="px-4 py-2 border-y border-border-subtle bg-background-tertiary/50 flex items-center gap-2">
            <span className="text-xs text-text-secondary">{selectedSeedIds.size} selected</span>
            <div className="flex-1" />
            <button
              onClick={bulkDuplicate}
              className="p-1.5 text-text-muted hover:text-primary transition-colors rounded hover:bg-background-tertiary"
              title="Duplicate selected"
            >
              <Copy className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={bulkDelete}
              className="p-1.5 text-text-muted hover:text-status-error transition-colors rounded hover:bg-background-tertiary"
              title="Delete selected"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Seed list */}
        <div className="flex-1 overflow-y-auto">
          {filteredSeeds.length === 0 ? (
            <div className="p-8 text-center">
              <Layers className="w-8 h-8 text-text-muted mx-auto mb-3" />
              {config.seeds.length === 0 ? (
                <>
                  <p className="text-sm font-medium text-text-primary mb-1">No seeds yet</p>
                  <p className="text-xs text-text-muted mb-4">Import from traces or create manually</p>
                  <button onClick={onImportFromTraces} className="btn-primary text-xs">
                    <FolderOpen className="w-3.5 h-3.5 mr-1" />
                    Import from Traces
                  </button>
                </>
              ) : (
                <>
                  <p className="text-sm font-medium text-text-primary mb-1">No matching seeds</p>
                  <p className="text-xs text-text-muted">Try adjusting your filters</p>
                </>
              )}
            </div>
          ) : (
            filteredSeeds.map((seed, filteredIdx) => {
              const globalIdx = config.seeds.findIndex(s => s.id === seed.id)
              const quality = seedQuality(seed)
              const isSelected = selectedSeedId === seed.id
              const isChecked = selectedSeedIds.has(seed.id)

              return (
                <div
                  key={seed.id}
                  className={clsx(
                    'group border-b border-border-subtle transition-colors cursor-pointer',
                    isSelected ? 'bg-primary/5' : 'hover:bg-background-tertiary/50'
                  )}
                >
                  <div className="flex items-start gap-2 p-3">
                    {/* Checkbox */}
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggleSelect(seed.id)}
                      onClick={(e) => e.stopPropagation()}
                      className="mt-1 rounded flex-shrink-0"
                    />

                    {/* Content */}
                    <div
                      className="flex-1 min-w-0"
                      onClick={() => setSelectedSeedId(seed.id)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-medium text-text-primary">#{globalIdx + 1}</span>
                        <span className={clsx(
                          'text-[10px] px-1.5 py-0.5 rounded-full',
                          seed.source === 'gold_trace' ? 'bg-status-success/20 text-status-success' :
                          seed.source === 'imported' ? 'bg-status-info/20 text-status-info' :
                          'bg-background-tertiary text-text-muted'
                        )}>
                          {seed.source === 'gold_trace' ? 'Gold' : seed.source === 'imported' ? 'Import' : 'Manual'}
                        </span>
                        {(seed.tags || []).slice(0, 2).map(tag => (
                          <span key={tag} className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                            {tag}
                          </span>
                        ))}
                        {(seed.tags || []).length > 2 && (
                          <span className="text-[10px] text-text-muted">+{(seed.tags || []).length - 2}</span>
                        )}
                      </div>
                      <p className="text-xs text-text-secondary truncate">
                        {Object.values(seed.data).filter(Boolean).join(' \u00b7 ').slice(0, 100) || 'Empty seed'}
                      </p>
                      {/* Quality bar */}
                      {config.columns.length > 0 && (
                        <div className="mt-1.5 flex items-center gap-2">
                          <div className="flex-1 h-1 bg-background-tertiary rounded-full overflow-hidden">
                            <div
                              className={clsx(
                                'h-full rounded-full',
                                quality === 1 ? 'bg-status-success' :
                                quality >= 0.5 ? 'bg-status-warning' :
                                'bg-status-error'
                              )}
                              style={{ width: `${quality * 100}%` }}
                            />
                          </div>
                          {quality === 1 && <CheckCircle className="w-3 h-3 text-status-success flex-shrink-0" />}
                        </div>
                      )}
                    </div>

                    {/* Reorder buttons */}
                    <div className="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                      <button
                        onClick={(e) => { e.stopPropagation(); moveSeed(seed.id, 'up') }}
                        disabled={globalIdx === 0}
                        className="p-0.5 text-text-muted hover:text-text-primary disabled:opacity-30 transition-colors"
                      >
                        <ChevronUp className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); moveSeed(seed.id, 'down') }}
                        disabled={globalIdx === config.seeds.length - 1}
                        className="p-0.5 text-text-muted hover:text-text-primary disabled:opacity-30 transition-colors"
                      >
                        <ChevronDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Bottom action bar */}
        <div className="p-3 border-t border-border-subtle flex gap-2">
          <button
            onClick={onImportFromTraces}
            className="btn-primary text-xs flex-1 flex items-center justify-center gap-1.5"
            data-tutorial="import-button"
          >
            <FolderOpen className="w-3.5 h-3.5" />
            Import from Traces
          </button>
          <button
            onClick={addSeed}
            className="btn-secondary text-xs flex-1 flex items-center justify-center gap-1.5"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Manual
          </button>
        </div>
      </div>

      {/* Right Panel - Seed Editor */}
      <div className="flex-1 overflow-hidden">
        {selectedSeed ? (
          <SeedEditor
            seed={selectedSeed}
            seedIndex={config.seeds.findIndex(s => s.id === selectedSeed.id)}
            columns={config.columns}
            onUpdate={updateSeed}
            onDelete={removeSeed}
            onNavigateToCreate={onNavigateToCreate}
          />
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <Layers className="w-10 h-10 text-text-muted mx-auto mb-3" />
              <p className="text-sm font-medium text-text-primary mb-1">Select a seed to edit</p>
              <p className="text-xs text-text-muted">
                {config.seeds.length === 0
                  ? 'Import from traces or add a manual seed to get started'
                  : 'Choose a seed from the list on the left'}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
