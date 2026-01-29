import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  GitBranch,
  ChevronRight,
  ChevronDown,
  Star,
  Loader2,
  Database,
  Cpu,
  GraduationCap,
  ArrowRight,
  ExternalLink
} from 'lucide-react'
import { clsx } from 'clsx'
import { modelsApi, ModelSummary } from '../../services/api'

interface LineageTreeProps {
  onSelectModel: (modelId: string) => void
  highlightModelId?: string
}

interface TreeNode {
  model_id: string
  display_name: string
  base_model: string
  parent_model: string | null
  training_strategy: string
  created_at: string
  starred: boolean
  custom_eval_pass_rate: number | null
  benchmark_avg_score: number | null
  children: TreeNode[]
  depth: number
}

function buildLineageTree(models: ModelSummary[]): TreeNode[] {
  // Create a map of model_id -> TreeNode
  const nodeMap = new Map<string, TreeNode>()

  // First pass: create all nodes
  models.forEach(model => {
    nodeMap.set(model.model_id, {
      model_id: model.model_id,
      display_name: model.display_name,
      base_model: model.base_model,
      parent_model: null, // Will be populated if available
      training_strategy: model.training_strategy,
      created_at: model.created_at,
      starred: model.starred,
      custom_eval_pass_rate: model.custom_eval_pass_rate,
      benchmark_avg_score: model.benchmark_avg_score,
      children: [],
      depth: 0
    })
  })

  // Group by base model
  const baseModelGroups = new Map<string, TreeNode[]>()
  nodeMap.forEach(node => {
    const base = node.base_model
    if (!baseModelGroups.has(base)) {
      baseModelGroups.set(base, [])
    }
    baseModelGroups.get(base)!.push(node)
  })

  // Create root nodes for each base model
  const roots: TreeNode[] = []
  baseModelGroups.forEach((children, baseModel) => {
    // Sort children by date
    children.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())

    // Set depth for all children
    children.forEach((child, i) => {
      child.depth = 1
    })

    // Create a virtual root for the base model
    roots.push({
      model_id: `base:${baseModel}`,
      display_name: baseModel.split('/').pop() || baseModel,
      base_model: baseModel,
      parent_model: null,
      training_strategy: 'base',
      created_at: children[0]?.created_at || new Date().toISOString(),
      starred: false,
      custom_eval_pass_rate: null,
      benchmark_avg_score: null,
      children: children,
      depth: 0
    })
  })

  // Sort roots by number of children (most popular first)
  roots.sort((a, b) => b.children.length - a.children.length)

  return roots
}

interface TreeNodeRowProps {
  node: TreeNode
  expanded: boolean
  onToggle: () => void
  onSelect: (modelId: string) => void
  isHighlighted: boolean
  isBase: boolean
}

function TreeNodeRow({ node, expanded, onToggle, onSelect, isHighlighted, isBase }: TreeNodeRowProps) {
  const hasChildren = node.children.length > 0
  const indent = node.depth * 24

  const strategyColors: Record<string, string> = {
    base: 'bg-gray-500/20 text-gray-500',
    sft: 'bg-primary/20 text-primary',
    dpo: 'bg-purple-500/20 text-purple-500',
    grpo: 'bg-orange-500/20 text-orange-500',
    distillation: 'bg-teal-500/20 text-teal-500',
  }

  return (
    <div
      className={clsx(
        'flex items-center gap-2 px-3 py-2 rounded-lg transition-colors',
        isHighlighted ? 'bg-primary/10 ring-1 ring-primary' : 'hover:bg-background-tertiary',
        !isBase && 'cursor-pointer'
      )}
      style={{ marginLeft: indent }}
      onClick={() => !isBase && onSelect(node.model_id)}
    >
      {/* Expand/Collapse button */}
      <button
        onClick={(e) => {
          e.stopPropagation()
          onToggle()
        }}
        className={clsx(
          'w-5 h-5 flex items-center justify-center rounded',
          hasChildren ? 'hover:bg-background-secondary' : 'invisible'
        )}
      >
        {hasChildren && (expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />)}
      </button>

      {/* Icon */}
      {isBase ? (
        <Database className="w-4 h-4 text-text-muted" />
      ) : (
        <Cpu className="w-4 h-4 text-primary" />
      )}

      {/* Name */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={clsx(
            'font-medium truncate',
            isBase ? 'text-text-muted' : 'text-text-primary'
          )}>
            {node.display_name}
          </span>
          {node.starred && <Star className="w-3.5 h-3.5 text-yellow-500 fill-yellow-500" />}
        </div>
        {!isBase && (
          <div className="text-xs text-text-muted">
            {new Date(node.created_at).toLocaleDateString()}
          </div>
        )}
      </div>

      {/* Strategy badge */}
      <span className={clsx(
        'px-2 py-0.5 rounded text-xs font-medium',
        strategyColors[node.training_strategy.toLowerCase()] || strategyColors.sft
      )}>
        {node.training_strategy.toUpperCase()}
      </span>

      {/* Metrics */}
      {!isBase && (
        <div className="flex items-center gap-4 text-sm">
          {node.custom_eval_pass_rate !== null && (
            <div className="text-right">
              <div className="text-text-primary font-medium">{node.custom_eval_pass_rate.toFixed(1)}%</div>
              <div className="text-xs text-text-muted">Custom</div>
            </div>
          )}
          {node.benchmark_avg_score !== null && (
            <div className="text-right">
              <div className="text-text-primary font-medium">{node.benchmark_avg_score.toFixed(1)}%</div>
              <div className="text-xs text-text-muted">Bench</div>
            </div>
          )}
        </div>
      )}

      {/* Children count for base models */}
      {isBase && hasChildren && (
        <span className="text-sm text-text-muted">
          {node.children.length} model{node.children.length !== 1 ? 's' : ''}
        </span>
      )}
    </div>
  )
}

export function LineageTree({ onSelectModel, highlightModelId }: LineageTreeProps) {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())

  // Fetch all models
  const fetchModels = useCallback(async () => {
    setIsLoading(true)
    const result = await modelsApi.list({ limit: 100, sort_by: 'created_at', sort_order: 'desc' })
    if (result.ok && result.data) {
      setModels(result.data.models)
      // Expand all base model nodes by default
      const tree = buildLineageTree(result.data.models)
      setExpandedNodes(new Set(tree.map(n => n.model_id)))
    }
    setIsLoading(false)
  }, [])

  useEffect(() => {
    fetchModels()
  }, [fetchModels])

  const tree = useMemo(() => buildLineageTree(models), [models])

  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev)
      if (next.has(nodeId)) {
        next.delete(nodeId)
      } else {
        next.add(nodeId)
      }
      return next
    })
  }

  const renderNode = (node: TreeNode, isBase: boolean = false) => {
    const isExpanded = expandedNodes.has(node.model_id)
    const isHighlighted = node.model_id === highlightModelId

    return (
      <div key={node.model_id}>
        <TreeNodeRow
          node={node}
          expanded={isExpanded}
          onToggle={() => toggleNode(node.model_id)}
          onSelect={onSelectModel}
          isHighlighted={isHighlighted}
          isBase={isBase}
        />
        {isExpanded && node.children.map(child => renderNode(child, false))}
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-primary animate-spin" />
      </div>
    )
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <GitBranch className="w-16 h-16 text-text-muted mb-4 opacity-30" />
        <h3 className="text-lg font-medium text-text-primary mb-2">No Model Lineage</h3>
        <p className="text-text-muted">
          Train models to see their lineage tree here
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {/* Legend */}
      <div className="flex items-center gap-4 mb-4 text-sm text-text-muted">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          <span>Base Model</span>
        </div>
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" />
          <span>Fine-tuned</span>
        </div>
        <div className="flex items-center gap-2">
          <GraduationCap className="w-4 h-4 text-teal-500" />
          <span>Distilled</span>
        </div>
      </div>

      {/* Tree */}
      <div className="space-y-1">
        {tree.map(root => renderNode(root, true))}
      </div>

      {/* Summary */}
      <div className="mt-6 pt-4 border-t border-border-subtle">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-semibold text-text-primary">{models.length}</div>
            <div className="text-sm text-text-muted">Total Models</div>
          </div>
          <div>
            <div className="text-2xl font-semibold text-text-primary">{tree.length}</div>
            <div className="text-sm text-text-muted">Base Models</div>
          </div>
          <div>
            <div className="text-2xl font-semibold text-text-primary">
              {models.filter(m => m.starred).length}
            </div>
            <div className="text-sm text-text-muted">Starred</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// Compact version for embedding in other components
export function LineageTreeCompact({ modelId, onSelectModel }: { modelId: string; onSelectModel: (id: string) => void }) {
  const [ancestors, setAncestors] = useState<string[]>([])

  // For now, just show base model chain
  // In future, could fetch full lineage from API
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-text-muted">Lineage:</span>
      <div className="flex items-center gap-1">
        <button
          className="px-2 py-0.5 rounded bg-background-tertiary hover:bg-background-secondary text-text-muted"
        >
          Base Model
        </button>
        <ArrowRight className="w-3 h-3 text-text-muted" />
        <span className="px-2 py-0.5 rounded bg-primary/20 text-primary">
          Current
        </span>
      </div>
    </div>
  )
}
