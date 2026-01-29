import {
  Sparkles,
  Hash,
  FileJson,
  User,
  Calendar,
  Code,
  Sliders,
  CheckCircle,
} from 'lucide-react'
import type { FactoryConfig } from '../../services/api'

export type TabId = 'create' | 'seeds' | 'settings' | 'jobs'

export const COLUMN_TYPES = [
  { id: 'llm', label: 'LLM Column', icon: Sparkles, description: 'AI-generated content from prompts' },
  { id: 'sampler', label: 'Sampler', icon: Hash, description: 'Random values from distribution' },
  { id: 'category', label: 'Category', icon: FileJson, description: 'Pick from predefined list' },
  { id: 'person', label: 'Person', icon: User, description: 'Synthetic person names' },
  { id: 'datetime', label: 'DateTime', icon: Calendar, description: 'Date/time values' },
  { id: 'expression', label: 'Expression', icon: Code, description: 'Jinja2 template' },
  { id: 'uuid', label: 'UUID', icon: Hash, description: 'Unique identifiers' },
  { id: 'gaussian', label: 'Gaussian', icon: Sliders, description: 'Normal distribution values' },
  { id: 'validator', label: 'Validator', icon: CheckCircle, description: 'Python validation code' },
] as const

export const PII_TYPES = [
  'person', 'email', 'phone', 'ssn', 'address', 'credit_card',
  'ip_address', 'medical_record', 'passport', 'drivers_license'
]

export const RISK_LEVELS = [
  { id: 'normal', label: 'Normal', color: 'text-text-secondary' },
  { id: 'elevated', label: 'Elevated', color: 'text-status-warning' },
  { id: 'high', label: 'High', color: 'text-status-error' },
] as const

export const DEFAULT_CONFIG: FactoryConfig = {
  columns: [],
  seeds: [],
  privacy: {
    enabled: false,
    epsilon: 8.0,
    pii_types: ['person', 'email', 'ssn'],
    replacement_strategy: 'synthetic'
  },
  prompt_optimization: {
    enabled: false,
    intensity: 'medium',
    max_demos: 4,
    metric_threshold: 0.8,
    target_metric: 'accuracy'
  },
  output: {
    row_count: 1000,
    format: 'jsonl',
    task_name: 'default_task',
    include_task_name: true,
    train_val_split: 0.9,
    include_negative_examples: false,
    negative_example_ratio: 0.1
  },
  safety: {
    enabled: true,
    block_dangerous_commands: true,
    require_confirmation_for_high_risk: true,
    max_risk_level: 'high',
    blocked_patterns: ['rm -rf', 'sudo', 'chmod 777', ':(){:|:&};:']
  },
  default_model: {
    model_id: 'claude-sonnet-4-5-20250929',
    temperature: 0.7,
    max_tokens: 1024
  }
}

export const SEED_CATEGORY_TAGS = [
  'git', 'docker', 'python', 'bash', 'javascript', 'typescript',
  'testing', 'refactor', 'debug', 'config', 'deployment', 'api'
] as const

export type SeedSourceFilter = 'all' | 'gold_trace' | 'imported' | 'manual'
