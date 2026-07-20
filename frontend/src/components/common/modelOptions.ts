export interface ModelOption {
  value: string
  label: string
}

export const FALLBACK_MODEL_OPTIONS: readonly ModelOption[] = [
  { value: 'claude-opus-4-5-20251101', label: 'Claude Opus 4.5' },
  { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5' },
  { value: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
  { value: 'Qwen/Qwen3.5-27B', label: 'Qwen3.5 27B' },
  { value: 'unsloth/gemma-4-12b-it', label: 'Gemma 4 12B' },
  { value: 'meta-llama/Llama-3.1-70B-Instruct', label: 'Llama 3.1 70B' },
  { value: 'deepseek-ai/DeepSeek-Coder-V2-Instruct', label: 'DeepSeek Coder V2' }
]
