// Current open models suggested for fine-tuning. These are examples, not
// requirements — the "Custom model…" option accepts any HuggingFace ID, so the
// platform never mandates a base model. Keep in step with
// bashgym/providers/detector.py RECOMMENDED_TRAINING_MODELS.

export interface BaseModelOption {
  value: string
  label: string
}

export interface BaseModelGroup {
  label: string
  models: BaseModelOption[]
}

export const BASE_MODEL_GROUPS: BaseModelGroup[] = [
  {
    label: 'Qwen2.5 Coder',
    models: [
      { value: 'Qwen/Qwen2.5-Coder-1.5B-Instruct', label: 'Qwen2.5 Coder 1.5B (fast default)' },
      { value: 'Qwen/Qwen2.5-Coder-7B-Instruct', label: 'Qwen2.5 Coder 7B' },
      { value: 'Qwen/Qwen2.5-Coder-32B-Instruct', label: 'Qwen2.5 Coder 32B (teacher-grade)' },
    ],
  },
  {
    label: 'Gemma 4',
    models: [
      {
        value: 'unsloth/gemma-4-12b-it',
        label: 'Gemma 4 12B Unified (BF16/LoRA training base)',
      },
      { value: 'google/gemma-4-E2B-it', label: 'Gemma 4 E2B (trains on a consumer GPU)' },
      { value: 'google/gemma-4-E4B-it', label: 'Gemma 4 E4B' },
      { value: 'google/gemma-4-26B-A4B-it', label: 'Gemma 4 26B MoE (4B active)' },
      { value: 'google/gemma-4-31B-it', label: 'Gemma 4 31B Dense (large GPU)' },
    ],
  },
  {
    label: 'Qwen3.5',
    models: [
      { value: 'Qwen/Qwen3.5-4B', label: 'Qwen3.5 4B (dense)' },
      { value: 'Qwen/Qwen3.5-9B', label: 'Qwen3.5 9B (dense)' },
      { value: 'Qwen/Qwen3.5-27B', label: 'Qwen3.5 27B (dense, large GPU)' },
      { value: 'Qwen/Qwen3.5-35B-A3B', label: 'Qwen3.5 35B-A3B (MoE, large GPU)' },
    ],
  },
  {
    label: 'DeepSeek',
    models: [
      {
        value: 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
        label: 'DeepSeek-Coder-V2-Lite (16B MoE)',
      },
    ],
  },
  {
    label: 'Other',
    models: [
      { value: 'meta-llama/Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B' },
      { value: 'microsoft/phi-4', label: 'Phi-4 (14B)' },
    ],
  },
]
