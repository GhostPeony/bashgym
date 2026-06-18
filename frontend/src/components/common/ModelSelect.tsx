import { useEffect, useId, useState } from 'react'
import { providersApi } from '../../services/api'

const PROVIDER_PREFIXES = ['anthropic/', 'openai/', 'nim/', 'hf/', 'ollama/', 'gemini/']

// The live catalog ids are provider-prefixed (e.g. "nim/deepseek-ai/...",
// "anthropic/claude-..."); the backends expect the bare id. Strip the prefix.
function bareId(id: string): string {
  for (const p of PROVIDER_PREFIXES) {
    if (id.startsWith(p)) return id.slice(p.length)
  }
  return id
}

interface Props {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

/**
 * Inference-model picker: a combobox of live models (Anthropic / NVIDIA NIM /
 * Ollama) sourced from the live provider catalog, so it never goes stale, with
 * free-text fallback for any model id. Replaces raw text inputs.
 */
export function ModelSelect({ value, onChange, placeholder, className }: Props) {
  const listId = useId()
  const [options, setOptions] = useState<{ value: string; label: string }[]>([])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await providersApi.getModels({ include_local: true, include_cloud: true })
        if (cancelled || !res.ok || !res.data) return
        const cats = res.data
        const seen = new Set<string>()
        const opts: { value: string; label: string }[] = []
        for (const m of [...cats.teacher, ...cats.inference]) {
          const v = bareId(m.id)
          if (!v || seen.has(v)) continue
          seen.add(v)
          opts.push({ value: v, label: m.name && m.name !== v ? `${v} — ${m.name}` : v })
        }
        setOptions(opts)
      } catch {
        /* keep empty — the input still works as free text */
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <>
      <input
        list={listId}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder ?? 'Select or type a model…'}
        className={className ?? 'input w-full text-sm'}
        spellCheck={false}
        autoComplete="off"
      />
      <datalist id={listId}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </datalist>
    </>
  )
}
