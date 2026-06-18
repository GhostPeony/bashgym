import { useState } from 'react'
import { BASE_MODEL_GROUPS } from './baseModels'

interface Props {
  value: string
  onChange: (value: string) => void
  className?: string
}

/**
 * Base-model picker for fine-tuning: a grouped dropdown of current suggested
 * open models plus a "Custom model…" option that reveals a free-text field for
 * any HuggingFace ID. Replaces raw text inputs so users select rather than type,
 * while still allowing any model (no mandated base model).
 */
export function BaseModelSelect({ value, onChange, className }: Props) {
  const isKnown = BASE_MODEL_GROUPS.some((g) => g.models.some((m) => m.value === value))
  const [custom, setCustom] = useState<boolean>(!!value && !isKnown)

  return (
    <div className="flex flex-col gap-2">
      <select
        value={custom ? '__custom__' : value}
        onChange={(e) => {
          if (e.target.value === '__custom__') {
            setCustom(true)
          } else {
            setCustom(false)
            onChange(e.target.value)
          }
        }}
        className={className ?? 'input w-full text-sm'}
      >
        <option value="">Select a base model…</option>
        {BASE_MODEL_GROUPS.map((g) => (
          <optgroup key={g.label} label={g.label}>
            {g.models.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </optgroup>
        ))}
        <option value="__custom__">Custom model…</option>
      </select>
      {custom && (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="input w-full text-sm"
          placeholder="org/model-name (any HuggingFace ID)"
          autoFocus
          spellCheck={false}
          autoComplete="off"
        />
      )}
    </div>
  )
}
