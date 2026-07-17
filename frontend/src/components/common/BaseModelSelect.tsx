import { useEffect, useState } from 'react'
import { systemInfoApi } from '../../services/api'
import { BASE_MODEL_GROUPS, type BaseModelGroup } from './baseModels'

interface Props {
  value: string
  onChange: (value: string) => void
  className?: string
  catalogOnly?: boolean
}

export function selectBaseModelGroups(
  discovered: BaseModelGroup[],
  catalogOnly: boolean
): BaseModelGroup[] {
  return catalogOnly ? discovered : [...BASE_MODEL_GROUPS, ...discovered]
}

/**
 * Base-model picker for fine-tuning: a grouped dropdown of current suggested
 * open models plus a "Custom model…" option that reveals a free-text field for
 * any HuggingFace ID. Replaces raw text inputs so users select rather than type,
 * while still allowing any model (no mandated base model).
 *
 * The static BASE_MODEL_GROUPS are the always-available fallback; on mount we
 * also live-discover current open models from HuggingFace (/api/models/discover)
 * so new releases (e.g. Qwen3.6) appear without a code change. Discovery failure
 * is non-fatal — the static groups still render.
 */
export function BaseModelSelect({ value, onChange, className, catalogOnly = false }: Props) {
  const [discovered, setDiscovered] = useState<BaseModelGroup[]>([])

  useEffect(() => {
    let cancelled = false
    systemInfoApi
      .discoverModels()
      .then((res) => {
        const found = res.data?.models
        if (cancelled || !found?.length) return
        const models = found.slice(0, 30).map((m) => {
          const size = m.params_billions ? ` (${m.params_billions}B)` : ''
          return { value: m.id, label: `${m.id}${size}` }
        })
        setDiscovered([{ label: 'Discovered (live from HuggingFace)', models }])
      })
      .catch(() => {
        /* discovery is best-effort; static groups remain */
      })
    return () => {
      cancelled = true
    }
  }, [])

  const groups = selectBaseModelGroups(discovered, catalogOnly)
  const isKnown = groups.some((g) => g.models.some((m) => m.value === value))
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
        {groups.map((g) => (
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
