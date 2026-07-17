import { BASE_MODEL_GROUPS, type BaseModelGroup } from './baseModels'
import { FALLBACK_MODEL_OPTIONS, type ModelOption } from './modelOptions'

export function selectBaseModelGroups(
  discovered: BaseModelGroup[],
  catalogOnly: boolean
): BaseModelGroup[] {
  return catalogOnly ? discovered : [...BASE_MODEL_GROUPS, ...discovered]
}

export function mergeModelOptions(
  catalogOptions: readonly ModelOption[],
  catalogOnly: boolean
): ModelOption[] {
  const merged = catalogOnly ? [] : [...FALLBACK_MODEL_OPTIONS]
  for (const option of catalogOptions) {
    if (!merged.some((existing) => existing.value === option.value)) {
      merged.push(option)
    }
  }
  return merged
}
