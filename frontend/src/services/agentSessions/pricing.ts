/**
 * Cost estimation for Claude models.
 *
 * Ported from bashgym/trace_capture/core.py (CLAUDE_PRICING / estimate_cost_usd)
 * — keep the two tables in sync. Rates are USD per million tokens; cache write
 * is 1.25x input, cache read is 0.1x input. Longest-prefix match resolves dated
 * model IDs (e.g. "claude-sonnet-4-5-20250929" → "claude-sonnet-4-5").
 */

import type { TokenTotals } from './types'

interface ModelPricing {
  input: number
  output: number
  cacheCreation: number
  cacheRead: number
}

const CLAUDE_PRICING: Record<string, ModelPricing> = {
  'claude-fable-5': { input: 10.0, output: 50.0, cacheCreation: 12.5, cacheRead: 1.0 },
  'claude-mythos-5': { input: 10.0, output: 50.0, cacheCreation: 12.5, cacheRead: 1.0 },
  'claude-opus-4-8': { input: 5.0, output: 25.0, cacheCreation: 6.25, cacheRead: 0.5 },
  'claude-opus-4-7': { input: 5.0, output: 25.0, cacheCreation: 6.25, cacheRead: 0.5 },
  'claude-opus-4-6': { input: 5.0, output: 25.0, cacheCreation: 6.25, cacheRead: 0.5 },
  'claude-opus-4-5': { input: 5.0, output: 25.0, cacheCreation: 6.25, cacheRead: 0.5 },
  'claude-opus-4-1': { input: 15.0, output: 75.0, cacheCreation: 18.75, cacheRead: 1.875 },
  'claude-opus-4': { input: 15.0, output: 75.0, cacheCreation: 18.75, cacheRead: 1.875 },
  'claude-sonnet-4-6': { input: 3.0, output: 15.0, cacheCreation: 3.75, cacheRead: 0.3 },
  'claude-sonnet-4-5': { input: 3.0, output: 15.0, cacheCreation: 3.75, cacheRead: 0.3 },
  'claude-sonnet-4': { input: 3.0, output: 15.0, cacheCreation: 3.75, cacheRead: 0.3 },
  'claude-haiku-4-5': { input: 1.0, output: 5.0, cacheCreation: 1.25, cacheRead: 0.1 }
}

const PRICING_KEYS_BY_LENGTH = Object.keys(CLAUDE_PRICING).sort((a, b) => b.length - a.length)

export function estimateCostUsd(model: string, totals: TokenTotals): number {
  let pricing = CLAUDE_PRICING[model]
  if (!pricing) {
    const key = PRICING_KEYS_BY_LENGTH.find((k) => model.startsWith(k))
    if (key) pricing = CLAUDE_PRICING[key]
  }
  if (!pricing) return 0

  const cost =
    (totals.input * pricing.input +
      totals.output * pricing.output +
      totals.cacheCreate * pricing.cacheCreation +
      totals.cacheRead * pricing.cacheRead) /
    1_000_000
  return Math.round(cost * 1e6) / 1e6
}

/**
 * The Claude journal doesn't record the context limit, so the meter assumes
 * the standard 200k window and steps up to the 1M tier only when the observed
 * occupancy proves it. Meters render with a "~" to signal the approximation.
 */
export const DEFAULT_CLAUDE_CONTEXT_WINDOW = 200_000
const EXTENDED_CLAUDE_CONTEXT_WINDOW = 1_000_000

export function contextWindowFor(observedTokens?: number): number {
  if (observedTokens && observedTokens > DEFAULT_CLAUDE_CONTEXT_WINDOW) {
    return EXTENDED_CLAUDE_CONTEXT_WINDOW
  }
  return DEFAULT_CLAUDE_CONTEXT_WINDOW
}
