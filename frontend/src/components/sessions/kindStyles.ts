import type { AgentSessionKind } from '../../services/agentSessions/types'

export const KIND_CHIP_BASE =
  'flex-shrink-0 px-1 py-px border-brutal rounded-brutal text-[8px] font-bold uppercase tracking-wider font-mono'

/**
 * Distinct identity per agent CLI: Claude wears the user's accent, Codex a
 * fixed Sky pastel — clearly different at a glance in mixed lists.
 */
export function kindChipClass(kind?: AgentSessionKind): string {
  if (kind === 'claude') return 'border-accent/60 bg-accent/10 text-accent'
  if (kind === 'codex') return 'border-[hsl(210_45%_55%)]/60 bg-[hsl(210_45%_55%)]/10 text-[hsl(210_45%_55%)]'
  return 'border-border-subtle bg-background-tertiary text-text-muted'
}
