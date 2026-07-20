const LEGACY_SHARED_SESSION_KEYS = new Set(['', 'bashgym-canvas'])

export function hermesWorkspaceSessionKey(configured: string, workspaceId: string): string {
  const workspace = workspaceId.trim() || 'default'
  const value = configured.trim()
  if (LEGACY_SHARED_SESSION_KEYS.has(value)) return `bashgym:${workspace}`
  return value.replaceAll('{workspace_id}', workspace)
}

export function composeAgentWorkspaceContext(
  authoritative: string | null,
  endpointDetails: string,
  contextError?: string
): string {
  const sections: string[] = [
    [
      '# BashGym Evidence Rules',
      '',
      '- Precedence: live runtime > durable ledger > workspace snapshot > curated GBrain > conversation memory.',
      '- Treat remembered conversation as unverified when it conflicts with current BashGym evidence.',
      '- Cite source IDs, run/campaign/model IDs, and observation times for current-state claims.',
      '- Expose missing or conflicting project context instead of blending experiments.'
    ].join('\n')
  ]
  if (authoritative?.trim()) sections.push(authoritative.trim())
  else {
    sections.push(
      [
        '# BashGym Workspace Context',
        '',
        'The authoritative workspace projection is temporarily unavailable.',
        contextError ? `Reason: ${contextError}` : ''
      ]
        .filter(Boolean)
        .join('\n')
    )
  }
  if (endpointDetails.trim()) sections.push(endpointDetails.trim())
  return sections.join('\n\n---\n\n')
}
