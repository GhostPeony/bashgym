export interface DesignerCanvasMetadata {
  originPanelId?: string
  runtimePid?: number
  runtimeDiscovered?: boolean
  config?: Record<string, unknown>
}

export interface DesignerPanelCandidate {
  id: string
  type: string
  adapterConfig?: Record<string, unknown>
}

function normalizedPath(value: unknown): string | undefined {
  if (typeof value !== 'string' || !value) return undefined
  return value.replaceAll('\\', '/').replace(/\/$/, '').toLowerCase()
}

/** Aggregate Data Designer jobs in one existing workspace node. */
export function selectDesignerPanelForJob(
  panels: readonly DesignerPanelCandidate[],
  job: { job_id: string; output_dir?: string | null },
  originPanelId?: string
): DesignerPanelCandidate | undefined {
  const designers = panels.filter((panel) => panel.type === 'designer')
  const exact = designers.find((panel) => {
    const config = panel.adapterConfig || {}
    return config.designerJobId === job.job_id || config.runtimeJobId === job.job_id
  })
  if (exact) return exact

  const outputDir = normalizedPath(job.output_dir)
  if (outputDir) {
    const sameOutput = designers.find(
      (panel) => normalizedPath(panel.adapterConfig?.outputDir) === outputDir
    )
    if (sameOutput) return sameOutput
  }

  if (originPanelId) {
    const origin = designers.find((panel) => panel.id === originPanelId)
    if (origin) return origin
  }

  return designers[0]
}
