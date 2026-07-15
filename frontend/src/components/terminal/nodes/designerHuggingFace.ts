import type { DesignerJobStatus } from '../../../services/api'

function compactOutputName(outputDir?: string): string | undefined {
  if (!outputDir) return undefined
  return outputDir.split(/[\\/]/).filter(Boolean).at(-1)
}

/** Build a stable Hugging Face dataset slug from a completed designer run. */
export function datasetRepoNameForJob(job: DesignerJobStatus): string {
  const source = job.job_name || compactOutputName(job.output_dir) || job.pipeline || job.job_id
  const slug = source
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^[._-]+|[._-]+$/g, '')
    .slice(0, 88)
    .replace(/[._-]+$/g, '')

  return `bashgym-${slug || 'designer-data'}`
}
