import assert from 'node:assert/strict'
import test from 'node:test'
import { selectDesignerPanelForJob } from './designerCanvasLifecycle'

test('aggregates concurrent Data Designer jobs in the existing node', () => {
  const panel = selectDesignerPanelForJob(
    [
      {
        id: 'designer-a',
        type: 'designer',
        adapterConfig: { designerJobId: 'job-a', status: 'running' }
      }
    ],
    { job_id: 'job-b' }
  )

  assert.equal(panel?.id, 'designer-a')
})

test('reuses the launching node and matches later updates by job id', () => {
  const panels = [
    { id: 'designer-a', type: 'designer', adapterConfig: { status: 'completed' } },
    {
      id: 'designer-b',
      type: 'designer',
      adapterConfig: { designerJobId: 'job-b', status: 'running' }
    }
  ]

  assert.equal(
    selectDesignerPanelForJob(panels, { job_id: 'job-new' }, 'designer-a')?.id,
    'designer-a'
  )
  assert.equal(selectDesignerPanelForJob(panels, { job_id: 'job-b' })?.id, 'designer-b')
})

test('reconciles a finalized manifest with the process node by output directory', () => {
  const panel = selectDesignerPanelForJob(
    [
      {
        id: 'designer-live',
        type: 'designer',
        adapterConfig: {
          runtimeJobId: 'runtime_designer_6276',
          outputDir: 'C:\\workspace\\output\\real_chunks',
          status: 'completed'
        }
      }
    ],
    {
      job_id: 'runtime_designer_manifest_123',
      output_dir: 'C:/workspace/output/real_chunks/'
    }
  )

  assert.equal(panel?.id, 'designer-live')
})

test('creates a Data Designer node only when none exists', () => {
  assert.equal(selectDesignerPanelForJob([], { job_id: 'job-new' }), undefined)
})
