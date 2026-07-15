import assert from 'node:assert/strict'
import test from 'node:test'
import type { DesignerJobStatus } from '../../../services/api'
import { datasetRepoNameForJob } from './designerHuggingFace'

test('builds a readable repository slug from a designer job name', () => {
  const job: DesignerJobStatus = {
    job_id: 'runtime_designer_manifest_done',
    status: 'completed',
    pipeline: 'generate_dd_train_pairs',
    num_records: 200,
    job_name: 'dd-train-pairs-20260709 / real_chunks',
  }

  assert.equal(
    datasetRepoNameForJob(job),
    'bashgym-dd-train-pairs-20260709-real_chunks',
  )
})

test('falls back to the generated output directory', () => {
  const job: DesignerJobStatus = {
    job_id: 'runtime_designer_manifest_done',
    status: 'completed',
    pipeline: '',
    num_records: 200,
    output_dir: 'C:\\workspace\\outputs\\fake_transcripts',
  }

  assert.equal(datasetRepoNameForJob(job), 'bashgym-fake_transcripts')
})
