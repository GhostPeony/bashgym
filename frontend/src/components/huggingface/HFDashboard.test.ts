import assert from 'node:assert/strict'
import test from 'node:test'

import { hfDashboardTabs } from './hfDashboardTabs'

test('presents provider jobs as observation rather than a paid cloud-training launcher', () => {
  const jobs = hfDashboardTabs().find((tab) => tab.id === 'training')

  assert.ok(jobs)
  assert.equal(jobs.label, 'Jobs')
  assert.equal(jobs.requiresPro, false)
  assert.ok(hfDashboardTabs().every((tab) => !tab.label.toLowerCase().includes('cloud training')))
})
