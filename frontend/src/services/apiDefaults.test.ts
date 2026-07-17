import assert from 'node:assert/strict'
import test from 'node:test'

import { API_BASE } from './api'

test('renderer fallback API URL matches the desktop-managed backend port', () => {
  assert.equal(API_BASE, 'http://127.0.0.1:8003/api')
})
