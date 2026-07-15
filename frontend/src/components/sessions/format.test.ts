import assert from 'node:assert/strict'
import test from 'node:test'
import { folderNameFromPath } from './format'

test('folderNameFromPath returns only the final folder for Windows and POSIX paths', () => {
  const paths = [
    'F:\\File location\\repos\\bashgym',
    'F:/File location/repos/bashgym/',
    '/home/developer/repos/bashgym/',
    '/Users/developer/repos/bashgym',
    'bashgym'
  ]

  for (const path of paths) {
    assert.equal(folderNameFromPath(path), 'bashgym')
  }
})

test('folderNameFromPath uses a useful fallback for missing or home paths', () => {
  assert.equal(folderNameFromPath(undefined, 'Codex'), 'Codex')
  assert.equal(folderNameFromPath('~', 'Codex'), 'Codex')
})
