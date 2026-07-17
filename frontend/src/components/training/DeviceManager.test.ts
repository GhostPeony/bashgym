import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

test('registered private-compute cards expose the logical device ID needed by campaign activation', async () => {
  const source = await readFile(new URL('./DeviceManager.tsx', import.meta.url), 'utf8')

  assert.match(source, /Device ID:\s*\{device\.id\}/)
  assert.doesNotMatch(source, /Device ID:\s*\{device\.(host|username|key_path|work_dir)\}/)
})
