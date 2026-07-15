import { readdirSync } from 'node:fs'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'

function findTests(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return findTests(path)
    return entry.name.endsWith('.test.ts') ? [path] : []
  })
}

const tests = findTests(join(process.cwd(), 'src'))
if (tests.length === 0) {
  throw new Error('No frontend tests found')
}

const result = spawnSync(
  process.execPath,
  ['--import', 'tsx', '--test', ...tests],
  { stdio: 'inherit' },
)

process.exit(result.status ?? 1)
