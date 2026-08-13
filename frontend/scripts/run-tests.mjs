import { readdirSync } from 'node:fs'
import { basename, join, relative } from 'node:path'
import { spawnSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'

export function findTests(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return findTests(path)
    return /\.test\.(?:ts|mjs)$/.test(entry.name) ? [path] : []
  })
}

function normalized(value) {
  return value.replaceAll('\\', '/').replace(/^\.\//, '').toLowerCase()
}

export function selectTests(tests, filters, cwd = process.cwd()) {
  const requested = filters.filter((value) => value && !value.startsWith('--'))
  if (requested.length === 0) return tests
  const selected = tests.filter((testPath) => {
    const candidates = [
      normalized(testPath),
      normalized(relative(cwd, testPath)),
      normalized(basename(testPath))
    ]
    return requested.some((filter) => {
      const wanted = normalized(filter)
      return candidates.some(
        (candidate) => candidate === wanted || candidate.endsWith(`/${wanted}`)
      )
    })
  })
  if (selected.length === 0) {
    throw new Error(`No frontend tests matched: ${requested.join(', ')}`)
  }
  return selected
}

export function main(args = process.argv.slice(2)) {
  const tests = [
    ...findTests(join(process.cwd(), 'src')),
    ...findTests(join(process.cwd(), 'scripts'))
  ]
  if (tests.length === 0) throw new Error('No frontend tests found')
  const selected = selectTests(tests, args)
  const result = spawnSync(process.execPath, ['--import', 'tsx', '--test', ...selected], {
    stdio: 'inherit'
  })
  return result.status ?? 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.exit(main())
}
