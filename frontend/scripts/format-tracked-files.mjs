import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const repositoryRoot = path.resolve(frontendRoot, '..')
const supportedFrontendExtensions = new Set([
  '.css',
  '.html',
  '.js',
  '.json',
  '.mjs',
  '.ts',
  '.tsx'
])

const trackedFiles = execFileSync('git', ['ls-files', '-z'], {
  cwd: repositoryRoot,
  encoding: 'buffer'
})
  .toString()
  .split('\0')
  .filter(Boolean)

const files = trackedFiles.filter((file) => {
  const extension = path.extname(file)
  return file.endsWith('.md') || (file.startsWith('frontend/') && supportedFrontendExtensions.has(extension))
})

const prettier = path.join(frontendRoot, 'node_modules', 'prettier', 'bin', 'prettier.cjs')
const check = process.argv.includes('--check')

execFileSync(process.execPath, [prettier, check ? '--check' : '--write', ...files], {
  cwd: repositoryRoot,
  stdio: 'inherit'
})
