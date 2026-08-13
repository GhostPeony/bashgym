import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { ghostPeonyIconPath } from './ghostPeonyIconAssets'

const logoSurfaces = [
  '../layout/NavigationBar.tsx',
  '../layout/Sidebar.tsx',
  '../home/HomeScreen.tsx',
  '../auth/LoginPage.tsx',
  '../download/DownloadPage.tsx'
]

test('packaged renderer branding uses file-relative public assets', () => {
  for (const surface of logoSurfaces) {
    const source = readFileSync(new URL(surface, import.meta.url), 'utf8')
    assert.match(source, /src="\.\/bashgym-peony\.png"/)
    assert.doesNotMatch(source, /src="\/bashgym-peony\.png"/)
  }

  assert.equal(ghostPeonyIconPath('app', 'color'), './bashgym-peony.png')
  assert.equal(ghostPeonyIconPath('training'), './node-icons/node-training-neutral.png')
})
