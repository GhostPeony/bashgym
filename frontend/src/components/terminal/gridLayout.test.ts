import assert from 'node:assert/strict'
import test from 'node:test'
import { getGridColumnSpan, getGridLayout, getGridTemplateRows } from './gridLayout'

test('uses one, two, then three dashboard columns', () => {
  assert.equal(getGridLayout(1).columns, 1)
  assert.equal(getGridLayout(2).columns, 2)
  assert.equal(getGridLayout(4).columns, 2)
  assert.equal(getGridLayout(5).columns, 3)
  assert.equal(getGridLayout(10).rows, 4)
})

test('balances incomplete final rows across the full dashboard width', () => {
  assert.deepEqual([0, 1, 2].map((index) => getGridColumnSpan(index, 3)), [2, 2, 4])
  assert.deepEqual([0, 1, 2, 3, 4].map((index) => getGridColumnSpan(index, 5)), [2, 2, 2, 3, 3])
  assert.equal(getGridColumnSpan(6, 7), 6)
  assert.deepEqual([6, 7].map((index) => getGridColumnSpan(index, 8)), [3, 3])
  assert.equal(getGridColumnSpan(9, 10), 6)
})

test('every grid row consumes all available tracks', () => {
  for (let total = 1; total <= 18; total += 1) {
    const { columns, tracks } = getGridLayout(total)
    for (let start = 0; start < total; start += columns) {
      const end = Math.min(total, start + columns)
      const rowSpan = Array.from(
        { length: end - start },
        (_, offset) => getGridColumnSpan(start + offset, total),
      ).reduce((sum, span) => sum + span, 0)
      assert.equal(rowSpan, tracks, `total=${total}, row=${Math.floor(start / columns)}`)
    }
  }
})

test('fills short grids and compacts dashboard-sized grids', () => {
  assert.equal(getGridTemplateRows(1), 'repeat(1, minmax(240px, 1fr))')
  assert.equal(getGridTemplateRows(4), 'repeat(2, minmax(240px, 1fr))')
  assert.equal(getGridTemplateRows(5), 'repeat(2, minmax(240px, 1fr))')
  assert.equal(getGridTemplateRows(7), 'repeat(3, 260px)')
  assert.equal(getGridTemplateRows(10), 'repeat(4, 260px)')
})
