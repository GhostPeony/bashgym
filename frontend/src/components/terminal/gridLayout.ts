export interface GridLayout {
  columns: number
  tracks: number
  rows: number
}

export function getGridLayout(total: number): GridLayout {
  const columns = total <= 1 ? 1 : total <= 4 ? 2 : 3
  return {
    columns,
    tracks: columns * 2,
    rows: Math.max(1, Math.ceil(total / columns)),
  }
}

export function getGridColumnSpan(index: number, total: number): number {
  const { columns, tracks } = getGridLayout(total)
  const remainder = total % columns
  const firstFinalRowIndex = total - (remainder || columns)

  if (index < firstFinalRowIndex || remainder === 0) return 2
  if (remainder === 1) return tracks
  if (columns === 3 && remainder === 2) return tracks / 2
  return 2
}

export function getGridTemplateRows(total: number): string {
  const { rows } = getGridLayout(total)
  return rows <= 2
    ? `repeat(${rows}, minmax(240px, 1fr))`
    : `repeat(${rows}, 260px)`
}
