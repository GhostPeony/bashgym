import assert from 'node:assert/strict'
import test from 'node:test'
import { useTerminalStore, type Panel } from './terminalStore'

test('keeps the panels reference when a panel config update is unchanged', () => {
  const originalPanels = useTerminalStore.getState().panels
  const panel: Panel = {
    id: 'designer-panel',
    type: 'designer',
    title: 'Data Designer',
    adapterConfig: {
      status: 'running',
      progress: { current: 20, total: 100, unit: 'seeds' },
    },
  }
  useTerminalStore.setState({ panels: [panel] })

  try {
    const before = useTerminalStore.getState().panels
    useTerminalStore.getState().updatePanelConfig('designer-panel', {
      status: 'running',
      progress: { current: 20, total: 100, unit: 'seeds' },
    })

    assert.equal(useTerminalStore.getState().panels, before)
  } finally {
    useTerminalStore.setState({ panels: originalPanels })
  }
})

test('replaces the panels reference when a panel config actually changes', () => {
  const originalPanels = useTerminalStore.getState().panels
  const panel: Panel = {
    id: 'designer-panel',
    type: 'designer',
    title: 'Data Designer',
    adapterConfig: { status: 'running' },
  }
  useTerminalStore.setState({ panels: [panel] })

  try {
    const before = useTerminalStore.getState().panels
    useTerminalStore.getState().updatePanelConfig('designer-panel', { status: 'completed' })

    assert.notEqual(useTerminalStore.getState().panels, before)
    assert.equal(
      useTerminalStore.getState().panels[0].adapterConfig?.status,
      'completed',
    )
  } finally {
    useTerminalStore.setState({ panels: originalPanels })
  }
})
