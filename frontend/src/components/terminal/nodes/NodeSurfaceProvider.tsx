import type { ReactNode } from 'react'
import { NodeSurfaceContext, type NodeSurface } from './nodeSurface'

export function NodeSurfaceProvider({
  surface,
  children
}: {
  surface: NodeSurface
  children: ReactNode
}) {
  return <NodeSurfaceContext.Provider value={surface}>{children}</NodeSurfaceContext.Provider>
}
