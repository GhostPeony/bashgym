import { createContext, useContext } from 'react'

export type NodeSurface = 'canvas' | 'grid'

export const NodeSurfaceContext = createContext<NodeSurface>('canvas')

export function useNodeSurface(): NodeSurface {
  return useContext(NodeSurfaceContext)
}
