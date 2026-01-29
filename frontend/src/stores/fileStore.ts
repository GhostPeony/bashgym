import { create } from 'zustand'

export interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  size?: number
  modified?: number
  children?: FileNode[]
  expanded?: boolean
  loading?: boolean
}

interface FileState {
  // Current root path
  rootPath: string

  // File tree
  tree: FileNode[]

  // Selection state
  selectedPath: string | null

  // Expanded directories
  expandedPaths: Set<string>

  // Loading state
  isLoading: boolean
  error: string | null

  // Actions
  setRootPath: (path: string) => Promise<void>
  loadDirectory: (path: string) => Promise<FileNode[]>
  toggleExpand: (path: string) => void
  selectFile: (path: string) => void
  refresh: () => Promise<void>
  goUp: () => Promise<void>
}

// IPC calls to Electron main process
const ipc = {
  readDirectory: async (path: string): Promise<FileNode[]> => {
    const result = await window.bashgym?.files?.readDirectory(path)
    if (!result?.success) {
      throw new Error(result?.error || 'Failed to read directory')
    }
    return result.files || []
  },

  getHomeDirectory: async (): Promise<string> => {
    const result = await window.bashgym?.files?.getHomeDirectory()
    return result || '~'
  },

  getParentDirectory: async (path: string): Promise<string> => {
    const result = await window.bashgym?.files?.getParentDirectory(path)
    return result || path
  }
}

export const useFileStore = create<FileState>((set, get) => ({
  rootPath: '',
  tree: [],
  selectedPath: null,
  expandedPaths: new Set(),
  isLoading: false,
  error: null,

  setRootPath: async (path: string) => {
    set({ isLoading: true, error: null })

    try {
      const files = await ipc.readDirectory(path)
      set({
        rootPath: path,
        tree: files,
        isLoading: false,
        expandedPaths: new Set()
      })
    } catch (error) {
      set({
        error: String(error),
        isLoading: false
      })
    }
  },

  loadDirectory: async (path: string): Promise<FileNode[]> => {
    try {
      const files = await ipc.readDirectory(path)
      return files
    } catch (error) {
      console.error('Failed to load directory:', path, error)
      return []
    }
  },

  toggleExpand: (path: string) => {
    set((state) => {
      const newExpanded = new Set(state.expandedPaths)
      if (newExpanded.has(path)) {
        newExpanded.delete(path)
      } else {
        newExpanded.add(path)
      }
      return { expandedPaths: newExpanded }
    })
  },

  selectFile: (path: string) => {
    set({ selectedPath: path })
  },

  refresh: async () => {
    const { rootPath, setRootPath } = get()
    if (rootPath) {
      await setRootPath(rootPath)
    }
  },

  goUp: async () => {
    const { rootPath, setRootPath } = get()
    if (rootPath) {
      const parentPath = await ipc.getParentDirectory(rootPath)
      if (parentPath !== rootPath) {
        await setRootPath(parentPath)
      }
    }
  }
}))

// Initialize with home directory on first use
let initialized = false
export const initializeFileStore = async () => {
  if (initialized) return
  initialized = true

  try {
    const homePath = await ipc.getHomeDirectory()
    useFileStore.getState().setRootPath(homePath)
  } catch (error) {
    console.error('Failed to initialize file store:', error)
  }
}
