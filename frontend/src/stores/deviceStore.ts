import { create } from 'zustand'
import { deviceApi, Device, NewDevice, SSHCandidate, PreflightResult } from '../services/api'

interface DeviceStore {
  devices: Device[]
  defaultDeviceId: string | null
  loading: boolean
  error: string | null
  loadedAt: number | null

  /** Fetch devices only if they have not been loaded this session. */
  ensureDevices: () => Promise<void>
  fetchDevices: () => Promise<void>
  addDevice: (device: NewDevice) => Promise<Device | null>
  updateDevice: (id: string, updates: Partial<NewDevice>) => Promise<void>
  removeDevice: (id: string) => Promise<void>
  runPreflight: (id: string) => Promise<PreflightResult | null>
  setDefault: (id: string) => Promise<void>
  discoverFromSSHConfig: () => Promise<SSHCandidate[]>
}

export const useDeviceStore = create<DeviceStore>((set, get) => ({
  devices: [],
  defaultDeviceId: null,
  loading: false,
  error: null,
  loadedAt: null,

  ensureDevices: async () => {
    const { loadedAt, loading } = get()
    if (loadedAt !== null || loading) return
    await get().fetchDevices()
  },

  fetchDevices: async () => {
    set({ loading: true, error: null })
    const result = await deviceApi.list()
    if (result.ok && result.data) {
      const devices = result.data
      const defaultDevice = devices.find(d => d.is_default)
      set({ devices, defaultDeviceId: defaultDevice?.id || null, loading: false, loadedAt: Date.now() })
    } else {
      set({ error: result.error || 'Failed to fetch devices', loading: false })
    }
  },

  addDevice: async (device: NewDevice) => {
    const result = await deviceApi.add(device)
    if (result.ok && result.data) {
      await get().fetchDevices()
      return result.data
    }
    set({ error: result.error || 'Failed to add device' })
    return null
  },

  updateDevice: async (id: string, updates: Partial<NewDevice>) => {
    const result = await deviceApi.update(id, updates)
    if (result.ok) {
      await get().fetchDevices()
    } else {
      set({ error: result.error || 'Failed to update device' })
    }
  },

  removeDevice: async (id: string) => {
    const result = await deviceApi.remove(id)
    if (result.ok) {
      await get().fetchDevices()
    } else {
      set({ error: result.error || 'Failed to remove device' })
    }
  },

  runPreflight: async (id: string) => {
    const result = await deviceApi.preflight(id)
    if (result.ok && result.data) {
      await get().fetchDevices()
      return result.data
    }
    set({ error: result.error || 'Preflight failed' })
    return null
  },

  setDefault: async (id: string) => {
    const result = await deviceApi.setDefault(id)
    if (result.ok) {
      await get().fetchDevices()
    } else {
      set({ error: result.error || 'Failed to set default' })
    }
  },

  discoverFromSSHConfig: async () => {
    const result = await deviceApi.discover()
    if (result.ok && result.data) {
      return result.data.candidates
    }
    set({ error: result.error || 'Discovery failed' })
    return []
  },
}))
