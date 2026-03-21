import { create } from 'zustand'
import { deviceApi, Device, NewDevice, SSHCandidate, PreflightResult } from '../services/api'

interface DeviceStore {
  devices: Device[]
  defaultDeviceId: string | null
  loading: boolean
  error: string | null

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

  fetchDevices: async () => {
    set({ loading: true, error: null })
    try {
      const devices = await deviceApi.list()
      const defaultDevice = devices.find(d => d.is_default)
      set({ devices, defaultDeviceId: defaultDevice?.id || null, loading: false })
    } catch (e: any) {
      set({ error: e.message || 'Failed to fetch devices', loading: false })
    }
  },

  addDevice: async (device: NewDevice) => {
    try {
      const added = await deviceApi.add(device)
      await get().fetchDevices()
      return added
    } catch (e: any) {
      set({ error: e.message || 'Failed to add device' })
      return null
    }
  },

  updateDevice: async (id: string, updates: Partial<NewDevice>) => {
    try {
      await deviceApi.update(id, updates)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to update device' })
    }
  },

  removeDevice: async (id: string) => {
    try {
      await deviceApi.remove(id)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to remove device' })
    }
  },

  runPreflight: async (id: string) => {
    try {
      const result = await deviceApi.preflight(id)
      await get().fetchDevices()
      return result
    } catch (e: any) {
      set({ error: e.message || 'Preflight failed' })
      return null
    }
  },

  setDefault: async (id: string) => {
    try {
      await deviceApi.setDefault(id)
      await get().fetchDevices()
    } catch (e: any) {
      set({ error: e.message || 'Failed to set default' })
    }
  },

  discoverFromSSHConfig: async () => {
    try {
      const result = await deviceApi.discover()
      return result.candidates
    } catch (e: any) {
      set({ error: e.message || 'Discovery failed' })
      return []
    }
  },
}))
