import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Browser-only config (no Electron plugin) for testing in browser
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@services': path.resolve(__dirname, './src/services')
    }
  }
})
