import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Web-only Vite config — no Electron plugins, no node-pty
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'studio-self-hosted-fonts',
      transformIndexHtml: (html) =>
        html.replace(/<link\b[^>]*https:\/\/fonts\.(?:googleapis|gstatic)\.com[^>]*>\s*/g, '')
    }
  ],
  define: {
    'import.meta.env.VITE_MODE': JSON.stringify('web')
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@services': path.resolve(__dirname, './src/services')
    }
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html')
      },
      // Exclude Electron-only dependencies from web bundle
      external: ['electron', 'node-pty']
    }
  }
})
