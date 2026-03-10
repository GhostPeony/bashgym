/**
 * Platform detection for web vs Electron builds.
 *
 * Uses the build-time VITE_MODE constant set in vite.config.web.ts / .env.web.
 * This enables Vite to tree-shake Electron-only code from web builds entirely,
 * since the condition resolves to a string literal at compile time.
 */
export const isWeb = import.meta.env.VITE_MODE === 'web'
export const isElectron = !isWeb
