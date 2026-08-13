/**
 * Platform detection for web vs Electron builds.
 *
 * `VITE_MODE` (set in vite.config.web.ts / .env.web) lets the web build
 * tree-shake Electron-only code at compile time. But the default dev server
 * (`npm run dev`) does NOT set it, so "not a web build" must not be treated as
 * "Electron" — opened in a plain browser it isn't, and assuming so makes
 * Electron-only code (e.g. window.bashgym) crash the app.
 *
 * We therefore also require the Electron preload bridge (`window.bashgym`,
 * exposed in electron/preload.ts) to be present at runtime. When the web build
 * sets VITE_MODE==='web', `isElectron` folds to a compile-time `false`, so
 * web-build tree-shaking is preserved; otherwise it falls back to detecting the
 * bridge at runtime, so `npm run dev` works correctly in a browser.
 */
const isWebBuild = import.meta.env?.VITE_MODE === 'web'
const hasElectronBridge =
  typeof window !== 'undefined' && Boolean((window as { bashgym?: unknown }).bashgym)

export const isElectron = !isWebBuild && hasElectronBridge
export const isWeb = !isElectron
