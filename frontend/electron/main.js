var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { app, BrowserWindow, ipcMain, shell, Menu, webContents, clipboard, nativeImage, safeStorage } from 'electron';
import path from 'path';
import fs from 'fs';
import os from 'os';
// Handle creating/removing shortcuts on Windows when installing/uninstalling
// This is only needed for production builds with Squirrel installer
// Removed: electron-squirrel-startup check (causes ESM/CJS issues in dev)
// Single instance lock - prevent multiple instances from running
var gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
    // Another instance is already running, quit immediately
    app.quit();
}
var mainWindow = null;
// Store for terminal processes
var terminals = new Map();
// Determine if we're in development
var isDev = !app.isPackaged;
// Credentials directory for secure storage
var credentialsDir = path.join(app.getPath('userData'), 'credentials');
function setupMenu() {
    var template = [
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' },
            ]
        },
        {
            label: 'View',
            submenu: [
                // Reload deliberately omitted — Ctrl+R kills terminals and wipes state
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { role: 'togglefullscreen' },
            ]
        }
    ];
    Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
function createWindow() {
    var isWin = process.platform === 'win32';
    mainWindow = new BrowserWindow(__assign(__assign({ width: 1400, height: 900, minWidth: 800, minHeight: 600, title: 'Bash Gym', titleBarStyle: isWin ? 'hidden' : 'hiddenInset' }, (isWin ? {} : { trafficLightPosition: { x: 16, y: 16 } })), { backgroundColor: '#0D0D0D', webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false, // Required for node-pty
            webviewTag: true // Required for browser panel screenshots
        } }));
    // Content Security Policy
    var devSources = isDev ? " 'unsafe-inline' 'unsafe-eval' http://localhost:*" : " 'unsafe-inline'";
    var csp = [
        "default-src 'self'",
        "script-src 'self'".concat(devSources),
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data: blob: https:",
        "connect-src 'self' http://localhost:* ws://localhost:* https://api.vercel.com https://api.neon.tech https://fonts.googleapis.com https://fonts.gstatic.com",
        "worker-src 'self' blob:",
        "child-src 'self'",
        "frame-src 'self' https:",
    ].join('; ');
    mainWindow.webContents.session.webRequest.onHeadersReceived(function (details, callback) {
        callback({
            responseHeaders: __assign(__assign({}, details.responseHeaders), { 'Content-Security-Policy': [csp] })
        });
    });
    // Load the app
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools(); // TEMP: debug black screen
    }
    else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
    // Inject electron-window class for platform-specific styling (e.g. window border on Windows)
    mainWindow.webContents.on('did-finish-load', function () {
        mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.webContents.executeJavaScript("document.documentElement.classList.add('electron-window')");
    });
    // Block Ctrl+R / Ctrl+Shift+R / F5 from refreshing the app — kills terminals and wipes state
    mainWindow.webContents.on('before-input-event', function (_event, input) {
        if (input.type !== 'keyDown')
            return;
        var isRefresh = (input.key === 'r' && input.control && !input.alt) ||
            (input.key === 'R' && input.control && input.shift && !input.alt) ||
            (input.key === 'F5' && !input.control && !input.alt);
        if (isRefresh) {
            // Forward Ctrl+R to renderer so browser panes can handle it
            mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.webContents.send('app-keydown', {
                key: input.key,
                ctrlKey: input.control,
                shiftKey: input.shift
            });
            _event.preventDefault();
        }
    });
    // Open external links in browser
    mainWindow.webContents.setWindowOpenHandler(function (_a) {
        var url = _a.url;
        shell.openExternal(url);
        return { action: 'deny' };
    });
    mainWindow.on('closed', function () {
        mainWindow = null;
        // Clean up all terminal processes
        terminals.forEach(function (term) { return term.kill(); });
        terminals.clear();
    });
}
// Only initialize if we got the lock
if (gotTheLock) {
    // Handle second instance - focus existing window
    app.on('second-instance', function () {
        if (mainWindow) {
            if (mainWindow.isMinimized())
                mainWindow.restore();
            mainWindow.focus();
        }
    });
    // App lifecycle
    app.whenReady().then(function () {
        setupMenu();
        createWindow();
        app.on('activate', function () {
            if (BrowserWindow.getAllWindows().length === 0) {
                createWindow();
            }
        });
    });
    app.on('window-all-closed', function () {
        if (process.platform !== 'darwin') {
            app.quit();
        }
    });
}
// IPC Handlers
// Get fresh environment with updated PATH (especially important on Windows)
function getFreshEnv() {
    var env = __assign({}, process.env);
    // Remove Claude Code's nesting guard so terminals can launch claude freely
    delete env.CLAUDECODE;
    if (process.platform === 'win32') {
        // On Windows, add common tool paths that might have been installed after app launch
        var userProfile = process.env.USERPROFILE || '';
        var localAppData = process.env.LOCALAPPDATA || '';
        var additionalPaths = [
            "".concat(localAppData, "\\Programs\\Ollama"),
            "".concat(localAppData, "\\Programs\\OpenCode"),
            "".concat(userProfile, "\\.local\\bin"),
            "".concat(userProfile, "\\scoop\\shims"),
        ].filter(function (p) { return p; });
        // Prepend to PATH so new tools are found first
        var currentPath = env.PATH || env.Path || '';
        env.PATH = __spreadArray(__spreadArray([], additionalPaths, true), [currentPath], false).join(';');
        env.Path = env.PATH;
    }
    return env;
}
// Terminal management
ipcMain.handle('terminal:create', function (_, id, cwd) { return __awaiter(void 0, void 0, void 0, function () {
    var pty, shell_1, shellArgs, ptyProcess, error_1;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, import('node-pty')];
            case 1:
                pty = _a.sent();
                shell_1 = process.platform === 'win32'
                    ? 'powershell.exe'
                    : process.env.SHELL || '/bin/bash';
                shellArgs = process.platform === 'win32'
                    ? ['-NoLogo']
                    : [];
                ptyProcess = pty.spawn(shell_1, shellArgs, {
                    name: 'xterm-256color',
                    cols: 80,
                    rows: 24,
                    cwd: cwd || process.env.HOME || process.cwd(),
                    env: getFreshEnv()
                });
                // Store reference (cast to ChildProcessWithoutNullStreams for type compatibility)
                terminals.set(id, ptyProcess);
                // Forward output to renderer
                ptyProcess.onData(function (data) {
                    mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.webContents.send("terminal:data:".concat(id), data);
                });
                ptyProcess.onExit(function (_a) {
                    var exitCode = _a.exitCode;
                    mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.webContents.send("terminal:exit:".concat(id), exitCode);
                    terminals.delete(id);
                });
                return [2 /*return*/, { success: true, id: id }];
            case 2:
                error_1 = _a.sent();
                console.error('Failed to create terminal:', error_1);
                return [2 /*return*/, { success: false, error: String(error_1) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('terminal:write', function (_, id, data) {
    var term = terminals.get(id);
    if (term) {
        term.write(data);
        return true;
    }
    return false;
});
ipcMain.handle('terminal:resize', function (_, id, cols, rows) {
    var term = terminals.get(id);
    if (term && term.resize) {
        term.resize(cols, rows);
        return true;
    }
    return false;
});
ipcMain.handle('terminal:kill', function (_, id) {
    var term = terminals.get(id);
    if (term) {
        term.kill();
        terminals.delete(id);
        return true;
    }
    return false;
});
// System info
ipcMain.handle('system:info', function () {
    return {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        electronVersion: process.versions.electron,
        cwd: process.cwd()
    };
});
// Window controls (for frameless window on Windows)
ipcMain.handle('window:minimize', function () {
    mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.minimize();
});
ipcMain.handle('window:maximize', function () {
    if (mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.isMaximized()) {
        mainWindow.unmaximize();
    }
    else {
        mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.maximize();
    }
});
ipcMain.handle('window:close', function () {
    mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.close();
});
ipcMain.handle('window:isMaximized', function () {
    var _a;
    return (_a = mainWindow === null || mainWindow === void 0 ? void 0 : mainWindow.isMaximized()) !== null && _a !== void 0 ? _a : false;
});
// Theme persistence
var currentTheme = 'dark';
ipcMain.handle('theme:get', function () { return currentTheme; });
ipcMain.handle('theme:set', function (_, theme) {
    currentTheme = theme;
    return currentTheme;
});
// API proxy (for development - avoids CORS issues)
ipcMain.handle('api:fetch', function (_, url, options) { return __awaiter(void 0, void 0, void 0, function () {
    var fetchOptions, response, text, data, error_2;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 3, , 4]);
                fetchOptions = __assign(__assign({}, options), { headers: __assign({ 'Content-Type': 'application/json', 'X-API-Key': process.env.BASHGYM_API_KEY || '' }, ((options === null || options === void 0 ? void 0 : options.headers) || {})) });
                return [4 /*yield*/, fetch(url, fetchOptions)];
            case 1:
                response = _a.sent();
                return [4 /*yield*/, response.text()];
            case 2:
                text = _a.sent();
                try {
                    data = JSON.parse(text);
                    return [2 /*return*/, { ok: response.ok, status: response.status, data: data }];
                }
                catch (_b) {
                    return [2 /*return*/, { ok: false, status: response.status, error: text || "HTTP ".concat(response.status) }];
                }
                return [3 /*break*/, 4];
            case 3:
                error_2 = _a.sent();
                return [2 /*return*/, { ok: false, error: String(error_2) }];
            case 4: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('files:readDirectory', function (_, dirPath) { return __awaiter(void 0, void 0, void 0, function () {
    var resolvedPath, entries, files, _i, entries_1, entry, fullPath, stats, _a, error_3;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                _b.trys.push([0, 8, , 9]);
                resolvedPath = dirPath.startsWith('~')
                    ? path.join(os.homedir(), dirPath.slice(1))
                    : dirPath;
                return [4 /*yield*/, fs.promises.readdir(resolvedPath, { withFileTypes: true })];
            case 1:
                entries = _b.sent();
                files = [];
                _i = 0, entries_1 = entries;
                _b.label = 2;
            case 2:
                if (!(_i < entries_1.length)) return [3 /*break*/, 7];
                entry = entries_1[_i];
                // Skip hidden files on Unix (starting with .)
                if (entry.name.startsWith('.'))
                    return [3 /*break*/, 6];
                fullPath = path.join(resolvedPath, entry.name);
                _b.label = 3;
            case 3:
                _b.trys.push([3, 5, , 6]);
                return [4 /*yield*/, fs.promises.stat(fullPath)];
            case 4:
                stats = _b.sent();
                files.push({
                    name: entry.name,
                    path: fullPath,
                    type: entry.isDirectory() ? 'directory' : 'file',
                    size: entry.isFile() ? stats.size : undefined,
                    modified: stats.mtimeMs
                });
                return [3 /*break*/, 6];
            case 5:
                _a = _b.sent();
                return [3 /*break*/, 6];
            case 6:
                _i++;
                return [3 /*break*/, 2];
            case 7:
                // Sort: directories first, then by name
                files.sort(function (a, b) {
                    if (a.type !== b.type) {
                        return a.type === 'directory' ? -1 : 1;
                    }
                    return a.name.localeCompare(b.name);
                });
                return [2 /*return*/, { success: true, files: files }];
            case 8:
                error_3 = _b.sent();
                return [2 /*return*/, { success: false, error: String(error_3) }];
            case 9: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('files:getHomeDirectory', function () {
    return os.homedir();
});
ipcMain.handle('files:getParentDirectory', function (_, filePath) {
    return path.dirname(filePath);
});
ipcMain.handle('files:readFile', function (_, filePath) { return __awaiter(void 0, void 0, void 0, function () {
    var content, error_4;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, fs.promises.readFile(filePath, 'utf-8')];
            case 1:
                content = _a.sent();
                return [2 /*return*/, { success: true, content: content }];
            case 2:
                error_4 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_4) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('files:exists', function (_, filePath) { return __awaiter(void 0, void 0, void 0, function () {
    var _a;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                _b.trys.push([0, 2, , 3]);
                return [4 /*yield*/, fs.promises.access(filePath)];
            case 1:
                _b.sent();
                return [2 /*return*/, true];
            case 2:
                _a = _b.sent();
                return [2 /*return*/, false];
            case 3: return [2 /*return*/];
        }
    });
}); });
// Browser screenshot via webContentsId — reliable alternative to webview.capturePage() in renderer
ipcMain.handle('browser:screenshot', function (_, webContentsId, rect) { return __awaiter(void 0, void 0, void 0, function () {
    var wc, fullImage, _a, imgW, imgH, scaleX, scaleY, cropped, error_5;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                _b.trys.push([0, 2, , 3]);
                wc = webContents.fromId(webContentsId);
                if (!wc)
                    return [2 /*return*/, { success: false, error: 'WebContents not found' }];
                return [4 /*yield*/, wc.capturePage()];
            case 1:
                fullImage = _b.sent();
                if (rect) {
                    _a = fullImage.getSize(), imgW = _a.width, imgH = _a.height;
                    scaleX = imgW / rect.vpW;
                    scaleY = imgH / rect.vpH;
                    cropped = fullImage.crop({
                        x: Math.round(rect.x * scaleX),
                        y: Math.round(rect.y * scaleY),
                        width: Math.max(1, Math.round(rect.width * scaleX)),
                        height: Math.max(1, Math.round(rect.height * scaleY))
                    });
                    return [2 /*return*/, { success: true, dataUrl: cropped.toDataURL() }];
                }
                return [2 /*return*/, { success: true, dataUrl: fullImage.toDataURL() }];
            case 2:
                error_5 = _b.sent();
                return [2 /*return*/, { success: false, error: String(error_5) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('files:writeTempFile', function (_, dataUrl, ext) { return __awaiter(void 0, void 0, void 0, function () {
    var filename, filePath, base64Data, error_6;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                filename = "bashgym_screenshot_".concat(Date.now(), ".").concat(ext);
                filePath = path.join(os.tmpdir(), filename);
                base64Data = dataUrl.replace(/^data:[^;]+;base64,/, '');
                return [4 /*yield*/, fs.promises.writeFile(filePath, Buffer.from(base64Data, 'base64'))];
            case 1:
                _a.sent();
                return [2 /*return*/, { success: true, path: filePath }];
            case 2:
                error_6 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_6) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('files:stat', function (_, filePath) { return __awaiter(void 0, void 0, void 0, function () {
    var stats, error_7;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, fs.promises.stat(filePath)];
            case 1:
                stats = _a.sent();
                return [2 /*return*/, {
                        success: true,
                        stats: {
                            isFile: stats.isFile(),
                            isDirectory: stats.isDirectory(),
                            size: stats.size,
                            modified: stats.mtimeMs,
                            created: stats.birthtimeMs
                        }
                    }];
            case 2:
                error_7 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_7) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
// Clipboard handlers — native Electron clipboard is reliable; navigator.clipboard.write() is not
ipcMain.handle('clipboard:writeImage', function (_, dataUrl) {
    try {
        var img = nativeImage.createFromDataURL(dataUrl);
        clipboard.writeImage(img);
        return { success: true };
    }
    catch (error) {
        return { success: false, error: String(error) };
    }
});
ipcMain.handle('clipboard:writeText', function (_, text) {
    try {
        clipboard.writeText(text);
        return { success: true };
    }
    catch (error) {
        return { success: false, error: String(error) };
    }
});
// Credential storage handlers — encrypt/decrypt via Electron safeStorage (OS-level encryption)
function safeCredentialPath(key) {
    var sanitized = key.replace(/[^a-zA-Z0-9_-]/g, '');
    if (!sanitized)
        throw new Error('Invalid credential key');
    var resolved = path.join(credentialsDir, sanitized);
    if (!resolved.startsWith(credentialsDir))
        throw new Error('Invalid credential path');
    return resolved;
}
ipcMain.handle('credentials:store', function (_, key, value) { return __awaiter(void 0, void 0, void 0, function () {
    var encrypted, error_8;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 3, , 4]);
                if (!safeStorage.isEncryptionAvailable()) {
                    return [2 /*return*/, { success: false, error: 'Encryption is not available on this system' }];
                }
                return [4 /*yield*/, fs.promises.mkdir(credentialsDir, { recursive: true })];
            case 1:
                _a.sent();
                encrypted = safeStorage.encryptString(value);
                return [4 /*yield*/, fs.promises.writeFile(safeCredentialPath(key), encrypted)];
            case 2:
                _a.sent();
                return [2 /*return*/, { success: true }];
            case 3:
                error_8 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_8) }];
            case 4: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('credentials:read', function (_, key) { return __awaiter(void 0, void 0, void 0, function () {
    var encrypted, value, error_9;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                if (!safeStorage.isEncryptionAvailable()) {
                    return [2 /*return*/, { success: false, error: 'Encryption is not available on this system' }];
                }
                return [4 /*yield*/, fs.promises.readFile(safeCredentialPath(key))];
            case 1:
                encrypted = _a.sent();
                value = safeStorage.decryptString(encrypted);
                return [2 /*return*/, { success: true, value: value }];
            case 2:
                error_9 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_9) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
ipcMain.handle('credentials:delete', function (_, key) { return __awaiter(void 0, void 0, void 0, function () {
    var error_10;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, fs.promises.unlink(safeCredentialPath(key))];
            case 1:
                _a.sent();
                return [2 /*return*/, { success: true }];
            case 2:
                error_10 = _a.sent();
                return [2 /*return*/, { success: false, error: String(error_10) }];
            case 3: return [2 /*return*/];
        }
    });
}); });
