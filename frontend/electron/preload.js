import { contextBridge, ipcRenderer } from 'electron';
// Expose protected methods to renderer
contextBridge.exposeInMainWorld('bashgym', {
    terminal: {
        create: function (id, cwd) { return ipcRenderer.invoke('terminal:create', id, cwd); },
        write: function (id, data) { return ipcRenderer.invoke('terminal:write', id, data); },
        resize: function (id, cols, rows) { return ipcRenderer.invoke('terminal:resize', id, cols, rows); },
        kill: function (id) { return ipcRenderer.invoke('terminal:kill', id); },
        onData: function (id, callback) {
            var channel = "terminal:data:".concat(id);
            var listener = function (_, data) { return callback(data); };
            ipcRenderer.on(channel, listener);
            return function () { return ipcRenderer.removeListener(channel, listener); };
        },
        onExit: function (id, callback) {
            var channel = "terminal:exit:".concat(id);
            var listener = function (_, exitCode) { return callback(exitCode); };
            ipcRenderer.on(channel, listener);
            return function () { return ipcRenderer.removeListener(channel, listener); };
        }
    },
    theme: {
        get: function () { return ipcRenderer.invoke('theme:get'); },
        set: function (theme) { return ipcRenderer.invoke('theme:set', theme); }
    },
    system: {
        info: function () { return ipcRenderer.invoke('system:info'); }
    },
    api: {
        fetch: function (url, options) { return ipcRenderer.invoke('api:fetch', url, options); }
    },
    files: {
        readDirectory: function (path) { return ipcRenderer.invoke('files:readDirectory', path); },
        getHomeDirectory: function () { return ipcRenderer.invoke('files:getHomeDirectory'); },
        getParentDirectory: function (path) { return ipcRenderer.invoke('files:getParentDirectory', path); },
        readFile: function (path) { return ipcRenderer.invoke('files:readFile', path); },
        exists: function (path) { return ipcRenderer.invoke('files:exists', path); },
        stat: function (path) { return ipcRenderer.invoke('files:stat', path); },
        writeTempFile: function (dataUrl, ext) { return ipcRenderer.invoke('files:writeTempFile', dataUrl, ext); }
    },
    browser: {
        screenshot: function (webContentsId, rect) {
            return ipcRenderer.invoke('browser:screenshot', webContentsId, rect);
        }
    },
    clipboard: {
        writeImage: function (dataUrl) { return ipcRenderer.invoke('clipboard:writeImage', dataUrl); },
        writeText: function (text) { return ipcRenderer.invoke('clipboard:writeText', text); }
    },
    credentials: {
        store: function (key, value) { return ipcRenderer.invoke('credentials:store', key, value); },
        read: function (key) { return ipcRenderer.invoke('credentials:read', key); },
        delete: function (key) { return ipcRenderer.invoke('credentials:delete', key); },
    },
    window: {
        minimize: function () { return ipcRenderer.invoke('window:minimize'); },
        maximize: function () { return ipcRenderer.invoke('window:maximize'); },
        close: function () { return ipcRenderer.invoke('window:close'); },
        isMaximized: function () { return ipcRenderer.invoke('window:isMaximized'); },
        onAppKeydown: function (callback) {
            var listener = function (_, data) { return callback(data); };
            ipcRenderer.on('app-keydown', listener);
            return function () { return ipcRenderer.removeListener('app-keydown', listener); };
        }
    }
});
