export interface TerminalAPI {
    create: (id: string, cwd?: string) => Promise<{
        success: boolean;
        id?: string;
        error?: string;
    }>;
    write: (id: string, data: string) => Promise<boolean>;
    resize: (id: string, cols: number, rows: number) => Promise<boolean>;
    kill: (id: string) => Promise<boolean>;
    onData: (id: string, callback: (data: string) => void) => () => void;
    onExit: (id: string, callback: (exitCode: number) => void) => () => void;
}
export interface ThemeAPI {
    get: () => Promise<'light' | 'dark'>;
    set: (theme: 'light' | 'dark') => Promise<'light' | 'dark'>;
}
export interface SystemAPI {
    info: () => Promise<{
        platform: string;
        arch: string;
        nodeVersion: string;
        electronVersion: string;
        cwd: string;
    }>;
}
export interface ApiProxy {
    fetch: (url: string, options?: RequestInit) => Promise<{
        ok: boolean;
        status?: number;
        data?: any;
        error?: string;
    }>;
}
export interface FileInfo {
    name: string;
    path: string;
    type: 'file' | 'directory';
    size?: number;
    modified?: number;
}
export interface FilesAPI {
    readDirectory: (path: string) => Promise<{
        success: boolean;
        files?: FileInfo[];
        error?: string;
    }>;
    getHomeDirectory: () => Promise<string>;
    getParentDirectory: (path: string) => Promise<string>;
    readFile: (path: string) => Promise<{
        success: boolean;
        content?: string;
        error?: string;
    }>;
    exists: (path: string) => Promise<boolean>;
    stat: (path: string) => Promise<{
        success: boolean;
        stats?: {
            isFile: boolean;
            isDirectory: boolean;
            size: number;
            modified: number;
            created: number;
        };
        error?: string;
    }>;
    writeTempFile: (dataUrl: string, ext: string) => Promise<{
        success: boolean;
        path?: string;
        error?: string;
    }>;
}
export interface WindowAPI {
    minimize: () => Promise<void>;
    maximize: () => Promise<void>;
    close: () => Promise<void>;
    isMaximized: () => Promise<boolean>;
    onAppKeydown: (callback: (data: {
        key: string;
        ctrlKey: boolean;
        shiftKey: boolean;
    }) => void) => () => void;
}
export interface BrowserAPI {
    screenshot: (webContentsId: number, rect?: {
        x: number;
        y: number;
        width: number;
        height: number;
        vpW: number;
        vpH: number;
    }) => Promise<{
        success: boolean;
        dataUrl?: string;
        error?: string;
    }>;
}
export interface ClipboardAPI {
    writeImage: (dataUrl: string) => Promise<{
        success: boolean;
        error?: string;
    }>;
    writeText: (text: string) => Promise<{
        success: boolean;
        error?: string;
    }>;
}
export interface CredentialsAPI {
    store: (key: string, value: string) => Promise<{
        success: boolean;
        error?: string;
    }>;
    read: (key: string) => Promise<{
        success: boolean;
        value?: string;
        error?: string;
    }>;
    delete: (key: string) => Promise<{
        success: boolean;
        error?: string;
    }>;
}
export interface BashGymAPI {
    terminal: TerminalAPI;
    theme: ThemeAPI;
    system: SystemAPI;
    api: ApiProxy;
    files: FilesAPI;
    window: WindowAPI;
    browser: BrowserAPI;
    clipboard: ClipboardAPI;
    credentials: CredentialsAPI;
}
declare global {
    interface Window {
        bashgym: BashGymAPI;
    }
}
