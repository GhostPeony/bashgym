# SSH Device Management: Plug-and-Play Remote Training

**Date**: 2026-03-20
**Status**: Approved
**Context**: Users need to configure SSH training targets through the frontend UI, not `.env` files. Devices should be discoverable from `~/.ssh/config`, testable via preflight, and selectable for training. Works for any SSH-accessible GPU machine.

---

## Scope

1. **Device Registry** — JSON-backed device storage at `~/.bashgym/devices.json` with CRUD API
2. **SSH Config Discovery** — Parse `~/.ssh/config` to find candidate devices
3. **Enhanced Preflight** — Gather GPU model, VRAM, CUDA version, hostname, OS during connection test
4. **Frontend Device Manager** — UI for discovering, adding, testing, and selecting devices
5. **Training Integration** — Training requests specify a `device_id`, trainer resolves to device config

---

## 1. Device Registry

### Storage: `~/.bashgym/devices.json`

```json
[
  {
    "id": "dgx-spark-01",
    "name": "DGX Spark",
    "host": "192.168.50.173",
    "port": 22,
    "username": "ponyo",
    "key_path": "~/.ssh/id_ed25519",
    "work_dir": "~/bashgym-training",
    "is_default": true,
    "added_at": "2026-03-20T10:00:00Z",
    "last_seen": "2026-03-20T12:30:00Z",
    "capabilities": {
      "python_version": "Python 3.12.3",
      "cuda_version": "12.4",
      "gpus": [{"name": "GH200", "vram_total_gb": 96, "vram_free_gb": 80}],
      "disk_free_gb": 473.0,
      "hostname": "dgx-spark",
      "os": "Linux 6.5.0"
    }
  }
]
```

### New file: `bashgym/api/device_routes.py`

| Method | Route | Purpose |
|--------|-------|---------|
| `GET /api/devices` | List all devices with last-known capabilities |
| `POST /api/devices` | Add a device (host, username, port, key_path, work_dir). Rejects duplicates (same host+username) |
| `PUT /api/devices/{id}` | Update a device |
| `DELETE /api/devices/{id}` | Remove a device |
| `POST /api/devices/{id}/preflight` | Run full preflight + capability scan (10s connection timeout) |
| `POST /api/devices/{id}/set-default` | Set as default training target |
| `POST /api/devices/discover` | Parse `~/.ssh/config`, return candidates |

### New file: `bashgym/device_registry.py`

Lives at package root (not in `bashgym/api/`) — pure data layer with no HTTP concerns, consistent with how `remote_trainer.py` lives in `bashgym/gym/` not `bashgym/api/`.

Handles:
- Load/save `devices.json` with `asyncio.Lock` for concurrent access safety (single process, multiple async handlers)
- Generate device IDs: `slugify(name)-{uuid4()[:8]}` (unique per creation, not deterministic)
- CRUD operations on the device list
- Get default device
- Merge capability data after preflight
- Normalize key paths on save: expand `~` via `Path.expanduser()`, convert `%USERPROFILE%` on Windows, store with forward slashes

### Auto-import from `.env`

On first API startup, if `devices.json` doesn't exist but `SSH_REMOTE_ENABLED=true` in env:
1. Create a device entry from `SSH_REMOTE_HOST`, `SSH_REMOTE_USER`, `SSH_REMOTE_PORT`, `SSH_REMOTE_KEY_PATH`, `SSH_REMOTE_WORK_DIR`
2. Set it as default
3. Save to `devices.json`

After import, the env vars still work as a fallback but `devices.json` takes precedence.

### Training integration

`TrainingRequest` gains an optional `device_id: Optional[str]` field. The **route handler in `routes.py`** (not the trainer) resolves device_id to SSHConfig:

1. If `device_id` is provided, resolve that device from registry
2. Else if registry has a default device, use that
3. Else fall back to `get_settings().ssh` (env var backward compat)
4. Construct `SSHConfig` from the resolved source
5. Store on `app.state.trainer.ssh_config` (new field) before calling train method

`_train_with_remote_ssh()` changes to use `self.ssh_config` if set, otherwise falls back to `get_settings().ssh`. This keeps the trainer decoupled from the device registry — the route handler owns the resolution logic.

---

## 2. SSH Config Discovery

### New file: `bashgym/device_discovery.py`

Lives at package root (not in `bashgym/api/`) — pure parsing logic with no HTTP concerns.

Parses `~/.ssh/config` (standard OpenSSH format) to find candidate devices.

**Parsing logic:**
1. Read `~/.ssh/config` (platform-aware: `%USERPROFILE%\.ssh\config` on Windows, `~/.ssh/config` on Unix)
2. Handle one level of `Include` directives (e.g., `Include ~/.ssh/config.d/*`) — expand globs and parse included files
3. Parse each `Host` block, extracting: `Host` (alias), `HostName`, `User`, `Port`, `IdentityFile`
4. Scope `IdentityFile` to the current `Host` block (don't inherit global defaults into block-specific entries)
5. Filter out: wildcard entries (`Host *`), localhost aliases, GitHub/GitLab/Bitbucket hosts
6. Cross-reference with `devices.json` to flag already-added devices

**Out of scope:** `Match` blocks (uncommon for device discovery).

**Response format from `POST /api/devices/discover`:**

```json
{
  "candidates": [
    {
      "ssh_alias": "dgx-spark",
      "host": "192.168.50.173",
      "username": "ponyo",
      "port": 22,
      "key_path": "~/.ssh/id_ed25519",
      "already_added": true,
      "existing_device_id": "dgx-spark-01"
    },
    {
      "ssh_alias": "gpu-server",
      "host": "10.0.1.50",
      "username": "admin",
      "port": 22,
      "key_path": "~/.ssh/id_rsa",
      "already_added": false,
      "existing_device_id": null
    }
  ],
  "ssh_config_path": "C:\\Users\\Cade\\.ssh\\config"
}
```

**Edge cases:**
- SSH config doesn't exist → return empty candidates with message
- Entries without `HostName` → use `Host` value as hostname (SSH default behavior)
- Entries without `User` → leave username empty, let user fill in
- Multiple `IdentityFile` directives → use the block-scoped one (not global), or first if multiple in same block

---

## 3. Enhanced Preflight (Capability Scan)

### Extended SSH commands

Run during `POST /api/devices/{id}/preflight` in a single SSH session:

| Command | Field | Parsing |
|---------|-------|---------|
| `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits` | gpus | Split CSV rows, each → `{name, vram_total_gb, vram_free_gb}` |
| `nvidia-smi \| head -3` | cuda_version | Regex `CUDA Version:\s*([\d.]+)` from header |
| `hostname` | hostname | Raw string, stripped |
| `uname -sr` | os | Raw string, stripped |
| `python3 --version` | python_version | Already implemented |
| `df -BG {work_dir}` | disk_free_gb | Already implemented |

### Extended PreflightResult

```python
@dataclass
class PreflightResult:
    ok: bool
    python_version: str
    disk_free_gb: float
    error: Optional[str]
    # New fields
    hostname: Optional[str] = None
    os_info: Optional[str] = None
    cuda_version: Optional[str] = None
    gpus: Optional[List[Dict[str, Any]]] = None
```

### Capability persistence

After successful preflight, the device route handler:
1. Updates the device's `capabilities` dict in `devices.json`
2. Sets `last_seen` to current timestamp
3. Returns the full device object with updated capabilities

GPU commands failing (no nvidia-smi) is not a preflight failure — just means no GPU info. The `ok` field reflects Python + disk checks only.

---

## 4. Frontend

### 4a. New component: `DeviceManager.tsx`

Replaces the hardcoded "DGX Spark" button area in TrainingConfig. Contains:

- **Device list** — cards for each configured device
- **Add Device button** — opens inline form (host, username, port, key path, work dir, name)
- **Discover from SSH Config button** — calls `/api/devices/discover`, shows candidates as selectable list
- **Empty state** — "No remote devices configured. Add one or discover from SSH config."

### 4b. Device card display

Each device card shows:
- **Name + host** (e.g., "DGX Spark — 192.168.50.173")
- **GPU summary** (e.g., "GH200 96GB, CUDA 12.4") or "No GPU info" if not scanned
- **Disk free** (e.g., "473 GB free")
- **Connection status** — green/yellow/red dot based on last preflight result
- **Last seen** timestamp
- **Actions**: Test Connection, Set Default, Edit, Remove

### 4c. Training Config integration

The backend selector changes from three buttons to:

- **Local** — trains on local GPU (same as now)
- **Remote Device** — shows dropdown of configured devices (default pre-selected). If no devices configured, shows "Add a device" link. Sets `use_remote_ssh: true` and `device_id: selectedDevice.id`
- **NeMo Cloud** — same as now (if enabled)

### 4d. New store: `deviceStore.ts`

Zustand store with:

```typescript
interface DeviceStore {
  devices: Device[]
  defaultDeviceId: string | null
  loading: boolean

  fetchDevices: () => Promise<void>
  addDevice: (device: NewDevice) => Promise<Device>
  updateDevice: (id: string, updates: Partial<Device>) => Promise<void>
  removeDevice: (id: string) => Promise<void>
  runPreflight: (id: string) => Promise<PreflightResult>
  setDefault: (id: string) => Promise<void>
  discoverFromSSHConfig: () => Promise<SSHCandidate[]>
}
```

### 4e. API client additions in `api.ts`

```typescript
export const deviceApi = {
  list: () => request<Device[]>('/devices'),
  add: (device: NewDevice) => request<Device>('/devices', { method: 'POST', body: device }),
  update: (id: string, device: Partial<Device>) => request<Device>(`/devices/${id}`, { method: 'PUT', body: device }),
  remove: (id: string) => request<void>(`/devices/${id}`, { method: 'DELETE' }),
  preflight: (id: string) => request<PreflightResult>(`/devices/${id}/preflight`, { method: 'POST' }),
  setDefault: (id: string) => request<void>(`/devices/${id}/set-default`, { method: 'POST' }),
  discover: () => request<DiscoverResult>('/devices/discover'),
}
```

---

## 5. Schema Updates

### `bashgym/api/schemas.py`

Add `device_id` to `TrainingRequest`:

```python
device_id: Optional[str] = Field(None, description="Target device ID for remote SSH training (uses default if omitted)")
```

### `bashgym/api/device_schemas.py`

New Pydantic models:

```python
class DeviceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    host: str = Field(..., min_length=1, description="Hostname or IP address")
    port: int = Field(22, ge=1, le=65535)
    username: str = Field(..., min_length=1)
    key_path: str = Field("~/.ssh/id_rsa", description="Path to SSH private key")
    work_dir: str = Field("~/bashgym-training", description="Remote working directory")

class DeviceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    host: Optional[str] = Field(None, min_length=1)
    port: Optional[int] = Field(None, ge=1, le=65535)
    username: Optional[str] = Field(None, min_length=1)
    key_path: Optional[str] = None
    work_dir: Optional[str] = None

class DeviceResponse(BaseModel):
    id: str
    name: str
    host: str
    port: int
    username: str
    key_path: str
    work_dir: str
    is_default: bool
    added_at: str
    last_seen: Optional[str]
    capabilities: Optional[Dict[str, Any]]

class SSHCandidate(BaseModel):
    ssh_alias: str
    host: str
    username: Optional[str]
    port: int = 22
    key_path: Optional[str]
    already_added: bool
    existing_device_id: Optional[str]

class DiscoverResponse(BaseModel):
    candidates: List[SSHCandidate]
    ssh_config_path: str
```

---

## Files Modified

| File | Change |
|------|--------|
| **New:** `bashgym/device_registry.py` | Device JSON storage, CRUD, asyncio.Lock, path normalization |
| **New:** `bashgym/device_discovery.py` | SSH config parser with Include support |
| **New:** `bashgym/api/device_routes.py` | REST endpoints for device management |
| **New:** `bashgym/api/device_schemas.py` | Pydantic models with field validation |
| **New:** `frontend/src/components/training/DeviceManager.tsx` | Device list, cards, add/discover UI |
| **New:** `frontend/src/stores/deviceStore.ts` | Zustand store for device state |
| `bashgym/api/routes.py` | Register device router, resolve device_id → SSHConfig in training handler |
| `bashgym/api/schemas.py` | Add `device_id` to TrainingRequest |
| `bashgym/gym/remote_trainer.py` | Extend PreflightResult, add GPU/system commands, 10s connection timeout |
| `bashgym/gym/trainer.py` | Accept ssh_config from caller in `_train_with_remote_ssh()` |
| `frontend/src/services/api.ts` | Add `deviceApi` client |
| `frontend/src/components/training/TrainingConfig.tsx` | Replace DGX Spark button with DeviceManager |
| `frontend/src/stores/trainingStore.ts` | Add `deviceId` to training config |

## What This Does NOT Change

- Local training (no SSH)
- NeMo Cloud training
- Remote training execution (RemoteTrainer internals unchanged, just gets config from a different source)
- Trace capture pipeline
- Any other frontend components

---

## Migration

- Existing `.env` SSH config auto-imports as a device on first startup
- No breaking changes — `use_remote_ssh: true` without `device_id` still works (uses default device)
- Old env vars remain as fallback if no devices.json exists
- `GET /api/ssh/preflight` kept as backward-compatible alias — internally resolves to default device preflight
- Frontend `SystemInfoPanel` updated to use device-based preflight for the default device instead of the old `sshApi.preflight()`
