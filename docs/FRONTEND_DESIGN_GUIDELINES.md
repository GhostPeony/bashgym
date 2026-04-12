# BashGym Frontend Design Guidelines

Reference for maintaining visual and interaction consistency across the bashgym UI.

## Stack

- **React 19** + **TypeScript**
- **Tailwind CSS** (custom design tokens, no component library)
- **Zustand 5** for state management
- **Recharts** for charting
- **Lucide React** for icons
- **XFlow/React** for DAG/pipeline visualization
- **ReactMarkdown + remark-gfm** for rendering markdown content

## Design Tokens

The app uses CSS custom properties (defined in `frontend/src/index.css` or Tailwind config) with a dark-first palette. Reference these via Tailwind classes:

| Token | Class | Usage |
|---|---|---|
| `--background` | `bg-background` | Page background |
| `--background-card` | `bg-background-card` | Card/panel surfaces |
| `--background-secondary` | `bg-background-secondary` | Hover states, subtle fills |
| `--text-primary` | `text-text-primary` | Primary text |
| `--text-secondary` | `text-text-secondary` | Secondary/label text |
| `--text-muted` | `text-text-muted` | Hints, timestamps, metadata |
| `--accent` | `text-accent`, `bg-accent` | Primary action color |
| `--accent-dark` | `border-accent-dark` | Active tab borders |
| `--accent-light` | `bg-accent-light` | Active tab background tint |
| `--border` | `border-border` | Standard borders |
| `--border-subtle` | `border-border-subtle` | Subtle/inner borders |
| `--status-success` | `text-status-success` | Success states |
| `--status-error` | `text-status-error` | Error states |
| `--status-warning` | `text-status-warning` | Warning states |

## Component Patterns

### Cards

```tsx
<div className="border-brutal shadow-brutal-sm rounded-brutal bg-background-card p-4">
  {/* content */}
</div>
```

### Buttons

| Variant | Class | When to use |
|---|---|---|
| Primary | `btn-primary` | Main actions (Create, Upload, Start) |
| Ghost | `btn-ghost` | Secondary actions (Disconnect, Cancel) |
| Icon | `btn-icon` | Toolbar icons (Refresh, Settings) |
| Destructive | `btn-ghost text-status-error` | Delete, Disconnect |

### Form inputs

```tsx
<label className="block text-xs font-mono text-text-secondary uppercase tracking-widest mb-1">
  Label Text
</label>
<input className="input w-full text-sm font-mono" />
```

### Tags / badges

```tsx
<span className="tag">
  <span className="flex items-center gap-1">
    <Sparkles className="w-3 h-3" />
    Pro
  </span>
</span>
```

### Status indicators

- Green dot: `w-2 h-2 rounded-full bg-status-success`
- Yellow dot: `w-2 h-2 rounded-full bg-status-warning`
- Red dot: `w-2 h-2 rounded-full bg-status-error`

### Error/success banners

```tsx
{/* Error */}
<div className="p-3 border-2 border-status-error rounded-brutal text-sm text-status-error">
  {error}
</div>

{/* Success */}
<div className="p-3 border-2 border-status-success rounded-brutal text-sm text-status-success">
  {message}
</div>
```

### Tabs

The standard tab pattern (used in HFDashboard, TrainingConfig, etc.):

```tsx
<div className="flex gap-1">
  {tabs.map((tab) => (
    <button
      key={tab.id}
      onClick={() => setActiveTab(tab.id)}
      className={clsx(
        'flex items-center gap-2 px-4 py-2 text-sm font-mono border-2 rounded-brutal transition-press',
        activeTab === tab.id
          ? 'bg-accent text-white border-accent-dark shadow-brutal-sm'
          : 'text-text-secondary border-border hover:text-text-primary hover:bg-background-secondary hover-press'
      )}
    >
      <tab.icon className="w-4 h-4" />
      {tab.label}
    </button>
  ))}
</div>
```

### Loading states

```tsx
<div className="flex justify-center py-8">
  <Loader2 className="w-6 h-6 animate-spin text-text-secondary" />
</div>
```

### Empty states

```tsx
<div className="text-center py-8 text-text-secondary">
  <IconComponent className="w-12 h-12 mx-auto mb-3 opacity-50" />
  <p>Descriptive message about what to do next.</p>
</div>
```

### Tables

```tsx
<div className="divide-y divide-border">
  {items.map((item) => (
    <div key={item.id} className="px-3 py-2 flex items-center justify-between text-sm">
      <span className="font-mono text-text-primary">{item.name}</span>
      <span className="text-text-muted font-mono text-xs">{item.meta}</span>
    </div>
  ))}
</div>
```

### Modals

Use the existing `<Modal>` component from `components/common/Modal.tsx`, or for inline overlays:

```tsx
<div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
  <div className="bg-background-card border-2 border-border rounded-brutal p-6 w-full max-w-lg shadow-brutal" onClick={e => e.stopPropagation()}>
    {/* content */}
  </div>
</div>
```

## Layout Structure

### Dashboard navigation

```
┌──────────┬──────────────────────────────────────┐
│          │  Header (title + actions)             │
│  Sidebar │──────────────────────────────────────│
│  (nav)   │  Tabs                                │
│          │──────────────────────────────────────│
│          │                                      │
│          │  Tab content (scrollable)             │
│          │                                      │
└──────────┴──────────────────────────────────────┘
```

- Sidebar: `Sidebar.tsx` — nav items map to `overlayView` values in `useUIStore`
- Main content: rendered by `MainLayout.tsx` based on `overlayView`
- Each dashboard (Training, HuggingFace, etc.) manages its own internal tabs

### HFDashboard tab layout

7 tabs: Training | Spaces | Datasets | Models | Buckets | Research | Traces

Each tab renders in the `flex-1 overflow-auto` content area below the header.

## Typography

- **Headings**: `font-brand` (the project's display font)
- **Body/labels**: `font-mono` for technical content, system font for prose
- **Sizes**: `text-xs` for metadata, `text-sm` for body, `text-lg` for section headers, `text-xl` for page titles
- **Uppercase labels**: `text-xs font-mono uppercase tracking-widest` for form labels

## Icons

Always use Lucide React. Import individually:

```tsx
import { Server, Database, Loader2 } from 'lucide-react'
```

Standard sizes:
- `w-3 h-3` — inline with text
- `w-4 h-4` — buttons, tabs, list items
- `w-5 h-5` — section headers
- `w-8 h-8` — page-level icons
- `w-12 h-12` — empty state illustrations

## State Management

Use Zustand stores (`frontend/src/stores/`). Existing stores:

- `useUIStore` — sidebar nav, overlay views, modal state
- `useTrainingStore` — training run state, metrics
- `cascadeStore` — cascade training state

For new features, prefer local `useState` over global stores unless the state needs to be shared across multiple components.

## API Integration

All API calls go through `frontend/src/services/api.ts`. Pattern:

```tsx
const result = await hfApi.someMethod(params)
if (result.ok && result.data) {
  // success
} else {
  setError(result.error || 'Something went wrong')
}
```

The `request<T>()` helper handles JSON parsing, error extraction, and returns `{ ok: boolean, data?: T, error?: string }`.

## File Organization

```
frontend/src/
├── components/
│   ├── huggingface/     # HF dashboard + all tabs
│   ├── training/        # Training dashboard + config + metrics
│   ├── factory/         # Data factory + designer
│   ├── common/          # Shared components (Button, Modal, etc.)
│   └── layout/          # MainLayout, Sidebar, NavigationBar
├── services/
│   ├── api.ts           # All API client methods (namespaced objects)
│   └── websocket.ts     # WebSocket client for live streaming
├── stores/              # Zustand stores
└── pages/               # Page-level components (if distinct from dashboard views)
```

New features should follow this structure. Tab components live in the same directory as their parent dashboard.
