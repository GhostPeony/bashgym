# Home Screen & Onboarding Design

> Make Bash Gym easier to pick up for both experienced users and newcomers

**Date:** 2026-01-27
**Status:** Approved

---

## Problem Statement

The current UI lacks a centered home page. Features are spread across 10+ sidebar items with no clear entry point or guidance. New users don't understand the flywheel concept or how the pieces connect.

## Design Principles

- **Coding-centric**: The Workspace is the primary activity; Training and Data Factory are supporting tools
- **Learn by doing**: Tutorial happens in the real UI, not a separate wizard
- **Progressive disclosure**: Show 3 core spaces upfront, tuck advanced features in Settings

---

## The Flywheel Concept

Visual asset: `/frontend/public/flywheel-bg.png`

```
EXECUTE → VERIFY → SYNTHESIZE → TRAIN → DEPLOY
```

Mapping to spaces:
- **Execute/Verify** → Workspace (where traces are captured)
- **Synthesize** → Data Factory (turn traces into training data)
- **Train/Deploy** → Training Center

---

## 1. Home Screen

The landing page that serves as both introduction and dashboard.

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    [Flywheel Infographic]                   │
│         EXECUTE → VERIFY → SYNTHESIZE → TRAIN → DEPLOY      │
│                                                             │
│      "Turn your coding sessions into smarter AI assistants" │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  WORKSPACE  │  │ DATA FACTORY│  │  TRAINING   │         │
│  │             │  │             │  │             │         │
│  │ Code with   │  │ Transform   │  │ Fine-tune   │         │
│  │ AI assist   │  │ traces into │  │ your own    │         │
│  │             │  │ training    │  │ models      │         │
│  │ 3 sessions  │  │ 127 → 412   │  │ 2 trained   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│                              [Settings ⚙]                   │
└─────────────────────────────────────────────────────────────┘
```

### Behavior

- **New users**: Overlay appears with "First time here? Let us show you around" + **Let's Go** / **Skip** buttons
- **Returning users**: See live stats on each card (sessions, traces → examples, models trained)
- **Navigation**: Click any card to enter that space. Logo returns to Home from anywhere.

---

## 2. Guided Tutorial

A contextual tour with persistent checklist for users who click "Let's Go".

### Checklist Panel

Docked to right side, minimizable to floating button:

```
Getting Started
─────────────────
☑ Welcome to Bash Gym
☐ Install capture hooks
☐ Import your first traces
☐ Generate training examples
☐ Start a training run
☐ View your trained model
```

- Current step highlighted
- Completed steps show green checkmarks
- Can minimize but stays accessible

### Tutorial Steps

| Step | Location | Tooltip Text | Action |
|------|----------|--------------|--------|
| 1. Install hooks | Settings | "Hooks capture your Claude Code sessions automatically. Click to install." | User clicks Install in hooks settings |
| 2. Import traces | Data Factory | "You already have session history! Let's import it." | User clicks Import button |
| 3. Generate examples | Data Factory | "Select traces and click Generate to create training data." | User selects traces, clicks Generate |
| 4. Start training | Training | "Your examples are ready. Configure and start a training run." | User clicks Start Training |
| 5. View model | Models | "Your model is ready! View it in the Models section." | Tutorial complete |

Each tooltip has:
- **"Got it"** - Dismisses tooltip, expects user to perform action
- **"Next"** - Skips to next step without performing action

---

## 3. Simplified Navigation

### Primary Navigation (Always Visible)

| Item | Icon | Description |
|------|------|-------------|
| Home | House/Logo | Returns to home screen |
| Workspace | Terminal | Coding environment (primary, emphasized) |
| Data Factory | Sparkles | Trace import, example generation |
| Training | BarChart | Configure runs, view progress |

### Secondary Features (In Settings)

Accessed via gear icon, organized in tabs or accordion:

- **Traces** - Detailed trace browser and management
- **Models** - Browse trained models, compare, lineage
- **Evaluator** - Run benchmarks against models
- **Router** - Teacher/student routing configuration
- **Guardrails** - Safety rules and filters
- **Profiler** - Performance monitoring
- **HuggingFace** - Cloud training integration
- **Integration** - External service connections

### Status Indicators

Move from sidebar to:
- Minimal status bar at bottom of screen, OR
- Small badges on relevant nav items (e.g., red dot on Training when run active)

---

## 4. Connected Spaces

Each space links to the next stage of the flywheel.

### Workspace

- **Recording indicator**: Shows "Session recording" when hooks active
- **Completion toast**: After session ends, show *"Session captured! 23 tool calls recorded."* with link to Data Factory

### Data Factory

- **Import section**: Shows available Claude Code history to import
- **Traces list**: Displays imported traces with quality scores
- **Generate action**: Clear button to transform traces → training examples
- **Ready prompt**: When examples ready, show *"142 examples ready for training"* with button to Training

### Training

- **Data source**: Shows available training data from Data Factory
- **Progress**: Clear indicators during training run
- **Completion**: *"Model ready!"* with links to test or deploy

---

## What Gets Removed

- `FlywheelMini` component from sidebar bottom
- 10 dashboard items from primary sidebar
- Disconnected navigation requiring feature discovery

---

## Implementation Notes

### New Components Needed

1. `HomeScreen.tsx` - The landing page with infographic and space cards
2. `OnboardingOverlay.tsx` - First-time user prompt
3. `TutorialChecklist.tsx` - Persistent checklist panel
4. `TutorialTooltip.tsx` - Contextual tooltip component
5. `useTutorialProgress.ts` - Hook to track tutorial state (localStorage)

### Files to Modify

- `Sidebar.tsx` - Simplify to 4 primary items + settings
- `MainLayout.tsx` - Add Home as default view, integrate tutorial
- `App.tsx` - Check first-time user state on mount

### Assets Required

- Copy flywheel infographic to `/frontend/public/flywheel-bg.png`

---

## Success Criteria

- [ ] New user can complete tutorial in under 5 minutes
- [ ] User understands the flywheel concept without reading docs
- [ ] Navigation reduced from 10+ items to 4 primary + settings
- [ ] Each space has clear pathway to the next
