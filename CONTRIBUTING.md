# Contributing to BashGym

Thanks for your interest in contributing! This guide will help you get set up.

---

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (LTS recommended)
- Git
- CUDA-capable GPU (optional — only needed for training)

### Clone and Install

```bash
git clone https://github.com/GhostPeony/bashgym.git
cd bashgym

# Isolated Python environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Package, CLI, and development dependencies
python -m pip install -e ".[dev]"
python -m pip install -e ".[training,dev]"  # optional, for ML work

# Frontend dependencies
cd frontend && npm ci && cd ..

# Environment
cp .env.example .env
# Add only provider credentials needed by the feature you are testing

# Verify the package and no-GPU AutoResearch control path
bashgym --help
bashgym campaign control-smoke --json
```

When testing authenticated campaign routes, create a disposable workspace-scoped
operator through the supported secret-reference path rather than placing raw
campaign tokens in commands or fixtures:

```bash
bashgym campaign provision-local-operator \
  --workspace-id <test-workspace> \
  --credential-ref BASHGYM_CAMPAIGN_TEST_OPERATOR --json
```

### Running Locally

```bash
# macOS / Linux
./dev.sh

# Windows (PowerShell)
.\dev.ps1

# Or start manually
python run_backend.py           # API on port 8003
cd frontend && npm run dev      # Vite on port 5173
```

---

## Code Style

### Python

- Python 3.10+ with type hints
- Formatter: **black** (100 char line length)
- Linter: **ruff**
- Dataclasses for configuration and result types
- Async where beneficial (httpx, training callbacks)

### TypeScript / React

- Existing ESLint configuration
- Functional components with hooks
- Zustand for state management
- Tailwind CSS for styling (follow the Botanical Brutalism design system)

### Design System

The UI follows **Botanical Brutalism** — structural honesty from brutalism (hard borders, monospace, grid layouts) tempered by organic warmth (nature-derived colors, serif brand typography). See the global `CLAUDE.md` for the full design token reference. Key rules:

- Borders: 2px solid with offset shadows (3-6px, no blur)
- Border radius: 0-4px in light mode, 8-16px in dark mode
- Typography: Cormorant Garamond (brand), Inter (body), JetBrains Mono (code/labels)
- Backgrounds: Warm parchment, never pure white

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=bashgym --cov-report=html

# Specific module
pytest tests/test_data_factory.py -v

# AutoResearch setup and managed-desktop authority contracts
pytest tests/campaigns/test_guided_setup.py tests/api/test_campaign_setup_routes.py tests/api/test_campaign_agent_routes.py -q

# Frontend suite
cd frontend
npm test

# Focused frontend files (exact basename or path; unmatched filters fail)
npm test -- GuidedAutoResearchSetup.test.ts campaignApi.test.ts campaignBridgeSecurity.test.ts

# Frontend contracts
npm run typecheck
npm run lint
```

---

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with clear, focused commits
3. Ensure tests pass (`pytest tests/ -v`)
4. Open a PR with:
   - A clear title describing the change
   - Description of what and why
   - Screenshots for UI changes
   - Documentation updates for public behavior, commands, configuration, or
     compatibility changes
5. Wait for review

### Documentation Changes

- Update the root `README.md` when a public capability or primary workflow
  changes, and update the canonical guide that owns the full procedure.
- Keep short discovery paths in entry-point docs; do not duplicate long command
  references across multiple pages.
- Use generic model, hardware, repository, and filesystem examples. Never commit
  operator names, private hostnames, personal paths, credentials, or a silent
  default model.
- Keep internal implementation plans and agent handoffs in ignored local
  planning files, not under the public `docs/` tree.
- Before opening a PR, run `git diff --check` and verify changed Markdown links.

### AutoResearch Boundaries

- Keep the six setup choices ordered: template, installation, model, data,
  compute, evaluation. The renderer may submit only registered logical IDs; it
  must not discover private paths or assert model or hardware reachability.
- Keep guided setup visible when the API is offline, but read-only. The write
  path is doctor, sealed validation, then campaign creation. Start is a separate
  server-revalidated human decision.
- Treat local/private registered SSH compute as the primary real-training path.
  Do not add a default model, implicit download, model substitution, hosted
  fallback, or required NeMo dependency.
- Campaign-agent authority may activate only after the current managed-desktop
  bootstrap exchange. Do not add renderer-controlled trust flags or expose raw
  session/delivery credentials through a generic renderer route.
- Electron main owns campaign-agent liveness, ephemeral keys, registration,
  claim, reconciliation, and revocation, and the Control Room keeps that state
  visible-disabled. The implemented Codex path launches one dedicated,
  scope-bound child with a fixed two-tool read-only proxy; ordinary terminals
  remain ineligible. Hermes parity and mutation-capable tools are not claimed.

### Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep the subject line under 72 characters
- Reference issues where applicable: "Fix #123"

---

## Project Architecture

See the [Project Structure](README.md#project-structure) in the README for the
current package map. Key directories:

| Directory       | Language   | Purpose                                   |
| --------------- | ---------- | ----------------------------------------- |
| `bashgym/`      | Python     | Backend — Arena, Judge, Factory, Gym, API |
| `frontend/src/` | TypeScript | React + Electron UI                       |
| `assistant/`    | Go         | Peony chat assistant (Discord/Telegram)   |
| `tests/`        | Python     | Test suite                                |
| `docs/`         | Markdown   | Durable public documentation              |

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
