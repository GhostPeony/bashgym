# Contributing to Bash Gym

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

# Python dependencies
pip install -r requirements.txt
pip install -r requirements-training.txt  # optional, for ML work

# Frontend dependencies
cd frontend && npm install && cd ..

# Environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
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
5. Wait for review

### Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep the subject line under 72 characters
- Reference issues where applicable: "Fix #123"

---

## Project Architecture

See the [Architecture Overview](README.md#architecture-overview) in the README for the full system diagram and layer descriptions. Key directories:

| Directory | Language | Purpose |
|-----------|----------|---------|
| `bashgym/` | Python | Backend — Arena, Judge, Factory, Gym, API |
| `frontend/src/` | TypeScript | React + Electron UI |
| `assistant/` | Go | Peony chat assistant (Discord/Telegram) |
| `tests/` | Python | Test suite |
| `docs/` | Markdown | Documentation |

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
