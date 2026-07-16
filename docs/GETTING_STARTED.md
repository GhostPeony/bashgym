# Getting Started with Bash Gym

This guide walks you from a fresh install to your first trained model.

---

## Prerequisites

- **Python 3.10+** — Backend and training
- **Node.js 18+** (LTS) — Frontend
- **Git** — Version control
- **CUDA-capable GPU** (optional) — Needed only for real local training. Private hardware over SSH and explicitly selected hosted backends are alternatives.
- **Provider API keys** (optional) — Add credentials only for teacher, augmentation, hosted training, or publication features you use.

---

## Step 1: Install and Configure (10 minutes)

```bash
# Clone the repository
git clone https://github.com/GhostPeony/bashgym.git
cd bashgym

# Create and activate an isolated environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install BashGym and register its CLI
python -m pip install -e .

# Optional real local/private training dependencies
python -m pip install -e ".[training]"

# Install frontend dependencies
cd frontend && npm ci && cd ..

# Create your environment file
cp .env.example .env
```

The copied file contains no credential or model default. Add only the provider
credentials you intend to use. Verify the package and durable AutoResearch
control plane without a GPU or API key:

```bash
bashgym --help
bashgym campaign control-smoke --json
```

---

## Step 2: Install Trace Capture Hooks (2 minutes)

These hooks silently capture sessions from all your AI coding tools as training data:

```bash
# Auto-detect installed tools and install hooks for all of them
python -m bashgym.trace_capture.setup

# Or bulk-import historical sessions (last 60 days)
python -m bashgym.trace_capture.setup import-all --days 60
```

Or use the dashboard **Settings > Agents** tab for one-click installation. Verify by checking that each configured tool shows "Installed."

---

## Step 3: Launch the App

```bash
# Windows (PowerShell) — backend + frontend together
.\dev.ps1

# macOS / Linux
./dev.sh

# For the full desktop experience (any platform)
.\dev.ps1 -Electron    # Windows
./dev.sh --electron     # macOS/Linux

# Docker backend API only (any platform)
docker compose up bashgym-api
```

Open `http://localhost:5173` in your browser (or the Electron window opens automatically).
Docker Compose does not currently package the frontend, so run the Vite or
Electron command separately when using the containerized API.

---

## Step 4: Accumulate Traces (Days 1-7)

Just use Claude Code normally on your projects. Every session is automatically captured. Check the **Home** screen to see your trace count grow, or open the **Traces** browser to see individual sessions with their quality scores.

Each trace is scored on 6 metrics and classified:

| Tier | Threshold | Usage |
|------|-----------|-------|
| **Gold** | >=90% success | Ready for SFT training |
| **Silver** | >=75% success | Good but not great |
| **Bronze** | >=60% success | Acceptable |
| **Failed** | <60% success | Used as negative examples for DPO training |

---

## Step 5: Review and Curate Traces

Open the **Traces** dashboard from the sidebar or home screen:

1. Browse traces by status (Gold/Silver/Bronze/Failed)
2. Filter by repository to focus on specific projects
3. Click a trace to see quality metrics breakdown
4. Manually **promote** good traces to Gold or **demote** bad ones
5. Use **Auto-Classify** to batch-sort traces with configurable thresholds

---

## Step 6: Generate Training Examples

From the **Traces** dashboard:

1. Click **Generate Examples** on individual gold traces, or
2. Open the **Data Factory** and run batch generation across all gold traces

The factory segments multi-task sessions into individual training examples, scrubs PII, and exports in NeMo JSONL format.

---

## Step 7: Train Your First Model

Open the **Training** dashboard:

1. Select **SFT** (Supervised Fine-Tuning) as the strategy
2. Choose a compatible, registered trainable base supported by an installed backend. Pin its immutable revision for a durable campaign; adapters and inference quants are not substitutes. Smaller models train on consumer GPUs, while larger ones use private or explicitly selected hosted targets.
3. Select which repos to train on (or use all gold traces)
4. Click **Start Training**
5. Watch the live loss curve and training logs

Training a small model with LoRA typically takes 30-90 minutes depending on your GPU and trace count. The model auto-exports to GGUF when complete.

### Cloud Training Alternative

No local GPU? Use HuggingFace Cloud Training:

1. Add `HF_TOKEN` to your `.env` file (requires a [HuggingFace Pro subscription](https://huggingface.co/subscribe/pro))
2. Open the **HuggingFace** dashboard from the sidebar
3. Submit a cloud training job — supports T4, A10G, A100, and H100 hardware starting at $0.60/hr

---

## Step 8: Evaluate the Model

Open the **Models** dashboard to see your trained model. From there:

1. Run **Custom Eval** (replays your own traces to test the student)
2. Run **Benchmarks** (HumanEval, MBPP, etc.) for standardized scores
3. Compare against previous models on the **Leaderboard**

---

## Step 9: Deploy and Route

1. Click **Deploy to Ollama** on your model to make it available locally
2. Open the **Router** dashboard
3. Set the strategy to **Progressive Handoff** (starts at 20% student, increases over time)
4. Monitor success rates — the router will only send tasks to the student when it's confident

---

## Step 10: Repeat

Every week, retrain with your accumulated traces. Each cycle produces a better student model that handles more tasks, reducing your API costs further.

---

## Next Steps

- **[Durable AutoResearch](training/autoresearch-campaign.md)** — Run the no-GPU control smoke, bind local/private compute, and execute a real baseline/candidate campaign
- **[Peony Assistant](../README.md#peony-assistant)** — Control Bash Gym via Discord or Telegram
- **[HuggingFace Integration](../README.md#huggingface-integration)** — Cloud training and model sharing
- **[API Reference](API.md)** — Full REST and WebSocket endpoint documentation

---

## Common Issues

**Port 8003 is already in use:**
```bash
# Windows (PowerShell)
Get-Process -Name "python" | Where-Object { $_.CommandLine -like "*uvicorn*" } | Stop-Process

# macOS/Linux
lsof -i :8003 | grep LISTEN
kill -9 <PID>
```

**Frontend can't connect to the API:**
Check that the backend is running. If using a different port, update `frontend/.env.local`:
```
VITE_API_URL=http://localhost:YOUR_PORT/api
VITE_WS_URL=ws://localhost:YOUR_PORT/ws
```

**Hooks show "Not installed":**
```bash
python -m bashgym.trace_capture.setup
```
Or use **Settings > Agents** in the app to install hooks from the UI.

**`npm install` fails:**
The frontend uses `node-pty` which requires native compilation. Ensure you have:
- Node.js 18+ (LTS recommended)
- Python 3.10+ (for node-gyp)
- On Windows: Visual Studio Build Tools with "Desktop development with C++"

**Training fails with "CUDA out of memory":**
1. Use QLoRA (4-bit quantization) — enabled by default
2. Reduce batch size to 1-2
3. Use the smaller 1.5B model instead of 7B
4. Close other GPU-consuming applications
