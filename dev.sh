#!/usr/bin/env bash
# dev.sh — Cross-platform dev script (macOS/Linux equivalent of dev.ps1)
#
# Usage:
#   ./dev.sh              # Backend + Frontend (browser)
#   ./dev.sh --backend    # Backend only
#   ./dev.sh --frontend   # Frontend only
#   ./dev.sh --electron   # Backend + Electron app
#   ./dev.sh --port 8003  # Custom backend port

set -euo pipefail

PORT=8003
BACKEND=true
FRONTEND=true
ELECTRON=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)  BACKEND=true; FRONTEND=false; shift ;;
    --frontend) BACKEND=false; FRONTEND=true; shift ;;
    --electron) ELECTRON=true; shift ;;
    --port)     PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./dev.sh [--backend] [--frontend] [--electron] [--port PORT]"
      echo ""
      echo "  --backend    Start API server only"
      echo "  --frontend   Start frontend only"
      echo "  --electron   Start backend + Electron app"
      echo "  --port PORT  Backend port (default: 8003)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

PIDS=()

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
  echo "Done."
}

trap cleanup EXIT INT TERM

if $BACKEND; then
  echo "Starting backend on port $PORT..."
  python run_backend.py --port "$PORT" &
  PIDS+=($!)
fi

if $FRONTEND; then
  if $ELECTRON; then
    echo "Starting Electron app..."
    (cd frontend && npm run electron:dev) &
  else
    echo "Starting frontend on port 5173..."
    (cd frontend && npm run dev) &
  fi
  PIDS+=($!)
fi

echo ""
echo "=== Bash Gym Dev Server ==="
echo "  API:  http://localhost:$PORT/api"
echo "  Docs: http://localhost:$PORT/docs"
if $FRONTEND && ! $ELECTRON; then
  echo "  UI:   http://localhost:5173"
fi
echo "  Press Ctrl+C to stop"
echo ""

wait
