#!/usr/bin/env python
"""
Run the Bash Gym API backend with hot reload for development.

Usage:
    python run_backend.py          # Default port 8001
    python run_backend.py --port 8002
"""

import subprocess
import sys
import os

def main():
    port = "8001"

    # Check for --port argument
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = sys.argv[idx + 1]

    # Set environment
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    cmd = [
        sys.executable, "-m", "uvicorn",
        "bashgym.api.routes:create_app",
        "--factory",
        "--host", "0.0.0.0",
        "--port", port,
        "--reload",  # Hot reload enabled!
        "--reload-dir", "bashgym",  # Watch bashgym directory
    ]

    print(f"Starting Bash Gym API on port {port} with hot reload...")
    print(f"API URL: http://localhost:{port}/api")
    print(f"Docs: http://localhost:{port}/docs")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
