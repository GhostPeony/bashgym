#!/usr/bin/env python3
"""
Bash Gym API Server Runner

Starts the FastAPI server for the Bash Gym frontend interface.

Usage:
    python -m bashgym.main
    python bashgym/main.py
    python bashgym/main.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
import os
from pathlib import Path

# Enable UTF-8 output on Windows
if sys.platform == "win32":
    os.system("")  # Enable ANSI escape codes on Windows
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    """Run the Bash Gym API server."""
    parser = argparse.ArgumentParser(
        description="Bash Gym API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with defaults (localhost:8000)
    python -m bashgym.main

    # Start with auto-reload for development
    python -m bashgym.main --reload

    # Start on a specific host and port
    python -m bashgym.main --host 0.0.0.0 --port 8080

    # Start with custom log level
    python -m bashgym.main --log-level debug
        """
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file for configuration"
    )

    args = parser.parse_args()

    # Load environment variables if specified
    if args.env_file:
        env_path = Path(args.env_file)
        if env_path.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path)
                print(f"Loaded environment from {env_path}")
            except ImportError:
                print("Warning: python-dotenv not installed, skipping .env loading")
        else:
            print(f"Warning: .env file not found at {env_path}")

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install it with: pip install uvicorn[standard]")
        sys.exit(1)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗  ██╗ ██████╗ ███████╗████████╗                 ║
║  ██╔════╝ ██║  ██║██╔═══██╗██╔════╝╚══██╔══╝                 ║
║  ██║  ███╗███████║██║   ██║███████╗   ██║                    ║
║  ██║   ██║██╔══██║██║   ██║╚════██║   ██║                    ║
║  ╚██████╔╝██║  ██║╚██████╔╝███████║   ██║                    ║
║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝   ╚═╝                    ║
║                                                               ║
║   ██████╗ ██╗   ██╗███╗   ███╗                               ║
║  ██╔════╝ ╚██╗ ██╔╝████╗ ████║                               ║
║  ██║  ███╗ ╚████╔╝ ██╔████╔██║                               ║
║  ██║   ██║  ╚██╔╝  ██║╚██╔╝██║                               ║
║  ╚██████╔╝   ██║   ██║ ╚═╝ ██║                               ║
║   ╚═════╝    ╚═╝   ╚═╝     ╚═╝                               ║
║                                                               ║
║  Self-Improving Agentic Development Gym                       ║
╚═══════════════════════════════════════════════════════════════╝

Starting API server...
  Host: {args.host}
  Port: {args.port}
  Workers: {args.workers}
  Reload: {args.reload}
  Log Level: {args.log_level}

API Documentation:
  Swagger UI: http://{args.host}:{args.port}/api/docs
  ReDoc: http://{args.host}:{args.port}/api/redoc

WebSocket:
  ws://{args.host}:{args.port}/ws

""")

    uvicorn.run(
        "bashgym.api.routes:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # reload doesn't support multiple workers
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
