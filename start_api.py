#!/usr/bin/env python
"""Start the BashGym API with Python 3.12"""
import subprocess
import sys
import os

os.chdir(r"C:\Users\Cade\projects\ghostwork")

# Start uvicorn
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "bashgym.api.routes:create_app",
    "--factory",
    "--host", "127.0.0.1",
    "--port", "8003"
])
