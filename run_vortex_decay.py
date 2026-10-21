#!/usr/bin/env python3
"""Run the CPU vortex-decay oracle (0.6.0 t5 decision metric).

Examples:
  python3 run_vortex_decay.py
  python3 run_vortex_decay.py --n=128 --t=4
  python3 run_vortex_decay.py --no-proj
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
node = shutil.which("node")
if not node:
	print("Node.js is required; install it and retry", file=sys.stderr)
	sys.exit(127)
result = subprocess.run([node, str(ROOT / "experiments" / "vortex-decay.js"), *sys.argv[1:]], cwd=ROOT)
sys.exit(result.returncode)
