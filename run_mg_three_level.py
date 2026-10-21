#!/usr/bin/env python3
"""Run the experimental CPU same-state multigrid oracle.

Examples:
  python3 run_mg_three_level.py
  python3 run_mg_three_level.py --scenario=cycle
  python3 run_mg_three_level.py --all
  python3 run_mg_three_level.py --n512
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
result = subprocess.run([node, str(ROOT / "experiments" / "mg-three-level.js"), *sys.argv[1:]], cwd=ROOT)
sys.exit(result.returncode)
