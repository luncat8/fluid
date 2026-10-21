#!/usr/bin/env python3
"""One-command pinned cycle residual capture; save the report under experiments/logs."""
import sys
sys.dont_write_bytecode = True
import run_gpu_capture

sys.argv = ["run_gpu_capture.py", "cycle-fixed-k20"] + sys.argv[1:]
run_gpu_capture.main()
