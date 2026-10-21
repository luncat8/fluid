#!/usr/bin/env python3
"""Pinned jet capture at 1 V-cycle. The quality peer of jet-fixed-k20."""
import sys
sys.dont_write_bytecode = True
import run_gpu_capture

sys.argv = ["run_gpu_capture.py", "jet-cycles1"] + sys.argv[1:]
run_gpu_capture.main()
