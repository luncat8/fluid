#!/usr/bin/env python3
"""Pinned jet capture at the shipping 2-cycle count, for the 1-cycle A/B."""
import sys
sys.dont_write_bytecode = True
import run_gpu_capture

sys.argv = ["run_gpu_capture.py", "jet-fixed-k20"] + sys.argv[1:]
run_gpu_capture.main()
