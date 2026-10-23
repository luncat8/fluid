#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
import run_gpu_capture
sys.argv = ["run_gpu_capture.py", "mac-k20"]
run_gpu_capture.main()
