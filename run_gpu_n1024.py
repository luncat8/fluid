#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
import run_gpu_capture
import sys
sys.argv = ["run_gpu_capture.py", "n1024"]
run_gpu_capture.main()
