#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
import run_gpu_capture
import sys
sys.argv = ["run_gpu_capture.py", "k60"]
run_gpu_capture.main()
