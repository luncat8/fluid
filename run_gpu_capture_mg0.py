#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
import run_browser_tests
import sys
sys.argv = ["run_browser_tests.py", "--query", "n=512&frames=600&scenarios=blobs,draw&mg=0"]
run_browser_tests.main()
