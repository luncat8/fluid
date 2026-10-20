#!/usr/bin/env python3
import run_browser_tests
import sys
sys.argv = ["run_browser_tests.py", "--query", "n=512&frames=600&scenarios=ice,cycle,jet,rain,arch"]
run_browser_tests.main()

