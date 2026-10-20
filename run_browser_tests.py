#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
"""Run the GPU browser diagnostics page on the COOP/COEP server for fine timers.

	python3 run_browser_tests.py
	python3 run_browser_tests.py --query "n=512&frames=600&scenarios=blobs,draw&mg=0"

Serving (instead of file://) cross-origin isolates the page so performance.now()
resolves finer than the clamped 0.1 ms quantum; the report prints the isolation
state and the observed quantum. Save the report from the page into
experiments/logs/ as <date>-GPU-browser-tests.log next to the other captures.
"""
import argparse

import run_index

PAGE = "experiments/browser-tests.html"


def main():
	parser = argparse.ArgumentParser(
		description="serve the Fluid GPU browser diagnostics with high-resolution timer isolation")
	parser.add_argument("port", nargs="?", type=int, default=8000)
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--query", default="",
		help="query string for the page, e.g. n=512&frames=600&scenarios=blobs,draw&mg=0")
	parser.add_argument("--no-open", action="store_true")
	args = parser.parse_args()

	query = args.query.lstrip("?")
	page_url = PAGE + ("?" + query if query else "")
	print("report: save as experiments/logs/<date>-GPU-browser-tests.log")
	run_index.serve(page_url, args.host, args.port, not args.no_open)


if __name__ == "__main__":
	main()
