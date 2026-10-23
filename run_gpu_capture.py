#!/usr/bin/env python3
import sys
sys.dont_write_bytecode = True
"""One-click launchers for the Fluid GPU browser diagnostics captures.

Run a named preset (no copy-pasted query string):

	python3 run_gpu_capture.py k40
	python3 run_gpu_capture.py fixed-k20

Or list them:

	python3 run_gpu_capture.py --list

With no arguments it runs the default preset (k20). Any extra arguments are
forwarded to run_browser_tests.py, so --query still works for a custom run:

	python3 run_gpu_capture.py --query "n=1024&frames=300&scenarios=ice&k=40"
"""
import run_browser_tests
import sys

PRESETS = {
	"k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&profile=1",
	"k40": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=40&profile=1",
	"k60": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=60&profile=1",
	"fixed-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&auto=0&ns=4&profile=1",
	"fixed-k40": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=40&auto=0&ns=4&profile=1",
	"fixed-k60": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=60&auto=0&ns=4&profile=1",
	"cycles1": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&cycles=1&auto=0&ns=4&profile=1",
	"jet-fixed-k20": "n=512&frames=600&scenarios=jet&k=20&cycles=2&auto=0&ns=4&profile=1",
	"jet-cycles1": "n=512&frames=600&scenarios=jet&k=20&cycles=1&auto=0&ns=4&profile=1",
	"jet-auto": "n=512&frames=600&scenarios=jet&k=20&profile=1",
	"n1024": "n=1024&frames=300&scenarios=blobs,dam,cycle,jet&k=20&profile=1",
	"n1024-fixed": "n=1024&frames=300&scenarios=blobs,dam,cycle,jet&k=20&auto=0&ns=4&profile=1",
	"cycle-fixed-k20": "n=512&frames=600&scenarios=cycle&k=20&cycles=2&auto=0&ns=4&profile=1",
	# 0.6.0 t16 device A/B for mac-cormack: same policies as their
	# semi-lagrangian peers, only &mac=1 differs.
	"mac-fixed-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&auto=0&ns=4&profile=1&mac=1",
	"mac-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&profile=1&mac=1",
	"jet-mac-auto": "n=512&frames=600&scenarios=jet&k=20&profile=1&mac=1",
	# 0.6.0 t8 device A/B for pass fusion (advectScalar + phaseSources).
	"fuse-fixed-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&auto=0&ns=4&profile=1&fuse=1",
	"fuse-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&profile=1&fuse=1",
	"mac-fuse-k20": "n=512&frames=600&scenarios=blobs,dam,cycle,jet&k=20&profile=1&mac=1&fuse=1",
}

DEFAULT = "k20"


def main():
	if "--list" in sys.argv or "-l" in sys.argv:
		print("available presets:")
		for name, query in PRESETS.items():
			print("  " + name + "  ->  " + query)
		return
	if "--query" in sys.argv:
		index = sys.argv.index("--query")
		query = sys.argv[index + 1]
		sys.argv = ["run_browser_tests.py", "--query", query]
		run_browser_tests.main()
		return
	no_open = "--no-open" in sys.argv
	preset = DEFAULT
	for arg in sys.argv[1:]:
		if arg in PRESETS:
			preset = arg
		elif arg not in ("--no-open",):
			print("unknown preset or option: " + arg, file=sys.stderr)
			print("use --list to see the available presets", file=sys.stderr)
			return 2
	query = PRESETS[preset]
	print("preset: " + preset + " -> " + query)
	new_argv = ["run_browser_tests.py", "--query", query]
	if no_open:
		new_argv.append("--no-open")
	sys.argv = new_argv
	run_browser_tests.main()


if __name__ == "__main__":
	result = main()
	sys.exit(result if result is not None else 0)
