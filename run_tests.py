#!/usr/bin/env python3
"""Run every headless suite. No arguments needed, no browser, no device.

	python3 run_tests.py              fast profile, one PASS/FAIL verdict (~10 s)
	python3 run_tests.py --full       add the long numerical histories (~10 min)
	python3 run_tests.py --gpu --copy  optional real-device parity, copy the log

The device pass is separate and optional: open experiments/api-test.html in a
WebGPU browser (file:// works) and press the run button.
"""
import argparse
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def setup_output():
	for stream in (sys.stdout, sys.stderr):
		if hasattr(stream, "reconfigure"):
			stream.reconfigure(encoding="utf-8", errors="replace")


def node_executable():
	return shutil.which("node")


def node_version(node):
	try:
		result = subprocess.run([node, "--version"], check=True, capture_output=True,
			text=True, encoding="utf-8", errors="replace")
		return result.stdout.strip()
	except (OSError, subprocess.CalledProcessError):
		return "unknown"


def log_path():
	directory = ROOT / "experiments" / "logs"
	directory.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().astimezone().strftime("%Y-%m-%d-%H-%M")
	path = directory / (stamp + "-test-suite.log")
	suffix = 1
	while path.exists():
		path = directory / (stamp + "-test-suite-" + str(suffix) + ".log")
		suffix += 1
	return path


def environment_line(node):
	return ("fluid tests · " + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
		+ " · node " + node_version(node)
		+ " · python " + platform.python_version()
		+ " · " + platform.system().lower() + " " + platform.machine())


def tee_run(command, log, header):
	started = time.monotonic()
	code = 130
	proc = None
	with log.open("w", encoding="utf-8", errors="replace") as output:
		print(header, flush=True)
		output.write(header + "\n")
		output.flush()
		try:
			proc = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
			for line in proc.stdout:
				print(line, end="", flush=True)
				output.write(line)
			code = proc.wait()
		except KeyboardInterrupt:
			if proc is not None and proc.poll() is None:
				proc.terminate()
				try:
					proc.wait(timeout=5)
				except subprocess.TimeoutExpired:
					proc.kill()
					proc.wait()
			print("\ninterrupted", flush=True)
			output.write("interrupted\n")
		except OSError as error:
			code = 127
			message = "could not start test suite: " + str(error)
			print(message, flush=True)
			output.write(message + "\n")
		output.write("\nexit " + str(code) + " · " + format(time.monotonic() - started, ".3f") + " s\n")
	return code


def clipboard_command():
	if sys.platform == "win32":
		return ["clip"]
	if sys.platform == "darwin":
		return ["pbcopy"]
	for command in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
		if shutil.which(command[0]):
			return command
	return None


def copy_log(log):
	command = clipboard_command()
	if command is None:
		print("clipboard utility not found; copy the saved log: " + str(log))
		return False
	try:
		with log.open("r", encoding="utf-8", errors="replace") as source:
			result = subprocess.run(command, input=source.read(), text=True,
				encoding="utf-8", errors="replace", check=False)
	except OSError as error:
		print("clipboard copy failed: " + str(error))
		return False
	if result.returncode:
		print("clipboard copy failed (exit " + str(result.returncode) + ")")
		return False
	print("copied full test log to clipboard")
	return True


def main():
	setup_output()
	parser = argparse.ArgumentParser(
		description="run every headless Fluid suite (no arguments needed)",
		epilog="exit code 0 means every suite passed; the log path is printed last")
	parser.add_argument("--full", action="store_true",
		help="also run the long numerical histories (default is the fast profile, ~10 s)")
	parser.add_argument("--quick", action="store_true", help="accepted for old habits; it is the default")
	parser.add_argument("--gpu", action="store_true",
		help="also run optional real WebGPU parity (requires the external webgpu package/backend)")
	parser.add_argument("--copy", action="store_true", help="copy the finished log to the system clipboard")
	args = parser.parse_args()
	node = node_executable()
	if not node:
		print("node was not found on PATH; install Node.js and retry", file=sys.stderr)
		return 127
	command = [node, str(ROOT / "experiments" / "run-all.js")]
	if not args.full:
		command.append("--quick")
	if args.gpu:
		command.append("--gpu")
	log = log_path()
	started = time.monotonic()
	code = tee_run(command, log, environment_line(node))
	elapsed = time.monotonic() - started
	verdict = "PASS · every headless suite passed" if code == 0 else "FAIL · see the failing suite above"
	print("\nresult: " + verdict + " · " + format(elapsed, ".1f") + " s · "
		+ ("full" if args.full else "fast") + " profile")
	print("log: " + str(log))
	if code == 0 and not args.full:
		print("device pass (optional, needs a WebGPU browser): "
			+ "python3 run_index.py --page experiments/api-test.html")
	if args.copy:
		copy_log(log)
	return code


if __name__ == "__main__":
	sys.exit(main())
