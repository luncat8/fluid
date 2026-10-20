#!/usr/bin/env python3
"""Serve the app or a local browser test page with cross-origin isolation headers."""
import argparse
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class COIHandler(SimpleHTTPRequestHandler):
	def end_headers(self):
		self.send_header("Cross-Origin-Opener-Policy", "same-origin")
		self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
		self.send_header("Cache-Control", "no-store")
		super().end_headers()


def main():
	parser = argparse.ArgumentParser(description="serve the Fluid lab with high-resolution timer isolation")
	parser.add_argument("port", nargs="?", type=int, default=8000)
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--page", default="index.html", help="page path relative to the repository root")
	parser.add_argument("--no-open", action="store_true")
	args = parser.parse_args()

	root = Path(__file__).resolve().parent
	page = Path(args.page)
	if page.is_absolute():
		parser.error("--page must be relative to the repository root")
	page_file = (root / page).resolve()
	if root not in page_file.parents or not page_file.is_file():
		parser.error("--page must name a file inside the repository: " + args.page)
	page_url = page_file.relative_to(root).as_posix()
	if not page_url:
		parser.error("--page must name a file")

	os.chdir(root)
	server = ThreadingHTTPServer((args.host, args.port), COIHandler)
	url_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
	url = "http://" + url_host + ":" + str(server.server_port) + "/" + page_url
	print("dir: " + str(root))
	print("server: " + url)
	print("timer isolation: COOP=same-origin, COEP=require-corp")
	print("to stop close or Ctrl+C.")

	if not args.no_open:
		threading.Timer(0.5, lambda: webbrowser.open(url)).start()

	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nstop...")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
