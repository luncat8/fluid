from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse
import threading
import webbrowser

class COIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

def main():
    parser = argparse.ArgumentParser(description="serve the fluid lab")
    parser.add_argument("port", nargs="?", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    import os
    os.chdir(root)

    server = ThreadingHTTPServer((args.host, args.port), COIHandler)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/index.html"
    print(f"dir: {root}")
    print(f"server: {url}")
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
