#!/usr/bin/env python3
"""Serve a cached visualization state independently from the stepping process.

Run viz.ts on an internal port, then place this proxy on the public port:
  python ai/python/scripts/viz_state_proxy.py --upstream http://127.0.0.1:9103 --port 9102
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:9103")
    ap.add_argument("--port", type=int, default=9102)
    ap.add_argument("--refresh-ms", type=int, default=250)
    args = ap.parse_args()

    # Atomic reference swaps under a short lock — never hold the lock during I/O.
    cache: dict[str, bytes] = {
        "state": b'{"frame":null,"proxy":"waiting for upstream"}',
        "page": b"OpenFront AI visualization is starting.",
    }
    lock = threading.Lock()

    def fetch(path: str, timeout: float = 0.4) -> bytes | None:
        try:
            with urllib.request.urlopen(f"{args.upstream}{path}", timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
            return None

    def refresh_state() -> None:
        while True:
            body = fetch("/state")
            if body:
                with lock:
                    cache["state"] = body
            time.sleep(max(0.05, args.refresh_ms / 1000))

    def refresh_page() -> None:
        while True:
            body = fetch("/", timeout=1.0)
            if body:
                with lock:
                    cache["page"] = body
            time.sleep(2.0)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            key = "state" if self.path.startswith("/state") else "page"
            with lock:
                body = cache[key]
            content_type = (
                "application/json; charset=utf-8"
                if key == "state"
                else "text/html; charset=utf-8"
            )
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    threading.Thread(target=refresh_state, name="proxy-state", daemon=True).start()
    threading.Thread(target=refresh_page, name="proxy-page", daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(
        f"viz state proxy → http://127.0.0.1:{args.port} "
        f"(upstream={args.upstream}, refresh={args.refresh_ms}ms)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
