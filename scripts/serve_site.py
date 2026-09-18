"""Static file server with HTTP Range support (so <video> can seek), stdlib only.

    python3 scripts/serve_site.py <dir> [port]     # binds 127.0.0.1 only; put `tunnel create <port>` in front

python -m http.server ignores Range headers, which makes browsers re-download whole MP4s and breaks seeking.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class RangeHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mp4": "video/mp4", ".json": "application/json"}

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or "Range" not in self.headers:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        rng = self.headers["Range"].strip()
        try:
            unit, spec = rng.split("=", 1)
            assert unit == "bytes"
            start_s, end_s = spec.split("-", 1)
            start = int(start_s) if start_s else max(0, size - int(end_s))
            end = int(end_s) if (end_s and start_s) else size - 1
            end = min(end, size - 1)
            assert 0 <= start <= end
        except (ValueError, AssertionError):
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "public, max-age=600")
        self.end_headers()
        f.seek(start)
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        while left > 0:
            chunk = source.read(min(1 << 20, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)
        self._range_left = None

    def end_headers(self):
        if self.command == "GET" and "Range" not in self.headers:
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter: one line per request, no per-chunk noise
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(RangeHandler, directory=root))
    print(f"serving {os.path.abspath(root)} on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
