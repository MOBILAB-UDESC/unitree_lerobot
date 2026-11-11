"""Short link for the Quest browser: http://<pc-ip>:8080 redirects to the full televuer URL.

Usage: python quest_link.py <pc-ip> [port]
"""

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ip = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
target = f"https://{ip}:8012/?ws=wss://{ip}:8012"


class Redirect(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", target)
        self.end_headers()

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("0.0.0.0", port), Redirect).serve_forever()
