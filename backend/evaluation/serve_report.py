"""
Simple server to view evaluation reports.

Usage:
    cd backend/evaluation
    python serve_report.py

Then open http://localhost:8080 in your browser.
"""

import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

RESULTS_DIR = Path(__file__).parent / "results"


class ReportHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        if self.path == "/api/reports":
            self.send_reports()
        elif self.path == "/":
            self.path = "/report.html"
            super().do_GET()
        else:
            super().do_GET()

    def send_reports(self):
        reports = []
        if RESULTS_DIR.exists():
            for f in sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True):
                try:
                    with open(f) as fp:
                        reports.append(json.load(fp))
                except Exception as e:
                    print(f"Error loading {f}: {e}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reports).encode())


def main():
    port = 8080
    server = HTTPServer(("localhost", port), ReportHandler)
    print(f"Serving evaluation reports at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
