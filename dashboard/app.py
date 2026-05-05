"""
VolleyVision Dashboard — Flask server.

Start with:
    python -m dashboard.app
or from project root:
    python dashboard/app.py

Serves the web UI at http://127.0.0.1:8000 and exposes:
    GET /api/clips          → manifest summary list
    GET /api/clips/<stem>   → full clip.json for one clip
    GET /output/<path>      → static files from output/ (videos, posters, JSONs)
"""

import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, abort

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
OUTPUT_DIR = BASE_DIR / "output"

app = Flask(__name__, template_folder="templates", static_folder="static")


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_manifest():
    """Return list of clip summaries.
    Prefers manifest.json; falls back to scanning *.clip.json files."""
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
        return data.get("clips", [])

    # Fallback: build summary from clip.json files
    clips = []
    for p in sorted(OUTPUT_DIR.glob("*.clip.json")):
        try:
            with open(p) as f:
                c = json.load(f)
            clips.append({
                "id":         c.get("id"),
                "title":      c.get("title"),
                "date":       c.get("date"),
                "duration_s": c.get("duration_s"),
                "hits":       sum(1 for e in c.get("events", []) if e.get("type") == "contact"),
                "thumb":      c.get("poster"),
                "status":     "analyzed",
            })
        except Exception:
            pass
    return clips


def _load_clip(stem: str):
    """Return full clip.json dict or None."""
    p = OUTPUT_DIR / f"{stem}.clip.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/clips")
def api_clips():
    return jsonify(_load_manifest())


@app.route("/api/clips/<stem>")
def api_clip_detail(stem):
    clip = _load_clip(stem)
    if clip is None:
        abort(404, description=f"Clip '{stem}' not found")
    return jsonify(clip)


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser, threading
    port = 8000
    print(f"[VolleyVision Dashboard] output dir: {OUTPUT_DIR}")
    print(f"[VolleyVision Dashboard] → http://127.0.0.1:{port}")
    # Open browser after a short delay so the server is ready
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="127.0.0.1", debug=False, port=port)
