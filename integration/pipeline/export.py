"""
VolleyVision — Clip data exporter.

Captures per-frame detections + Kalman state during a video pipeline run and
writes a `<stem>.clip.json` next to the rendered `<stem>_tracked.mp4`.

Usage from video_pipeline.py:

    from pipeline.export import ClipExporter

    exporter = ClipExporter(
        input_path=input_path, output_dir=output_dir,
        fps=fps, width=width, height=height, total_frames=total,
        px_per_m=px_per_m, net_width_px=net_width_px, net_height_px=net_height_px,
        model_path=model_path, conf=conf,
    )

    # ── inside the per-frame loop ──
    exporter.record(
        frame_idx=frame_idx,
        detection=tracker._best_detection(result),  # raw (px, px) or None
        kalman=tracker.position,                    # smoothed (px, px) or None
        prev_pos=prev_pos,                          # to compute vx/vy
        speed_kmh=display_speed,
        is_lost=(tracker._lost > 0 and tracker._kf_ready),
    )

    # ── after the loop ──
    exporter.finalize(peak_speed_kmh=peak_speed)

The exporter is **pure data** — does not affect video rendering or your
existing CLI behavior.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Optional, Tuple

SCHEMA = "vv-clip/1"
PIPELINE_VERSION = "0.2.0"


class ClipExporter:
    def __init__(
        self,
        input_path: str,
        output_dir: str,
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        px_per_m: Optional[float],
        net_width_px: Optional[int] = None,
        net_height_px: Optional[int] = None,
        model_path: str = "models/best3.pt",
        conf: float = 0.40,
    ):
        self.input_path = input_path
        self.output_dir = output_dir
        self.stem = Path(input_path).stem

        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.total = int(total_frames)
        self.px_per_m = px_per_m
        self.net_width_px = net_width_px
        self.net_height_px = net_height_px
        self.model_path = model_path
        self.conf = conf

        # Per-frame buffers
        self.detections: list[dict] = []   # sparse (only when YOLO returns a ball)
        self.trajectory: list[dict] = []   # dense (one per processed frame)
        self.speeds: list[float] = []

    # ── recording ─────────────────────────────────────────────────────────

    def record(
        self,
        frame_idx: int,
        detection: Optional[Tuple[float, float]],
        kalman: Optional[Tuple[float, float]],
        prev_pos: Optional[Tuple[float, float]] = None,
        speed_kmh: float = 0.0,
        is_lost: bool = False,
    ):
        """Call once per processed frame."""
        # Raw detection (sparse)
        if detection is not None:
            dx, dy = detection
            self.detections.append({
                "f": int(frame_idx),
                "x": round(dx / self.width, 5),
                "y": round(dy / self.height, 5),
                "conf": None,  # caller can set; left null for now
            })

        # Kalman trajectory (dense — also written when lost)
        entry: dict = {"f": int(frame_idx)}
        if kalman is not None:
            kx, ky = kalman
            entry["x"] = round(kx / self.width, 5)
            entry["y"] = round(ky / self.height, 5)
            if prev_pos is not None:
                vx = (kx - prev_pos[0]) / self.width
                vy = (ky - prev_pos[1]) / self.height
                entry["vx"] = round(vx, 6)
                entry["vy"] = round(vy, 6)
            entry["lost"] = bool(is_lost)
            entry["speed_kmh"] = round(float(speed_kmh), 2)
            self.speeds.append(float(speed_kmh))
        else:
            entry["x"] = None
            entry["y"] = None
            entry["lost"] = True
            entry["speed_kmh"] = 0.0
        self.trajectory.append(entry)

    # ── event detection (very simple v1) ──────────────────────────────────

    def _detect_events(self) -> list[dict]:
        """Detect apex points + contacts from the dense trajectory.

        Apex   = local minimum in y (image coords; lower y = higher in real world).
        Contact = sharp velocity-direction change (vy flips sign with high |dvy|).
        """
        events: list[dict] = []
        traj = [t for t in self.trajectory if t["x"] is not None]
        if len(traj) < 8:
            return events

        # Apex: local y-minima with at least N frames each side
        N = 4
        for i in range(N, len(traj) - N):
            y = traj[i]["y"]
            if y is None:
                continue
            window = traj[i - N: i + N + 1]
            if all(w["y"] is not None and y <= w["y"] for w in window) and y < 0.45:
                # Convert to meters above court (rough — needs calibration + a court-y-baseline).
                height_m = None
                if self.px_per_m:
                    # crude: assume court baseline ~ 0.85 of frame height
                    baseline_y_px = self.height * 0.85
                    apex_y_px = y * self.height
                    height_m = round((baseline_y_px - apex_y_px) / self.px_per_m, 2)
                events.append({
                    "f": traj[i]["f"],
                    "t": round(traj[i]["f"] / self.fps, 3),
                    "type": "apex",
                    "x": traj[i]["x"],
                    "y": traj[i]["y"],
                    "height_m": height_m,
                })

        # Contact: vy sign-flip with magnitude > threshold
        for i in range(2, len(traj) - 2):
            v_prev = traj[i - 1].get("vy")
            v_curr = traj[i].get("vy")
            if v_prev is None or v_curr is None:
                continue
            if v_prev * v_curr < 0 and abs(v_prev - v_curr) > 0.012:
                events.append({
                    "f": traj[i]["f"],
                    "t": round(traj[i]["f"] / self.fps, 3),
                    "type": "contact",
                    "actor": None,
                })

        events.sort(key=lambda e: e["f"])
        return events

    # ── finalize ──────────────────────────────────────────────────────────

    def finalize(self, peak_speed_kmh: float = 0.0) -> str:
        events = self._detect_events()
        avg_speed = (sum(self.speeds) / len(self.speeds)) if self.speeds else 0.0

        tracked = sum(1 for t in self.trajectory if not t.get("lost") and t.get("x") is not None)
        lost = len(self.trajectory) - tracked
        apexes = [e for e in events if e["type"] == "apex" and e.get("height_m") is not None]
        apex_m = max((e["height_m"] for e in apexes), default=None)

        contacts = [e for e in events if e["type"] == "contact"]
        tempo = None
        if len(contacts) >= 2:
            gaps = [contacts[i + 1]["t"] - contacts[i]["t"] for i in range(len(contacts) - 1)]
            gaps = [g for g in gaps if 0.4 < g < 3.0]
            if gaps:
                tempo = round(sum(gaps) / len(gaps), 2)

        clip = {
            "schema": SCHEMA,
            "id": self.stem,
            "title": self.stem,
            "date": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "video": f"{self.stem}_tracked.mp4",
            "raw_video": str(Path(self.input_path).resolve()),
            "poster": f"{self.stem}.poster.jpg",
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "frame_count": self.total,
            "duration_s": round(self.total / self.fps, 3) if self.fps else None,
            "calibration": {
                "px_per_m": round(self.px_per_m, 2) if self.px_per_m else None,
                "net_width_px": self.net_width_px,
                "net_height_px": self.net_height_px,
                "homography": None,
            },
            "detections": self.detections,
            "trajectory": self.trajectory,
            "events": events,
            "metrics": {
                "peak_speed_kmh": round(float(peak_speed_kmh), 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "apex_m": apex_m,
                "tempo_s": tempo,
                "tracked_frames": tracked,
                "lost_frames": lost,
                "track_continuity": round(tracked / max(len(self.trajectory), 1), 3),
            },
            "pose": [],
            "model": Path(self.model_path).name,
            "conf_threshold": self.conf,
            "pipeline_version": PIPELINE_VERSION,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"{self.stem}.clip.json")
        with open(out_path, "w") as fh:
            json.dump(clip, fh, indent=2)
        print(f"[VolleyVision] Wrote {out_path}  ({len(self.trajectory)} frames, {len(events)} events)")

        # Update manifest
        update_manifest(self.output_dir)
        return out_path


def update_manifest(output_dir: str) -> str:
    """Rebuild output/manifest.json from all *.clip.json files in the dir."""
    out = Path(output_dir)
    clips = []
    for p in sorted(out.glob("*.clip.json")):
        try:
            with open(p) as fh:
                c = json.load(fh)
            clips.append({
                "id": c.get("id"),
                "title": c.get("title"),
                "date": c.get("date"),
                "duration_s": c.get("duration_s"),
                "hits": sum(1 for e in c.get("events", []) if e.get("type") == "contact"),
                "thumb": c.get("poster"),
                "status": "analyzed",
            })
        except Exception as e:
            print(f"[manifest] skip {p.name}: {e}")

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump({"schema": "vv-manifest/1", "clips": clips}, fh, indent=2)
    print(f"[VolleyVision] Updated {manifest_path} ({len(clips)} clips)")
    return str(manifest_path)
