"""VolleyVision pipeline.

run()         two-pass (default): YOLO once with every candidate cached -> linker over the
              whole clip with zoom search -> touch detection -> render.
run_legacy()  the original single-pass tracker (YOLO conf >= 0.40 + Kalman), kept for comparison.
"""
import cv2
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from core.ball_tracker import BallTracker
from core.touch_detector import detect_touches, Touch


def run_legacy(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/best4.pt",
    conf: float = 0.4,
    trail_len: int = 40,
    show_preview: bool = False,
    raw: bool = False,
    touches: bool = True,
) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames")

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_tracked.mp4")

    # Always render to a temp file first: touches are only known after the
    # whole clip is tracked, so they're overlaid in a second (cheap, no-YOLO)
    # pass, then the result is transcoded with ffmpeg if available.
    has_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_path   = os.path.join(tempfile.gettempdir(), f"{stem}_noaudio.mp4")
    writer     = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = BallTracker(model_path=model_path, conf=conf, trail_len=trail_len)
    track_rows = []

    t0 = time.time()
    for frame_idx, result in enumerate(
        tracker.model.predict(source=input_path, stream=True, conf=conf, verbose=False, imgsz=1280)
    ):
        if raw:
            out = result.plot()
        else:
            pos = tracker.update_from_result(result)
            out = tracker.draw(result.orig_img)
            row = {"f": frame_idx}
            if tracker.last_det is not None:
                row["det"] = [round(tracker.last_det[0], 1), round(tracker.last_det[1], 1)]
                row["conf"] = round(tracker.last_conf, 3)
            if pos is not None:
                row["pos"] = [round(pos[0], 1), round(pos[1], 1)]
            track_rows.append(row)

        writer.write(out)

        if show_preview:
            cv2.imshow("VolleyVision", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 60 == 0:
            print(f"  {frame_idx / max(total, 1) * 100:5.1f}%  frame {frame_idx}/{total}  [{time.time()-t0:.1f}s]")

    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    render_path = tmp_path
    if not raw:
        track_path = os.path.join(output_dir, f"{stem}_track.json")
        with open(track_path, "w") as fh:
            json.dump({"schema": "vv-track/1", "video": str(input_path), "fps": fps,
                       "width": width, "height": height, "frames": track_rows}, fh)
        print(f"[VolleyVision] Trajectory  →  {track_path}")

        if touches:
            pts = [(r["f"], *r["det"]) for r in track_rows if "det" in r]
            found = detect_touches(*map(np.array, zip(*pts)), fps, height) if pts else []
            touch_path = os.path.join(output_dir, f"{stem}_touches.json")
            with open(touch_path, "w") as fh:
                json.dump({"schema": "vv-touches/1", "video": str(input_path), "fps": fps,
                           "touches": [t.to_dict() for t in found]}, fh, indent=1)
            print(f"[VolleyVision] {len(found)} touches  →  {touch_path}")
            for t in found:
                print(f"    {t.t:6.2f}s  frame {t.frame:5d}  turn {t.angle_deg:3.0f}°  "
                      f"{t.speed_in:5.0f} → {t.speed_out:5.0f} px/s{'  (during gap)' if t.in_gap else ''}")
            if found:
                annotated = os.path.join(tempfile.gettempdir(), f"{stem}_touches.mp4")
                overlay_touches(tmp_path, annotated, found, fps)
                os.remove(tmp_path)
                render_path = annotated

    if has_ffmpeg:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", render_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            out_path,
        ], check=True, capture_output=True)
        os.remove(render_path)
        print("[VolleyVision] Transcoded to H.264 + audio muxed")
    else:
        shutil.move(render_path, out_path)
        print("[VolleyVision] ffmpeg not found — audio not preserved")

    print(f"[VolleyVision] Done in {time.time()-t0:.1f}s  →  {out_path}")
    return out_path


def overlay_touches(src: str, dst: str, touches: list[Touch], fps: float, hold_s: float = 0.5):
    """Re-read a rendered video and mark each touch with an expanding, fading
    ring at the contact point, held for hold_s after the contact frame.
    Touch data (count, timing, velocities) lives in <stem>_touches.json only."""
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    hold = max(1, int(round(hold_s * fps)))
    scale = h / 1080
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for t in touches:
            if not (t.frame <= f < t.frame + hold):
                continue
            fade = 1 - (f - t.frame) / hold
            c = (0, int(80 + 175 * fade), 255)
            cv2.circle(img, (int(t.x), int(t.y)), int((28 + 30 * (1 - fade)) * scale), c,
                       max(2, int(4 * scale)), cv2.LINE_AA)
        writer.write(img)
        f += 1
    cap.release()
    writer.release()


# ── two-pass pipeline ─────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/best4.pt",
    det_conf: float = 0.05,
    imgsz: int = 1280,
    linker: str = "rule",
    zoom: bool = True,
    redetect: bool = False,
    trail_len: int = 40,
    show_preview: bool = False,
    touches: bool = True,
    ring: bool = False,
    tracknet: str | None = "models/tracknet/FT_TrackNet_best.pt",
) -> str:
    from ultralytics import YOLO
    from core.linker import link as rule_link, LinkParams
    from core.refine import link_with_zoom, video_dup_flags, content_box
    from core.render import draw_trail, draw_ball, draw_touch_rings, finish_video
    from tools.dump_detections import dump

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stem = Path(input_path).stem
    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames  ({linker} linker)")
    t0 = time.time()

    # 1. detection pass (cached)
    det_dir = Path(output_dir) / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    det_path = det_dir / f"{stem}_dets.json"
    model = YOLO(model_path)
    dets = json.load(open(det_path)) if det_path.exists() else None
    stale = (dets is None or redetect or dets.get("imgsz") != imgsz or dets.get("conf", det_conf) > det_conf
             or dets.get("model", model_path) != model_path)
    if stale:
        print(f"[VolleyVision] detection pass (YOLO on every frame, cached afterwards)...")
        dump(input_path, model, det_dir, det_conf, imgsz)
        dets = json.load(open(det_path))
        dets["model"] = model_path
    else:
        print(f"[VolleyVision] detections: cached ({det_path})")

    # 2. linking (+ zoom search)
    if zoom:
        track = link_with_zoom(dets, input_path, model, LinkParams(), linker=linker,
                               log=lambda m: print("[VolleyVision]" + m))
    else:
        from core.profile import get as get_profile, link_args
        get_profile(input_path, dets)
        a = link_args(dets)
        if linker == "cost":
            from core.cost_linker import link as cost_link
            track = cost_link(dets["frames"], a["fps"], dets["width"], dets["height"],
                              dup_flags=a["dup_flags"], content=a["content"], times=a["times"])
        else:
            track = rule_link(dets["frames"], a["fps"], dets["width"], dets["height"], **{k: v for k, v in a.items() if k != "fps"})
    with open(det_path, "w") as fh:                       # keep zoom finds / repeat flags cached
        json.dump(dets, fh, separators=(",", ":"))

    # 2b. TrackNet (multi-frame model): fills gaps and extends track ends where YOLO saw
    #     nothing, inside the corridor physics predicts. Never overrides a YOLO detection.
    if tracknet:
        from core.tracknet import candidates as tracknet_candidates
        from core.tracknet_fill import apply as tracknet_fill
        tn = tracknet_candidates(input_path, dets, tracknet, cache_dir=str(det_dir),
                                 log=lambda m: print(m))
        if tn is not None:
            ball_px = (dets.get("profile") or {}).get("ball_px") or float(np.median(
                [max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= 0.4] or [30.0]))
            got = tracknet_fill(track, tn, ball_px)
            print(f"[VolleyVision] TrackNet filled {got['gap']} gap frames, extended track ends by {got['end']} frames")

    s = track.summary()
    print(f"[VolleyVision] ball tracked in {s['coverage']:.1%} of frames "
          f"(detected {s['detected']}, weak {s['weak']}, tracknet {s['tracknet']}, filled {s['filled']}, none {s['none']})")
    os.makedirs(output_dir, exist_ok=True)

    # 3. contacts: split the track into flights (one parabola each); each boundary is a
    #    contact, classified touch / bounce / other using where the people are
    found = []
    if touches:
        from core.flights import from_track
        from core.contacts import classify
        flights, found = from_track(track)
        if found:
            person_model = YOLO(str(Path(model_path).parent / "yolov8n.pt")) \
                if (Path(model_path).parent / "yolov8n.pt").exists() else YOLO("yolov8n.pt")
            classify(found, track, input_path, person_model, dets.setdefault("persons", {}))
            with open(det_path, "w") as fh:               # cache person boxes too
                json.dump(dets, fh, separators=(",", ":"))
        contact_path = os.path.join(output_dir, f"{stem}_contacts.json")
        with open(contact_path, "w") as fh:
            json.dump({"schema": "vv-contacts/1", "video": str(input_path), "fps": fps,
                       "flights": [{"f0": fl.f0, "f1": fl.f1, "n": int(fl.n), "rms": round(fl.rms, 2),
                                    "cx": [round(v, 5) for v in fl.cx], "cy": [round(v, 5) for v in fl.cy]}
                                   for fl in flights],
                       "contacts": [c.to_dict() for c in found]}, fh, indent=1)
        # gaps inside a flight follow that flight's parabola (real detections are kept)
        from core.flights import apply_flights
        track.pos, track.state = apply_flights(track, flights, found)
        kinds = {k: sum(c.kind == k for c in found) for k in ("touch", "bounce", "other")}
        print(f"[VolleyVision] {len(flights)} flights, {kinds['touch']} touches, {kinds['bounce']} bounces, "
              f"{kinds['other']} other contacts  →  {contact_path}")
        for c in found:
            print(f"    {c.t:6.2f}s  frame {c.frame:7.1f}  {c.kind:6s}{'  (ball hidden at contact)' if c.in_gap else ''}")

    track_path = os.path.join(output_dir, f"{stem}_track.json")
    with open(track_path, "w") as fh:
        json.dump({"schema": "vv-track/2", "video": str(input_path), "fps": fps, "width": width,
                   "height": height, "linker": linker, "frames": track.to_rows()}, fh)
    print(f"[VolleyVision] trajectory  →  {track_path}")

    # 4. render
    out_path = os.path.join(output_dir, f"{stem}_tracked.mp4")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{stem}_render.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    cap = cv2.VideoCapture(input_path)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        lo = max(0, f - trail_len + 1)
        draw_trail(img, [track.pos[i] if i < len(track.pos) else None for i in range(lo, f + 1)])
        if f < len(track.pos) and track.pos[f] is not None:
            draw_ball(img, track.pos[f], ring=ring)
        draw_touch_rings(img, f, [c for c in found if c.kind == "touch"], fps)
        draw_touch_rings(img, f, [c for c in found if c.kind == "bounce"], fps, color=(255, 255, 255))
        writer.write(img)
        if show_preview:
            cv2.imshow("VolleyVision", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                show_preview = False
                cv2.destroyAllWindows()
        f += 1
    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()
    used_ffmpeg = finish_video(tmp_path, input_path, out_path)
    print(f"[VolleyVision] {'H.264 + audio' if used_ffmpeg else 'ffmpeg not found — no audio'}")
    print(f"[VolleyVision] Done in {time.time() - t0:.1f}s  →  {out_path}")
    return out_path
