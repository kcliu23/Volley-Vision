"""VolleyVision pipeline: run(input_path, ...) tracks the ball, finds contacts, renders.

pipeline="finder-first" (default): WASB on every frame, YOLO only on crops around its
    candidates, hybrid linker (core/finder_first.py) -> flights + contacts -> render.
pipeline="classic": YOLO on every frame (cached), rule linker + zoom search, TrackNet gap
    filling -> flights + contacts -> render.
The old single-pass tracker (run_legacy) is archived in core/archive/legacy_pipeline.py.
"""
import cv2
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np



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
    render: bool = True,
    pipeline: str = "finder-first",
    finder: str = "models/wasb/wasb_ft2.pth.tar",
    strict_p: float | None = 0.6,
) -> str:
    """pipeline: "finder-first" (default; uses finder + strict_p, strict_p=None for the loose
    crop rule) or "classic" (uses linker, zoom, tracknet)."""
    from ultralytics import YOLO
    from core.linker import link as rule_link, LinkParams
    from core.refine import link_with_zoom
    from core.render import draw_trail, draw_ball, draw_touch_rings, finish_video
    from tools.dump_detections import dump
    from core import paths

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stem = Path(input_path).stem
    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames  "
          f"({pipeline + (' pipeline' if pipeline == 'finder-first' else f', {linker} linker')})")
    t0 = time.time()

    if pipeline == "finder-first":
        # finder-first (default): WASB on every frame, YOLO only on crops around its top
        # candidates, hybrid linker (core/finder_first.py)
        from core import finder_first
        model = YOLO(model_path)
        track, info = finder_first.track(input_path, model, finder, strict_p=strict_p, root=output_dir,
                                         log=lambda m: print(m))
        det_path = paths.cache_dir(stem, output_dir) / "sample_dets.json"
        dets = json.load(open(det_path))
        arc_finder = finder
    else:
        # classic: YOLO on every frame, rule linker + zoom search, TrackNet fills gaps
        arc_finder = tracknet
        # 1. detection pass (cached)
        det_path = paths.ensure(paths.dets(stem, output_dir))
        model = YOLO(model_path)
        dets = json.load(open(det_path)) if det_path.exists() else None
        stale = (dets is None or redetect or dets.get("imgsz") != imgsz or dets.get("conf", det_conf) > det_conf
                 or dets.get("model", model_path) != model_path)
        if stale:
            print(f"[VolleyVision] detection pass (YOLO on every frame, cached afterwards)...")
            dump(input_path, model, det_path, det_conf, imgsz)
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
        #     The finder is TrackNet or WASB (core.finders picks by checkpoint: --tracknet <ckpt>).
        if tracknet:
            from core import finders
            from core.tracknet_fill import apply as tracknet_fill
            name = finders.kind(tracknet)
            tn = finders.candidates(input_path, dets, tracknet, cache_path=finders.cache_path(stem, tracknet, output_dir),
                                    log=lambda m: print(m))
            if tn is not None:
                ball_px = (dets.get("profile") or {}).get("ball_px") or float(np.median(
                    [max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= 0.4] or [30.0]))
                got = tracknet_fill(track, finders.correct(tn, ball_px, tracknet), ball_px)
                print(f"[VolleyVision] {name} filled {got['gap']} gap frames, extended track ends by {got['end']} frames")

    s = track.summary()
    print(f"[VolleyVision] ball tracked in {s['coverage']:.1%} of frames "
          f"(detected {s['detected']}, weak {s['weak']}, tracknet {s['tracknet']}, filled {s['filled']}, none {s['none']})")
    clip_dir = paths.clip_dir(stem, output_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

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
        contact_path = str(paths.contacts(stem, output_dir))
        with open(contact_path, "w") as fh:
            json.dump({"schema": "vv-contacts/1", "video": str(input_path), "fps": fps,
                       "flights": [{"f0": fl.f0, "f1": fl.f1, "n": int(fl.n), "rms": round(fl.rms, 2),
                                    "cx": [round(v, 5) for v in fl.cx], "cy": [round(v, 5) for v in fl.cy]}
                                   for fl in flights],
                       "contacts": [c.to_dict() for c in found]}, fh, indent=1)
        # gaps inside a flight follow that flight's parabola (real detections are kept)
        from core.flights import apply_flights
        from core import finders
        track.pos, track.state = apply_flights(track, flights, found, place_tracknet=finders.place_on_arc(arc_finder))
        kinds = {k: sum(c.kind == k for c in found) for k in ("touch", "bounce", "other")}
        print(f"[VolleyVision] {len(flights)} flights, {kinds['touch']} touches, {kinds['bounce']} bounces, "
              f"{kinds['other']} other contacts  →  {contact_path}")
        for c in found:
            print(f"    {c.t:6.2f}s  frame {c.frame:7.1f}  {c.kind:6s}{'  (ball hidden at contact)' if c.in_gap else ''}")

    track_path = str(paths.track(stem, output_dir))
    with open(track_path, "w") as fh:
        json.dump({"schema": "vv-track/2", "video": str(input_path), "fps": fps, "width": width,
                   "height": height, "linker": "finder-first" if pipeline == "finder-first" else linker,
                   "frames": track.to_rows()}, fh)
    print(f"[VolleyVision] trajectory  →  {track_path}")

    # 4. render
    if not render:
        print(f"[VolleyVision] Done in {time.time() - t0:.1f}s (no render)")
        return track_path
    out_path = str(paths.video(stem, output_dir))
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
