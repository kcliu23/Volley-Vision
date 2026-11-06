"""Calibrate a clip's camera by clicking known court points (once per camera position).

    python -m tools.calibrate_court videos/test74.mov                # men's / coed net (2.43 m)
    python -m tools.calibrate_court videos/test74.mov --net women    # 2.24 m

Shows the clip's empty court (the median of frames across the clip, so players fade out; the
camera must not move during the clip). The name of the point to click is at the top: click it,
and the tool moves to the next one. Points you can't see: press s to skip. As soon as 6 points
are in, the camera is solved after every click and the court is drawn back onto the picture
(green lines), with the fit error. Where the green lines sit on the real lines, it's right.

Convention: "near" = the half closer to the camera, "far" = beyond the net; "left" / "right" =
the sidelines as seen in the picture. "net top" / "net bottom" / "antenna top" are where the
antenna (on the sideline) crosses the top tape / the bottom band of the net, and its tip.

  click          place the current point (a magnifier follows the mouse)
  s / Tab        skip this point          [ / ]  previous / next point
  u / Backspace  remove the current point (or the last one placed)
  Enter          save to data/cameras/<clip>.json     q / Esc  quit (asks to save if changed)

Re-solve saved calibrations from their clicked points (after a solver change; no window):
    python -m tools.calibrate_court holdout/holdout*.mov --resolve

Good calibrations: 8+ points spread over the picture, including some net/antenna points
(heights) and some far-court points. rms under ~2 px; the green lines on the real lines.
Only click points you can actually see: a guessed corner (a line too faint to see, an antenna
that isn't there) pulls the whole camera off. Skip it (s).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.court import NET_HEIGHT, ORDER, keypoints, lines  # noqa: E402

MAX_W, MAX_H, TOP_H = 1500, 900, 64
CAM_DIR = Path("data/cameras")


def background(video: str, n: int = 25):
    """Median of n frames spread over the clip: players move, the court doesn't."""
    import cv2
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    want = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    frames, f = [], 0
    while len(frames) < len(want):
        ok = cap.grab()
        if not ok:
            break
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                frames.append(img)
        f += 1
    cap.release()
    if not frames:
        raise SystemExit(f"can't read {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def resolve_saved(video: str):
    """Solve a saved calibration again from its clicked points and save it back."""
    from core.camera import Camera, solve
    stem = Path(video).stem
    out = CAM_DIR / f"{stem}.json"
    if not out.exists():
        print(f"[calibrate] {stem}: no saved calibration ({out}); click it first")
        return
    d = json.load(open(out))
    old = Camera.load(out)
    cam = solve(d["points"], old.width, old.height, d.get("net_height", 2.43))
    cam.save(out, {k: d[k] for k in ("video", "net_height", "points") if k in d})
    print(f"[calibrate] {stem}: rms {old.info.get('rms_px')} -> {cam.info['rms_px']} px; focal {old.f:.0f} -> {cam.f:.0f}; "
          f"k1 {old.k1:+.2f} -> {cam.k1:+.2f}; camera {[round(float(v), 2) for v in old.position]} -> "
          f"{[round(float(v), 2) for v in cam.position]} m  → {out}")


def main():
    import cv2
    from core.camera import solve
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", nargs="+")
    ap.add_argument("--net", default="men", help="men (2.43 m), women (2.24 m), or a height in meters")
    ap.add_argument("--resolve", action="store_true", help="re-solve the saved calibration(s) from their points, no window")
    a = ap.parse_args()
    if a.resolve:
        for v in a.video:
            resolve_saved(v)
        return
    if len(a.video) > 1:
        raise SystemExit("one clip at a time when clicking (several only with --resolve)")
    a.video = a.video[0]
    net = NET_HEIGHT.get(a.net) or float(a.net)
    stem = Path(a.video).stem
    out = CAM_DIR / f"{stem}.json"
    pts: dict = {}
    if out.exists():
        old = json.load(open(out))
        pts = {k: tuple(v) for k, v in old.get("points", {}).items()}
        net = old.get("net_height", net)
        print(f"[calibrate] {stem}: loaded {len(pts)} points from {out}")
    print(f"[calibrate] {stem}: building the empty-court picture ...", flush=True)
    bg = background(a.video)
    H, W = bg.shape[:2]
    scale = min(MAX_W / W, (MAX_H - TOP_H) / H, 1.0)
    shown = cv2.resize(bg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else bg.copy()
    kp = keypoints(net)
    cur = next((i for i, n in enumerate(ORDER) if n not in pts), 0)
    state = {"mouse": None, "click": None}
    cam, err, changed = None, "", False

    def resolve():
        nonlocal cam, err
        cam, err = None, ""
        if len(pts) >= 6:
            try:
                cam = solve(pts, W, H, net)
            except Exception as e:                       # bad clicks: say so, keep going
                err = str(e)

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    WIN = "VolleyVision court calibration"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    resolve()
    while True:
        img = np.zeros((TOP_H + shown.shape[0], shown.shape[1], 3), np.uint8)
        img[TOP_H:] = shown
        if cam is not None:                              # the court drawn back onto the picture
            for p, q in lines(net):
                seg = np.linspace(p, q, 40)
                ok = cam.depth(seg) > 0.1
                uv = cam.project(seg[ok]) * scale
                for (x1, y1), (x2, y2) in zip(uv[:-1], uv[1:]):
                    cv2.line(img, (int(x1), int(y1) + TOP_H), (int(x2), int(y2) + TOP_H), (60, 230, 60), 1, cv2.LINE_AA)
        for n, (u, v) in pts.items():
            c = (int(u * scale), int(v * scale) + TOP_H)
            col = (0, 255, 255) if n == ORDER[cur] else (255, 120, 0)
            cv2.circle(img, c, 5, col, 2, cv2.LINE_AA)
            cv2.putText(img, n, (c[0] + 7, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            if cam is not None:                          # where the solved camera puts this point
                pu = cam.project([kp[n]])[0] * scale
                cv2.line(img, c, (int(pu[0]), int(pu[1]) + TOP_H), (0, 0, 255), 1, cv2.LINE_AA)
        if state["mouse"] and state["mouse"][1] >= TOP_H:  # magnifier
            mx, my = state["mouse"]
            x0, y0 = int(mx / scale), int((my - TOP_H) / scale)
            r = 40
            crop = bg[max(0, y0 - r):y0 + r, max(0, x0 - r):x0 + r]
            if crop.size:
                z = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                zh, zw = z.shape[:2]
                ox = 8 if mx > shown.shape[1] / 2 else img.shape[1] - zw - 8
                oy = TOP_H + 8
                img[oy:oy + zh, ox:ox + zw] = z
                cxz, cyz = ox + (x0 - max(0, x0 - r)) * 3, oy + (y0 - max(0, y0 - r)) * 3
                cv2.line(img, (cxz - 12, cyz), (cxz + 12, cyz), (0, 255, 255), 1)
                cv2.line(img, (cxz, cyz - 12), (cxz, cyz + 12), (0, 255, 255), 1)
        name = ORDER[cur]
        top = f"click: {name.upper()}   ({cur + 1}/{len(ORDER)}, {len(pts)} placed)   s skip  [ ] move  u remove  Enter save  q quit"
        cv2.putText(img, top, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if cam is not None:
            i = cam.info
            fit = (f"fit: {i['rms_px']:.1f} px rms   focal {cam.f:.0f} px (+-{i.get('f_std', float('nan')):.0f})"
                   f"   FOV {cam.fov_deg:.0f} deg   camera at x={cam.position[0]:.1f} y={cam.position[1]:.1f} "
                   f"z={cam.position[2]:.2f} m")
            worst = max(i["residual_px"].items(), key=lambda kv: kv[1])
            fit += f"   worst point: {worst[0]} ({worst[1]:.0f} px)"
            col = (120, 255, 120) if i["rms_px"] < 3 else (80, 160, 255)
        else:
            fit, col = (err or f"place {max(0, 6 - len(pts))} more point(s) to solve"), (180, 180, 180)
        cv2.putText(img, fit, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)
        key = cv2.waitKey(20) & 0xFF

        if state["click"] is not None:
            x, y = state["click"]
            state["click"] = None
            if y >= TOP_H:
                pts[name] = (round(x / scale, 1), round((y - TOP_H) / scale, 1))
                changed = True
                resolve()
                nxt = [i for i in range(cur + 1, len(ORDER)) if ORDER[i] not in pts]
                cur = nxt[0] if nxt else cur
        if key == 255:
            continue
        k = chr(key) if key < 128 else ""
        if k in ("s", "\t", "]"):
            cur = (cur + 1) % len(ORDER)
        elif k == "[":
            cur = (cur - 1) % len(ORDER)
        elif k == "u" or key == 8 or key == 127:
            if name in pts:
                pts.pop(name)
            elif pts:
                pts.pop(list(pts)[-1])
            changed = True
            resolve()
        elif key in (13, 10):
            if cam is None:
                print("[calibrate] not solved yet: place at least 6 points")
                continue
            cam.save(out, {"video": a.video, "net_height": net, "points": {k: list(v) for k, v in pts.items()}})
            changed = False
            print(f"[calibrate] {stem}: saved  →  {out}   (rms {cam.info['rms_px']} px, "
                  f"camera at {[round(float(v), 2) for v in cam.position]} m, FOV {cam.fov_deg:.0f} deg)")
        elif k == "q" or key == 27:
            if changed and cam is not None:
                print("[calibrate] unsaved changes: press Enter to save, q again to quit without saving")
                changed = False
                continue
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
