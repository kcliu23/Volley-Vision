"""Calibrate a clip's camera by clicking known court points (once per camera position).

    python -m tools.calibrate_court videos/test74.mov                # men's / coed net (2.43 m)
    python -m tools.calibrate_court videos/test74.mov --net women    # 2.24 m

Shows the clip's empty court (the median of frames across the clip, so players fade out; the
camera must not move during the clip). What to click is named at the top. Two kinds:
  LINES first (net tape, net band, sidelines, centre / attack / end lines): click 2-3 points
     ANYWHERE along the line, spread out, then ] for the next one (it moves on by itself after 3).
     No corner needed: use these for faint lines and for the net (the net is wider than the court).
  POINTS after (corners, where the antenna crosses the net): click the exact spot, only if you
     can really see it.
Anything you can't see: s to skip. Once there are enough clicks, the camera is solved after every
click and the court is drawn back onto the picture (green lines), with the fit error. Where the
green lines sit on the real lines, it's right.

Convention: "near" = the half closer to the camera, "far" = beyond the net; "left" / "right" =
the sidelines as seen in the picture. "net top" / "net bottom" / "antenna top" are where the
antenna (on the sideline) crosses the top tape / the bottom band of the net, and its tip.

  click          place a point / add a click on the current line (a magnifier follows the mouse)
  next:  ]  n  ->  Tab  s  or the "next >" button     back:  [  b  <-  or the "< back" button
                 (go back to any line or point to add clicks, undo them, or redo them)
  u / Backspace  remove the current point or the current line's last click (or the last one placed)
  e              contrast boost on / off: faint lines (far side) stand out
  Enter          save to data/cameras/<clip>.json     q / Esc  quit (asks to save if changed)

Re-solve saved calibrations from their clicked points (after a solver change; no window):
    python -m tools.calibrate_court holdout/holdout*.mov --resolve

Good calibrations: the net tape and band, both sidelines and 2+ cross lines (lines), plus the
corners you can clearly see (points). rms under ~2-3 px; the green lines on the real lines.
Only click points you can actually see: a guessed corner (a line too faint to see, an antenna
that isn't there) pulls the whole camera off. Skip it (s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


from core.court import LINE_ORDER, NET_HEIGHT, ORDER, keypoints, lines

ITEMS = LINE_ORDER + ORDER          # lines first, then points
LINE_CLICKS = 3                     # clicks per line before moving on by itself

MAX_W, MAX_H, TOP_H = 1500, 900, 96
BACK_BTN, NEXT_BTN = (10, 64, 200, 90), (210, 64, 400, 90)     # x1, y1, x2, y2 in the top bar
LEFT_KEYS = {63234, 65361, 2424832, 81}       # arrow left: macOS, Linux, Windows, (masked)
RIGHT_KEYS = {63235, 65363, 2555904, 83}
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


def enhance(img):
    """Local contrast boost: faint painted lines on a light floor stand out (grey picture)."""
    import cv2
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(L)
    blur = cv2.GaussianBlur(L, (0, 0), 8)
    L = cv2.addWeighted(L, 2.5, blur, -1.5, 0)
    return cv2.cvtColor(L, cv2.COLOR_GRAY2BGR)


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
    cam = solve(d.get("points", {}), old.width, old.height, d.get("net_height", 2.43), line_clicks=d.get("line_clicks"))
    cam.save(out, {k: d[k] for k in ("video", "net_height", "points", "line_clicks") if k in d})
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
    lcl: dict = {}                                      # line name -> [(u, v), ...]
    if out.exists():
        old = json.load(open(out))
        pts = {k: tuple(v) for k, v in old.get("points", {}).items()}
        lcl = {k: [tuple(p) for p in v] for k, v in old.get("line_clicks", {}).items()}
        net = old.get("net_height", net)
        print(f"[calibrate] {stem}: loaded {len(pts)} points and {sum(map(len, lcl.values()))} line clicks from {out}")
    print(f"[calibrate] {stem}: building the empty-court picture ...", flush=True)
    bg = background(a.video)
    H, W = bg.shape[:2]
    scale = min(MAX_W / W, (MAX_H - TOP_H) / H, 1.0)
    plain = bg
    boosted = enhance(bg)                               # faint lines made visible (key e)
    views = {False: plain, True: boosted}
    small = {k: (cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else v.copy())
             for k, v in views.items()}
    state_view = {"boost": False}
    shown = small[False]
    kp = keypoints(net)
    cur = next((i for i, n in enumerate(ITEMS) if n not in pts and len(lcl.get(n, [])) < 2), 0)
    state = {"mouse": None, "click": None}
    cam, err, changed = None, "", False

    def resolve():
        nonlocal cam, err
        cam, err = None, ""
        if 2 * len(pts) + sum(len(v) for v in lcl.values() if len(v) >= 2) >= 10:
            try:
                cam = solve(pts, W, H, net, line_clicks=lcl)
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
            col = (0, 255, 255) if n == ITEMS[cur] else (255, 120, 0)
            cv2.circle(img, c, 5, col, 2, cv2.LINE_AA)
            cv2.putText(img, n, (c[0] + 7, c[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            if cam is not None:                          # where the solved camera puts this point
                pu = cam.project([kp[n]])[0] * scale
                cv2.line(img, c, (int(pu[0]), int(pu[1]) + TOP_H), (0, 0, 255), 1, cv2.LINE_AA)
        for n, cl in lcl.items():                        # line clicks, and the line through them
            col = (0, 255, 255) if n == ITEMS[cur] else (255, 0, 200)
            P = [(int(u * scale), int(v * scale) + TOP_H) for u, v in cl]
            for c in P:
                cv2.circle(img, c, 4, col, 1, cv2.LINE_AA)
            if len(P) >= 2:
                cv2.line(img, min(P), max(P), col, 1, cv2.LINE_AA)
            if P:
                cv2.putText(img, n, (P[0][0] + 7, P[0][1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        if state["mouse"] and state["mouse"][1] >= TOP_H:  # magnifier
            mx, my = state["mouse"]
            x0, y0 = int(mx / scale), int((my - TOP_H) / scale)
            r = 40
            crop = views[state_view["boost"]][max(0, y0 - r):y0 + r, max(0, x0 - r):x0 + r]
            if crop.size:
                z = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                zh, zw = z.shape[:2]
                ox = 8 if mx > shown.shape[1] / 2 else img.shape[1] - zw - 8
                oy = TOP_H + 8
                img[oy:oy + zh, ox:ox + zw] = z
                cxz, cyz = ox + (x0 - max(0, x0 - r)) * 3, oy + (y0 - max(0, y0 - r)) * 3
                cv2.line(img, (cxz - 12, cyz), (cxz + 12, cyz), (0, 255, 255), 1)
                cv2.line(img, (cxz, cyz - 12), (cxz, cyz + 12), (0, 255, 255), 1)
        name = ITEMS[cur]
        is_line = name in LINE_ORDER
        what = (f"LINE: {name.upper()} - click 2-3 spots anywhere along it ({len(lcl.get(name, []))} so far)"
                if is_line else f"POINT: {name.upper()} - only if you can see it")
        top = f"{what}   ({cur + 1}/{len(ITEMS)})   u undo  e boost  Enter save  q quit"
        cv2.putText(img, top, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        def done_mark(n):
            return "*" if (n in pts or len(lcl.get(n, [])) >= 2) else ""
        prv, nxt_ = ITEMS[(cur - 1) % len(ITEMS)], ITEMS[(cur + 1) % len(ITEMS)]
        for (x1, y1, x2, y2), label in ((BACK_BTN, "< back"), (NEXT_BTN, "next >")):
            cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(img, label, (x1 + 55, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"back: {prv}{done_mark(prv)}    next: {nxt_}{done_mark(nxt_)}    (* = done)",
                    (415, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if cam is not None:
            i = cam.info
            fit = (f"fit: {i['rms_px']:.1f} px rms   focal {cam.f:.0f} px (+-{i.get('f_std', float('nan')):.0f})"
                   f"   FOV {cam.fov_deg:.0f} deg   camera at x={cam.position[0]:.1f} y={cam.position[1]:.1f} "
                   f"z={cam.position[2]:.2f} m")
            worst = max(list(i["residual_px"].items()) + list(i.get("line_px", {}).items()), key=lambda kv: kv[1])
            fit += f"   worst: {worst[0]} ({worst[1]:.0f} px)"
            if i.get("warning"):
                fit = "!! " + i["warning"] + "   " + fit
            col = (120, 255, 120) if i["rms_px"] < 3 else (80, 160, 255)
        else:
            fit, col = (err or "more clicks needed: net tape, net band, both sidelines and a line across the court"), (180, 180, 180)
        cv2.putText(img, fit, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.imshow(WIN, img)
        raw = cv2.waitKeyEx(20)
        key = raw & 0xFF if raw != -1 else 255
        if raw in LEFT_KEYS or (raw > 255 and (raw & 0xFF) == 81 and raw != -1):
            key, raw = ord("["), ord("[")
        elif raw in RIGHT_KEYS or (raw > 255 and (raw & 0xFF) == 83):
            key, raw = ord("]"), ord("]")

        if state["click"] is not None:
            x, y = state["click"]
            state["click"] = None
            inside = lambda b: b[0] <= x <= b[2] and b[1] <= y <= b[3]   # noqa: E731
            if inside(BACK_BTN):
                key = ord("[")
            elif inside(NEXT_BTN):
                key = ord("]")
            elif y >= TOP_H:
                uvc = (round(x / scale, 1), round((y - TOP_H) / scale, 1))
                if is_line:
                    lcl.setdefault(name, []).append(uvc)
                    done = len(lcl[name]) >= LINE_CLICKS
                else:
                    pts[name] = uvc
                    done = True
                changed = True
                resolve()
                if done:
                    nxt = [i for i in range(cur + 1, len(ITEMS)) if ITEMS[i] not in pts and len(lcl.get(ITEMS[i], [])) < 2]
                    cur = nxt[0] if nxt else cur
        if key == 255:
            continue
        k = chr(key) if key < 128 else ""
        if k in ("s", "\t", "]", "n"):
            cur = (cur + 1) % len(ITEMS)
        elif k in ("[", "b"):
            cur = (cur - 1) % len(ITEMS)
        elif k == "e":
            state_view["boost"] = not state_view["boost"]
            shown = small[state_view["boost"]]
        elif k == "u" or key == 8 or key == 127:
            if name in pts:
                pts.pop(name)
            elif lcl.get(name):
                lcl[name].pop()
                if not lcl[name]:
                    lcl.pop(name)
            elif pts:
                pts.pop(list(pts)[-1])
            changed = True
            resolve()
        elif key in (13, 10):
            if cam is None:
                print("[calibrate] not solved yet: more clicks needed")
                continue
            cam.save(out, {"video": a.video, "net_height": net, "points": {k: list(v) for k, v in pts.items()},
                           "line_clicks": {k: [list(p) for p in v] for k, v in lcl.items()}})
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
