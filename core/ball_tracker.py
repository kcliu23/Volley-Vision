import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


class BallTracker:

    _MAX_LOST    = 3
    _SMOOTH_WIN  = 3

    _MIN_BALL_PX = 13
    _MAX_BALL_PX = 85

    # Gate around the Kalman-predicted position: a candidate farther than this
    # is treated as a different object (e.g. a second ball), not the tracked one.
    # Grows a bit per coasted frame, since the prediction gets less certain
    # the longer it goes uncorrected.
    _GATE_BASE_PX   = 200
    _GATE_GROWTH    = 0.5

    # Weak-lock watchdog: recovers from a persistent wrong lock (tracking a
    # decoy that stays visible every frame, so _MAX_LOST never triggers).
    # Out-of-frame loss doesn't touch this path — no ball visible means no
    # in-gate detection at all, so it falls straight to the _MAX_LOST reset
    # below and re-acquires ungated once the ball reappears anywhere.
    _WEAK_CONF   = 0.50   # locked-on detection weaker than this counts as "shaky"
    _STRONG_CONF = 0.75   # an outside candidate this strong counts as a rival
    _WATCHDOG_N  = 6      # consecutive frames both streaks must hold before switching

    def __init__(
        self,
        model_path: str = "models/best4.pt",
        conf: float = 0.40,
        trail_len: int = 40,
        ignore_regions: list[tuple[int, int, int, int]] | None = None,
    ):
        self.model          = YOLO(model_path)
        self.conf           = conf
        self.ignore_regions = ignore_regions or []   # [(x1,y1,x2,y2), ...]

        self.trail: deque[tuple[float, float] | None] = deque(maxlen=trail_len)
        self.position: tuple[float, float] | None = None

        self._kf       = self._build_kf()
        self._kf_ready = False
        self._lost     = 0
        self._prev_pos: tuple[float, float] | None = None
        self._weak_streak  = 0
        self._decoy_streak = 0
        self._pending_pos: tuple[float, float] | None = None

    # ── Kalman ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_kf() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-1
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        return kf

    def _kf_init(self, x, y):
        kf = self._build_kf()
        state = np.array([[x],[y],[0],[0]], np.float32)
        kf.statePre = kf.statePost = state.copy()
        self._kf = kf
        self._kf_ready = True

    def _kf_correct(self, x, y) -> tuple[float, float]:
        self._kf.predict()
        r = self._kf.correct(np.array([[x],[y]], np.float32))
        return float(r[0]), float(r[1])

    def _kf_predict(self) -> tuple[float, float]:
        r = self._kf.predict()
        return float(r[0]), float(r[1])

    # ── Detection ─────────────────────────────────────────────────────────────

    def _in_ignore_region(self, cx: float, cy: float) -> bool:
        for x1, y1, x2, y2 in self.ignore_regions:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _candidates(self, result) -> list[tuple[float, float, float]]:
        """All size- and region-filtered detections this frame, as (conf, cx, cy)."""
        out = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w < self._MIN_BALL_PX or h < self._MIN_BALL_PX:
                continue
            if w > self._MAX_BALL_PX or h > self._MAX_BALL_PX:
                continue
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            if self._in_ignore_region(cx, cy):
                continue
            out.append((float(box.conf[0]), cx, cy))
        return out

    def _select_detection(self, candidates, ref_xy):
        """Pick which candidate is the ball, and flag a sustained rival for the watchdog.

        With a predicted position (ref_xy) to compare against: prefer the
        candidate NEAREST to it, not the most confident one — a decoy ball can
        outscore the real one on confidence but shouldn't be nearer than the
        real one's predicted spot. Anything outside the gate is ignored, same
        as the old jump-rejection, except the gate now grows a little per
        coasted frame instead of staying fixed.

        Without a prediction (cold start / just reset) there's nothing to
        compare position against, so this falls back to highest confidence
        anywhere in frame — this is also what re-acquires the ball after
        it's been out of frame for a while, deliberately ungated since it
        may reappear somewhere unrelated to where it left. That also means
        a single spurious detection during a long blind gap (a knee pad, a
        logo) would otherwise be snapped onto immediately, so a fresh lock
        needs the same spot to show up two frames running before it commits.
        """
        if not candidates:
            self._pending_pos = None
            return None, None, None

        if ref_xy is None:
            conf, cx, cy = max(candidates, key=lambda c: c[0])
            if (self._pending_pos is not None
                    and np.hypot(cx - self._pending_pos[0], cy - self._pending_pos[1]) <= self._GATE_BASE_PX):
                self._pending_pos = None
                return (cx, cy), conf, None
            self._pending_pos = (cx, cy)
            return None, None, None

        self._pending_pos = None

        gate = self._GATE_BASE_PX * (1 + self._GATE_GROWTH * min(self._lost, self._MAX_LOST))
        dists = [(c, np.hypot(c[1] - ref_xy[0], c[2] - ref_xy[1])) for c in candidates]
        in_gate  = [(c, d) for c, d in dists if d <= gate]
        outside  = [(c, d) for c, d in dists if d > gate]

        det, det_conf = None, None
        if in_gate:
            (conf, cx, cy), _ = min(in_gate, key=lambda cd: cd[1])
            det, det_conf = (cx, cy), conf

        rival = max(outside, key=lambda cd: cd[0][0]) if outside else None
        return det, det_conf, rival

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update_from_result(self, result) -> tuple[float, float] | None:
        candidates = self._candidates(result)

        # Peek the predicted position without consuming it — .predict() is
        # safe to call more than once per frame since it only reads statePost,
        # which _kf_correct()/_kf_predict() below will still update once.
        ref_xy = None
        if self._kf_ready:
            p = self._kf.predict()
            ref_xy = (float(p[0]), float(p[1]))

        det, det_conf, rival = self._select_detection(candidates, ref_xy)

        # Weak-lock watchdog — only meaningful while a lock exists to be weak.
        if ref_xy is not None:
            self._weak_streak = self._weak_streak + 1 if (det is None or det_conf < self._WEAK_CONF) else 0
            strong_rival = rival is not None and rival[0][0] >= self._STRONG_CONF
            self._decoy_streak = self._decoy_streak + 1 if strong_rival else 0

            if self._weak_streak >= self._WATCHDOG_N and self._decoy_streak >= self._WATCHDOG_N:
                conf, cx, cy = rival[0]
                det, det_conf = (cx, cy), conf
                self._kf_ready = False   # re-init fresh at the new lock, don't blend with the old track
                self._weak_streak = self._decoy_streak = 0

        if det is not None:
            x, y = det
            if not self._kf_ready:
                self._kf_init(x, y)
            x, y = self._kf_correct(x, y)
            self._lost = 0
            pos = (x, y)
        else:
            self._lost += 1
            if self._kf_ready and self._lost <= self._MAX_LOST:
                pos = self._kf_predict()
            else:
                self._kf = self._build_kf()
                self._kf_ready = False
                pos = None
                self._weak_streak = self._decoy_streak = 0

        self._prev_pos = pos
        self.trail.append(pos)
        self.position = pos
        return pos

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()

        total_pts = len([p for p in self.trail if p is not None])

        segment: list[tuple[float, float]] = []
        drawn = 0
        for p in self.trail:
            if p is not None:
                segment.append(p)
            else:
                if len(segment) > 1:
                    pts = self._smooth(segment)
                    for i in range(1, len(pts)):
                        drawn += 1
                        alpha = drawn / max(total_pts, 1)
                        cv2.line(out,
                                 (int(pts[i-1][0]), int(pts[i-1][1])),
                                 (int(pts[i][0]),   int(pts[i][1])),
                                 (int(255*alpha), int(200*(1-alpha)), 30),
                                 max(1, int(3*alpha)), cv2.LINE_AA)
                segment = []

        if len(segment) > 1:
            pts = self._smooth(segment)
            for i in range(1, len(pts)):
                drawn += 1
                alpha = drawn / max(total_pts, 1)
                cv2.line(out,
                         (int(pts[i-1][0]), int(pts[i-1][1])),
                         (int(pts[i][0]),   int(pts[i][1])),
                         (int(255*alpha), int(200*(1-alpha)), 30),
                         max(1, int(3*alpha)), cv2.LINE_AA)

        if self.position:
            cx, cy = int(self.position[0]), int(self.position[1])
            cv2.circle(out, (cx, cy), 10, (0, 255, 255), -1)
            cv2.circle(out, (cx, cy), 12, (0, 180, 180),  2)
        return out

    def _smooth(self, pts: list) -> list:
        half = self._SMOOTH_WIN // 2
        smoothed = []
        for i in range(len(pts)):
            window = pts[max(0, i - half): i + half + 1]
            smoothed.append((
                sum(p[0] for p in window) / len(window),
                sum(p[1] for p in window) / len(window),
            ))
        return smoothed

    def reset(self):
        self.trail.clear()
        self.position  = None
        self._kf       = self._build_kf()
        self._kf_ready = False
        self._lost     = 0
        self._prev_pos = None
        self._weak_streak  = 0
        self._decoy_streak = 0
        self._pending_pos  = None
