import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO


class BallTracker:

    _MAX_LOST    = 3
    _MAX_JUMP_PX = 200
    _SMOOTH_WIN  = 3

    _MIN_BALL_PX = 13
    _MAX_BALL_PX = 85

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

    def _best_detection(self, result) -> tuple[float, float] | None:
        best = None
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
            conf = float(box.conf[0])
            if best is None or conf > best[0]:
                best = (conf, cx, cy)
        return (best[1], best[2]) if best else None

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update_from_result(self, result) -> tuple[float, float] | None:
        det = self._best_detection(result)

        if det is not None and self._prev_pos is not None:
            if np.hypot(det[0] - self._prev_pos[0], det[1] - self._prev_pos[1]) > self._MAX_JUMP_PX:
                det = None

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
