import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Shared CLAHE instance — same settings as auto_label.py so inference matches training
_clahe = cv2.createCLAHE(clipLimit=0.03, tileGridSize=(8, 8))

def apply_clahe(frame):
    # Apply CLAHE only to the L (lightness) channel in LAB space to avoid color shifts
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    enhanced = cv2.merge((_clahe.apply(l), a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class BallTracker:

    # Base gate radius — expands as more frames are lost
    _GATE_RADIUS = 120.0

    # How much the gate grows per lost frame, so ball can still be caught after fast movement
    _GATE_EXPAND_PER_FRAME = 10.0

    # Gravity added to vy each frame (pixels) during prediction
    _GRAVITY = 0.5

    def __init__(
        self,
        model_path: str = "models/best.pt",
        conf: float = 0.30,
        trail_len: int = 40,
        max_lost: int = 12,
    ):
        self.model     = YOLO(model_path)
        self.conf      = conf
        self.max_lost  = max_lost

        names = self.model.names
        ball_ids = [i for i, n in names.items() if "ball" in n.lower()]
        self.ball_ids = ball_ids or list(names.keys())

        self.trail: deque[tuple[float, float] | None] = deque(maxlen=trail_len)
        self.position: tuple[float, float] | None = None

        self._kf       = self._build_kf()
        self._kf_ready = False
        self._lost     = 0

        # Last known velocity — preserved across resets so new tracking starts with momentum
        self._last_vx  = 0.0
        self._last_vy  = 0.0

        # Static-fixture filter — ignores detections that haven't moved for _fp_age frames
        self._fp_buf    : deque[tuple[float, float]] = deque(maxlen=10)
        self._fp_radius = 25.0
        self._fp_age    = 6

    # ── Kalman ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_kf() -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        kf.errorCovPost        = np.eye(4, dtype=np.float32)
        return kf

    def _kf_init(self, x, y, vx=0.0, vy=0.0):
        # Seed with last known velocity instead of zero so tracking resumes with momentum
        self._kf.statePre  = np.array([[x],[y],[vx],[vy]], np.float32)
        self._kf.statePost = np.array([[x],[y],[vx],[vy]], np.float32)
        self._kf_ready = True

    def _kf_correct(self, x, y):
        self._kf.predict()
        r = self._kf.correct(np.array([[x],[y]], np.float32))
        # Save velocity after each successful correction
        self._last_vx = float(self._kf.statePost[2])
        self._last_vy = float(self._kf.statePost[3])
        return float(r[0]), float(r[1])

    def _kf_predict(self):
        r = self._kf.predict()
        # Apply gravity so arc curves downward naturally during lost frames
        self._kf.statePost[3] += self._GRAVITY
        return float(r[0]), float(r[1])

    def _kf_predicted_pos(self):
        state = self._kf.statePost
        return float(state[0]), float(state[1])

    # ── Static filter ─────────────────────────────────────────────────────────

    def _is_static(self, x, y) -> bool:
        if len(self._fp_buf) < self._fp_age:
            return False
        return all(
            np.hypot(x - px, y - py) < self._fp_radius
            for px, py in list(self._fp_buf)[-self._fp_age:]
        )

    # ── Per-frame update ──────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> tuple[float, float] | None:
        # Adaptive confidence: lower threshold when ball is lost to help re-acquire
        adaptive_conf = self.conf * (0.7 if 0 < self._lost <= self.max_lost else 1.0)

        det = self._detect(frame, adaptive_conf)

        if det is not None:
            x, y = det
            if not self._kf_ready:
                # Soft re-init: seed with last known velocity so tracking resumes smoothly
                self._kf_init(x, y, vx=self._last_vx, vy=self._last_vy)
            sx, sy = self._kf_correct(x, y)
            self._lost = 0

            # If we were interpolating during lost frames, replace the None entries
            # in the trail with linearly interpolated positions up to current position
            self._fill_trail_gap(sx, sy)

            pos = (sx, sy)

        else:
            self._lost += 1
            if self._kf_ready and self._lost <= self.max_lost:
                sx, sy = self._kf_predict()
                pos = (sx, sy)
            else:
                # Soft reset: preserve velocity so next re-acquisition starts with momentum
                self._kf       = self._build_kf()
                self._kf_ready = False
                pos            = None

        self.trail.append(pos)
        self.position = pos
        return pos

    def _fill_trail_gap(self, curr_x, curr_y):
        # Find how many trailing None entries exist in the trail (the lost-frame gap)
        trail_list = list(self.trail)
        gap = 0
        for p in reversed(trail_list):
            if p is None:
                gap += 1
            else:
                break

        if gap == 0:
            return

        # Find the last known position before the gap
        last_known = None
        for p in reversed(trail_list[:-gap] if gap > 0 else trail_list):
            if p is not None:
                last_known = p
                break

        if last_known is None:
            return

        # Replace the None entries with linearly interpolated positions
        lx, ly = last_known
        for i, idx in enumerate(range(len(trail_list) - gap, len(trail_list))):
            t = (i + 1) / (gap + 1)
            interp = (lx + t * (curr_x - lx), ly + t * (curr_y - ly))
            self.trail[idx] = interp

    def _detect(self, frame: np.ndarray, conf: float) -> tuple[float, float] | None:
        results = self.model(apply_clahe(frame), conf=conf, verbose=False)[0]

        candidates = []
        for box in results.boxes:
            if int(box.cls[0]) not in self.ball_ids:
                continue
            cx = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            cy = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            candidates.append((float(box.conf[0]), cx, cy))
        candidates.sort(reverse=True)

        # Gate expands the longer the ball is lost, so fast-moving balls can still be caught
        if self._kf_ready:
            px, py = self._kf_predicted_pos()
            gate = self._GATE_RADIUS + self._GATE_EXPAND_PER_FRAME * self._lost
            candidates = [
                (c, x, y) for c, x, y in candidates
                if np.hypot(x - px, y - py) <= gate
            ]

        for _, cx, cy in candidates:
            if self._is_static(cx, cy):
                self._fp_buf.append((cx, cy))
                continue
            self._fp_buf.append((cx, cy))
            return (cx, cy)

        if candidates:
            self._fp_buf.append((candidates[0][1], candidates[0][2]))
        return None

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray) -> np.ndarray:
        out   = frame.copy()
        trail = [p for p in self.trail if p is not None]

        for i in range(1, len(trail)):
            alpha = i / len(trail)
            cv2.line(out,
                     (int(trail[i-1][0]), int(trail[i-1][1])),
                     (int(trail[i][0]),   int(trail[i][1])),
                     (int(255*alpha), int(200*(1-alpha)), 30),
                     max(1, int(3*alpha)), cv2.LINE_AA)

        if self.position:
            cx, cy = int(self.position[0]), int(self.position[1])
            cv2.circle(out, (cx, cy), 10, (0, 255, 255), -1)
            cv2.circle(out, (cx, cy), 12, (0, 180, 180),  2)

        return out

    def reset(self):
        self.trail.clear()
        self.position  = None
        self._kf       = self._build_kf()
        self._kf_ready = False
        self._lost     = 0
        self._last_vx  = 0.0
        self._last_vy  = 0.0
        self._fp_buf.clear()
