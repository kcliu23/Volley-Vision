# Moving through frames in the labeling window

The user steps frame by frame to put a mark on the exact contact frame. The frame on screen must be the frame number in the top bar, or every mark lands on the wrong picture. Stepping back has to stay fast late in a 47,000-frame game.

## Sub-features

- `step`: `a` / `d` move 1 frame, and `A` / `D` move 10 frames.
- `jump-mark`: `M` goes to the next marked touch.
- `play`: space plays and pauses, and `w` / `s` change speed (0.25x–4x; 2x keeps time since commit 6cee588).
- `seek`: `FrameSource` jumps straight to a frame only after `constant_rate()` and a seek-vs-decode pixel check pass. Every jump is rechecked by its timestamp, and a miss switches seeking off. Without seeking, it decodes from frame 0.

## How to get to it (user POV)

- In `tools.review`, press `p` or `M` to get near a touch, then `a` until the hand meets the ball.

## Driving it with drive_review.py

Preconditions: a sandbox for `full_rally_test` (long, so seeking matters), and `doctor.sh` reports ffprobe ok.

Steps:
- **Step back.** `--keys "p a a a A"`. The top bar frame numbers in the PNGs go f, f-1, f-2, f-3, f-13.
- **Picture matches the number.** Decode the same frame from frame 0 in a separate script (`cv2.VideoCapture`, `grab()` up to it), resize by the tool's scale (`min(1500/W, (860-TIMELINE_H-TOP_H)/H, 1)`), and compare it with the PNG's frame area below `TOP_H`. It should match up to JPEG noise and differ clearly from f±1 in a moving scene.
- **Stays fast.** Time `src.load` for 200 steps back from frame ~40,000 with `tools.review.FrameSource`. Expect one wait of about 2 s per ~240 frames, the rest instant. Before the fix, each press took about 28 s.
- **Seek fallback.** `python -m pytest tests/test_review_frames.py`. It covers a capture whose seeks land one frame late.
- **Proof.** Keep the PNG sequence and the comparison numbers in the evidence directory.

## Gotchas

- The start log line `video seeks (fast jumps)` or `decodes from the start on jumps back` says which mode is on. It's decided on the first jump, not at open.
- Phone `.mov` files can have uneven frame times, so they never seek. That's correct, just slow.
- Playback in the fake window runs as fast as the polls (`wait:N`). Don't time 2x speed through `drive_review.py`; use `tools.review.play_steps` timing instead.
