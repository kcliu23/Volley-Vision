# VolleyVision

Volleyball ball tracking: YOLOv8 detection + Kalman filter, drawn as a trail on the output video, plus touch (contact) detection from the ball trajectory.

## Usage

```bash
pip install -r requirements.txt
python main.py path/to/video.mp4
```

Outputs in `output/`:

- `<name>_tracked.mp4` — ball trail, with each detected touch ringed and numbered
- `<name>_track.json` — per-frame raw detection + Kalman position
- `<name>_touches.json` — detected touches (frame, time, position, velocity in/out, turn angle)

The tracked video is written to `output/<name>_tracked.mp4`. If `ffmpeg` is installed, the output is H.264 with the original audio.

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output` | Directory for the tracked video |
| `--model` | `models/best4.pt` | YOLO weights |
| `--conf` | `0.40` | Detection confidence threshold |
| `--trail-len` | `40` | Ball trail length in frames |
| `--no-preview` | off | Disable the live preview window (press `q` to quit it) |
| `--raw` | off | Show raw YOLO boxes (no Kalman / trail) |
| `--no-touches` | off | Skip touch detection and overlay |

### Re-running touch detection only

Touch detection works from the saved trajectory, so it can be tuned without re-running YOLO:

```bash
python -m tools.detect_touches output/test10_track.json
python -m tools.detect_touches output/test10_track.json --frames tests/test10.mov   # contact sheets per touch
```

## Layout

```
main.py                 CLI entry point
core/
  ball_tracker.py       YOLO + Kalman tracker, trail drawing
  touch_detector.py     Touch detection (split-parabola impulse fit on raw detections)
  video_pipeline.py     Read video, track, detect touches, write output
models/best4.pt         Trained weights
training/               Training notebook and dataset scripts
tools/                  Dataset helpers, detect_touches.py
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.
