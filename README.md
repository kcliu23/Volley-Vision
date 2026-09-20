# VolleyVision

Volleyball ball tracking: YOLOv8 detection + Kalman filter, drawn as a trail on the output video.

## Usage

```bash
pip install -r requirements.txt
python main.py path/to/video.mp4
```

The tracked video is written to `output/<name>_tracked.mp4`. If `ffmpeg` is installed, the output is H.264 with the original audio.

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `output` | Directory for the tracked video |
| `--model` | `models/best4.pt` | YOLO weights |
| `--conf` | `0.40` | Detection confidence threshold |
| `--trail-len` | `40` | Ball trail length in frames |
| `--no-preview` | off | Disable the live preview window (press `q` to quit it) |
| `--raw` | off | Show raw YOLO boxes (no Kalman / trail) |

## Layout

```
main.py                 CLI entry point
core/
  ball_tracker.py       YOLO + Kalman tracker, trail drawing
  video_pipeline.py     Read video, track, write output
models/best4.pt         Trained weights
training/               Training notebook and dataset scripts
tools/                  Dataset helpers (bbox stats, Roboflow upload)
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.
