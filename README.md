# VolleyVision

**Volleyball analytics from a single phone video.** VolleyVision follows the ball through every
rally, finds each touch, rebuilds the ball's flight in 3D, and identifies the players, all from one
camera behind the court.

## What it does

| | |
|---|---|
| **Ball tracking** | Finds the ball in every frame, through blur, occlusion and screen-recording glitches; on the ball in about 95% of frames on held-out test clips. |
| **Touch detection** | Splits each rally into flights and finds the touches between them: serve, pass, set, hit, block, dig. |
| **3D ball flight** | Turns the 2D track into the ball's path in metres: height, speed, apex, net clearance, where it lands. |
| **Players** | Body pose for everyone on court, including servers behind the far baseline, and player numbering with re-identification guesses. |
| **Outputs** | An overlay video, an interactive 3D viewer, and JSON for your own analysis. |

## How it works

1. **2D ball tracking.** A multi-frame ball finder (WASB) proposes candidates on every frame, YOLO
   confirms them on small crops, and a physics-aware linker picks one consistent path for the clip.
2. **Court calibration.** Click a few court landmarks once per camera position; the camera's
   position and lens are solved from them.
3. **People.** A pose model tracks every player, with a zoomed pass on the far court.
4. **3D reconstruction.** Each rally is fitted as a chain of physical flights (gravity, air drag,
   true frame timing), joined at the touches and anchored to the player who touched the ball.

## Requirements

- Python 3.10+, macOS or Linux; an Apple-silicon or CUDA GPU is recommended
- `ffmpeg` (optional: keeps the original audio in rendered videos)
- Model weights in `models/`

```bash
pip install -e ".[ml]"
pip install numba            # optional: about 3.5x faster 3D fitting
```

## Quick start

```bash
# 1. Calibrate the court (once per camera position)
python -m tools.calibrate_court videos/match.mov

# 2. Track, find people, reconstruct in 3D, and build the viewer
python -m main videos/match.mov

# 3. Optional: the overlay videos too
python -m main videos/match.mov --video
```

Results land in `output/<clip>/`:

| File | What it is |
|---|---|
| `ball3d_view.html` | Interactive 3D view of every rally; open it in a browser |
| `ball3d.mp4` | The video with the ball's track, height and player numbers (with `--video`) |
| `tracked.mp4` | The video with the 2D ball track and touches (with `--video`) |
| `ball3d.json` | The ball in metres for every frame, plus per-flight speed, apex and net crossing |
| `track.json`, `contacts.json` | The 2D track and the touches |

A 22-second clip takes a few minutes on an Apple-silicon Mac; most of it is pose estimation.

## Labeling your own clips

Labels improve accuracy and train the models. One command walks a clip through every step:
court, ball and touches, people, player numbers, then a list of frames worth re-checking.

```bash
python -m tools.annotate videos/match.mov
python -m tools.check_labels match --summary
```

Clips in `holdout/` are test clips: they are scored, never trained on.

## Accuracy

Measured on 10 held-out clips with every frame checked by hand:

| | |
|---|---|
| Ball found within one ball-width | about 95% of frames |
| Touches found | 96% |
| 3D ball within reach of a player at hits and sets | 97% and 90% |

Accuracy is best on raw recordings filmed from behind the baseline with the camera raised.
Screen recordings and very low cameras make the 3D less certain near the net.

## Project layout

```
main.py         the whole pipeline for one clip
core/           the library: ball tracking, geometry, people, 3D
tools/          helpers: calibration, labeling, single steps, viewers
evaluation/     benchmarks and accuracy checks
tests/          unit tests and the 3D accuracy gate
training/       model fine-tuning notebooks
```

## Development

```bash
pip install -e ".[ml,dev]"
pytest && ruff check .
```


## Acknowledgements

Ball finding builds on [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) (MIT) and TrackNet;
detection and pose use [Ultralytics YOLO](https://github.com/ultralytics/ultralytics); player
re-identification uses LightMBN through [BoxMOT](https://github.com/mikel-brostrom/boxmot).
Check each project's license before redistributing.
