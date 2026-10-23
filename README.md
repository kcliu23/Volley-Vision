# VolleyVision

Volleyball ball tracking and touch detection from a single camera.

Two passes: YOLOv8 runs **once** per clip and every ball candidate is cached; a **linker**
then builds the ball's path over the whole clip (physics-based: gravity arcs, speed and
size limits, one ball) and zoom-searches wherever the ball was missed. The path is split
into **flights** (one parabola each); the boundaries are **contacts**, classified as touch,
bounce or other from where the players are.

## Usage

```bash
pip install -r requirements.txt
python main.py tests/test42.MP4
```

The first run of a clip includes the YOLO pass (minutes on CPU); re-runs reuse the cached
detections and take seconds plus zoom search.

| Flag | Default | Description |
|------|---------|-------------|
| `--linker` | `rule` | `rule` (best measured) or `cost` (global optimization) |
| `--no-zoom` | off | Skip zoom search (faster, fewer recovered frames) |
| `--redetect` | off | Re-run the YOLO pass even if cached |
| `--ring` | off | Hollow ring marker, so the real ball stays visible underneath |
| `--preview` | off | Show the render while it's written |
| `--trail-len` | `40` | Ball trail length in frames |
| `--no-touches` | off | Skip contact detection and overlay |
| `--output` | `output` | Output directory |
| `--model` | `models/best4.pt` | YOLO weights |
| `--legacy` | off | Old single-pass tracker (YOLO conf ≥ 0.40 + Kalman); `--conf`, `--raw` apply to it |

Outputs in `output/`:

- `<clip>_tracked.mp4` — ball trail; orange ring at each touch, white ring at each bounce (H.264 + original audio if `ffmpeg` is installed)
- `<clip>_track.json` — per frame: `state` (`detected` / `weak` / `filled` / `none`) and `pos`
- `<clip>_contacts.json` — flights (one parabola each) and the contacts between them: frame (can fall inside a gap), position, velocity in/out, `kind` = `touch` / `bounce` / `other` (net, wall, unclear)
- `detections/<clip>_dets.json` — cached YOLO candidates, repeated-frame flags, picture area, zoom finds

## Accuracy

Measured on 10 hand-labelled clips (`data/gt/`, 1,024 frames with the ball visible), share of
frames where the track is within one ball-width of the labelled ball:

| | old tracker | rule linker | cost linker |
|---|---|---|---|
| on the ball | 74.8% | **95.2%** | 93.1% |
| wrong ball / place | 8.3% | 0.9% | 2.3% |
| missed | 17.0% | 3.9% | 4.6% |

## Development tools

```bash
python -m tools.regression [--linker cost]      # 11 known-tricky moments, seconds, no YOLO
python -m tools.eval_gt                          # score old / rule / cost trackers on data/gt labels
python -m tools.label_gt --suggested             # label ground truth (click the ball every 5th frame)
python -m tools.try_linker tests/x.mov --ring    # side-by-side video: old tracker vs linker
python -m tools.dump_detections tests/*.mov      # pre-compute the YOLO pass for many clips
python -m tools.analyze_detections output/detections/x_dets.json --track ... --video ...
python -m tools.detect_touches output/x_track.json --frames tests/x.mov   # touch contact sheets
```

## Layout

```
main.py                 CLI entry point
core/
  video_pipeline.py     run(): detect (cached) -> link + zoom search -> touches -> render; run_legacy()
  linker.py             rule-based linker (default) + shared gap filling
  cost_linker.py        cost-based linker (Viterbi over candidates)
  refine.py             zoom search, repeated-frame detection, picture-area detection
  flights.py            split the track into flights (parabola each, dynamic programming); contacts between them
  contacts.py           classify contacts touch / bounce / other using person boxes (YOLOv8n, contact frames only)
  touch_detector.py     older touch detector (used by --legacy)
  render.py             trail / marker / touch rings, ffmpeg encode
  ball_tracker.py       legacy single-pass YOLO + Kalman tracker
tools/                  dev tools above, dataset helpers
data/gt/                ground-truth labels (committed)
models/best4.pt         trained ball weights; models/yolov8n.pt: COCO person detector
training/               training notebook and dataset scripts
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.
