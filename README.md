# VolleyVision

Volleyball ball tracking and touch detection from a single camera.

Two passes: YOLOv8 runs **once** per clip and every ball candidate is cached; a **linker**
then builds the ball's path over the whole clip (physics-based: gravity arcs, speed and
size limits, one ball) and zoom-searches wherever the ball was missed. **TrackNet**, a
model that looks at 8 frames at once, then fills gaps and track ends where YOLO saw
nothing, inside the corridor the ball's arc predicts. The path is split into **flights**
(one parabola each); the boundaries are **contacts**, classified as touch, bounce or other
from where the players are.

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
| `--tracknet` | `models/tracknet/FT_TrackNet_best.pt` | TrackNet checkpoint (stage skipped if the file is missing) |
| `--no-tracknet` | off | Skip the TrackNet stage |
| `--output` | `output` | Output directory |
| `--model` | `models/best4.pt` | YOLO weights |
| `--legacy` | off | Old single-pass tracker (YOLO conf ≥ 0.40 + Kalman); `--conf`, `--raw` apply to it |

Outputs, one folder per clip (`output/<clip>/`; all paths are defined in `core/paths.py`):

- `tracked.mp4` — ball trail; orange ring at each touch, white ring at each bounce (H.264 + original audio if `ffmpeg` is installed)
- `track.json` — per frame: `state` (`detected` / `weak` / `tracknet` / `filled` / `none`) and `pos`
- `contacts.json` — flights (one parabola each) and the contacts between them: frame (can fall inside a gap), position, velocity in/out, `kind` = `touch` / `bounce` / `other` (net, wall, unclear)
- `cache/dets.json` — YOLO candidates, clip profile (timestamps, repeated frames, picture area, ball size, camera motion), zoom finds, person boxes
- `cache/tracknet.json` — TrackNet candidates (the caches are rebuilt automatically if deleted)

`output/dev/` holds comparison videos and experiments; `output/archive/` holds files from older
versions. `python -m tools.organize_output --apply` moves an old flat `output/` into this layout.

## Any video

Settings are in **seconds** and **ball-widths** (1 ball-width ≈ 0.21 m at the ball's depth), not
frames and pixels, and fits use each frame's real timestamp. Simulated on the labelled clips:
30 fps 95.0%, 15 fps 91.5%, 720p-like 95.3%, 4K-like 95.3% (original 95.3%). Camera motion is
measured but not yet compensated, so handheld footage is still weaker.

## Accuracy

Measured on 10 hand-labelled clips (`data/gt/`, 1,024 frames with the ball visible), share of
frames where the track is within one ball-width of the labelled ball:

| | old tracker | rule linker | cost linker | rule linker + TrackNet (default) |
|---|---|---|---|---|
| on the ball | 74.8% | 95.4% | 93.1% | **96.4%** |
| wrong ball / place | 8.3% | 0.9% | 2.3% | 0.8% |
| missed | 17.0% | 3.7% | 4.6% | 2.8% |
| false ball when hidden | 30.1% | 8.0% | 5.5% | 9.8% |

## More training data: autolabel → review → export

```bash
python -m tools.autolabel tests/test40.mov            # or --dir <folder>: propose labels for every frame
python -m tools.review tests/test40.mov               # check rally by rally; Enter accepts, click fixes
python -m tools.export_training --name v2 \
    --tracknet-base training/tracknet/datasets/tracknet_v1   # YOLO + TrackNet training sets (old + new)
```

- **autolabel** runs the whole pipeline (no render) and writes `data/proposals/<clip>.json`: a
  ball position and box for every frame, flagged where it's worth a closer look (physics
  guesses, TrackNet-only, picture edge, jumps, YOLO/TrackNet disagreement, odd contacts), and
  the clip split into rallies.
- **review** plays each rally with the proposal drawn on it: accept it with Enter, or pause and
  click the ball / `n` no ball / `h` hidden / `i` fill a straight line between two clicks.
  Saves to `data/verified/<clip>.json` after every action. `--test` (or `t`) makes a clip a
  test clip: it is never exported for training. `?` prints all keys.
- **export_training** writes versioned sets, never overwriting: `training/datasets/yolo_<name>/`
  in the same Roboflow layout as `volleyball_combined` (`<clip>_<frame>.jpg`, class 0, empty
  label = no ball; frames already in `volleyball_combined` are skipped; `data_with_combined.yaml`
  trains on old + new) and `training/tracknet/datasets/tracknet_<name>/` (every frame; hidden =
  not visible). For Colab, tar the TrackNet folder and use it in place of `tracknet_data.tar`.
- Fine-tune from the current checkpoint on old + new data; keep a new model only if it beats
  the current one on `data/gt` (`tools.eval_gt`) and passes `tools.regression --tracknet`.

## Development tools

```bash
python -m tools.regression [--linker cost] [--tracknet]   # 11 known-tricky moments, seconds, no YOLO
python -m tools.eval_gt                          # score old / rule / cost / rule+TrackNet on data/gt labels
python -m tools.eval_tracknet --tag ft           # TrackNet alone / pooled with YOLO, per clip
python -m tools.label_gt --suggested             # label ground truth (click the ball every 5th frame)
python -m tools.try_linker tests/x.mov --ring    # side-by-side video: old tracker vs linker
python -m tools.dump_detections tests/*.mov      # pre-compute the YOLO pass for many clips
python -m tools.analyze_detections output/x/cache/dets.json --track output/x/track.json --video tests/x.mov
python -m tools.detect_touches output/x/track.json --frames tests/x.mov   # touch contact sheets
python -m tools.tracknet_dump --gt --ckpt <ckpt> --tag <tag>               # TrackNet candidates for benchmarking
```

## Layout

```
main.py                 CLI entry point
core/
  paths.py              where every file lives (output/<clip>/..., data/..., training/...)
  video_pipeline.py     run(): detect (cached) -> link + zoom search -> TrackNet -> contacts -> render; run_legacy()
  tracknet.py           TrackNet heatmaps -> candidates (network from TrackNetV3, MIT), cached per clip
  tracknet_fill.py      TrackNet stage: fill gaps / extend track ends inside the physics corridor
  profile.py            clip profile: real frame timestamps, repeated frames, picture area, ball size,
                        camera motion — all thresholds work in seconds and ball-widths
  linker.py             rule-based linker (default) + shared gap filling
  cost_linker.py        cost-based linker (Viterbi over candidates)
  refine.py             zoom search, repeated-frame detection, picture-area detection
  flights.py            split the track into flights (parabola each, dynamic programming); contacts between them
  contacts.py           classify contacts touch / bounce / other using person boxes (YOLOv8n, contact frames only)
  touch_detector.py     older touch detector (used by --legacy)
  render.py             trail / marker / touch rings, ffmpeg encode
  ball_tracker.py       legacy single-pass YOLO + Kalman tracker
tools/                  autolabel / review / export_training, evaluation and dev tools above
data/gt/                test labels, every 5th frame (committed; never trained on)
data/proposals/         autolabel proposals            data/verified/   your checked labels (commit these)
models/best4.pt         trained ball weights; models/yolov8n.pt: COCO person detector
models/tracknet/        TrackNet checkpoint (FT_TrackNet_best.pt: fine-tuned on 17 of your clips)
training/               YOLO notebook + datasets; training/tracknet/: Colab fine-tune notebook,
                        patch script, datasets (tracknet_v1, ...)
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.
