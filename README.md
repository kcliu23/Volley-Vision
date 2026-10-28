# VolleyVision

Volleyball ball tracking and touch detection from a single camera.

**Finder-first (default):** **WASB**, a ball-finding network that looks at 3 frames at once,
runs on every frame and proposes where the ball might be. **YOLO** checks only a small crop
around WASB's top candidate (where it's likely enough to be the ball). A **joint linker**
then picks the ball's path over the whole clip in one pass: every candidate weighed by a
probability learned from your labels (score calibration) plus physics (gravity arcs, speed and
size limits, one ball). The path is split into **flights** (one parabola each); the boundaries
are **contacts**, classified as touch, bounce or other from where the players are.

The older **classic** pipeline (YOLO on every frame, rule linker + zoom search, TrackNet gap
filling) is still there: `--pipeline classic`.

## Usage

```bash
pip install -r requirements.txt
python main.py videos/test42.MP4
```

The first run of a clip includes the sparse YOLO pass (every 20th frame), the WASB pass and
the YOLO crops; re-runs reuse the caches and take seconds.

| Flag | Default | Description |
|------|---------|-------------|
| `--pipeline` | `finder-first` | `finder-first` or `classic` |
| `--finder` | `models/wasb/wasb_ft2.pth.tar` | (finder-first) ball finder, WASB |
| `--crop-top`, `--crop-p` | `1`, `0.6` | (finder-first) YOLO-check up to this many WASB candidates per frame, those with calibrated P(ball) ≥ this |
| `--loose` | off | (finder-first) top 2 candidates, no threshold: fewer misses, more false balls, ~2.5× the YOLO work |
| `--ring` | off | Hollow ring marker, so the real ball stays visible underneath |
| `--preview` | off | Show the render while it's written |
| `--trail-len` | `40` | Ball trail length in frames |
| `--no-touches` | off | Skip contact detection and overlay |
| `--output` | `output` | Output directory |
| `--model` | `models/best4.pt` | YOLO weights |
| `--linker`, `--no-zoom`, `--redetect`, `--tracknet`, `--no-tracknet` | | classic pipeline only |

Outputs, one folder per clip (`output/<clip>/`; all paths are defined in `core/paths.py`):

- `tracked.mp4` — ball trail; orange ring at each touch, white ring at each bounce (H.264 + original audio if `ffmpeg` is installed)
- `track.json` — per frame: `state` (`detected` / `weak` = YOLO-confirmed, `tracknet` = finder only, `filled` = on the arc, `none`) and `pos`
- `contacts.json` — flights (one parabola each) and the contacts between them: frame (can fall inside a gap), position, velocity in/out, `kind` = `touch` / `bounce` / `other` (net, wall, unclear)
- `cache/sample_dets.json` — finder-first: sparse YOLO pass, clip profile (timestamps, repeated frames, picture area, ball size, camera motion), person boxes
- `cache/tracknet_wasb_ft2.json` — WASB candidates; `cache/crops_wasb_ft2_strict0.6.json` — YOLO crop finds
- `cache/dets.json`, `cache/tracknet.json` — classic pipeline (and autolabel): YOLO on every frame, TrackNet candidates
- caches are rebuilt automatically if deleted

`output/dev/` holds comparison runs and experiments; `output/archive/` holds files from older versions.

## Accuracy

Measured on the 10 `holdout/` clips (never trained on, every frame checked, game ball only),
share of ball frames where the track is within one ball-width of your label:

| | on the ball | wrong place | missed | false ball when hidden | YOLO work |
|---|---|---|---|---|---|
| classic | 92.2% | 0.5% | 7.3% | 2.7% | every frame + zoom rounds |
| **finder-first, strict (default)** | **95.2%** | 1.2% | **3.6%** | 3.4% | ~39% of every frame (+5% sparse pass) |
| finder-first, loose | 96.3% | 0.9% | 2.8% | 4.9% | ~100% of every frame |

Measured 2026-10-28 after the simplification pass (one joint linker, no tracklet layer; WASB-only
points penalised by 0.5 instead of 1.0, which found more of the ball for a few more false balls).
Before it: 94.8% on-ball, 4.1% missed, 2.3% false.

Known gap: a second ball in play on a neighbouring court can win over the game ball for a moment
when it's seen more confidently (regression case test34). A court-area cue (homography) is the
planned fix; `--pipeline classic` handles that clip.

Positions: YOLO-confirmed points are within ~0.05 ball-widths of your label; points only WASB
saw (blur, partly hidden; ~40% of frames) are ~0.17 ball-widths (~4 px) off, WASB's heatmap
resolution. Settings are in **seconds** and **ball-widths** (1 ball-width ≈ 0.21 m at the ball's
depth), not frames and pixels, and fits use each frame's real timestamp. Camera motion is
measured but not yet compensated, so handheld footage is weaker.

## More training data: autolabel → review → export

```bash
python -m tools.autolabel videos/test40.mov            # or --dir <folder>: propose labels for every frame
python -m tools.review videos/test40.mov               # check rally by rally; Enter accepts, click fixes
python -m tools.export_training --name v3 \
    --tracknet-base training/tracknet/datasets/tracknet_v1   # YOLO + WASB/TrackNet training sets (old + new)
```

- **autolabel** runs finder-first with the loose crop rule (fewest misses: a miss costs more
  clicks than a false ball), plus YOLO on every frame (other-ball suggestions and hard negatives
  need it), and writes `data/proposals/<clip>.json`: a ball position and box for every frame,
  flagged where it's worth a closer look (physics guesses, finder-only, picture edge, jumps,
  finder disagreement, odd contacts), and the clip split into rallies. `--classic` for the old pipeline.
- **review** plays each rally with the proposal drawn on it: accept it with Enter, or pause and
  click the ball / `n` no ball / `N` a run of no ball / `h` hidden / `i` fill a straight line
  between two clicks / `I` fill between all your clicks in the rally; Shift+click (or `e`)
  confirms another ball through the whole clip, `o` jumps to the next suggestion; `m` + 1–6/0
  marks a touch (serve, pass, set, attack, block, dig, other), `M` jumps between marks.
  Saves to `data/verified/<clip>.json` after every action. Clips in `holdout/` are always test
  clips (never exported). `?` prints all keys.
- **export_training** writes versioned sets, never overwriting: `training/datasets/yolo_<name>/`
  in the same Roboflow layout as `volleyball_combined` and `training/tracknet/datasets/tracknet_<name>/`
  (every frame, the format WASB trains on). Splits by whole clip; holdout clips go to the test split.
- **Training:** `training/yolo/finetune_yolo.ipynb` (YOLO), `training/wasb/finetune_wasb.ipynb` (WASB), on Colab.
- `python -m tools.calibrate --finder <ckpt>`: score calibration — maps each source's score
  (YOLO / zoom crops / the finder) to a comparable probability, fit on your checked clips
  (`models/score_calibration.json`, one model per finder). Run it for every new finder checkpoint.
- **Keep a new model only if it wins on the holdout clips:**
  `python -m tools.scoreboard --yolo models/best5.pt` or `--finder models/wasb/<new>.pth.tar`.

## Development tools

```bash
python -m tools.regression [--linker finder-first|joint|cost] [--tracknet]   # 12 known-tricky moments
python -m tools.compare_finders --runs --finders tracknet wasb:models/wasb/wasb_ft2.pth.tar   # finders alone
python -m tools.eval_finder_first --finder models/wasb/wasb_ft2.pth.tar     # classic vs YOLO+finder vs finder-first crop rules
python -m tools.ablate [--variants joint classic]   # test a pipeline change on cached inputs (holdout/dev/checked + regression)
python -m tools.dump_detections videos/*.mov      # pre-compute the full YOLO pass for many clips
```

## Layout

```
main.py                 CLI entry point
core/
  paths.py              where every file lives (output/<clip>/..., data/..., training/...)
  video_pipeline.py     run(): finder-first (default) or classic -> flights + contacts -> render
  finder_first.py       sparse YOLO pass, WASB, YOLO crops, joint linker
  finders.py            one interface for the ball finders (WASB, TrackNet): candidates, caches, calibration name
  wasb.py, wasb_hrnet.py   WASB (HRNet, from nttcom/WASB-SBDT, MIT) heatmaps -> candidates
  tracknet.py           TrackNet heatmaps -> candidates (network from TrackNetV3, MIT)
  calibration.py        detection score -> probability, per source
  joint_linker.py       the ball's path over the whole clip: calibrated candidates + physics, one pass
  linker.py             rule-based linker (classic) + shared gap filling
  cost_linker.py        cost-based linker (classic option; shared helpers)
  refine.py             zoom search (classic), YOLO crops, repeated-frame and picture-area detection
  tracknet_fill.py      classic TrackNet stage: fill gaps / extend track ends inside the physics corridor
  profile.py            clip profile: real frame timestamps, repeated frames, picture area, ball size, camera motion
  flights.py            split the track into flights (parabola each); contacts between them
  contacts.py           classify contacts touch / bounce / other using person boxes (YOLOv8n)
  device.py             CUDA / MPS / CPU choice, batched YOLO (VV_DEVICE, VV_BATCH)
  other_balls.py        other volleyballs in the picture (suggestions for the review tool)
  render.py             trail / marker / touch rings, ffmpeg encode
  archive/              retired: single-pass tracker + old touch detector, VballNet, TrackNet-zoom experiment,
                        tracklet layer (tracklets, tracklet_linker: no gain over the joint linker alone)
tools/                  autolabel, review, export_training, build_tracknet_data, calibrate, scoreboard,
                        regression, ablate, compare_finders, eval_finder_first, dump_detections
tools/archive/          retired tools (eval_gt, eval_joint, label_gt, compare_tracking, tracknet_offset, fit_linker,
                        old Roboflow upload, organize_output) and dev/ (linker debugging, visualisation)
videos/                 your clip pool: everything labelled for training (not in git)
holdout/                test clips: never trained on, only scored (not in git)
data/gt/                old dev labels, every 5th frame (10 clips)
data/proposals/         autolabel proposals            data/verified/   your checked labels (commit these)
models/best4.pt         YOLO ball weights (best5.pt: candidate); models/yolov8n.pt: COCO person detector
models/wasb/            wasb_ft2.pth.tar (fine-tuned, default) + wasb_volleyball_best.pth.tar (authors' weights)
models/tracknet/        FT_TrackNet_best.pt + .offset.json (classic pipeline)
models/score_calibration.json   score calibration (tools.calibrate)
models/archive/         retired weights and downloads (not in git)
training/yolo/          YOLO fine-tuning notebook (Colab)
training/wasb/          WASB fine-tuning notebook + train_wasb.py (Colab)
training/tracknet/      dataset_labels.json + datasets/ (TrackNet-format exports WASB trains on; not in git)
training/datasets/      volleyball_combined (old training set), yolo_<name> exports; sources/ (not in git)
training/legacy/        old auto-label -> Roboflow -> merge scripts; TrackNet notebooks
output/<clip>/          pipeline outputs; output/dev/ experiments; output/archive/ old files
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.
