# VolleyVision

Volleyball ball tracking and touch detection from a single camera.

**Finder-first (default):** **WASB**, a ball-finding network that looks at 3 frames at once,
runs on every frame and proposes where the ball might be. **YOLO** checks only a small crop
around WASB's top candidate (where it's likely enough to be the ball). A **joint linker**
then picks the ball's path over the whole clip in one pass: every candidate weighed by a
probability learned from your labels (score calibration) plus physics (gravity arcs, speed and
size limits, one ball). The path is split into **flights** (one parabola each); where a flight
says the ball must be but nothing confirmed it, YOLO checks a crop there (**physics-guided
search**) and the linker runs again. The flight boundaries are **contacts**, classified as touch,
bounce or other from where the players are.

The older **classic** pipeline (YOLO on every frame, rule linker + zoom search, TrackNet gap
filling) is still there: `--pipeline classic`.

## Usage

```bash
pip install -e ".[ml,dev]"          # once: installs the project and its dependencies (pyproject.toml)
pre-commit install                  # once: ruff checks every commit
python main.py videos/test42.MP4
```

## Development

```bash
ruff check .                        # style + likely bugs (config in pyproject.toml)
pytest                              # fast tests, ~5 s; tests that need local clips skip without them
pytest -m slow                      # + the 3D accuracy gate on synthetic rallies, ~1.5 min
```

CI (`.github/workflows/ci.yml`) runs ruff and the fast tests on every push, and the slow gate
weekly or by hand. Hand-made data (labels, calibrations) is saved through `core.common.safe_io`
(temp file + swap, rolling backups): use it for anything a person spent time on.

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
| `--no-guided` | off | (finder-first) Skip the physics-guided search |
| `--output` | `output` | Output directory |
| `--model` | `models/best4.pt` | YOLO weights |
| `--linker`, `--no-zoom`, `--redetect`, `--tracknet`, `--no-tracknet` | | classic pipeline only |

Outputs, one folder per clip (`output/<clip>/`; all paths are defined in `core/common/paths.py`):

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
| finder-first, strict, no guided search (`--no-guided`) | 95.2% | 1.2% | 3.6% | 3.4% | ~39% of every frame (+5% sparse pass) |
| **finder-first, strict + guided search (default)** | **+0.3** | **−0.2** | **−0.1** | **−0.2** | + ~0.5 crops per frame (≈ +25% of every frame) |
| finder-first, loose | 96.3% | 0.9% | 2.8% | 4.9% | ~100% of every frame |

Measured 2026-10-28 after the simplification pass (one joint linker, no tracklet layer; WASB-only
points penalised by 0.5 instead of 1.0, which found more of the ball for a few more false balls).
Before it: 94.8% on-ball, 4.1% missed, 2.3% false. Guided search (measured with `evaluation.ablate`
on the same inputs, so shown as the change): holdout 95.3 → 95.6% on-ball, dev 91.5 → 92.6%,
checked clips 97.1 → 97.8%, wrong and false balls equal or lower, the same regression cases.
A physics-first tracker (`core/ball/flight_model.py`: arcs found and chained directly) was also built
and measured; it scored lower (holdout 92.8%) and is kept as an experimental `evaluation.ablate` variant.

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
  `python -m evaluation.scoreboard --yolo models/best5.pt` or `--finder models/wasb/<new>.pth.tar`.

## Development tools

```bash
python -m evaluation.regression [--linker finder-first|joint|cost] [--tracknet]   # 12 known-tricky moments
python -m evaluation.compare_finders --runs --finders tracknet wasb:models/wasb/wasb_ft2.pth.tar   # finders alone
python -m evaluation.eval_finder_first --finder models/wasb/wasb_ft2.pth.tar     # classic vs YOLO+finder vs finder-first crop rules
python -m evaluation.ablate [--variants joint classic]   # test a pipeline change on cached inputs (holdout/dev/checked + regression)
python -m tools.dump_detections videos/*.mov      # pre-compute the full YOLO pass for many clips
```

## Layout

```
main.py                 CLI entry point: track the ball in one clip
core/                   the library (no command-line code)
  pipeline.py           run(): finder-first (default) or classic -> flights + contacts -> render
  ball/                 finding the ball in 2D
    finder_first.py       sparse YOLO pass, WASB, YOLO crops, joint linker
    finders.py            one interface for the ball finders (WASB, TrackNet)
    wasb.py, wasb_hrnet.py  WASB (HRNet, nttcom/WASB-SBDT, MIT) heatmaps -> candidates
    tracknet.py           TrackNet heatmaps -> candidates (classic)
    confidence.py         detection score -> probability that it's the ball, per source
    joint_linker.py       the ball's path over the whole clip: candidates + physics, one pass
    linker.py, cost_linker.py, refine.py, tracknet_fill.py, physics_search.py   classic pipeline stages
    profile.py            clip profile: frame timing, picture area, ball size, camera motion
    flights.py, flight_model.py   the track as flights (gravity arcs) and the contacts between them
    contacts.py           touch / bounce / other, from where the people are
    other_balls.py        other volleyballs in the picture (suggestions for the review tool)
  geometry/             camera.py (calibration: pixels <-> meters), court.py (court and net)
  scene3d/              ball3d.py (physics fit), rally3d.py (global rally fit), params3d.py (every set number),
                        legacy_v1.py (first 3D mechanism, for comparison)
  people/               players.py (positions, jumps), reid.py (one id per player)
  common/               paths.py (where every file lives), safe_io.py (safe saves + backups),
                        frametime.py, device.py, render.py
  archive/              retired code
tools/                  the commands you run: autolabel, review, label_players, calibrate_court, pose, players,
                        scene3d, ball3d_run/_video/_viewer, export_training, build_tracknet_data, calibrate,
                        dump_detections, resnap_labels, todo
evaluation/             how good is each part: scoreboard, regression, ablate, compare_finders, eval_finder_first,
                        failures, reid_eval, sim_rally + sim_camera (synthetic rallies and cameras)
tests/                  pytest: unit tests + labeling-tool replays + the slow 3D accuracy gate
tools/archive/          retired tools and dev/ (linker debugging, visualisation)
videos/                 your clip pool: everything labelled for training (not in git)
holdout/                test clips: never trained on, only scored (not in git)
data/gt/                old dev labels, every 5th frame (10 clips)
data/verified/          your checked ball labels       data/player_ids/   your player labels
data/cameras/           court calibrations             data/proposals/    autolabel suggestions (not in git)
models/best4.pt         YOLO ball weights (best5.pt: candidate); models/yolov8n.pt: COCO person detector
models/yolov8m*.pt      person + pose models (downloaded automatically, not in git)
models/wasb/            wasb_ft2.pth.tar (fine-tuned, default) + wasb_volleyball_best.pth.tar (authors' weights)
models/tracknet/        FT_TrackNet_best.pt + .offset.json (classic pipeline)
models/score_calibration.json   score calibration (tools.calibrate)
training/yolo/          YOLO fine-tuning notebook (Colab)
training/wasb/          WASB fine-tuning notebook + train_wasb.py (Colab)
training/tracknet/      dataset_labels.json + datasets/ (not in git)
training/datasets/      volleyball_combined, yolo_<name> exports (not in git)
training/legacy/        old auto-label -> Roboflow -> merge scripts; TrackNet notebooks
output/<clip>/          pipeline outputs (not in git)
```

Earlier work on court calibration, 3D height estimation and a web dashboard lives on the `wip/court-calibration` branch.

