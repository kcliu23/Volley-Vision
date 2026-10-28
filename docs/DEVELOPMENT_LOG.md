# VolleyVision development log

A record of how the ball tracker evolved: what was built, why, what it scored, and what didn't
work. Newest decisions at the bottom of each phase; the roadmap ahead is at the end.

**Goal:** robust, precise volleyball ball tracking from one phone camera, as the base for
contacts → court calibration (homography) → analyses (ball height, set speed, attack point,
contact timing). Guiding principle since October 2026: as few hand-set rules as possible;
learned models plus physics, and every change measured before it ships.

---

## How results are measured

| Set | What | Used for |
|---|---|---|
| `data/gt` (dev) | 10 clips, every 5th frame labeled; test7/12/17 also in training | early development, still reported |
| `data/verified` (checked) | 16+ clips (test45–63), every frame checked | training data; also scored ("did it learn them") |
| `holdout/` | 10 clips, every frame checked, **never trained on** | **the real test; decisions are made here** (since 2026-10-27) |
| regression cases | 12 known-tricky moments from real bugs (`tools.regression`) | a change that re-breaks an old bug fails here |

"On-ball" = the track is within one ball-width of the label. Also reported: wrong place, missed,
false ball (a ball reported where there is none).

---

## Phase 1 — single-pass YOLO tracker (May – Sep 2026)

- **May 5:** initial commit. YOLOv8 ball detector (`best4.pt`, trained on Roboflow sets merged into
  `volleyball_combined`), single pass: YOLO conf ≥ 0.40 per frame + a Kalman filter. Also an early
  dashboard and export scripts (later moved to the `wip/court-calibration` branch).
- **Sep 19–23:** simplified to a plain ball-tracking pipeline; ball selection reworked to prefer the
  candidate nearest the Kalman prediction over the most confident one, with a growing gate and a
  watchdog for wrong locks.

**Result (dev):** 74.8% on-ball, 8.3% wrong, 17.0% missed, **30.1% false balls**. One frame at a
time, a high confidence threshold and a short memory: it lost the ball often and locked onto junk.

## Phase 2 — two passes: detect once, link the whole clip (Oct 22–23)

- YOLO runs **once** with every candidate cached (low threshold); a **rule linker** then builds the
  path over the whole clip (gravity arcs, speed and size limits, one ball) and **zoom-searches**
  where the ball was missed (YOLO on crops, re-link, up to 6 rounds).
- Touch detection from the trajectory, then **contacts from flight segmentation**: the track split
  into parabolas; the boundaries are contacts, classified touch / bounce / other from person boxes.
- A cost-based linker (Viterbi) was tried as an alternative.

**Result (dev):** rule linker **95.4%** on-ball (0.9% wrong, 3.7% missed, 8.0% false); cost linker
93.1%. The jump from 75% came from linking globally instead of frame by frame.

## Phase 3 — TrackNet and any-video robustness (Oct 23–24)

- **TrackNetV3** (a multi-frame ball network) benchmarked: pretrained weights didn't beat YOLO +
  linker. Fine-tuned on our clips and used to **fill gaps and extend track ends** inside the corridor
  physics predicts.
- **Clip profile:** real frame timestamps (phones record uneven timing), repeated frames, picture
  area, ball size, camera motion. All settings moved to **seconds and ball-widths** instead of frames
  and pixels, so one configuration works on any video (simulated: 30 fps 95.0%, 15 fps 91.5%,
  720p- and 4K-like 95.3%).
- TrackNet's position offset (~0.35 ball-widths too high) handled as a per-checkpoint workaround.

**Result (dev):** rule linker + TrackNet **96.4%** on-ball (0.8% wrong, 2.8% missed, 9.8% false).

## Phase 4 — a labeling workflow (Oct 24–25)

- **autolabel → review → export:** autolabel proposes a ball position and box for every frame, with
  doubt flags and the clip split into rallies; **review** plays it rally by rally (Enter accepts, a
  click fixes, `n` no ball, `h` hidden, fills between clicks); **export** writes YOLO and TrackNet
  training sets.
- Export matches `volleyball_combined` exactly (resize to 1280, 3× Roboflow-style augmentation,
  every visible ball labeled), splits by **whole clip** (neighbouring frames leak otherwise) and
  always includes hard negatives (no-ball frames where YOLO fired anyway: hands, heads, clocks).
- **Other balls:** labeled only when confirmed in review (shift+click follows a ball through the
  clip). YOLO's own confident "other balls" were mostly clocks and lights.
- `tests/` renamed `videos/`; **`holdout/`** added as a test set never trained on.
- Score calibration: each source's score (YOLO, zoom, TrackNet) mapped to a learned probability.
- A **tracklets / hybrid linker** added for test34 (a second ball on the far court).
- Fixes from real bugs: TrackNet points placed on the fitted arc (median error 0.33 → 0.16 bw); a
  single bad point no longer wipes a long stretch (test53: stretches split at long gaps); arc fills
  limited to gaps ≤ 0.5 s (no invented 70-frame arcs).

## Phase 5 — speed and a better ball finder (Oct 26)

- **Speed:** YOLO on the GPU (CUDA / Apple MPS, CPU fallback), batched. Mac detection pass ~38
  frames/s vs ~4 on a CPU.
- **Video models explored:** **WASB** (HRNet, BMVC 2023, MIT) chosen for its precision; VballNet
  tried and dropped. WASB fine-tuned on our data (`wasb_ft2`).

**Finder alone, dev clips:** TrackNet finds the ball in 91.8% of frames (top pick right 56.5%, 0.38 bw
error); WASB as released 53.9%; **`wasb_ft2` 94.1%** (top pick right 79.0%, 0.23 bw), but 15.3% false
balls on its own.

## Phase 6 — holdout and the finder-first pipeline (Oct 27)

- The 10 holdout clips were labeled: the first honest test. Dev and holdout disagreed on the
  ranking; holdout decides.
- **Finder-first** became the default: WASB on every frame finds the ball; YOLO only checks small
  **crops** around its likely candidates; the hybrid linker chooses; flights finish. The classic
  pipeline stays behind `--pipeline classic`.

**Holdout:**

| | on-ball | missed | false | YOLO work |
|---|---|---|---|---|
| classic (YOLO every frame + rule linker + zoom + TrackNet) | 92.2% | 7.3% | 2.7% | every frame + zoom rounds |
| finder-first, strict crops | 94.8% | 4.1% | 2.3% | ~39% |
| finder-first, loose crops | 96.2% | 2.8% | 3.7% | ~100% |

**Tried and dropped the same day:**
- WASB retraining against false balls (more ball-free windows, hard-negative mining): recall fell
  86% → 77% in round 3; a gentler round 4 halved false balls but still lost 4–6 points of recall.
  The pipeline already filters most of WASB's false balls, so recall matters more.
- Snapping WASB-only points to faint YOLO boxes: no effect (YOLO usually has no box there at all).
- A learned crop-rule / linker-weight fit (`tools/fit_linker.py`): small chain gains, no end-to-end gain.
- `best5` (retrained YOLO) vs `best4`: a wash; YOLO only confirms now, so it matters less.

Repo cleanup: the single-pass tracker, VballNet, the TrackNet-zoom experiment and unused tools moved
to `core/archive/` and `tools/archive/`.

## Phase 7 — simplification by measurement (Oct 28)

Built `tools/ablate.py`: score any variant of the tracker on the same cached inputs (holdout, dev,
checked, regression cases). Then removed or tuned one piece at a time:

| Change | Evidence | Decision |
|---|---|---|
| Tracklet layer + rival/fixture filters before the joint linker | joint linker alone: same holdout (94.5 vs 94.6%), same dev, better checked (96.8 vs 95.1%), same regression | **removed** |
| Penalty on WASB-only points (1.0) | 0.5: +71 ball frames on holdout for +17 false; best total over 36 clips | **1.0 → 0.5** |
| Sizing pass every 10th frame | every 20th: ball size within ~1% | **halved** |
| Sizing-pass YOLO boxes as candidates | free; dev 89.6 → 91.6%, holdout unchanged | **added (anchors)** |
| Other joint-linker terms (speed, touch, re-entry, still, size, agreement) | each one off made it worse | kept |
| Flights fill limit 0.5 s | longer fills raise on-ball but put more points in the wrong place | kept |

**Fixes found along the way:**
- **Near-camera balls** (test62): WASB barely responds to a ball 2–3× normal size; training gets
  **zoom augmentation** (30% of windows zoomed 1.5–3×) from the next retrain.
- **Playback speed:** phones record variable frame rate (test60: 60 and 30 fps parts); rendering at
  one fixed rate played parts too fast or slow. Now each frame is shown for as long as it really lasted.
- **Smoothness:** WASB-only points wobbled (median jitter 0.158 bw vs classic 0.092). Placing them on
  the flight arc again: 0.100, on-ball unchanged.
- **Label precision:** hand clicks wobble 2.4× more than accepted proposals. Review clicks, fills and
  shift+clicks now **snap to YOLO boxes**; `tools/resnap_labels.py` tidies existing labels.
- **YOLO crops batched** across frames on the GPU (identical results).

**Current holdout result:** **95.2% on-ball**, 1.2% wrong, 3.6% missed, 3.4% false balls.

**Speed, test62 (8.4 s, 501 frames) from scratch:** Mac ~65 s new vs ~83 s classic; cloud CPU 431 s
vs 535 s. YOLO work ~3× lower; WASB is now the slowest stage.

---

## The pipeline today

1. **Sizing pass:** YOLO on every 20th full frame: ball size (the ball-width unit), clip profile,
   anchor candidates.
2. **WASB on every frame:** up to 3 candidates per frame; calibrated probabilities from your labels.
3. **YOLO crops:** around WASB's top candidate where P ≥ 0.6 (batched): confirms and pinpoints.
4. **Joint linker:** one pass picks the path: evidence + motion physics + touches + limits + gaps.
5. **Flights:** parabolas, WASB-only points on the arc, gaps ≤ 0.5 s filled, contacts at the breaks.
6. **Render** in real time.

Autolabel: the same, with loose crops (fewest misses) and full-frame YOLO kept for other-ball
suggestions and hard negatives.

## Lessons

- **Global beats local.** The biggest single jump (75% → 95%) came from linking the whole clip
  instead of frame by frame.
- **Holdout or it didn't happen.** Dev and holdout disagreed on the pipeline ranking; validation F1
  during training disagreed with the pipeline result.
- **Remove by measurement.** Two of three linking stages turned out to be unnecessary once WASB was
  good; each was added for a real reason at the time.
- **Know what a metric rewards.** Distance to hand labels penalised smoothing (labels wobble too);
  jitter had to be measured separately.
- **Better data beat clever training settings** for WASB, every time we tried both.
- **Physics is used too late.** It cleans up after the choices instead of making them; that's the
  next redesign.

---

## Roadmap ahead

**Step 1 — robust tracking (now)**
1. **Contact scorer** against touch marks (`m` in review; holdout first): found / missed / fake, timing error.
2. **Physics backbone** (redesign of linking): find flights as robust parabola fits through the
   candidates → chain flights that meet at contacts (one ball; a far-court ball's arcs don't connect)
   → **search where physics predicts** (YOLO crops at predicted positions) → refit → output from the
   arcs (smooth by construction, sub-frame contact times, each frame marked observed / predicted).
   Built in steps: physics-guided search on the current pipeline first, then the flight chainer,
   each measured on holdout, regression and contact timing.
3. **Plan B layers for where WASB fails:** sizing-pass anchors (done) → physics-guided crops →
   physics through occlusions → full-frame YOLO fallback for stretches where WASB goes quiet →
   flag what nothing finds.
4. **WASB:** more varied labeled clips; next retrain with zoom augmentation and other-ball masking
   in the loss; maybe a higher input resolution.
5. **Richer calibration features** (size, shape, motion) against false balls.

**Step 2 — contacts:** touch type from the volleyball sequence (serve, 3 touches per side, net
crossings); who touched it (nearest player → player tracking with ReID).

**Step 3 — court calibration:** click 4+ court points once per camera position → floor
homography; plus known heights (net) → camera calibration for 3D; the court area also becomes a
cue for "which ball is in play".

**Step 4 — analyses:** ball height, set height and tempo, attack point and height, serve and attack
speed, landing zones, rally stats, per-player stats; validated against known references.

**Speed (later):** WASB in half precision / CoreML on the Mac, skip dead time between rallies,
decode the video once, rally-sized chunks for full 20-minute games.
