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
| regression cases | 12 known-tricky moments from real bugs (`evaluation.regression`) | a change that re-breaks an old bug fails here |

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

Built `evaluation/ablate.py`: score any variant of the tracker on the same cached inputs (holdout, dev,
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

## Phase 8 — physics backbone experiment (Oct 28)

Idea: the ball between touches is a gravity arc, so build the track from physics: find arcs in
all candidates, chain the arcs that meet at contacts, search where the arcs predict the ball.

- **Physics-guided search** (`core/ball/physics_search.py`): after the first pass, YOLO crops where each
  flight predicts the ball but nothing confirmed it (gaps inside flights, just before/after
  contacts), then the linker runs again. **Consistent small gain on every set:** holdout 95.3 →
  95.6%, dev 91.5 → 92.6%, checked 97.1 → 97.8% on-ball; wrong and false equal or lower; same
  regression cases. Cost ~0.5 extra crops per frame. **Adopted as default** (`--no-guided` to skip).
- **Physics-first tracker** (`core/ball/flight_model.py`): seed-and-grow robust arc fitting (each new
  point must agree with the whole arc), gravity estimated from the clip's own arcs, arcs chained
  and cut where they meet. **Scored lower:** holdout 92.8% on-ball (3.2% wrong, 7.6% false) vs
  95.3%; it breaks on short, fast flights around spikes and digs (regression cases test12, test21)
  and didn't fix the second ball (test34). Looser tolerances didn't help. Kept as an experimental
  `evaluation.ablate` variant (`arcs`), to re-test after WASB improves.
- **Conclusion:** WASB as the backbone, physics as the assist (choosing, smoothing, filling,
  guided search, contacts; and exact 3D physics later for the analyses). The remaining losses are
  mostly frames no detector proposes (about 60% of lost frames), which only a better WASB fixes.

---

## Phase 9 — labeling tools and the first 3D step (Oct 29 – Nov 4)

Labeling: `tools.todo` (review queue: verified / old data / dev set / to review, hardest first);
autolabel fixed (loose crop rule crash) and batch-safe; review: "WASB only" doubt label, `N`
follows the proposal (not its state), Shift+click instant (follows balls YOLO saw through its
saved boxes; `g` fills between two clicks for balls YOLO can't see; `F` = slower crop / look
following). Learned on the way: tiny resting background balls are invisible to YOLO even on crops,
and other-ball labels only feed YOLO (WASB trains on the game ball alone).

3D, step 1 (one camera, screen recordings, so everything is solved from the scene):
- **Calibration** (`core/geometry/court.py`, `core/geometry/camera.py`, `tools/calibrate_court.py`): click >= 6 known
  court / net points; the solver finds focal length + pose (PnP over a focal search, then joint
  least squares, with fit uncertainty). Synthetic test (`evaluation.sim_camera`): ~1 cm ball-ray error
  (95%: 2-5 cm) from calibration alone, also with a partial court. test74: 6 floor points; the
  solved camera puts the net's tape and bottom band where they are for a 2.43 m net (independent
  check); camera 3.8 m behind the near end line, 1.36 m high, 60 deg FOV. Lesson: the court frame
  must be right-handed (+y = left sideline) or the camera solves under the floor.
- **Ball in 3D** (`core/scene3d/ball3d.py`, `tools/ball3d_run.py`): each rally's flights fitted together
  as gravity parabolas in meters: reprojection + ball size + touch links (with each touch's exact
  time solved) + floor bounces, robust loss. Synthetic rallies through test74's camera
  (`tools.sim_ball3d`, now `evaluation.sim_rally` "ideal", 3 px noise, 10% missing, 2% wrong points): apex heights within 1-1.5 cm,
  touch points 4 cm, positions 5 cm median; weakest = depth along the line of sight on the far
  court (rebounds ~20 cm). With test74's current calibration uncertainty: positions ~20 cm,
  heights ~5 cm, so calibration is the main error source now.
- **test74 (real):** YOLO boxes run 13% larger than the ball (blur), learned per clip from gravity
  on long flights and corrected. 12 of 15 flights are clean free flights (2D fit 1-5 px); every
  net crossing clears the tape (never told about the net); the landing flight reaches the floor;
  apexes 2.9-6.3 m. Tracker output and hand labels give the same apexes on clean flights.
- **Air drag** (Nov 4): a volleyball is light for its size (k = rho Cd A / 2m ~ 0.035 /m, terminal
  ~17 m/s). Synthetic rallies with drag: the no-drag model was 34 cm off per frame and 2 m at
  landing; with drag 5 cm and 12 cm. Drag is fixed at the ball's physics value on real clips
  (fitting it per rally was unstable with few flights).
- **Gaps** in test74's 3D (630 of 854 frames at first): no ball in play (after the rally), the ball
  above the picture, and flights rejected because they held a touch the 2D split missed. Fixes:
  re-split a flight where one arc becomes two (touch at a reachable height <= 3.5 m, clearly better
  than one arc) -> found the block at 228/229 and a touch behind a player at 608/616; frames where
  the ball left the picture at an edge must project outside it; edge points count less; the
  clean-flight check uses max(8 px, 2x the flight's own jitter) (hand clicks wobble ~5 px).
  Now 740 frames in 3D; short or very uncertain flights are marked unsure.
- Tools: `tools.ball3d_video` (floor shadow + height drawn on the video), `tools.ball3d_viewer`
  (interactive 3D page: orbit / from the phone / side / top / far end; the phone view matches the
  video's court lines). Touch check against players' feet (person detector): calibration measures
  people at 1.69 m median; clear touches sit above the toucher.
- Next: a touch score (ball floor spot vs the toucher's feet) as the 3D accuracy number; fit the
  camera together with the flights; propagate calibration uncertainty; `m` marks for contacts.

## Phase 10 — players, the global rally fit, true frame times (Nov 4 – 5)

- **Players** (`tools/players.py`, `core/people/players.py`): person detector + ByteTrack every frame;
  split tracks stitched; feet -> floor spot; jumps = the feet rise along a gravity parabola
  (0.15-1.2 m) between a takeoff and a landing spot; standing height from the box top. test74: 57
  tracks, 12 jumps (blockers ~0.5 m, the hitter 0.7-0.85 m); no appearance Re-ID yet.
- **Global rally fit** (`core/scene3d/rally3d.py`): touches and arcs chosen together. Candidates (touch
  hints, 2D turns, gap edges, a 0.15 s grid) -> fast no-drag arc per pair -> dynamic programming
  for the cheapest chain -> drag refine (drop / add touches) -> joint fit -> dense re-check of bad
  flights. Score = fit error + a cost per touch + continuity + reach + "a touch must change the
  velocity" + "a touch must be near a player" (floor bounces exempt); a wrong point costs at most
  a capped amount (cheaper to ignore than to explain with touches). Synthetic (`evaluation.sim_rally`,
  20 hard rallies: block, occlusions, out of the picture, 10% missing, 2% wrong points, imperfect
  hints): v1 11.8 cm median / 117 m at 95%, 11% teleports, touches 88% / 80%; global fit 5.4 /
  45 cm, 0.5% teleports, touches 100% / 100%. Caveat: the rallies use the fit's own physics.
- **True frame times** (`core/common/frametime.py`): screen recordings repeat a frame every ~9 frames and
  skip one 3-5 frames later; in between the picture is a frame late (35% of test75's frames, 16%
  of test74's). Found from the picture alone (change between frames ~0 at a repeat, ~2x at a
  skip, paired). Before the fix the fit explained the timing error with fake touches. test75: fit
  error 1.2-6.6 px -> 0.6-2.7 px, 7 touches -> the 4 real ones + a floor bounce; test74: 1-7 px ->
  1-3 px on almost every flight.
- **Solid floor and net** in the joint fit: a flight can't pass through either.
- **Cleanup:** every set number in `core/scene3d/params3d.py`, marked with how it was set (synthetic /
  real / guess); `tools/ball3d_run.py` is load -> solve -> save; observation helpers in
  `core/scene3d/ball3d.py`, player reach in `core/people/players.py`; v1 moved to `core/scene3d/legacy_v1.py` (to delete
  once checked on real touch labels); one simulator (`evaluation/sim_rally.py`, with "ideal" = true
  touches); one command per clip (`tools/scene3d.py`); frame times computed at the start of the
  pipeline. Same results before and after (test74, test75: identical flights and positions).
- Still weak: the block exchange on test74 (230-236, 8 px); short flights' depth; touch accuracy
  on real clips is unmeasured until the holdout touch marks exist.

---

## The pipeline today

0. **Frame times:** repeated / skipped frames of screen recordings -> each frame's true time.
1. **Sizing pass:** YOLO on every 20th full frame: ball size (the ball-width unit), clip profile,
   anchor candidates.
2. **WASB on every frame:** up to 3 candidates per frame; calibrated probabilities from your labels.
3. **YOLO crops:** around WASB's top candidate where P ≥ 0.6 (batched): confirms and pinpoints.
4. **Joint linker:** one pass picks the path: evidence + motion physics + touches + limits + gaps.
5. **Physics-guided search:** YOLO crops where the flights predict the ball but nothing confirmed it; link again.
6. **Flights:** parabolas, WASB-only points on the arc, gaps ≤ 0.5 s filled, contacts at the breaks.
7. **Render** in real time.

3D (`python -m tools.scene3d videos/<clip>.mov --labels`): frame times -> players -> global
rally fit (ball in meters, touches) -> 3D viewer. Needs the court calibration first.

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
2. ~~**Physics backbone** (redesign of linking)~~ tried Oct 28 (Phase 8): guided search adopted; the
   physics-first tracker scored lower and stays experimental. Original plan: find flights as robust parabola fits through the
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
