---
name: verify-volleyvision
description: Drive VolleyVision the way its user does and prove a change works. Covers the labeling window (tools.review keys, driven headless with scripted keys and screenshots), the pipeline CLI (python -m main), rally scoring and the cut video. Always in a throwaway sandbox, never on the user's real labels. Use after changing tools/review.py, core/scene3d/rallies.py, main.py, evaluation/ or tools/rally_video.py, or when asked to prove behavior rather than unit-test it.
---

# Verify VolleyVision

VolleyVision is a Python project with no server. A user touches it through three surfaces:

- **`tools.review`**: an OpenCV window where they label by key presses. This is the primary surface and where their work lives.
- **`python -m main <video>`**: the pipeline, which writes `output/<clip>/`.
- **`evaluation.*` and `tools.rally_video`**: CLI reports and the cut video.

The user's labels in `data/verified/` are hand-made, irreplaceable, and often open in their own review window while you work. **Every drive runs in a sandbox made by `scripts/sandbox.sh`.** Never run a tool from the repo root with a key script, and never write `data/verified`, `data/player_ids` or `output/<clip>/rallies.json` in the repo.

`<skill>` below is `.claude/skills/verify-volleyvision`, and `<repo>` is the repo root.

## Launch

There's no server to start. "Launch" means making a sandbox for the clip under test:

```bash
SB=$(<skill>/scripts/sandbox.sh <clip> | tail -1)      # e.g. full_rally_test, holdout2, test
```

**Ready** means `scripts/doctor.sh "$SB"` prints `doctor: OK`.

**What the sandbox contains:**

| In the sandbox | What it is |
|---|---|
| `videos/`, `holdout/`, `models/`, `training/` | Symlinks, read only |
| `data/{proposals,verified,cameras,player_ids}/<clip>.json` | Copies |
| `output/<clip>/` | A copy with no `.mp4` files |

**Running a tool in it:** run any tool from inside `$SB` with `PYTHONPATH=<repo>`, so the code is the working tree's:

```bash
cd "$SB" && PYTHONPATH=<repo> python -m evaluation.rallies_eval <clip>
```

**Clips:**
- Short clips with full labels are `holdout1`–`holdout10` (in `holdout/*.mov`) and `test*` (in `videos/`).
- The one full game is `full_rally_test` (`videos/full_rally_test.mp4`, 47,704 frames). It's the only clip that still has open rally guesses, so use it for `p`/`P`/`X` journeys.

## Doctor

```bash
<skill>/scripts/doctor.sh [<sandbox>]
```

It's read only. It checks:
- python, cv2, numpy, sklearn and ffprobe are available;
- the git branch and how many files are modified;
- **whether the user's own `tools.review` is running** (if so, leave it alone, never send it keys);
- whether the sandbox is ours and its labels are copies, not the real files.

Run it first, and again whenever anything looks off.

## Drive

**The labeling window, headless:** `scripts/drive_review.py` runs the real `tools.review.review()` loop. The OpenCV window is replaced with a fake: scripted keys come back from `cv2.waitKey`, and each picture the tool draws is saved.

```bash
EV=$HOME/.cache/vv-verify/evidence/$(date +%Y%m%d-%H%M%S)-<what>
cd "$SB" && PYTHONPATH=<repo> python <skill>/scripts/drive_review.py videos/full_rally_test.mp4 \
    --keys "p X" --evidence "$EV"
```

**Key tokens:**
- a single character: `p P X m 1 9 v a d A D M u w s` (see `python -m tools.review -h` / `?`);
- `esc`, `enter`, `space` (play and pause);
- `wait:N`: N idle polls, so playback advances;
- `click:X,Y`: a left click in window pixels. The window has a top bar of `TOP_H` pixels above the frame.

A final `q` (save and quit) is added. The script refuses to run outside a sandbox.

**The pipeline:**

```bash
cd "$SB" && PYTHONPATH=<repo> python -m main videos/test.MP4 --no-players
```

- `test.MP4` takes about 12 s.
- Without `--no-players`, pose runs, which is slow.
- It writes `$SB/output/test/`.

**Reports:**

```bash
cd "$SB" && PYTHONPATH=<repo> python -m evaluation.rallies_eval <clip> --detail
cd "$SB" && PYTHONPATH=<repo> python -m tools.rally_video videos/<clip>.mp4
```

The per-feature recipes are in `features/`.

## Evidence

Evidence goes in `$HOME/.cache/vv-verify/evidence/<timestamp>-<what>/`, outside the sandbox, so cleanup never removes it.

**What `drive_review.py` writes:**
- `000_start.png`;
- `NNN_<key>.png`: the window after each key;
- `keys.txt`;
- `labels_before.json`, `labels_after.json`;
- `diff.txt`: which `contacts`, `not_rally` and `unseen` entries changed.

**Proof standards:**
1. Drive the real key path (`drive_review.py` → `review()`), never `Review` methods called directly.
2. Show the action and the resulting state: the screenshot after the key, plus the saved-label diff. A message on screen alone isn't proof; the diff is.
3. Check side effects. The sandbox's `data/verified/<clip>.json` changed as expected, and **the repo's real file did not**: compare its `not_rally`/`contacts` keys, or its mtime against the user's own activity.
4. For the pipeline, the proof is the files written plus a report run on them (`rallies_eval`), not just exit code 0.
5. Before claiming a pixel or frame result, look at the PNG (Read it).

## Cleanup

```bash
<skill>/scripts/cleanup.sh "$SB"
```

- It removes only a directory that `sandbox.sh` made (it checks for the `.vv-repo` marker).
- Symlinked inputs are unlinked, not followed.
- Evidence in `~/.cache/vv-verify/evidence/` stays.
- Never kill a `tools.review` process: the running one is the user's.

## Helpers

| Script | Use |
|---|---|
| `scripts/sandbox.sh <clip> [<dir>]` | Make a sandbox. The last line printed is its path |
| `scripts/doctor.sh [<sandbox>]` | Read-only health check. Exit code 1 if it fails |
| `scripts/drive_review.py <video> --keys "..." --evidence <dir> [--test]` | Drive the labeling window from inside a sandbox |
| `scripts/cleanup.sh <sandbox>` | Remove a sandbox |

Keep the feature map honest as the app changes with `/pstack:maintain-verification-skill`.
