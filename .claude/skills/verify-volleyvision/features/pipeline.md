# The pipeline on a clip

`python -m main <video>` takes one clip from video to results, with none of the user's labels used. The steps are 2D ball, people (pose), 3D ball, and viewer, and optionally videos. It needs the court calibration in `data/cameras/<clip>.json` first.

## Sub-features

- `track2d`: `output/<clip>/track.json` and `cache/`.
- `pose`: `pose.json`. Skipped with `--no-players`, and kept from the last run unless `--redo` is given.
- `ball3d`: `ball3d.json`, `touches.json`, `rallies.json`, `events.json`, `stats.json`.
- `viewer`: `ball3d_view.html`.
- `videos` (`--video`): `tracked.mp4`, `ball3d.mp4`, `rallies.mp4`.
- `2d-only` (`--2d-only`): only the 2D track.

## How to get to it (user POV)

- `python -m main videos/game2.mp4`, after `python -m tools.calibrate_court videos/game2.mp4`.
- At the end it prints a time-per-step table.

## Driving it with the shell

Preconditions: a sandbox for `test` (it has a camera and is short, about 12 s without pose).

Steps:
- **Run.** `cd "$SB" && PYTHONPATH=<repo> python -m main videos/test.MP4 --no-players`. Exit code is 0, and the output ends with `[volleyvision] test: time per step`.
- **Files.** `ls "$SB/output/test"` shows `track.json`, `ball3d.json`, `touches.json`, `rallies.json` and `ball3d_view.html`, all with fresh mtimes.
- **Content.** `python -c "import json; print(json.load(open('output/test/rallies.json'))['rallies'])"` shows a list of rallies with `f0`, `f1`, `end` and `why`.
- **Scored.** `PYTHONPATH=<repo> python -m evaluation.rallies_eval test` shows `1 found` for `test`.
- **Proof.** Save the run's stdout and the `rallies_eval` output to the evidence directory.

## Gotchas

- Without `data/cameras/<clip>.json`, only the 2D track is made, and it still exits 0. Check that `ball3d.json` exists.
- Pose on a full game takes tens of minutes. Use a short clip or `--no-players`.
- `full_rally_test`'s real `output/` holds the user's review guesses (`rallies.json`). Never rerun it outside a sandbox.
- MPS/CUDA availability changes speed, not results.
