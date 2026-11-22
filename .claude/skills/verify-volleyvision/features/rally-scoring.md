# Rally scores and the cut video

The user reads how well the rally finder does against their marks, and watches the game with the dead time cut out.

## Sub-features

- `rallies-eval`: `python -m evaluation.rallies_eval <clips> [--detail]`. It scores only the checked part, and reports:
  - found, missed and fake rallies (a fake you rejected with `X` is reported as such);
  - play kept and dead time cut;
  - starts within 3 frames of the serve mark and ends within 0.5 s. `P`-accepted and `v` serves are left out.
- `rally-v1`: `python -m evaluation.rally_v1 <clips> [--rules <dir>]`. It compares the learned detector, the rule and v1+rule, leaving one clip out.
- `rally-video`: `python -m tools.rally_video videos/<clip>.mp4` writes `output/<clip>/rallies.mp4`. Each rally starts 2.5 s (`PRE_S`) before its start and ends 1.5 s after.

## How to get to it (user POV)

- `python -m evaluation.rallies_eval full_rally_test` after a labeling session.
- `python -m tools.rally_video videos/full_rally_test.mp4` to watch.

## Driving it with the shell

Preconditions: a sandbox for `full_rally_test` (or a holdout), with `output/<clip>/rallies.json` copied in.

Steps:
- **Score.** `cd "$SB" && PYTHONPATH=<repo> python -m evaluation.rallies_eval full_rally_test --detail`. The output shows:
  - a row with the checked span;
  - `missed:` and `fake:` lines;
  - `start - your serve: within 3 frames` and `end - your rally over: within 0.5 s`.
- **Score moves with a label.** Drive `p X` with `drive_review.py`, then rerun. The `fake ... (N of them you rejected with X)` count goes up by one, or the span grows.
- **Compare detectors.** `PYTHONPATH=<repo> python -m evaluation.rally_v1 full_rally_test holdout1 ... holdout10`, which takes about 1 min. It ends with three `ALL` lines: v1, rule and v1+rule.
- **Cut video.** `PYTHONPATH=<repo> python -m tools.rally_video videos/<clip>.mp4` writes `output/<clip>/rallies.mp4`.
  - Check the frame count with `ffprobe -count_packets`. It should be about the sum of (f1 - f0 + 1 + 4 s × fps) over the rallies, minus overlaps.
  - Grab a frame 2 s into the first rally to see the server before the hit.
- **Proof.** Save the report text, and for the video one PNG per checked moment, in the evidence directory.

## Gotchas

- The sandbox has no `.mp4` files, so `rally_video` reads the original video, not `ball3d.mp4`. The overlay differs from the user's; the cut doesn't.
- `rally_v1` needs `rallies.json` for each clip. For `test*` clips that comes from a scratch rerun (`--rules <dir>`).
- A serve with no rally-over mark yet is left out of the truth, so the rule's rally there shows as `fake`. That's a labeling gap, not a detector fake.
- The scores change while the user labels. Compare before/after in the same sandbox, not against numbers in notes.
