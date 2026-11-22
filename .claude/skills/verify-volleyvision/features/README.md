# VolleyVision feature map

One file per user-facing feature. Each says what the user does, how to drive it in a sandbox, and what observable end state proves it works.

| Feature | File | Surface |
|---|---|---|
| Rally guesses: accept, reject or mark a serve / rally over | [review-rally-guesses.md](review-rally-guesses.md) | `tools.review` keys |
| Moving through frames: step, jump, play, and the frame shown is the frame named | [review-frames.md](review-frames.md) | `tools.review` keys |
| The pipeline on a clip | [pipeline.md](pipeline.md) | `python -m main` |
| Rally scores and the cut video | [rally-scoring.md](rally-scoring.md) | `evaluation.rallies_eval`, `evaluation.rally_v1`, `tools.rally_video` |

**Not mapped yet:**
- ball labeling in `tools.review` (clicks, `i`/`I` fill, `Enter` accept);
- other balls (shift+click, `g`, `r`/`R`);
- `tools.label_players`;
- `tools.calibrate_court`, which is interactive clicking.
