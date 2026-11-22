# Rally guesses in the labeling window

The rally finder's rallies (`output/<clip>/rallies.json`) show up in `tools.review` as cyan guesses: each start is a guessed serve and each end a guessed rally over. The user checks each guess and does one of three things:
- accepts it as is;
- rejects it as neither;
- marks the true frame themselves.

Those marks are the answer key for every rally number.

## Sub-features

- `guess-jump`: `p` goes to the next open guess. The top bar names it.
- `guess-accept`: `P` saves the guess as the user's mark (`contacts[f]` = `serve` or `dead`).
- `guess-reject`: `X` saves `not_rally[f]`, and the guess is no longer open. `u` undoes it.
- `mark-serve` / `mark-over`: `m` then `1` marks a serve, `m` then `9` marks a rally over. `m` on a marked frame removes the mark.
- `mark-unseen`: `v` on a marked touch toggles "not seen" (added to `unseen`). The serve isn't timed by `evaluation.rallies_eval`.
- `guess-cleared`: a user mark within 90 frames (`GUESS_NEAR`) of a guess of the same kind closes that guess.

## How to get to it (user POV)

- Open `python -m tools.review videos/full_rally_test.mp4`.
- Press `p`, look at the frame, then press `P`, `X`, or step with `a`/`d` and mark with `m 1` / `m 9`.

## Driving it with drive_review.py

Preconditions:
- A sandbox for `full_rally_test` exists. The holdouts have no open guesses.
- `doctor.sh "$SB"` prints `OK`.
- The start log line reads `N guessed serves / rally overs to check` with N > 0.

Steps:
- **Reject.** `--keys "p X"`.
  - `diff.txt` has `not_rally[<f>]: None -> serve|dead`.
  - `003_X.png` top bar says `guess rejected at <f> (<N-1> left; u undoes)`.
  - Proven on 2026-11-22: `not_rally[16800]: None -> dead`.
- **Undo.** `--keys "p X u"`. `diff.txt` says `no label changes`.
- **Accept.** `--keys "p P"`. `diff.txt` has `contacts[<f>]: None -> serve|dead`, and `<f>` is the frame in `002_p.png`'s top bar.
- **Mark yourself.** `--keys "p a a m 1"`. `diff.txt` has `contacts[<f-2>]: None -> serve`, and the guess at `<f>` is closed: the next `p` goes elsewhere.
- **Not seen.** `--keys "p a m 1 v"`. `diff.txt` has `contacts[<f-1>]` and `unseen +<f-1>`.
- **Proof.** Keep `diff.txt` and the PNG after the deciding key. Then run `cd "$SB" && PYTHONPATH=<repo> python -m evaluation.rallies_eval full_rally_test`: an `X` counts toward "you rejected with X", and a `v` serve shows in "serves not seen (v) left out".

## Gotchas

- `m` is a two-key sequence: `m` then a digit. A stray key in between cancels it.
- `x` (lower case) is another key. Reject is capital `X`.
- Guesses come from the sandbox's copy of `rallies.json`. Rerunning the rally finder in the sandbox moves the guesses, but marks and `X`s stay.
- Some open guesses sit inside the part the user has already checked (2026-11-22: frame 16800). That's the user's labeling state, not a bug in the tool.
- The real `data/verified/full_rally_test.json` changes while the user labels. Compare its keys for the frames you touched, not the whole file.
