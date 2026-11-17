"""One pass over a clip: every label it needs, in order, then what to re-check.

    python -m tools.annotate holdout/holdout1.mov
    python -m tools.annotate videos/test20.mov --net women

Each step says what's there and asks: Enter = do it, s = skip it, q = stop here (everything you
did is saved; run it again to carry on). A step that is already done says so and suggests
skipping.

    1  court        tools.calibrate_court   the camera from the court lines (net height checked)
    2  people       tools.pose              boxes + body points (far back court zoomed: servers);
                                            runs again if the saved run is older than the zoom
                                            (makes the ball proposals first if there are none)
    3  players      tools.label_players     numbers (1-6 near, 7-12 far; a to take the guesses)
    4  ball, touch  tools.review            the ball in every frame + touch marks (m, type, and
                                            who touched it: the numbers from step 3); b jumps to
                                            frames to re-check
    5  3D + check   tools.ball3d_run, tools.check_labels
                                            builds the 3D, then lists every frame to re-check

Players come before touches, so "who touched it" uses the numbers you just gave.
Holdout clips (holdout*) are marked as test clips in step 4: never trained on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from core.common import paths


def _ask(what: str, done: bool) -> str:
    hint = "done already: s to skip, Enter to redo" if done else "Enter to do it, s to skip"
    try:
        r = input(f"\n── {what}  [{hint}, q to stop] ").strip().lower()
    except EOFError:
        r = "q"
    return "q" if r.startswith("q") else ("s" if r.startswith("s") else "go")


def _run(args: list[str]) -> int:
    print("   $ python -m " + " ".join(args), flush=True)
    return subprocess.call([sys.executable, "-m"] + args)


def _camera_state(stem: str) -> tuple[bool, str]:
    p = Path("data/cameras") / f"{stem}.json"
    if not p.exists():
        return False, "not calibrated"
    c = json.load(open(p))
    rms = c.get("rms")
    return True, f"calibrated (line error {rms:.1f} px, net {c.get('net_height', '?')} m)" if isinstance(rms, (int, float)) \
        else "calibrated"


def _review_state(stem: str) -> tuple[bool, str]:
    vp = Path(paths.verified(stem))
    if not vp.exists():
        return False, "no ball labels yet"
    V = json.load(open(vp))
    from core.common.label_checks import all_flags
    n_lab = sum(isinstance(v, list) for v in (V.get("labels") or {}).values())
    n_touch = len(V.get("contacts") or {})
    n_who = len(V.get("touched_by") or {})
    flags = all_flags(V)
    return True, (f"{n_lab} ball frames, {n_touch} touch marks ({n_who} with who touched it), "
                  f"{len(flags)} to re-check (b in the tool)")


def _pose_state(stem: str, out: str) -> tuple[bool, str]:
    p = paths.clip_dir(stem, out) / "pose.json"
    if not p.exists():
        return False, "no poses yet"
    head = open(p).read(400)                        # the header is first: no need to load it all
    zoom = '"far_zoom"' in head and '"far_zoom":null' not in head.replace(" ", "")
    return zoom, "poses with the far court zoom" if zoom else "poses WITHOUT the far court zoom (servers missed): rerun"


def _players_state(stem: str) -> tuple[bool, str]:
    p = Path("data/player_ids") / f"{stem}.json"
    if not p.exists():
        return False, "no player numbers yet"
    ids = json.load(open(p))
    nums = sorted({b.get("player") for v in (ids.get("frames") or {}).values() for b in v
                   if isinstance(b.get("player"), int)})
    return True, f"numbered: {', '.join('#' + str(n) for n in nums)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--net", default=None, help="for step 1: men / women / a height (default: the calibration tool's)")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    stem = Path(a.video).stem
    test = stem.startswith("holdout")
    print(f"[annotate] {stem}" + ("  (holdout: a test clip, never trained on)" if test else ""))

    done, what = _camera_state(stem)
    r = _ask(f"1 court: {what}", done)
    if r == "q":
        return
    if r == "go":
        _run(["tools.calibrate_court", a.video] + (["--net", a.net] if a.net else []))

    done, what = _pose_state(stem, a.output)
    r = _ask(f"2 people: {what}", done)
    if r == "q":
        return
    if r == "go":
        if not Path(paths.proposals(stem)).exists():       # the 2D touches tell pose where to look closely
            print(f"   no ball proposals for {stem} yet: making them first (tools.autolabel)")
            _run(["tools.autolabel", a.video])
        _run(["tools.pose", a.video, "--redo", "--output", a.output])

    done, what = _players_state(stem)
    r = _ask(f"3 players: {what}", False)               # after a new pose run: number the new people
    if r == "q":
        return
    if r == "go":
        _run(["tools.label_players", a.video, "--output", a.output])

    done, what = _review_state(stem)
    r = _ask(f"4 ball + touches: {what}", False)       # always worth a look: b shows what to re-check
    if r == "q":
        return
    if r == "go":
        if not Path(paths.proposals(stem)).exists():
            print(f"   no ball proposals for {stem} yet: making them first (tools.autolabel)")
            _run(["tools.autolabel", a.video])
        _run(["tools.review", a.video] + (["--test"] if test else []))

    r = _ask("5 3D + what to re-check", False)
    if r == "q":
        return
    if r == "go":
        _run(["tools.ball3d_run", a.video, "--output", a.output])
        _run(["tools.check_labels", stem, "--output", a.output])
    print(f"\n[annotate] {stem}: done. Run again any time: steps already done can be skipped.")


if __name__ == "__main__":
    main()
