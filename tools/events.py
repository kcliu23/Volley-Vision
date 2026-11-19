"""events.json + stats.json from the saved 3D (touches.json, rallies.json): no rerun.

    python -m tools.events videos/match.mov            # or a clip name; + --labels for the 3D from your labels

Each touch gets its rally, team, player, type (core.events: the cheapest sequence the rules allow,
from the ball's height and the 3D's sides), zone, and receptions a grade. Stats: core.stats.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.common import paths
from core.events import build
from core.stats import team_stats


def run(stem: str, output: str, sfx: str = "") -> tuple[dict, dict] | None:
    d = paths.clip_dir(stem, output)
    tp, rp = d / f"touches{sfx}.json", d / f"rallies{sfx}.json"
    if not tp.exists() or not rp.exists():
        print(f"[events] {stem}: no {tp.name} / {rp.name} (run the 3D first: python -m main)")
        return None
    T = json.load(open(tp))
    ev = build(T["touches"], json.load(open(rp))["rallies"], stem)
    st = team_stats(ev)
    json.dump(ev, open(d / f"events{sfx}.json", "w"), indent=1)
    json.dump(st, open(d / f"stats{sfx}.json", "w"), indent=1)
    return ev, st


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    for v in a.videos:
        stem = Path(v).stem
        r = run(stem, a.output, "_labels" if a.labels else "")
        if r is None:
            continue
        ev, st = r
        from collections import Counter
        c = Counter(e["type"] for e in ev["events"])
        print(f"[events] {stem}: {len(ev['rallies'])} rallies, {len(ev['events'])} touches: "
              + ", ".join(f"{k} {n}" for k, n in c.most_common()))
        for team, s in st["teams"].items():
            q = s["reception_quality"]
            print(f"[events] {stem}: {team:4s} serves {s['serves']}  receptions {s['receptions']} "
                  f"(A {q['A']} B {q['B']} C {q['C']} err {q['error']})  sets {s['sets']}  attacks {s['attacks']}"
                  f"  blocks {s['blocks']}  digs {s['digs']}")
        print(f"[events] {stem}: timing (median s) {st['timing_s']}  side-out {st['side_out']}")


if __name__ == "__main__":
    main()
