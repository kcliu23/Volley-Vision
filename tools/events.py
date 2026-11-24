"""events.json + stats.json from the saved 3D (touches.json, rallies.json): no rerun.

    python -m tools.events videos/match.mov            # or a clip name; + --labels / --player-ids: that 3D run

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
    ap.add_argument("--player-ids", action="store_true", help="the 3D from tools.ball3d_run --player-ids")
    a = ap.parse_args()
    for v in a.videos:
        stem = Path(v).stem
        r = run(stem, a.output, "_labels" if a.labels else ("_ids" if a.player_ids else ""))
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
        for line in report(st):
            print(f"[events] {stem}: {line}")


def report(st: dict) -> list[str]:
    """Receiving by zone, drawn as each half court seen from that team facing the net, and per player."""
    def pct(g):
        return "   -" if g["positive"] is None else f"{g['positive'] * 100:3.0f}%"

    def cell(g):
        return f"{g['n']:2d} {pct(g)}"
    out = []
    for team, other in (("near", "far"), ("far", "near")):
        z = st["teams"][team]["reception_by_zone"]
        z = {int(k): v for k, v in z.items()}
        out.append(f"{team} receiving by zone (receptions, positive %), facing the net"
                   f" = where {other} served to:")
        out.append(f"    net:  4 {cell(z[4])}   3 {cell(z[3])}   2 {cell(z[2])}")
        out.append(f"          5 {cell(z[5])}   6 {cell(z[6])}   1 {cell(z[1])}")
    for k, p in sorted(st.get("players", {}).items(), key=lambda kv: (str(kv[1]["team"]), kv[0].zfill(3))):
        g = p["reception"]
        other = ", ".join(f"{t} {n}" for t, n in sorted(p["touches"].items()) if t != "reception")
        out.append(f"#{k:>2s} {str(p['team']):4s} receptions {g['n']:2d} (A {g['A']} B {g['B']} C {g['C']} err {g['error']})"
                   f" positive {pct(g)}   other touches: {other or '-'}")
    return out


if __name__ == "__main__":
    main()
