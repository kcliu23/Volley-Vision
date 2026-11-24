"""The first stats from events.json (notes/ANALYSIS_DESIGN.md section 4): counts over the events.
Per team, per FIVB zone 1-6 (where the receiver stood), and per player (touches with a number:
tools.ball3d_run --player-ids).

    serve       attempts; aces (reception graded error) and errors need the rally's end: later
    reception   attempts, A / B / C / error, positive % (A + B); also per zone and per player.
                A team's serve targets are the other team's reception zones (how well they
                received where we served); aces the 3D missed aren't in them
    attack      attempts (kills / errors / blocked: need the rally's end, later)
    timing      serve -> reception, reception -> set, set -> attack (s, medians); rally length
    side-out    the receiving team wins the rally / rallies with a known winner
"""
from __future__ import annotations

from statistics import median


def _med(v):
    return round(median(v), 2) if v else None


def _grades(rec: list[dict]) -> dict:
    q = {g: sum(e["quality"] == g for e in rec) for g in ("A", "B", "C", "error")}
    graded = sum(q.values())
    return {"n": len(rec), **q, "positive": round((q["A"] + q["B"]) / graded, 2) if graded else None}


def team_stats(ev: dict) -> dict:
    E, R = ev["events"], ev["rallies"]
    out = {}
    for team in ("near", "far"):
        mine = [e for e in E if e["team"] == team]
        rec = [e for e in mine if e["type"] == "reception"]
        q = {g: sum(e["quality"] == g for e in rec) for g in ("A", "B", "C", "error")}
        graded = sum(q.values())
        out[team] = {
            "serves": sum(e["type"] == "serve" for e in mine),
            "receptions": len(rec), "reception_quality": q,
            "reception_positive": round((q["A"] + q["B"]) / graded, 2) if graded else None,
            "sets": sum(e["type"] == "set" for e in mine),
            "attacks": sum(e["type"] == "attack" for e in mine),
            "blocks": sum(e["type"] == "block" for e in mine),
            "digs": sum(e["type"] == "dig" for e in mine),
            "reception_by_zone": {z: _grades([e for e in rec if e["zone"] == z]) for z in range(1, 7)},
        }
    players = {}
    for e in E:
        if e.get("player") is not None:
            p = players.setdefault(str(e["player"]), {"team": e["team"], "touches": {}})
            p["touches"][e["type"]] = p["touches"].get(e["type"], 0) + 1
    for k, p in players.items():
        p["reception"] = _grades([e for e in E if str(e.get("player")) == k and e["type"] == "reception"])
    gaps = {"serve_to_reception": [], "reception_to_set": [], "set_to_attack": []}
    pairs = {("serve", "reception"): "serve_to_reception", ("reception", "set"): "reception_to_set",
             ("set", "attack"): "set_to_attack"}
    for a, b in zip(E, E[1:]):
        k = pairs.get((a["type"], b["type"]))
        if k and a["rally"] == b["rally"] and a["t"] is not None and b["t"] is not None:
            gaps[k].append(b["t"] - a["t"])
    fps_t = [(e["rally"], e["t"]) for e in E if e["t"] is not None]
    lengths = []
    for r in R:
        ts = [t for ri, t in fps_t if ri == r["id"]]
        if len(ts) >= 2:
            lengths.append(max(ts) - min(ts))
    known = [r for r in R if r["winner"] and r["serve_team"]]
    side_out = [r["winner"] != r["serve_team"] for r in known]
    return {"teams": out, "players": players, "timing_s": {k: _med(v) for k, v in gaps.items()} | {"rally_length": _med(lengths)},
            "side_out": round(sum(side_out) / len(side_out), 2) if side_out else None,
            "rallies": len(R), "rallies_with_winner": len(known)}
