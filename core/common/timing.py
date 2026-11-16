"""Where a run's time goes: named laps, printed as they finish and as one table at the end.

    T = Laps("[VolleyVision]")
    ... ; T.lap("sparse YOLO pass")
    ... ; T.lap("WASB finder")
    T.report()                     # each lap's seconds and share of the total
"""
from __future__ import annotations

import time


class Laps:
    def __init__(self, tag: str = "", log=print):
        self.tag, self.log = tag, log
        self.t0 = self.t = time.time()
        self.laps: list[tuple[str, float]] = []

    def lap(self, name: str) -> float:
        now = time.time()
        dt, self.t = now - self.t, now
        self.laps.append((name, dt))
        self.log(f"{self.tag} {name}: {dt:.1f} s")
        return dt

    def report(self, title: str = "time") -> None:
        tot = max(time.time() - self.t0, 1e-9)
        self.log(f"{self.tag} {title}: {tot:.1f} s")
        for name, dt in sorted(self.laps, key=lambda x: -x[1]):
            self.log(f"{self.tag}   {dt:7.1f} s  {100 * dt / tot:4.0f}%  {name}")
