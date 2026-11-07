"""python -m tests.test_safe_io : saving hand-made data can't lose it."""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.safe_io import backup, backups, load_json, save_json  # noqa: E402


def main():
    d = Path(tempfile.mkdtemp())
    p = d / "clip.json"
    res = []
    save_json(p, {"v": 1})
    res.append(("save", json.load(open(p)) == {"v": 1} and not list(d.glob("*.tmp"))))
    backup(p)
    p.write_text("{ broken")                                   # crash mid-write in an old version
    res.append(("damaged -> backup", load_json(p, log=lambda m: None) == {"v": 1}))
    n = len(backups(p))
    time.sleep(1.1)
    backup(p)                                                  # a damaged file never becomes a backup
    res.append(("no damaged backup", len(backups(p)) == n))
    for i in range(7):
        save_json(p, {"v": i})
        backup(p, keep=5)
        time.sleep(1.05)
    res.append(("keep 5", len(backups(p)) == 5))
    fails = 0
    for name, ok in res:
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
