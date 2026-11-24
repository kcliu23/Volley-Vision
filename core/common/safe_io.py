"""Saving hand-made data (labels, calibrations) so it can't be lost.

    save_json(path, obj)            write a temp file, then swap it in: a crash or Ctrl-C mid-write
                                    leaves the old file, never a half-written one
    load_json(path)                 read; if the file is damaged, the newest readable backup instead
    backup(path, keep=5)            a timestamped copy in a hidden folder next to it
                                    (<dir>/.backups/<stem>.backup-<time>.json), newest `keep` kept;
                                    call once per session, before the first change. Hidden, so tools
                                    that list <dir>/*.json never mistake a backup for a clip.
    file_key(path)                  a short fingerprint of a file's content (to notice it changed)
    g = Guarded(path); g.save(obj)  save_json, but never over changes made elsewhere since this
                                    session loaded or last saved the file (a second window on the
                                    same clip, an iCloud copy synced back): those changes stay, and
                                    this session's go to <dir>/.conflicts/<stem>.conflict-<time>.json
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def save_json(path, obj, **dump_kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, **dump_kw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


BACKUP_DIR = ".backups"


def backups(path) -> list[Path]:
    """Backups of `path`, oldest first (also old ones saved next to the file, before .backups/)."""
    path = Path(path)
    pat = f"{path.stem}.backup-*.json"
    return sorted([*path.parent.glob(pat), *(path.parent / BACKUP_DIR).glob(pat)], key=lambda b: b.name)


def load_json(path, log=print):
    """The file's content; if it is damaged, the newest backup that reads. Raises if none does."""
    path = Path(path)
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        for b in reversed(backups(path)):
            try:
                d = json.load(open(b))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            log(f"[safe_io] {path} is damaged ({e}); loaded the backup {b}")
            return d
        raise


def backup(path, keep: int = 5, content=None) -> Path | None:
    """Timestamped copy of `path` (or of `content`, the data as loaded, when the file itself is
    damaged), keeping the newest `keep`. Never copies a damaged file over the backups."""
    path = Path(path)
    if content is None:
        if not path.exists():
            return None
        try:
            content = json.load(open(path))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None                                     # damaged: don't let it push out a good backup
    out = path.parent / BACKUP_DIR / f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json"
    save_json(out, content, separators=(",", ":"))
    for b in backups(path)[:-keep]:
        b.unlink()
    return out


def file_key(path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Guarded:
    """A hand-label file one window owns. Make it right after loading the file."""

    def __init__(self, path, log=print):
        self.path, self.log = Path(path), log
        self.key = file_key(self.path) if self.path.exists() else None

    def save(self, obj, **dump_kw) -> Path:
        now = file_key(self.path) if self.path.exists() else None
        if now != self.key:                                 # someone else wrote it: keep theirs
            orig = self.path
            self.path = orig.parent / ".conflicts" / f"{orig.stem}.conflict-{time.strftime('%Y%m%d-%H%M%S')}.json"
            self.log(f"[safe_io] !!! {orig} was changed by another window or a sync since it was opened. "
                     f"Not overwriting it: this window now saves to {self.path}. Merge by hand.")
        save_json(self.path, obj, **dump_kw)
        self.key = file_key(self.path)
        return self.path
