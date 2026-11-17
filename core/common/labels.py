"""Text labels drawn over video, one look for every overlay: a clean, cool-toned HUD.

Design (ui-ux-pro-max, "HUD / FUI" kept restrained): sharp dark glass plates, coloured
edges, a short accent bar, monospaced figures (Fira Code); the two sides in two strongly
different hues (cyan / fuchsia) so they separate at a glance. Contrast checked
against the gym's own backgrounds (test20, 2026-11-10): the text sits on the plate's dark fill
(88% opaque), so it stays readable over white walls, the maple floor and the dark net alike:

    near side   cyan-300  #67E8F9     9.9:1 or more
    far side    fuchsia-400 #E879F9  5.8:1 or more
    unnamed     slate-400 #94A3B8   4.9:1 or more
    values      slate-50  #F8FAFC   (the ball's height)

Colour never carries meaning alone: the number is always text, and the side colours are named
in a small legend (Labels.legend).

Labels are drawn with Pillow at 3x, scaled down, cached as small RGBA sprites and alpha-blended
into the frame. Sizes follow the frame height (designed at 1080 px).

    L = Labels(frame_height)
    L.chips(img, [("#4", (x, y), Labels.NEAR), ...])        # (x, y) = each plate's bottom centre
    L.tag(img, "2.9 m", (x, y))                             # (x, y) = the plate's left middle
    L.legend(img)                                           # bottom left: what the colours mean
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SS = 3                                       # supersampling for smooth text and edges


class Labels:
    # colour tokens, RGB (Tailwind palette)
    NEAR = (103, 232, 249)                   # cyan-300: the camera's side of the net
    FAR = (232, 121, 249)                    # fuchsia-400: the far side
    UNNAMED = (148, 163, 184)                # slate-400: nobody numbered them
    VALUE = (248, 250, 252)                  # slate-50: measured values (ball height)
    PLATE = (8, 13, 24)                      # the dark glass

    def __init__(self, frame_height: int):
        self.s = frame_height / 1080

    def _plate(self, text, color, size, bar=True):
        return _sprite(text, round(size * self.s), tuple(color), round(2 * self.s * SS) / SS,
                       round(5 * self.s) if bar else 0)

    def chip(self, img, text: str, bottom_centre, color=NEAR, size: float = 28):
        self.chips(img, [(text, bottom_centre, color, size)])

    def chips(self, img, items):
        """Several plates at once, nudged apart so none covers another: items are (text,
        bottom_centre, color[, size[, opacity 0-1]]); a plate that would overlap one already placed
        moves up just clear of it (players close together in the picture get a short stack)."""
        gap = max(2, round(4 * self.s))
        placed = []                                     # (x0, y0, x1, y1)
        for it in sorted(items, key=lambda it: -it[1][1]):
            text, (cx, by), color = it[:3]
            spr = self._plate(text, color, it[3] if len(it) > 3 else 28)
            h, w = spr.shape[:2]
            x0, y1 = int(cx - w / 2), int(by)
            moved = True
            while moved:
                moved = False
                for a0, b0, a1, b1 in placed:
                    if x0 < a1 and a0 < x0 + w and y1 - h < b1 and b0 < y1:
                        y1 = b0 - gap
                        moved = True
            placed.append((x0, y1 - h, x0 + w, y1))
            _blend(img, spr, x0, y1 - h, it[4] if len(it) > 4 else 1.0)

    def tag(self, img, text: str, left_middle, color=VALUE, size: float = 30):
        spr = self._plate(text, color, size, bar=False)
        _blend(img, spr, int(left_middle[0]), int(left_middle[1] - spr.shape[0] / 2))

    def legend(self, img, items=(("NEAR SIDE", NEAR), ("FAR SIDE", FAR))):
        """What the colours mean, small, bottom left."""
        x, y = round(24 * self.s), img.shape[0] - round(24 * self.s)
        for text, color in items:
            spr = self._plate(text, color, 20)
            _blend(img, spr, x, y - spr.shape[0])
            x += spr.shape[1] + round(8 * self.s)


def _font(px: int):
    from PIL import ImageFont
    for path in (FONTS / "FiraCode-SemiBold.ttf", FONTS / "FiraSans-SemiBold.ttf", "DejaVuSansMono-Bold.ttf"):
        try:
            return ImageFont.truetype(str(path), px)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=512)
def _sprite(text, px, color, edge, bar):
    """A sharp dark plate holding `text` in `color`, a 1-2 px edge in `color`, and (bar > 0) a
    short accent bar on the left: uint8 array (h, w, 4), BGR + alpha."""
    from PIL import Image, ImageDraw
    f = _font(px * SS)
    x0, _, x1, _ = f.getbbox(text, anchor="ls")
    padx, pady = int(0.45 * px * SS), int(0.36 * px * SS)
    lead = bar * SS * 2 if bar else 0
    tw, th = x1 - x0, int(0.70 * px * SS)               # cap height: digits and capitals sit centred
    W, H = lead + tw + 2 * padx, th + 2 * pady
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = max(1, int(0.12 * H))                            # nearly square corners: sharp, not pills
    e = max(1, int(edge * SS))
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=r, fill=Labels.PLATE + (224,), outline=color + (230,), width=e)
    if bar:
        d.rectangle((e, e + int(0.18 * H), e + bar * SS - 1, H - 1 - e - int(0.18 * H)), fill=color + (255,))
    d.text((lead + padx - x0, pady + th), text, font=f, anchor="ls", fill=color + (255,))
    im = im.resize((max(1, W // SS), max(1, H // SS)), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(im)[..., [2, 1, 0, 3]])


def _blend(img, spr, x, y, opacity: float = 1.0):
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y:yb - y, xa - x:xb - x]
    al = s[..., 3:4].astype(np.float32) / 255 * opacity
    roi = img[ya:yb, xa:xb]
    roi[:] = (roi * (1 - al) + s[..., :3] * al).astype(np.uint8)
