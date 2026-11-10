"""Text labels drawn over video: solid pills with a real font, shared by the overlay videos.

Design (ui-ux-pro-max: Fira Sans for data, semantic colour tokens, WCAG contrast) and checked
on test20's frames behind real labels (2026-11-10):

    chip    a player's number: a solid pill in the side's colour, white text, a thin white ring
            so it separates from dark backgrounds (the net, dark walls); the saturated fill
            separates it from light ones. A thin coloured outline on a dark fill (the first
            version) dropped to 1.1:1 against dark backgrounds: the side was unreadable there.
    tag     the ball's height: an amber pill, dark text, a dark ring (amber alone fell to 2.4:1
            against the light gym walls).

Text contrast: white on blue-600 5.2:1, on rose-600 4.7:1, slate-200 on slate-600 6.2:1, dark on
amber 10.7:1 (all >= 4.5). Sizes follow the frame height (designed at 1080 px): a 30 px label on
a 1206 px video viewed 1300 px wide is ~15 px on screen.

OpenCV's built-in fonts are blocky and its shapes jagged, so labels are drawn with Pillow (Fira
Sans SemiBold, assets/fonts, SIL Open Font License) at 3x, scaled down, cached as small RGBA
sprites and alpha-blended into the frame.

    L = Labels(frame_height)
    L.chip(img, "#4", (x, y), fill=Labels.NEAR)             # (x, y) = the chip's bottom centre
    L.tag(img, "2.9 m", (x, y))                             # (x, y) = the tag's left middle
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SS = 3                                       # supersampling for smooth text and corners


class Labels:
    # colour tokens, RGB (Tailwind palette)
    NEAR = (37, 99, 235)                     # blue-600: the camera's side of the net
    FAR = (225, 29, 72)                      # rose-600: the far side
    UNNAMED = (71, 85, 105)                  # slate-600: a player nobody numbered
    BALL = (251, 191, 36)                    # amber-400: the ball's height
    INK = (15, 23, 42)                       # slate-900
    WHITE = (255, 255, 255)
    SOFT = (226, 232, 240)                   # slate-200

    def __init__(self, frame_height: int):
        self.s = frame_height / 1080

    def chip(self, img, text: str, bottom_centre, fill=NEAR, text_rgb=WHITE, size: float = 30):
        self.chips(img, [(text, bottom_centre, fill, text_rgb, size)])

    def _chip_sprite(self, text, fill, text_rgb, size):
        return _sprite(text, round(size * self.s), tuple(fill) + (245,), tuple(text_rgb),
                       Labels.WHITE + (220,), max(1, round(2 * self.s)))

    def chips(self, img, items):
        """Several chips at once, nudged apart so none covers another: items are (text,
        bottom_centre, fill, text_rgb, size); a chip that would overlap one already placed moves
        up just clear of it (players close together in the picture get a short stack)."""
        gap = max(2, round(3 * self.s))
        placed = []                                     # (x0, y0, x1, y1)
        for text, (cx, by), fill, text_rgb, size in sorted(items, key=lambda it: -it[1][1]):
            spr = self._chip_sprite(text, fill, text_rgb, size)
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
            _blend(img, spr, x0, y1 - h)

    def tag(self, img, text: str, left_middle, fill=BALL, size: float = 32):
        spr = _sprite(text, round(size * self.s), tuple(fill) + (250,), Labels.INK,
                      Labels.INK + (200,), max(1, round(2 * self.s)))
        h, _w = spr.shape[:2]
        _blend(img, spr, int(left_middle[0]), int(left_middle[1] - h / 2))


def _font(px: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(str(FONTS / "FiraSans-SemiBold.ttf"), px)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", px)
        except OSError:
            return ImageFont.load_default()


@lru_cache(maxsize=512)
def _sprite(text, px, fill_rgba, text_rgb, ring_rgba, ring):
    """A pill holding `text`: uint8 array (h, w, 4), BGR + alpha."""
    from PIL import Image, ImageDraw
    f = _font(px * SS)
    x0, _, x1, _ = f.getbbox(text, anchor="ls")
    padx, pady = int(0.5 * px * SS), int(0.34 * px * SS)
    tw, th = x1 - x0, int(0.70 * px * SS)               # cap height: digits and capitals sit centred
    W, H = tw + 2 * padx, th + 2 * pady
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 2, fill=ring_rgba)
    r = ring * SS
    d.rounded_rectangle((r, r, W - 1 - r, H - 1 - r), radius=H // 2 - r, fill=fill_rgba)
    d.text((padx - x0, pady + th), text, font=f, anchor="ls", fill=tuple(text_rgb) + (255,))
    im = im.resize((max(1, W // SS), max(1, H // SS)), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(im)[..., [2, 1, 0, 3]])


def _blend(img, spr, x, y):
    h, w = spr.shape[:2]
    H, W = img.shape[:2]
    xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if xa >= xb or ya >= yb:
        return
    s = spr[ya - y:yb - y, xa - x:xb - x]
    al = s[..., 3:4].astype(np.float32) / 255
    roi = img[ya:yb, xa:xb]
    roi[:] = (roi * (1 - al) + s[..., :3] * al).astype(np.uint8)
