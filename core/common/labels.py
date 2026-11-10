"""Text labels drawn over video: rounded chips with a real font, shared by the overlay videos.

OpenCV's built-in fonts are blocky and its shapes have jagged edges, so labels are drawn with
Pillow (Fira Sans, assets/fonts, SIL Open Font License) at 3x and scaled down, then cached as
small RGBA sprites and alpha-blended into the frame. One look for every label:

    chip    a pill: dark translucent fill, white text, an outline in the label's colour
    tag     a solid pill (the ball's height): coloured fill, dark text

Sizes follow the frame height (designed at 1080 px), so a 4K and a 720p video look the same.

    L = Labels(frame_height)
    L.chip(img, "#4", (x, y), outline=Labels.NEAR)          # (x, y) = the chip's bottom centre
    L.tag(img, "2.9 m", (x, y), anchor="left")              # (x, y) = the tag's left middle
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SS = 3                                       # supersampling for smooth text and corners


class Labels:
    # colour tokens, RGB (Tailwind palette: blue-500, rose-500, slate-400, amber-400, slate-900)
    NEAR = (59, 130, 246)
    FAR = (244, 63, 94)
    MUTED = (148, 163, 184)
    BALL = (251, 191, 36)
    INK = (15, 23, 42)
    WHITE = (248, 250, 252)

    def __init__(self, frame_height: int):
        self.s = frame_height / 1080

    def chip(self, img, text: str, bottom_centre, outline=MUTED, size: float = 26, text_rgb=WHITE):
        spr = _sprite(text, round(size * self.s), outline, Labels.INK + (205,), text_rgb, round(3 * self.s))
        h, w = spr.shape[:2]
        _blend(img, spr, int(bottom_centre[0] - w / 2), int(bottom_centre[1] - h))

    def tag(self, img, text: str, point, anchor: str = "left", fill=BALL, size: float = 28):
        spr = _sprite(text, round(size * self.s), None, fill + (240,), Labels.INK, 0)
        h, w = spr.shape[:2]
        x = int(point[0]) if anchor == "left" else int(point[0] - w)
        _blend(img, spr, x, int(point[1] - h / 2))


def _font(px: int):
    from PIL import ImageFont
    for name in ("FiraSans-SemiBold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(str(FONTS / name) if name.startswith("Fira") else name, px)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=512)
def _sprite(text, px, outline, fill_rgba, text_rgb, line):
    """An RGBA pill holding `text`, as a premultiplied-free uint8 array (h, w, 4) in BGR + alpha."""
    from PIL import Image, ImageDraw
    f = _font(px * SS)
    x0, _, x1, _ = f.getbbox(text, anchor="ls")
    padx, pady = int(0.55 * px * SS), int(0.36 * px * SS)
    tw, th = x1 - x0, int(0.70 * px * SS)               # cap height: digits and capitals sit centred
    W, H = tw + 2 * padx, th + 2 * pady
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=H // 2, fill=tuple(fill_rgba),
                        outline=(tuple(outline) + (255,)) if outline else None, width=line * SS)
    d.text((padx - x0, pady + th), text, font=f, anchor="ls", fill=tuple(text_rgb) + (255,))
    im = im.resize((max(1, W // SS), max(1, H // SS)), Image.LANCZOS)
    a = np.asarray(im)
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


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
