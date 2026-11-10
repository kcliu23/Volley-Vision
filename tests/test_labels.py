"""Overlay labels (core.common.labels): drawn inside the frame, never on top of each other."""
import numpy as np

from core.common.labels import Labels


def test_chips_do_not_overlap_and_stay_in_frame():
    img = np.zeros((1080, 1920, 3), np.uint8)
    L = Labels(1080)
    items = [(f"#{i}", (500 + 10 * i, 400), Labels.NEAR, Labels.WHITE, 30) for i in range(4)]
    L.chips(img, items)
    drawn = img.any(axis=2)
    rows = np.where(drawn.any(axis=1))[0]
    assert rows.max() <= 400 and rows.min() > 400 - 4 * 60     # stacked upwards from their spot
    # four separate chips: four separate runs of rows with ink in them
    runs = np.sum(np.diff(rows) > 1) + 1
    assert runs == 4


def test_chip_at_the_edge_is_clipped_not_crashing():
    img = np.zeros((200, 200, 3), np.uint8)
    Labels(200).chip(img, "#12", (2, 10))
    Labels(200).tag(img, "2.9 m", (190, 5))
    assert img.any()
