"""Contact sheet of every WEAK and FILLED frame from a link result, for eyeballing."""
import json, sys, cv2, numpy as np
link, video, out = sys.argv[1], sys.argv[2], sys.argv[3]
states = sys.argv[4].split(",") if len(sys.argv) > 4 else ["weak", "filled"]
R = {x["f"]: x for x in json.load(open(link))["frames"]}
want = {f for f, x in R.items() if x["state"] in states and not x.get("dup")}
cap = cv2.VideoCapture(video); tiles = []; f = 0
col = {"detected": (0, 255, 0), "weak": (0, 255, 255), "filled": (255, 0, 255)}
while True:
    ok, im = cap.read()
    if not ok: break
    if f in want:
        x, y = R[f]["pos"]; H, W = im.shape[:2]; S = 110
        x0, y0 = int(np.clip(x - S, 0, W - 2 * S)), int(np.clip(y - S, 0, H - 2 * S))
        t = im[y0:y0 + 2 * S, x0:x0 + 2 * S].copy()
        cv2.circle(t, (int(x - x0), int(y - y0)), 20, col[R[f]["state"]], 2)
        cv2.putText(t, f"{f} {R[f]['state'][0]}", (3, 18), 0, .55, (0, 0, 255), 2)
        tiles.append(cv2.resize(t, (150, 150)))
    f += 1
cols = 14
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
g = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
cv2.imwrite(out, g, [cv2.IMWRITE_JPEG_QUALITY, 82]); print(len(want), g.shape)
