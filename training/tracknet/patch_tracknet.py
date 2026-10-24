"""Patch a fresh TrackNetV3 checkout (github.com/qaz812345/TrackNetV3, MIT) for VolleyVision fine-tuning.

    python patch_tracknet.py /path/to/TrackNetV3 /path/to/tracknet_data

  - IMG_FORMAT png -> jpg (our frames are JPEG)
  - dataset root -> the given data dir
  - .cuda() -> .to(DEVICE) so the same code runs on GPU (Colab) or CPU (smoke test)
  - train.py: start from the checkpoint in env INIT_CKPT (fine-tuning) unless resuming
  - train.py: data-loader workers capped at the CPU count (Colab has 2)
  - dataset.py / test.py: test-split detection by folder, not by "test" anywhere in the path
  - dataset.py: faster training loader (cached background, preallocated float32 arrays);
    output identical to the original for our setup (heatmap, bg concat, no frame mixup)
Idempotent: running it twice is harmless.
"""
import re
import sys
from pathlib import Path

repo, data = Path(sys.argv[1]), Path(sys.argv[2]).resolve()

g = repo / "utils/general.py"
s = g.read_text()
s = s.replace("IMG_FORMAT = 'png'", "IMG_FORMAT = 'jpg'")
g.write_text(s)

FAST = '''

# ── VolleyVision: fast training loader (same output as __getitem__ for heatmap/concat, no frame mixup) ──
_VV_MED, _VV_GRID = {}, {}
_vv_orig_getitem = Shuttlecock_Trajectory_Dataset.__getitem__

def _vv_getitem(self, idx):
    if self.data_mode != 'heatmap' or self.frame_alpha > 0 or self.bg_mode != 'concat' or not hasattr(self, 'img_config'):
        return _vv_orig_getitem(self, idx)
    data_idx = self.data_dict['id'][idx]
    frame_file = self.data_dict['frame_file'][idx]
    coor = np.array(self.data_dict['coor'][idx], dtype=np.float64)
    vis = self.data_dict['vis'][idx]
    w, h = self.img_config['img_shape'][data_idx[0][0]]
    ws, hs = self.img_config['img_scaler'][data_idx[0][0]]
    H, W, L = self.HEIGHT, self.WIDTH, self.seq_len
    match_dir, rally_id, _ = parse.parse(os.path.join('{}', 'frame', '{}', '{}.' + IMG_FORMAT), frame_file[0])
    mf = os.path.join(match_dir, 'median.npz')
    if not os.path.exists(mf):
        mf = os.path.join(match_dir, 'frame', rally_id, 'median.npz')
    med = _VV_MED.get(mf)
    if med is None:
        m = Image.fromarray(np.load(mf)['median'].astype('uint8')).resize(size=(W, H))
        med = _VV_MED[mf] = np.moveaxis(np.array(m), -1, 0)
    if (H, W) not in _VV_GRID:
        y, x = np.meshgrid(np.linspace(1, W, W), np.linspace(1, H, H))[::-1]
        _VV_GRID[(H, W)] = (y, x)
    yy, xx = _VV_GRID[(H, W)]
    frames = np.empty((L + 1, 3, H, W), np.float32)
    frames[0] = med
    heat = np.zeros((L, H, W), np.float32)
    s2 = self.sigma ** 2
    for i in range(L):
        img = Image.open(frame_file[i])
        if img.size != (W, H):
            img = img.resize(size=(W, H))
        frames[i + 1] = np.moveaxis(np.asarray(img), -1, 0)
        cx, cy = int(coor[i][0] / ws), int(coor[i][1] / hs)
        if not (cx == cy == 0):
            heat[i] = (((yy - (cy + 1)) ** 2 + (xx - (cx + 1)) ** 2) <= s2) * self.mag
    frames = frames.reshape(-1, H, W)
    frames /= 255.
    coor[:, 0] = coor[:, 0] / w
    coor[:, 1] = coor[:, 1] / h
    return data_idx, frames, heat, coor, vis

Shuttlecock_Trajectory_Dataset.__getitem__ = _vv_getitem
'''

d = repo / "dataset.py"
s = d.read_text()
s = re.sub(r"^data_dir = .*$", f"data_dir = '{data}'", s, count=1, flags=re.M)
# upstream picks the test-set label folder whenever "test" appears ANYWHERE in the path
# (our clips are named testN): check the split folder instead
s = s.replace("if 'test' in rally_dir:", "if os.sep + 'test' + os.sep in rally_dir.replace(self.root_dir, ''):")
if "_vv_getitem" not in s:
    s += FAST
d.write_text(s)

DEV = "DEVICE =__import__('torch').device('cuda' if __import__('torch').cuda.is_available() else 'cpu')\n"
for name in ("train.py", "test.py"):
    p = repo / name
    s = p.read_text()
    if "DEVICE = " not in s:
        s = DEV + s
    s = s.replace(".cuda()", ".to(DEVICE)")
    p.write_text(s)

tp = repo / "test.py"
s = tp.read_text()
s = s.replace("if 'test' in rally_dir else", "if os.sep + 'test' + os.sep in rally_dir else")
tp.write_text(s)

t = repo / "train.py"
s = t.read_text()
anchor = "    train_fn = train_tracknet if args.model_name == 'TrackNet' else train_inpaintnet\n"
inject = ("    if os.environ.get('INIT_CKPT') and not args.resume_training:\n"
          "        _ck = torch.load(os.environ['INIT_CKPT'], map_location=DEVICE, weights_only=False)\n"
          "        model.load_state_dict(_ck['model'])\n"
          "        print(f\"Fine-tuning from {os.environ['INIT_CKPT']}\")\n")
if "INIT_CKPT" not in s:
    assert anchor in s, "train.py layout changed; patch by hand"
    s = s.replace(anchor, inject + anchor)
s = s.replace("num_workers = args.batch_size if args.batch_size <= 16 else 16",
              "num_workers = min(args.batch_size, 16, os.cpu_count() or 2)")
s = s.replace("torch.load(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'))",
              "torch.load(os.path.join(args.save_dir, f'{args.model_name}_cur.pt'), weights_only=False)")
t.write_text(s)
print("patched", repo)
