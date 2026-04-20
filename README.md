# 🏐 VolleyVision

Volleyball analytics platform — ball trajectory tracking, jump analysis, and rally detection.

## Setup

```bash
pip install -r requirements.txt
```

Download YOLOv8 nano weights (auto-downloads on first run):
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Move the downloaded file to models/yolov8n.pt
```

## Usage

**Analyse a video file:**
```bash
python main.py video path/to/match.mp4
```

**Live camera:**
```bash
python main.py live
```

**With real-world calibration** (measure net width in pixels from your footage):
```bash
python main.py video match.mp4 --ref-px 480 --ref-meters 9
```

## Controls (live mode)
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Save snapshot |
| R | Reset tracker |

## Project Modules

| Module | Status |
|--------|--------|
| Ball Tracking + Trajectory | ✅ Milestone 1 |
| Player Pose + Jump Height | 🔜 Milestone 2 |
| Rally Detection + Cropping | 🔜 Milestone 3 |
| Streamlit Dashboard | 🔜 Milestone 4 |
