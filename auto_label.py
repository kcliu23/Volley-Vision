from ultralytics import YOLO
import cv2
import os


model = YOLO("models/YOLOv11m.pt")
# model = YOLO("models/best.pt")
source_video = "tests/test14.mov"
project_name = "auto_labels_clean"
video_stem = os.path.splitext(os.path.basename(source_video))[0]

cap = cv2.VideoCapture(source_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open: {source_video}")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

idx = 1
while os.path.exists(os.path.join(project_name, f"{video_stem}_{idx}")):
    idx += 1
run_dir = os.path.join(project_name, f"{video_stem}") if idx == 1 else os.path.join(project_name, f"{video_stem}_{idx}")
image_save_path = os.path.join(run_dir, "images")
label_save_path = os.path.join(run_dir, "labels")
os.makedirs(image_save_path, exist_ok=True)
os.makedirs(label_save_path, exist_ok=True)
print(f"[auto_label] Saving to: {run_dir}")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25, verbose=False)
    r = results[0]

    base_name = f"{video_stem}_{frame_idx}"

    cv2.imwrite(os.path.join(image_save_path, f"{base_name}.jpg"), frame)

    detected = r.boxes is not None and len(r.boxes) > 0
    if detected:
        with open(os.path.join(label_save_path, f"{base_name}.txt"), "w") as f:
            for box in r.boxes.xywhn:
                x, y, w, h = box.tolist()
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    
    if detected:
        conf = float(r.boxes.conf[0])
        print(f"  {frame_idx}/{total}  ball: DETECTED  conf: {conf:.2f}")
    else:
        print(f"  {frame_idx}/{total}  ball: not found")

    frame_idx += 1

cap.release()

# Write data.yaml for Roboflow
with open(os.path.join(run_dir, "data.yaml"), "w") as f:
    f.write("nc: 1\nnames: ['volleyball']\n")

print(f"Done! Images: {image_save_path}")
print(f"      Labels: {label_save_path}")
