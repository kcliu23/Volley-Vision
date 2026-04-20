from ultralytics import YOLO
import cv2
import os

# CLAHE enhances local contrast so the ball stands out against the background.
# We apply it before YOLO detection AND save the enhanced image as training data,
# so the model always sees the same type of input at both train and inference time.
clahe = cv2.createCLAHE(
    clipLimit=0.03,      # limits over-amplification of noise in uniform areas
    tileGridSize=(8, 8) # divides frame into 8x8 tiles for localized enhancement
)

def apply_clahe(frame):
    # CLAHE works on single-channel images.
    # Convert BGR → LAB, apply only to the L (lightness) channel, then convert back.
    # This boosts contrast without shifting colors.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


model = YOLO("models/YOLOv11m.pt")
source_video = "tests/test9.mov"
project_name = "auto_labels_clean"
video_stem = os.path.splitext(os.path.basename(source_video))[0]

# Read video frame-by-frame so we can apply CLAHE before YOLO sees each frame
cap = cv2.VideoCapture(source_video)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open: {source_video}")

run_dir = None
image_save_path = None
label_save_path = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply CLAHE before passing to YOLO — detection runs on the enhanced frame
    enhanced_frame = apply_clahe(frame)

    results = model.predict(
        enhanced_frame,
        conf=0.25,
        verbose=False,
        project=project_name,
        name="volleyball_data"
    )
    r = results[0]

    # Set up output folders using the first result's save_dir
    if run_dir is None:
        run_dir = r.save_dir
        image_save_path = os.path.join(run_dir, "images")
        label_save_path = os.path.join(run_dir, "labels")
        os.makedirs(image_save_path, exist_ok=True)
        os.makedirs(label_save_path, exist_ok=True)

    base_name = f"{video_stem}_{frame_idx}"

    # Save the CLAHE-enhanced frame — matches what YOLO saw during detection
    cv2.imwrite(os.path.join(image_save_path, f"{base_name}.jpg"), enhanced_frame)

    if r.boxes is not None and len(r.boxes):
        with open(os.path.join(label_save_path, f"{base_name}.txt"), "w") as f:
            for box in r.boxes.xywhn:
                x, y, w, h = box.tolist()
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    frame_idx += 1

cap.release()

# Write data.yaml for Roboflow
with open(os.path.join(run_dir, "data.yaml"), "w") as f:
    f.write("nc: 1\nnames: ['volleyball']\n")

print(f"Done! Images: {image_save_path}")
print(f"      Labels: {label_save_path}")
