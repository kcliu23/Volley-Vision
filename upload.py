from roboflow import Roboflow
import os
import glob
import random

rf = Roboflow(api_key="6BezSMC6wWscvbI535Zu")

workspaceId = "laundry"
projectId = "volleyball-lcy2z"
project = rf.workspace(workspaceId).project(projectId)

image_dir = "auto_labels_clean/test14/images"
label_dir = "auto_labels_clean/test14/labels"

# 70% train, 20% valid, 10% test
SPLIT_RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}

images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
random.shuffle(images)

n = len(images)
n_train = int(n * SPLIT_RATIOS["train"])
n_valid = int(n * SPLIT_RATIOS["valid"])

splits = (
    [(img, "train") for img in images[:n_train]] +
    [(img, "valid") for img in images[n_train:n_train + n_valid]] +
    [(img, "test")  for img in images[n_train + n_valid:]]
)

print(f"Uploading {n} images  —  train:{n_train}  valid:{n_valid}  test:{n - n_train - n_valid}")

for i, (image_path, split) in enumerate(splits):
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, f"{base}.txt")

    project.upload(
        image_path=image_path,
        annotation_path=label_path if os.path.exists(label_path) else None,
        split=split,
        num_retry_uploads=3,
    )

    if i % 50 == 0:
        print(f"  {i}/{n} uploaded")

print("Done!")
