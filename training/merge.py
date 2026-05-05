import os, shutil

# Merge images
for split in ["train", "valid", "test"]:
    src = f"volleyball_v6/{split}/images"
    dst = f"volleyball_combined/{split}/images"
    for f in os.listdir(src):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")

# Merge labels
for split in ["train", "valid", "test"]:
    src = f"volleyball_v6/{split}/labels"
    dst = f"volleyball_combined/{split}/labels"
    for f in os.listdir(src):
        shutil.copy(f"{src}/{f}", f"{dst}/{f}")
        
        