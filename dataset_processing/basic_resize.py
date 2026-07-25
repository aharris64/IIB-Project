# Resize to 224x224

# Resize so the smaller side is 224
# Crop from there if not square

import os
from pathlib import Path
from PIL import Image

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
source = os.path.join(DATASETS_ROOT, "Dataset", "raw")
destination = os.path.join(DATASETS_ROOT, "Dataset", "basic_resize_224")

SOURCE_ROOT = Path(source)
DEST_ROOT = Path(destination)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
TARGET_SIZE = 224
IMAGE_EXTS = {".jpg", ".png"}

DEST_ROOT.mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    src_cls_dir = SOURCE_ROOT / cls
    dst_cls_dir = DEST_ROOT / cls
    dst_cls_dir.mkdir(parents=True, exist_ok=True)

    for img_path in src_cls_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")

            w, h = img.size

            # Resize so shorter side = 224
            if w < h:
                new_w = TARGET_SIZE
                new_h = int(h * TARGET_SIZE / w)
            else:
                new_h = TARGET_SIZE
                new_w = int(w * TARGET_SIZE / h)

            img = img.resize((new_w, new_h), Image.BICUBIC) # Uses Bicubic

            # Center crop to 224x224
            left = (new_w - TARGET_SIZE) // 2
            top = (new_h - TARGET_SIZE) // 2
            right = left + TARGET_SIZE
            bottom = top + TARGET_SIZE

            img = img.crop((left, top, right, bottom))

            out_path = dst_cls_dir / img_path.name
            img.save(out_path, quality=95)

print("Resize to 224x224 complete.")