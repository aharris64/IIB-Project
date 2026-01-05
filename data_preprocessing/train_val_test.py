import random
from collections import defaultdict
from pathlib import Path
import shutil

SOURCE = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Dataset"
DESTINATION = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\train_test_val"

DATASET = "basic_resize_224"

SOURCE_ROOT = Path(SOURCE + "\\" + DATASET)
DEST_ROOT = Path(DESTINATION + "\\" + DATASET)

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}
SEED = 42
SPLIT = (0.70, 0.15, 0.15)

random.seed(SEED)
items = []

for label_dir in ["normal", "papilledema", "pseudopapilledema"]:
    for img_path in (SOURCE_ROOT / label_dir).iterdir():
        if img_path.suffix.lower() not in {".jpg", ".png"}:
            continue

        # filename: <label>_<imgID>_<DATASET>.jpg
        dataset = img_path.stem.split("_")[-1]

        items.append({
            "path": img_path,
            "label": label_dir,
            "dataset": dataset
        })

buckets = defaultdict(list)

for item in items:
    buckets[(item["dataset"], item["label"])].append(item)

train, val, test = [], [], []

for bucket in buckets.values():
    random.shuffle(bucket)
    n = len(bucket)
    print(n)

    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])

    train.extend(bucket[:n_train])
    val.extend(bucket[n_train:n_train + n_val])
    test.extend(bucket[n_train + n_val:])

print("train:", len(train))
print("val:", len(val))
print("test:", len(test))

for split in ["train", "val", "test"]:
    for cls in CLASSES:
        (DEST_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": train,
    "val": val,
    "test": test,
}

for split_name, split_items in SPLITS.items():
    for item in split_items:
        src = item["path"]
        dst = DEST_ROOT / split_name / item["label"] / src.name
        shutil.copy2(src, dst)