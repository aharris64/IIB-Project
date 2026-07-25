"""Splits a processed dataset (DATASET, under DATASETS_ROOT/Processed Datasets/) into
train/val/test, stratified per (class, diagnostic acronym) bucket so each diagnostic
sub-category is split proportionally rather than just each class as a whole.

Excludes specific acronyms from the papilledema class (PAPILLEDEMA_EXCLUDE) and, when
HAS_AUGMENTATION is set, keeps augmented copies of an image out of val/test (only
originals go there) so the same source image never appears in both train and eval
augmented variants are only ever added to train.
"""

import math
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# ---- Config ----

DATASET          = "disc_centred_r4.0_cl34_augmented_lowres14"   # <-- change to any folder name
HAS_AUGMENTATION = True   # False for quarters dataset (or any dataset with no augmented variants)

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
SOURCE_ROOT = Path(DATASETS_ROOT) / "Processed Datasets" / DATASET
DEST_ROOT   = Path(DATASETS_ROOT) / "Processed Datasets" / "train_test_val_low_res" / DATASET

CLASSES    = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SPLIT = (0.70, 0.15, 0.15)   # train, val, test
SEED  = 42

# These acronyms are labelled "papilledema" in the raw dataset, but actually denote the
# broader category of optic disc oedema rather than confirmed papilloedema specifically
PAPILLEDEMA_EXCLUDE = {"EDD", "RFM", "IFD"}

assert abs(sum(SPLIT) - 1.0) < 1e-9, "Split fractions must sum to 1.0"

random.seed(SEED)


# ---- Augmentation detection (only used when HAS_AUGMENTATION = True) ----

_AUG_RE = re.compile(r'(_hflip|_crop_(tl|tr|bl|br)|_q_(tl|tr|bl|br))$')

def is_augmented(stem: str) -> bool:
    """True if stem ends in an augmentation suffix (_hflip, _crop_tl, _q_tr, etc.)."""
    if not HAS_AUGMENTATION:
        return False
    return bool(_AUG_RE.search(stem))

def original_stem(stem: str) -> str:
    """Strip augmentation suffix to get the base image stem."""
    return _AUG_RE.sub("", stem)


# ---- Acronym extraction ----

_ACRONYM_RE = re.compile(r'_([A-Z]{2,})$')

def extract_acronym(stem: str) -> str:
    """Return the trailing diagnostic acronym (e.g. EDD, RFM) from a filename stem, or ""."""
    base = original_stem(stem)
    m = _ACRONYM_RE.search(base)
    return m.group(1) if m else ""


# ---- Split helpers ----

def floor_split(n: int, fractions: tuple) -> tuple[int, int, int]:
    """Split n items into (train, val, test) counts matching fractions, remainder to test."""
    n_train = math.floor(n * fractions[0])
    n_val   = math.floor(n * fractions[1])
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test


# ---- Collect images ----

paths_by_class_acronym: dict[str, dict[str, dict[str, list[Path]]]] = {
    cls: defaultdict(lambda: defaultdict(list)) for cls in CLASSES
}

excluded = 0

for cls in CLASSES:
    cls_dir = SOURCE_ROOT / cls
    if not cls_dir.exists():
        print(f"[INFO] Directory not found, skipping: {cls_dir}")
        continue

    for img_path in sorted(cls_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        stem    = img_path.stem
        acronym = extract_acronym(stem)
        base    = original_stem(stem)

        if cls == "papilledema" and acronym in PAPILLEDEMA_EXCLUDE:
            excluded += 1
            continue

        paths_by_class_acronym[cls][acronym][base].append(img_path)

print(f"Excluded {excluded} papilledema files with acronyms {PAPILLEDEMA_EXCLUDE}\n")


# ---- Stratified split — per class, per acronym bucket ----

splits: dict[str, dict[str, list[Path]]] = {
    "train": defaultdict(list),
    "val":   defaultdict(list),
    "test":  defaultdict(list),
}

print(f"{'Class':<25} {'Acronym':<10} {'Orig':>5} {'Train':>6} {'Val':>5} {'Test':>5}")
print("-" * 60)

for cls in CLASSES:
    acronym_buckets = paths_by_class_acronym[cls]
    if not acronym_buckets:
        continue

    for acronym, base_dict in sorted(acronym_buckets.items()):
        originals = list(base_dict.keys())
        random.shuffle(originals)
        n = len(originals)
        n_train, n_val, n_test = floor_split(n, SPLIT)

        train_bases = originals[:n_train]
        val_bases   = originals[n_train : n_train + n_val]
        test_bases  = originals[n_train + n_val :]

        # Train: all variants (original + augmented if HAS_AUGMENTATION)
        for base in train_bases:
            splits["train"][cls].extend(base_dict[base])

        # Val / test: originals only (augmented stripped if HAS_AUGMENTATION,
        # otherwise every file is treated as an original)
        for base in val_bases:
            for p in base_dict[base]:
                if not is_augmented(p.stem):
                    splits["val"][cls].append(p)

        for base in test_bases:
            for p in base_dict[base]:
                if not is_augmented(p.stem):
                    splits["test"][cls].append(p)

        print(f"{cls:<25} {acronym:<10} {n:>5} {n_train:>6} {n_val:>5} {n_test:>5}")

print("-" * 60)
for split_name in ("train", "val", "test"):
    total = sum(len(v) for v in splits[split_name].values())
    print(f"  {split_name}: {total} files")
print()


# ---- Copy files ----

for split_name in ("train", "val", "test"):
    for cls in CLASSES:
        (DEST_ROOT / split_name / cls).mkdir(parents=True, exist_ok=True)

for split_name, class_paths in splits.items():
    for cls, paths in class_paths.items():
        for src in paths:
            dst = DEST_ROOT / split_name / cls / src.name
            shutil.copy2(src, dst)

print("Split complete.")
print(f"Output: {DEST_ROOT}")