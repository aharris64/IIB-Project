from pathlib import Path
import re
import csv

import cv2
import pandas as pd

# -------------------- CONFIG --------------------
name = "disc_candidates_050226"

ROOT = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Processed Datasets")
SOURCE_ROOT = ROOT / name

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

START_ROW = 2286

RATING_KEYS = {"1", "2", "3", "4"}

cand_regex = re.compile(r"^(?P<stem>.+?)_cand(?P<cand>\d+)(?:_vb[-\d.]+)?$", re.IGNORECASE)

out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)
simple_file = out_dir / f"{name}_candidates_.csv"


def parse_candidate_name(p: Path):
    """
    Expected filenames like:
      image_0123_cand01.png
      image_0123_cand01_vb0.532.png
    Returns: (base_image_filename_with_ext, cand_index)
    """
    m = cand_regex.match(p.stem)
    if m:
        return m.group("stem") + p.suffix.lower(), int(m.group("cand"))
    return p.name, 1

def collect_candidate_images():
    items = []
    for cls in CLASSES:
        cls_dir = SOURCE_ROOT / cls
        if not cls_dir.exists():
            continue

        paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        for p in paths:
            base_image, cand_idx = parse_candidate_name(p)
            items.append({
                "class": cls,
                "path": p,
                "candidate_image": p.name,
                "base_image": base_image,
                "candidate_index": cand_idx,
            })

    items.sort(key=lambda d: (d["class"], d["base_image"], d["candidate_index"], d["candidate_image"]))
    return items

items = collect_candidate_images()
if len(items) == 0:
    raise ValueError(f"No candidate images found under {SOURCE_ROOT}.")

# Load existing csv
rated = {}  # (image_no_ext, candidate_num) -> rating
if simple_file.exists():
    df_prev = pd.read_csv(simple_file)
    if {"image", "candidate_num", "rating"}.issubset(df_prev.columns):
        for _, r in df_prev.iterrows():
            if pd.isna(r["rating"]):
                continue
            rated[(str(r["image"]), int(r["candidate_num"]))] = int(r["rating"])


i = max(START_ROW - 1, 0)
last_saved = None

while 0 <= i < len(items):
    it = items[i]
    cls = it["class"]
    img_path = it["path"]
    cand_name = it["candidate_image"]
    cand_idx = it["candidate_index"]

    image_no_ext = Path(it["base_image"]).stem
    rating_key = (image_no_ext, cand_idx)

    img = cv2.imread(str(img_path))
    if img is None:
        i += 1
        continue

    current_rating = rated.get(rating_key, None)

    cv2.imshow(cand_name, img)

    while True:
        title = (
            f"{cls} | image={image_no_ext} | cand={cand_idx} | file={cand_name} | "
            f"rating={current_rating} | "
            f"[1-4]=rate  n=next  b=back  ESC=quit | {i+1}/{len(items)}"
        )
        cv2.setWindowTitle(cand_name, title)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            i = len(items)
            break

        if key == ord("b"):
            cv2.destroyAllWindows()
            i = max(i - 1, 0)
            break

        if key == ord("n"):
            cv2.destroyAllWindows()
            i += 1
            break

        ch = chr(key) if 0 <= key < 256 else ""
        if ch in RATING_KEYS:
            current_rating = int(ch)
            rated[rating_key] = current_rating

            last_saved = {
                "image": image_no_ext,
                "candidate_num": cand_idx,
                "rating": current_rating,
                "index": i + 1,  # global position in items
            }

            cv2.destroyAllWindows()
            i += 1
            break

# Save csv
rows = []
for idx0, it in enumerate(items, start=1):
    image_no_ext = Path(it["base_image"]).stem
    cand_idx = it["candidate_index"]
    r = rated.get((image_no_ext, cand_idx), None)
    if r is None:
        continue
    rows.append({
        "image": image_no_ext,
        "candidate_num": cand_idx,
        "rating": int(r),
        "index": idx0,
    })

with open(simple_file, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image", "candidate_num", "rating", "index"])
    w.writeheader()
    w.writerows(rows)

print("Saved:", simple_file)

if last_saved is not None:
    print(
        "Last saved rating:\n"
        f"  image         : {last_saved['image']}\n"
        f"  candidate_num : {last_saved['candidate_num']}\n"
        f"  rating        : {last_saved['rating']}\n"
        f"  index         : {last_saved['index']}"
    )
else:
    print("No new ratings saved in this session.")
