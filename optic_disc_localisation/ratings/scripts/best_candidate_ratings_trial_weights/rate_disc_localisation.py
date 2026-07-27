"""Interactive OpenCV labeling tool: rate disc-localisation quality (1-4) and vessel-ok
(0/9) per image via keypresses, resuming from START_CLASS/START_NUM. """

import os
from pathlib import Path

import cv2
import pandas as pd

name = "disc_localisation_030226"

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
ROOT = Path(DATASETS_ROOT) / "Processed Datasets" / "disc_localisation"

SOURCE_ROOT = ROOT / name

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

START_CLASS = "pseudopapilledema"  
START_NUM = 172
last_saved = None

def collect_images():
    items = []
    for cls in CLASSES:
        cls_dir = SOURCE_ROOT / cls
        paths = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        for i, p in enumerate(paths, start=1):
            items.append({
                "class": cls,
                "path": p,
                "name": p.name,
                "class_index": i,   # 1-based within class
            })
    return items

items = collect_images()

start_global = None
for gi, it in enumerate(items):
    if it["class"] == START_CLASS and it["class_index"] == START_NUM:
        start_global = gi
        break

if start_global is None:
    raise ValueError(f"Could not find {START_CLASS} image #{START_NUM}. Check START_NUM and sorting.")

# Load existing ratings if present
out_dir = Path(__file__).resolve().parents[2] / "data" / "best_candidate_ratings_trial_weights"
out_file = out_dir / f"{name}_manual.csv"

rated = {}  # (class, name) -> {"rating": int, "vessel": int}
if out_file.exists():
    df_prev = pd.read_csv(out_file)
    for _, r in df_prev.iterrows():
        rated[(r["class"], r["image"])] = {
            "rating": int(r["rating"]),
            "vessel": int(r["vessel_ok"])
        }

# Main loop: same image, two-step input (rating then vessel)
i = start_global
while 0 <= i < len(items):
    it = items[i]
    cls, img_path, img_name = it["class"], it["path"], it["name"]

    img = cv2.imread(str(img_path))
    if img is None:
        i += 1
        continue

    entry = rated.get((cls, img_name), {})
    current_rating = entry.get("rating", None)
    current_vessel = entry.get("vessel", None)

    cv2.imshow(img_name, img)

    if current_rating is None:
        step = "rating"
    elif current_vessel is None:
        step = "vessel"
    else:
        step = "rating"
    while True:
        title = (
            f"{cls} [{it['class_index']}] | {img_name} | "
            f"rating={current_rating} | vessel={current_vessel} | "
            f"ENTER {step.upper()} | global={i+1}/{len(items)}"
        )
        cv2.setWindowTitle(img_name, title)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            i = len(items)  # exit outer loop
            break

        # navigation (allowed anytime)
        if key == ord("b"):
            cv2.destroyAllWindows()
            i = max(i - 1, 0)
            break
        if key == ord("n"):
            cv2.destroyAllWindows()
            i += 1
            break

        # ensure dict exists
        if (cls, img_name) not in rated:
            rated[(cls, img_name)] = {}

        if step == "rating":
            if key in map(ord, ["1", "2", "3", "4"]):
                current_rating = int(chr(key))
                rated[(cls, img_name)]["rating"] = current_rating
                step = "vessel"  # now collect vessel on SAME image
            else:
                # ignore unknown keys during rating step
                pass

        elif step == "vessel":
            if key in map(ord, ["0", "9"]):
                current_vessel = int(chr(key))
                rated[(cls, img_name)]["vessel"] = current_vessel

                # both labels captured -> record last_saved and move on
                last_saved = {
                    "image": img_name,
                    "class": cls,
                    "rating": current_rating,
                    "vessel_ok": current_vessel,
                    "class_index": it["class_index"],
                    "global_index": i + 1
                }
                cv2.destroyAllWindows()
                i += 1
                break
            else:
                # ignore unknown keys during vessel step
                pass

    # outer loop continues

# Save to csv
rows = []
for it in items:
    k = (it["class"], it["name"])
    if k in rated:
        rows.append({
            "image": it["name"],
            "class": it["class"],
            "rating": rated[k].get("rating", None),
            "vessel_ok": rated[k].get("vessel", None),
            "class_index": it["class_index"]
        })

df = pd.DataFrame(rows)
df.to_csv(out_file, index=False)

print("Saved ratings to", out_file)

if last_saved is not None:
    print(
        f"Last saved rating:\n"
        f"  class        : {last_saved['class']}\n"
        f"  image        : {last_saved['image']}\n"
        f"  rating       : {last_saved['rating']}\n"
        f"  vessel       : {last_saved['vessel_ok']}\n"
        f"  class index  : {last_saved['class_index']}\n"
        f"  global index : {last_saved['global_index']}"
    )
else:
    print("No new ratings saved in this session.")
