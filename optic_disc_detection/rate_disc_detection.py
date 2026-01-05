from pathlib import Path

import cv2
import pandas as pd

SOURCE_ROOT = Path(r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Dataset\disc_detection")

CLASSES = ["normal", "papilledema", "pseudopapilledema"]
IMAGE_EXTS = {".jpg", ".png"}

START_CLASS = "normal"  
START_NUM = 602
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
out_file = Path(__file__).resolve().parent / "ratings.csv"
rated = {}  # key: (class, name) -> rating
if out_file.exists():
    df_prev = pd.read_csv(out_file)
    for _, r in df_prev.iterrows():
        rated[(r["class"], r["image"])] = int(r["rating"])

# Main loop with back support
i = start_global
while 0 <= i < len(items):
    it = items[i]
    cls, img_path, img_name = it["class"], it["path"], it["name"]

    img = cv2.imread(str(img_path))
    if img is None:
        i += 1
        continue

    current_rating = rated.get((cls, img_name), None)
    title = f"{cls} [{it['class_index']}] | {img_name} | rating={current_rating} | global={i+1}/{len(items)}"
    cv2.imshow(img_name, img)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:  # ESC
        break
    elif key in map(ord, ["1", "2", "3", "4"]):
        rating = int(chr(key))
        rated[(cls, img_name)] = rating
        last_saved = {
            "image": img_name,
            "class": cls,
            "rating": rating,
            "class_index": it["class_index"],
            "global_index": i + 1
        }
        i += 1
    elif key == ord("n"):  # skip forward
        i += 1
    elif key == ord("b"):  # back one image
        i = max(i - 1, 0)
    else:
        # ignore unknown keys, stay on same image
        pass

# Save
rows = []
for it in items:
    k = (it["class"], it["name"])
    if k in rated:
        rows.append({
            "image": it["name"],
            "class": it["class"],
            "rating": rated[k],
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
        f"  class index  : {last_saved['class_index']}\n"
        f"  global index : {last_saved['global_index']}"
    )
else:
    print("No new ratings saved in this session.")
