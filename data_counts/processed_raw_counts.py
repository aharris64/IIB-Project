"""Prints a resolution breakdown per label x diagnostic-acronym class, plus a summary
table and grand total, for the Processed Datasets/raw pipeline stage (post initial
processing, pre disc-centring) — filenames are matched against PATTERN to extract the
trailing 3-letter acronym (EDD, IFD, RFM, PPE, WHC)."""

import os
import re
from collections import defaultdict
from PIL import Image

# ---- Configuration ----
DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
ROOT_FOLDER = os.path.join(DATASETS_ROOT, "Processed Datasets", "raw")
LABELS      = {"normal", "papilledema", "pseudopapilledema"}
CLASSES     = {"EDD", "IFD", "RFM", "PPE", "WHC"}
PATTERN     = re.compile(r"^.+_\d{4}_([A-Z]{3})\.", re.IGNORECASE)

 
# { label: { class: { (width, height): count } } }
data    = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
skipped = []
 
for label in sorted(LABELS):
    label_path = os.path.join(ROOT_FOLDER, label)
    if not os.path.isdir(label_path):
        continue
 
    for filename in os.listdir(label_path):
        match = PATTERN.match(filename)
        if not match:
            continue
 
        cls = match.group(1).upper()
        if cls not in CLASSES:
            continue
 
        filepath = os.path.join(label_path, filename)
        try:
            with Image.open(filepath) as img:
                resolution = img.size      # (width, height)
            data[label][cls][resolution] += 1
        except Exception as e:
            skipped.append((label, filename, str(e)))
 
# ---- Print results ----
SEP_MAJOR = "=" * 60
SEP_MINOR = "-" * 40
 
for label in sorted(LABELS):
    print(SEP_MAJOR)
    label_data = data.get(label, {})
    label_total = sum(sum(res.values()) for res in label_data.values())
    print(f"LABEL: {label.upper()}   ({label_total} total images)")
    print(SEP_MAJOR)
 
    if not label_data:
        print("  No matching images found.\n")
        continue
 
    for cls in sorted(CLASSES):
        res_counts = label_data.get(cls)
        if not res_counts:
            print(f"\n  [{cls}]  — not present")
            continue
 
        cls_total = sum(res_counts.values())
        print(f"\n  [{cls}]  ({cls_total} image{'s' if cls_total != 1 else ''})")
        for res, count in sorted(res_counts.items()):
            print(f"    {res[0]:>6} x {res[1]:<6}  →  {count:>4} image{'s' if count != 1 else ''}")
 
    print()
 
# ---- Summary table ----
print(SEP_MAJOR)
print("SUMMARY  (image count per label × class)")
print(SEP_MAJOR)
 
col_w = 22
header = f"{'Label':<{col_w}}" + "".join(f"{cls:>8}" for cls in sorted(CLASSES)) + f"{'TOTAL':>8}"
print(header)
print("-" * len(header))
 
for label in sorted(LABELS):
    label_data = data.get(label, {})
    row = f"{label:<{col_w}}"
    row_total = 0
    for cls in sorted(CLASSES):
        count = sum(label_data.get(cls, {}).values())
        row_total += count
        row += f"{count:>8}"
    row += f"{row_total:>8}"
    print(row)
 
# Column totals
print("-" * len(header))
totals_row = f"{'TOTAL':<{col_w}}"
grand_total = 0
for cls in sorted(CLASSES):
    col_total = sum(
        sum(data[lbl].get(cls, {}).values())
        for lbl in LABELS
    )
    grand_total += col_total
    totals_row += f"{col_total:>8}"
totals_row += f"{grand_total:>8}"
print(totals_row)
 
# ---- Skipped files ----
if skipped:
    print(f"\n⚠  Skipped {len(skipped)} file(s) (could not be opened):")
    for label, fname, reason in skipped:
        print(f"  [{label}] {fname}: {reason}")