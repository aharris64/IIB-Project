"""Prints a per-class and total breakdown of image dimensions in the raw dataset."""

from PIL import Image
import os
from collections import Counter

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
normal = os.path.join(DATASETS_ROOT, "Combined Dataset", "Normal")
pseudopapilloedema = os.path.join(DATASETS_ROOT, "Combined Dataset", "Pseudopapilledema")
papilloedema = os.path.join(DATASETS_ROOT, "Combined Dataset", "Papilledema")

def count_sizes(folder):

    size_counts = Counter()

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            with Image.open(path) as img:
                size_counts[img.size] += 1  # (width, height)
        except Exception:
            pass  # skip non-images or corrupted files

    return size_counts


normal_size_counts = count_sizes(normal)
print("Normal:")
for (w, h), count in normal_size_counts.items():
    print(f"{count} {w}x{h}")

pseudo_size_counts = count_sizes(pseudopapilloedema)
print("Pseudopapilledema:")
for (w, h), count in pseudo_size_counts.items():
    print(f"{count} {w}x{h}")

papilledema_size_counts = count_sizes(papilloedema)
print("papilledema:")
for (w, h), count in papilledema_size_counts.items():
    print(f"{count} {w}x{h}")

total_size_counts = (normal_size_counts + pseudo_size_counts + papilledema_size_counts)
print("Total:")
for (w, h), count in total_size_counts.items():
    print(f"{count} {w}x{h}")