
import os
from collections import Counter

normal = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Combined Dataset\Normal"
pseudopapilloedema = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Combined Dataset\Pseudopapilledema"
papilloedema = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\Combined Dataset\Papilledema"


def count_image_types(folder):
    type_counts = Counter()

    for filename in os.listdir(folder):
        ext = os.path.splitext(filename)[1].lower()  # '.jpg', '.png', etc.
        if ext:  # skip files without extension
            type_counts[ext] += 1

    return type_counts

normal_type_counts = count_image_types(normal)
print("Normal:")
for ext, count in normal_type_counts.items():
    print(f"{count} {ext}")

pseudo_type_counts = count_image_types(pseudopapilloedema)
print("Pseudopapilledema:")
for ext, count in pseudo_type_counts.items():
    print(f"{count} {ext}")

papillo_type_counts = count_image_types(papilloedema)
print("Papilledema:")
for ext, count in papillo_type_counts.items():
    print(f"{count} {ext}")

total_type_counts = (normal_type_counts + pseudo_type_counts + papillo_type_counts)
print("Total:")
for ext, count in total_type_counts.items():
    print(f"{count} {ext}")