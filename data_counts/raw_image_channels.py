from PIL import Image
import os
from collections import Counter

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
normal = os.path.join(DATASETS_ROOT, "Combined Dataset", "Normal")
pseudopapilloedema = os.path.join(DATASETS_ROOT, "Combined Dataset", "Pseudopapilledema")
papilloedema = os.path.join(DATASETS_ROOT, "Combined Dataset", "Papilledema")

def count_channels(folder):
    channel_counts = Counter()

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            with Image.open(path) as img:
                c = len(img.getbands())
                channel_counts[c] += 1
        except Exception:
            pass

    return channel_counts

normal_channel_counts = count_channels(normal)
print("Normal:")
for channels, count in normal_channel_counts.items():
    print(f"{count} images with {channels} channels")

pseudo_channel_counts = count_channels(pseudopapilloedema)
print("Pseudopapilledema:")
for channels, count in pseudo_channel_counts.items():
    print(f"{count} images with {channels} channels")

pailledema_channel_counts = count_channels(papilloedema)
print("papilledema:")
for channels, count in pailledema_channel_counts.items():
    print(f"{count} images with {channels} channels")

total_channel_counts = (normal_channel_counts + pseudo_channel_counts + pailledema_channel_counts)
print("Total:")
for channels, count in total_channel_counts.items():
    print(f"{count} images with {channels} channels")

