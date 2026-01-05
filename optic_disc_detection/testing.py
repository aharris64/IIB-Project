import os
from pathlib import Path
from PIL import Image

from disc_detection import detect_disc
from initial_processing import resize
from save_results import save_image, save_candidate_overlay

# Test Images
# name  = "normal_0001_EDD.jpg"
# name = "PPE 94.jpg"
# name = "EDD Disc Edema32.jpg"
# name = "papilledema_0223_PPE.jpg"
# name = "papilledema_0459_RFM.png"
# name = "papilledema_0136_IFD.jpg"

current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
results_folder = os.path.join(current_path, "results")

TARGET_SIZE = 512

for file in os.listdir(test_images):
    test_image = os.path.join(test_images, file)

    with Image.open(test_image) as img:
        img = img.convert("RGB")
        resize_img = resize(img, TARGET_SIZE)

    best = detect_disc(resize_img)

    save_candidate_overlay(resize_img, [best], results_folder, Path(test_image).name)

