import os
from pathlib import Path
from PIL import Image

from disc_detection import detect_disc
from initial_processing import resize


current_path = Path(__file__).resolve().parent
test_images = os.path.join(current_path, "test_images")
results_folder = os.path.join(current_path, "results")

TARGET_SIZE = 512

def one_image(name):
    test_image = os.path.join(test_images, name)

    with Image.open(test_image) as img:
        img = img.convert("RGB")
        resize_img = resize(img, TARGET_SIZE)

    detect_disc(resize_img, None)



def all_images(use_weights, sigma_att, sigma_blur):

    for file in os.listdir(test_images):
        test_image = os.path.join(test_images, file)

        with Image.open(test_image) as img:
            img = img.convert("RGB")
            resize_img = resize(img, TARGET_SIZE)

        save_name = Path(test_image).stem + "_W" + str(use_weights) + "_sa" + str(sigma_att) + "_sb" + str(sigma_blur) + ".png"
        print(save_name)
        detect_disc(resize_img, use_weights, sigma_att, sigma_blur, save_name)


all_images(False, 200, 15)

# one_image("papilledema_0483_RFM.png")

# one_image("pseudopapilledema_0009_PPE.jpg")
# one_image("pseudopapilledema_0011_PPE.jpg")
# one_image("pseudopapilledema_0015_PPE.jpg")
# one_image("pseudopapilledema_0029_PPE.jpg")
# one_image("pseudopapilledema_0202_PPE.jpg")
# one_image("pseudopapilledema_0296_WHC.jpg")
# one_image("pseudopapilledema_0317_WHC.jpg")
# one_image("pseudopapilledema_0320_WHC.jpg")
# one_image("pseudopapilledema_0321_WHC.jpg")
# one_image("pseudopapilledema_0361_WHC.jpg")