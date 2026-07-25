import os
import cv2
import numpy as np
import random

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "./Datasets")
image_address = os.path.join(DATASETS_ROOT, "Combined Dataset", "Normal", "IFD 1ffa962e-8d87-11e8-9daf-6045cb817f5b..JPG")

def gaussian_subtraction():
    # Load image
    image = cv2.imread(image_address)

    # Check if loaded
    if image is None:
        raise ValueError("Image not found. Check the file path.")
    else:
        print("Image found")

    cv2.imshow('window_name', cv2.resize(image, (0,0), fx=0.25, fy=0.25))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Apply to 
    blur_small = cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    blur_large = cv2.GaussianBlur(image, (21, 21), sigmaX=3)

    # Difference of Gaussians
    dog = cv2.subtract(blur_small, blur_large)
    dog_boost = dog * 5    # try 3, 5, 10 etc.
    dog_norm  = cv2.normalize(dog_boost, None, 0, 255, cv2.NORM_MINMAX)
    dog_norm  = dog_norm.astype("uint8")
    dog_visible = cv2.convertScaleAbs(dog_norm, alpha=5, beta=0)  # alpha = contrast
    cv2.imshow("DoG visible", cv2.resize(dog_visible, (0,0), fx=0.25, fy=0.25))
    # cv2.imshow('window_name', cv2.resize(dog_norm, (0,0), fx=0.25, fy=0.25))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate(img, angle=15):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    x =  cv2.warpAffine(img, M, (w, h))
    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return 

def hflip(img):
    x = cv2.flip(img, 1)
    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adjust_brightness_contrast(img, alpha=1.2, beta=20):
    # alpha: contrast, beta: brightness
    x = cv2.convertScaleAbs(img, alpha=alpha, beta=beta) 
    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def random_crop(img, crop_ratio=0.9):
    h, w = img.shape[:2]
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y = random.randint(0, h - new_h)
    x = random.randint(0, w - new_w)
    crop = img[y:y+new_h, x:x+new_w]
    x = cv2.resize(crop, (w, h))
    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_gaussian_noise(img, mean=0, std=15):
    noise = np.random.normal(mean, std, img.shape)
    noisy_img = img + noise
    x = np.clip(noisy_img, 0, 255).astype(np.uint8)
    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def apply_gamma(img, gamma=1.2):
    img_float = img.astype(np.float32) / 255.0
    
    gamma_corrected = img_float ** gamma
    
    x =  (gamma_corrected * 255).astype(np.uint8)

    cv2.imshow("DoG visible", x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_address_2 = os.path.join(DATASETS_ROOT, "Combined Dataset", "Normal", "PPE 265.jpg")
image = cv2.imread(image_address_2)
apply_gamma(image)