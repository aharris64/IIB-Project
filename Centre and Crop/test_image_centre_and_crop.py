from PIL import Image
import cv2
import numpy as np

path = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\Centre and Crop"
name  = "normal_0001_EDD.jpg"

with Image.open(path + "\\" + name) as img:
    img = img.convert("RGB")
    
    # Resize to shorter side 512 - to make image analysis faster
    target_size = 512
    w, h = img.size

    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    resize_img = img.resize((new_w, new_h), Image.BICUBIC) # Uses Bicubic

    resize_img.save(path + "\\resized.jpg")


    # Extract red channel
    red_img = np.array(resize_img)[:, :, 0].astype(np.float32) / 255.0
    cv2.imwrite(path + "\\red.png", (red_img * 255).astype(np.uint8))

    
        
    # # Mask to remove zero values (black outer edge)
    # THRESHOLD_VALUE = 5/255
    # border_mask = (red_img > THRESHOLD_VALUE).astype(np.uint8)
    # coverage = border_mask.mean()

    # if coverage > 0.001:
    #     # Clean small gaps/noise
    #     k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    #     border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, k, iterations=1)

    #     # Keep largest component (retina FOV)
    #     num, labels, stats, _ = cv2.connectedComponentsWithStats(border_mask, connectivity=8)
    #     if num > 1:
    #         largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #         border_mask = (labels == largest).astype(np.uint8)
    # else:
    #     # if the amount removed is small assume there is no border
    #     border_mask[:] = 1

    # cv2.imwrite(path + "\\border_mask.png", border_mask * 255)

    # # Replace outer border with  median pixels to not interfere with optic disc detection
    # border_mask_f = border_mask.astype(bool)

    # # Fill value from real retina pixels (robust)
    # fill_value = float(np.median(red_img[border_mask_f])) if np.any(border_mask_f) else 0.0
    # red_filled = red_img.copy()
    # red_filled[~border_mask_f] = fill_value

    # cv2.imwrite(path + "\\red_filled.png", (red_filled * 255).astype(np.uint8))

    # # Blur (large sigma)
    # large_s_img = cv2.GaussianBlur(red_filled, (0,0), sigmaX=25)
    # cv2.imwrite(path + "\\large_s.png", (large_s_img * 255).astype(np.uint8))

    # # Background subtraction + clip (top-hat style)
    # subtraction_clip = np.clip(red_img - large_s_img, 0, 1)
    # subtraction_clip[~border_mask_f] = 0.0
    # cv2.imwrite(path + "\\subtraction_clip.png", (subtraction_clip * 255).astype(np.uint8))

    # # Blur (large sigma)
    # small_s_img = cv2.GaussianBlur(subtraction_clip, (0,0), sigmaX=2)
    # cv2.imwrite(path + "\\small_s.png", (small_s_img * 255).astype(np.uint8))