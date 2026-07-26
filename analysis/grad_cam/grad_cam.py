"""Grad-CAM visualisation (via the pytorch-grad-cam library, `pip install grad-cam`):
runs Grad-CAM over one image or a sample from DATA_DIR, and saves one overlay PNG per
image directly into OUT_DIR, mirroring DATA_DIR's per-class subfolder layout. Edit the
CONFIG section below, then run directly."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, XGradCAM, EigenGradCAM, HiResCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).parents[2]))
from classifier.models import build_model

# ---- Constants ----
CLASS_NAMES = {0: "normal", 1: "papilledema", 2: "pseudopapilledema"}
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

CAM_METHODS = {
    "GradCAM":         GradCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "XGradCAM":        XGradCAM,
    "EigenGradCAM":    EigenGradCAM,
    "HiResCAM":        HiResCAM,
}


def get_target_layers(model, model_name: str) -> list:
    """Return the single best target layer for model_name — pytorch-grad-cam expects
    a list of layers; this is the last convolutional block before global pooling,
    the standard Grad-CAM choice."""
    if model_name in ("mobilenet_v3", "mobilenet_v3_small"):
        return [model.features[-1]]
    elif model_name == "mobilenet_v2":
        return [model.features[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    elif model_name == "resnet":
        return [model.layer4[-1]]
    elif model_name == "squeezenet":
        return [model.features[-1]]
    elif model_name in ("efficientnet_lite0", "efficientnet_lite1", "ghostnet"):
        return [model.blocks[-1]]
    else:
        raise ValueError(f"No target layer defined for '{model_name}'.")


def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    """Normalised tensor → float32 RGB array in [0, 1] (required by show_cam_on_image)."""
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1).astype(np.float32)


def predict(model, input_tensor: torch.Tensor, device):
    """Run model on input_tensor; return (predicted class index, its softmax probability)."""
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs  = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        pred_prob  = float(probs[0, pred_class].item())
    return pred_class, pred_prob


def collect_images(data_dir: Path, n: int, class_filter: int, seed: int):
    """Walk an ImageFolder-style directory -> list of (path, label, class_name) tuples,
    label being the class's index in sorted subfolder order (matches CLASS_NAMES)."""
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    label_map  = {d.name: i for i, d in enumerate(class_dirs)}
    print("Label map:", label_map)

    items = []
    for d in class_dirs:
        label = label_map[d.name]
        if class_filter is not None and label != class_filter:
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"):
            for fp in d.glob(ext):
                items.append((fp, label, d.name))

    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n]


# ---- Config ----

# Model
MODEL_NAME  = "mobilenet_v3_small"   # must match a key in build_model()
NUM_CLASSES = 3
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT     = REPO_ROOT / "outputs" / "runs"
MODEL_RUN = "mobilenet_v3_small_20260530_061821"
WEIGHTS = Path(MODEL_ROOT) / MODEL_RUN / "best_model_state_dict.pt"

# Input — set IMAGE to a single file path, or DATA_DIR to a test folder.
# Leave IMAGE as None to sample from DATA_DIR.
IMAGE    = None
DATA_DIR = REPO_ROOT / "analysis" / "grad_cam" / "test_images"       # ImageFolder layout

N            = 100       # images to sample from DATA_DIR (ignored if IMAGE is set)
CLASS_FILTER = None  # int: only show images from this true class; None = all classes
                       #   0 = normal | 1 = papilledema | 2 = pseudopapilledema
TARGET_CLASS = None    # int: force heatmap to explain this class; None = predicted class

# CAM method — one of: GradCAM | GradCAMPlusPlus | XGradCAM | EigenGradCAM | HiResCAM
METHOD = "HiResCAM"

# Output — one overlay PNG per image, saved under OUT_DIR/<class_name>/ (mirroring
# DATA_DIR's layout); with IMAGE set instead of DATA_DIR, there's no class subfolder
# to mirror, so the overlay is saved directly under OUT_DIR.
OUT_DIR = REPO_ROOT / "analysis" / "grad_cam" / "grad_cam_results"
SEED    = 42


def main():
    """Load the model + weights configured above, run Grad-CAM over the configured
    image(s), and save one overlay PNG per image directly to OUT_DIR (no figure,
    nothing displayed on screen)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Method : {METHOD}")

    # ---- Load model ----
    model = build_model(MODEL_NAME, NUM_CLASSES, freeze="none")
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded : {WEIGHTS}")

    # ---- Target layers ----
    target_layers = get_target_layers(model, MODEL_NAME)
    print(f"Target layer : {target_layers[0].__class__.__name__}")

    # ---- Collect images ----
    if IMAGE is not None:
        items = [(Path(IMAGE), None, None)]
    else:
        items = collect_images(Path(DATA_DIR), N, CLASS_FILTER, SEED)
    print(f"Processing {len(items)} image(s) …\n")

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run CAM, saving each overlay as it's produced ----
    CamClass = CAM_METHODS[METHOD]

    with CamClass(model=model, target_layers=target_layers) as cam:
        for img_path, label, class_name in items:
            img_pil  = Image.open(img_path).convert("RGB")
            input_t  = TRANSFORM(img_pil).unsqueeze(0).to(device)

            pred_class, pred_prob = predict(model, input_t, device)

            cam_target = TARGET_CLASS if TARGET_CLASS is not None else pred_class
            targets    = [ClassifierOutputTarget(cam_target)]

            grayscale_cam = cam(input_tensor=input_t, targets=targets)[0]
            original      = unnormalise(input_t)
            overlay_img   = show_cam_on_image(original, grayscale_cam, use_rgb=True)

            dst_dir = out_dir / class_name if class_name is not None else out_dir
            dst_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(dst_dir / f"{img_path.stem}_gradcam.png"),
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
            )

            true_name = CLASS_NAMES.get(label, "?") if label is not None else "?"
            pred_name = CLASS_NAMES.get(pred_class)
            tick      = "✓" if label == pred_class else ("✗" if label is not None else "")
            print(f"  {tick}  {img_path.name:<40}  "
                  f"true={true_name:<20} pred={pred_name:<20} ({pred_prob:.1%})")

    print(f"\nDone. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
