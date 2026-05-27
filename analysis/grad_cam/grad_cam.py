"""
gradcam.py  —  Grad-CAM visualisation using pytorch-grad-cam library

Install:  pip install grad-cam

Edit the CONFIG section below, then run:  python gradcam.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# pytorch-grad-cam
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, XGradCAM, EigenGradCAM, HiResCAM
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── project models ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parents[2]))
from cnn.models import build_model

# ── constants ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Target-layer registry
# Last convolutional block before global pooling — standard Grad-CAM choice.
# ─────────────────────────────────────────────────────────────────────────────
def get_target_layers(model, model_name: str) -> list:
    """
    pytorch-grad-cam expects a list of layers.
    Returns the single best layer for each architecture.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def unnormalise(tensor: torch.Tensor) -> np.ndarray:
    """Normalised tensor → float32 RGB array in [0, 1] (required by show_cam_on_image)."""
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1).astype(np.float32)


def predict(model, input_tensor: torch.Tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs  = F.softmax(logits, dim=1)
        pred_class = int(logits.argmax(dim=1).item())
        pred_prob  = float(probs[0, pred_class].item())
    return pred_class, pred_prob


def collect_images(data_dir: Path, n: int, class_filter: int, seed: int):
    """Walk an ImageFolder-style directory → list of (Path, label) tuples."""
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
                items.append((fp, label))

    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Figure builder
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(results: list, method_name: str, save_path: Path = None):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Grad-CAM  ({method_name})", fontsize=12, fontweight="bold")

    for i, r in enumerate(results):
        pred_name = CLASS_NAMES.get(r["pred_class"], str(r["pred_class"]))
        true_name = (CLASS_NAMES.get(r["true_label"], str(r["true_label"]))
                     if r["true_label"] is not None else "?")
        correct   = (r["true_label"] == r["pred_class"]
                     if r["true_label"] is not None else None)
        tick      = "✓" if correct else ("✗" if correct is False else "")
        colour    = "green" if correct else ("red" if correct is False else "black")

        axes[i][0].imshow(r["original"])
        axes[i][0].set_title(f"{r['path'].name}\nTrue: {true_name}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(r["overlay"])
        axes[i][1].set_title(
            f"{tick}  Pred: {pred_name}  ({r['pred_prob']:.1%})",
            fontsize=8, color=colour
        )
        axes[i][1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    plt.show()


# =============================================================================
# CONFIG  —  edit this section
# =============================================================================

# Model
MODEL_NAME  = "mobilenet_v3_small"   # must match a key in build_model()
NUM_CLASSES = 3
MODEL_ROOT     = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\runs"
MODEL_RUN = "mobilenet_v3_small_20260520_141418"
WEIGHTS = Path(MODEL_ROOT) / MODEL_RUN / "best_model_state_dict.pt"

# Input — set IMAGE to a single file path, or DATA_DIR to a test folder.
# Leave IMAGE as None to sample from DATA_DIR.
IMAGE    = None
DATA_DIR = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\analysis\grad_cam\test_images"       # ImageFolder layout

N            = 100       # images to sample from DATA_DIR (ignored if IMAGE is set)
CLASS_FILTER = 1   # int: only show images from this true class; None = all classes
                       #   0 = normal | 1 = papilledema | 2 = pseudopapilledema
TARGET_CLASS = None    # int: force heatmap to explain this class; None = predicted class

# CAM method — one of: GradCAM | GradCAMPlusPlus | XGradCAM | EigenGradCAM | HiResCAM
METHOD = "HiResCAM"

# Output
OUT_DIR = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Code\IIB-Project\analysis\grad_cam\grad_cam_results"
SEED    = 42

# =============================================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Method : {METHOD}")

    # ── load model ────────────────────────────────────────────────────────────
    model = build_model(MODEL_NAME, NUM_CLASSES, freeze="none")
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded : {WEIGHTS}")

    # ── target layers ─────────────────────────────────────────────────────────
    target_layers = get_target_layers(model, MODEL_NAME)
    print(f"Target layer : {target_layers[0].__class__.__name__}")

    # ── collect images ────────────────────────────────────────────────────────
    if IMAGE is not None:
        items = [(Path(IMAGE), None)]
    else:
        items = collect_images(Path(DATA_DIR), N, CLASS_FILTER, SEED)
    print(f"Processing {len(items)} image(s) …\n")

    # ── run CAM ───────────────────────────────────────────────────────────────
    CamClass = CAM_METHODS[METHOD]

    results = []
    with CamClass(model=model, target_layers=target_layers) as cam:
        for img_path, label in items:
            img_pil  = Image.open(img_path).convert("RGB")
            input_t  = TRANSFORM(img_pil).unsqueeze(0).to(device)
            original = unnormalise(input_t)

            pred_class, pred_prob = predict(model, input_t, device)

            cam_target = TARGET_CLASS if TARGET_CLASS is not None else pred_class
            targets    = [ClassifierOutputTarget(cam_target)]

            grayscale_cam = cam(input_tensor=input_t, targets=targets)[0]
            overlay_img   = show_cam_on_image(original, grayscale_cam, use_rgb=True)

            results.append({
                "path":       img_path,
                "original":   (original * 255).astype(np.uint8),
                "overlay":    overlay_img,
                "pred_class": pred_class,
                "pred_prob":  pred_prob,
                "true_label": label,
            })

            true_name = CLASS_NAMES.get(label, "?")
            pred_name = CLASS_NAMES.get(pred_class)
            tick      = "✓" if label == pred_class else ("✗" if label is not None else "")
            print(f"  {tick}  {img_path.name:<40}  "
                  f"true={true_name:<20} pred={pred_name:<20} ({pred_prob:.1%})")

    # ── save outputs ──────────────────────────────────────────────────────────
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"gradcam_{MODEL_NAME}_{METHOD}.png"
    make_figure(results, METHOD, save_path=fig_path)

    for r in results:
        cv2.imwrite(
            str(out_dir / f"{r['path'].stem}_gradcam.png"),
            cv2.cvtColor(r["overlay"], cv2.COLOR_RGB2BGR)
        )

    print(f"\nDone. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()