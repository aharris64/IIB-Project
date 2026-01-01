import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

# Config
DATA_ROOT = r"C:\Users\adam6\OneDrive\Documents\University\Engineering\Engineering IIB\IIB Project\Datasets\train_test_val\basic_resize_224"
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DEVICE = "cpu"
PATIENCE = 5            # stop if val macro-F1 doesn't improve for this many epochs

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])


# -----------------------
# DATASETS & LOADERS
# -----------------------
train_dir = os.path.join(DATA_ROOT, "train")
val_dir   = os.path.join(DATA_ROOT, "val")
test_dir  = os.path.join(DATA_ROOT, "test")

train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)
test_ds  = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)

print("Class mapping:", train_ds.class_to_idx)
print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))


# -----------------------
# CLASS WEIGHTS (IMBALANCE)
# -----------------------
counts = np.bincount([y for _, y in train_ds.samples])
# inverse frequency weighting
weights = counts.sum() / (len(counts) * counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print("Train class counts:", counts.tolist())
print("Class weights:", class_weights.tolist())


# -----------------------
# MODEL (MobileNetV1 from timm)
# -----------------------
model = timm.create_model(
    "mobilenetv1_100",
    pretrained=True,
    num_classes=len(train_ds.classes)
).to(DEVICE)


# -----------------------
# LOSS & OPTIMIZER
# -----------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# -----------------------
# EVALUATION
# -----------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return total_loss / max(n, 1), macro_f1, bal_acc, cm


# -----------------------
# TRAIN LOOP + EARLY STOP
# -----------------------
best_val_f1 = -1.0
best_epoch = 0
bad_epochs = 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()

    running_loss = 0.0
    n = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        n += y.size(0)

    train_loss = running_loss / max(n, 1)
    val_loss, val_f1, val_bal_acc, val_cm = evaluate(val_loader)

    dt = time.time() - t0

    improved = val_f1 > best_val_f1 + 1e-4
    if improved:
        best_val_f1 = val_f1
        best_epoch = epoch
        bad_epochs = 0
        torch.save(model.state_dict(), "best_mobilenetv1_cpu.pt")
    else:
        bad_epochs += 1

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | {dt:.1f}s | "
        f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
        f"val macroF1 {val_f1:.4f} | val balAcc {val_bal_acc:.4f}"
    )
    print("Val confusion matrix:\n", val_cm)

    if bad_epochs >= PATIENCE:
        print(f"Early stopping: no val macro-F1 improvement for {PATIENCE} epochs.")
        break


# -----------------------
# TEST EVAL (best checkpoint)
# -----------------------
model.load_state_dict(torch.load("best_mobilenetv1_cpu.pt", map_location=DEVICE))
test_loss, test_f1, test_bal_acc, test_cm = evaluate(test_loader)

print("\nBEST EPOCH:", best_epoch, "BEST VAL macroF1:", best_val_f1)
print("\nTEST RESULTS")
print(f"Loss: {test_loss:.4f}")
print(f"Macro-F1: {test_f1:.4f}")
print(f"Balanced Acc: {test_bal_acc:.4f}")
print("Confusion matrix:\n", test_cm)
