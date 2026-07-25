"""Single-phase training loop. cnn/run_model.ipynb calls train() once (freeze="none"/"head")
or twice back-to-back (freeze="two_phase": head-only, then unfreeze + full fine-tune) —
this module has no phase logic of its own, it just runs a plain train/early-stop loop."""

from evaluate import evaluate
from sklearn.metrics import f1_score
import copy

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch over loader; return the average training loss."""
    model.train()

    total_loss = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

    avg_loss = total_loss / n

    return avg_loss

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience):
    """Train for up to num_epochs, stopping early after `patience` epochs with no
    val_loss improvement. Returns (best_epoch, best_state_dict, history), where
    history is a list of {epoch, train_loss, val_loss, macro_f1} per epoch."""

    best = None
    best_state = None
    best_epoch = None
    bad_epochs = 0
    history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, y_true, y_pred, y_prob = evaluate(model, val_loader, device, criterion)

        macro_f1 = f1_score(y_true, y_pred, average="macro")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_f1": macro_f1
        })

        # Try monitoring val_loss first, switch to macro_f1 if its better
        metric = val_loss
        improved = (best is None) or (metric < best)

        if improved:
            best = metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

            bad_epochs = 0
        else:
            bad_epochs += 1

        print(
            f"[Epoch {epoch+1}] "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val macro-F1: {macro_f1:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    return best_epoch, best_state, history