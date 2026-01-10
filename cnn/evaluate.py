import numpy as np
import torch

@torch.no_grad() # No gradient tracking
def evaluate(model, loader, device, criterion):
    """
    Run evaluation for a classification model
    Computes computes the average loss and
    collects ground-truth labels, predicted class labels, predicted class probabilities
    """

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1) 

        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    avg_loss = total_loss / n

    return avg_loss, y_true, y_pred, y_prob