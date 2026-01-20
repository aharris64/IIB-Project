import json
import matplotlib.pyplot as plt
import os


def load_history(run_root):
    """
    Loads history from json file given root
    """
    history_file = os.path.join(run_root, "history.json")
    with open(history_file) as f:
        history = json.load(f)

    return history

def plot_loss(run_root, train_loss=True, val_loss=True, save_path=None):
    """
    Plot validation and training loss over epochs
    """
    history = load_history(run_root)

    epochs = [e["epoch"] for e in history]
    train_losses = [e["train_loss"] for e in history]
    val_losses = [e["val_loss"] for e in history]

    if train_loss:
        plt.plot(epochs, train_losses, label="Train Loss")
    if val_loss:
        plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_macro_F1(run_root, save_path=None):
    """
    Plot macro F1 over epochs
    """
    history = load_history(run_root)

    epochs = [e["epoch"] for e in history]
    macro_f1s = [e["macro_f1"] for e in history]

    plt.plot(epochs, macro_f1s, label="Macro F1")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()