"""Training-curve plotting: works for a single run (run_roots is one path — plots
"Train Loss"/"Validation Loss"/"Macro F1" plainly) or for comparing several runs
(run_roots is a list — one line per run, labelled by each run's config.json MODEL_NAME).

Handles two-phase training runs transparently: history.json for a two-phase run is
{"phase1": [...], "phase2": [...]} rather than a flat list, so it's flattened into one
continuous epoch axis first, with a dashed vertical line marking where phase 2 starts.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

from plotting.common import setup_style

setup_style()


def load_history(run_root):
    """Load history.json for run_root, plus MODEL_NAME from its config.json if present.

    Returns (model_name, history); model_name is None if config.json is missing.
    """
    history_file = os.path.join(run_root, "history.json")
    with open(history_file) as f:
        history = json.load(f)

    model_name = None
    config_file = os.path.join(run_root, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            model_name = json.load(f).get("MODEL_NAME")

    return model_name, history


def _flatten_phases(history):
    """Flatten a two-phase {"phase1": [...], "phase2": [...]} history into one
    continuous epoch-indexed list. Returns (flat_history, phase_boundary_epoch);
    phase_boundary_epoch is None if history was already a flat list."""
    if not isinstance(history, dict):
        return history, None

    offset = 0
    all_epochs = []
    phase_boundary = None
    phases = list(history.values())
    for i, phase in enumerate(phases):
        for e in phase:
            all_epochs.append({**e, "epoch": e["epoch"] + offset})
        if i == 0 and len(phases) > 1:
            phase_boundary = offset + len(phase)
        offset += len(phase)

    return all_epochs, phase_boundary


def plot_loss(run_roots, train_loss=True, val_loss=True, save_path=None):
    """Plot training/validation loss over epochs for one run, or one line per run
    (labelled by MODEL_NAME) if run_roots is a list — see module docstring."""
    single = isinstance(run_roots, (str, Path))
    roots = [run_roots] if single else run_roots

    phase_boundary = None
    for r in roots:
        model_name, history = load_history(r)
        history, boundary = _flatten_phases(history)
        if boundary is not None:
            phase_boundary = boundary

        epochs = [e["epoch"] for e in history]
        if train_loss:
            label = "Train Loss" if single else f"{model_name}"
            plt.plot(epochs, [e["train_loss"] for e in history], label=label)
        if val_loss:
            label = "Validation Loss" if single else f"{model_name}"
            plt.plot(epochs, [e["val_loss"] for e in history], label=label)

    if phase_boundary is not None:
        plt.axvline(x=phase_boundary, color='gray', linestyle='--', linewidth=1.5, label='Phase boundary')

    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()


def plot_macro_F1(run_roots, save_path=None):
    """Plot macro-F1 over epochs for one run, or one line per run (labelled by
    MODEL_NAME) if run_roots is a list — see module docstring."""
    single = isinstance(run_roots, (str, Path))
    roots = [run_roots] if single else run_roots

    phase_boundary = None
    for r in roots:
        model_name, history = load_history(r)
        history, boundary = _flatten_phases(history)
        if boundary is not None:
            phase_boundary = boundary

        epochs = [e["epoch"] for e in history]
        macro_f1s = [e["macro_f1"] for e in history]
        label = "Macro F1" if single else f"{model_name} Macro F1"
        plt.plot(epochs, macro_f1s, label=label)

    if phase_boundary is not None:
        plt.axvline(x=phase_boundary, color='gray', linestyle='--', linewidth=1.5, label='Phase boundary')

    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    plt.show()
