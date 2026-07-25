"""Single source of training config, read by classifier/run_model.ipynb for one training run.

Edit these values and re-run the notebook to launch a new experiment; a snapshot of
this file's values is saved to <run_dir>/config.json for each run.
"""

# Free-text label for this run, saved into run_meta.json for your own record-keeping.
EXPERIMENT_NOTE = (
    "Mobilenetv3 with disc centering, augmentation, removed manually rated class 3 and 4, removed some datasets, two-phase, lowres14"
)

# Model
MODEL_NAME = "mobilenet_v3_small"  # see classifier/models.py:build_model for valid names
NUM_CLASSES = 3
# "none": train end-to-end from the start.
# "head": freeze the backbone permanently, train only the classifier head.
# "two_phase": train the head alone for PHASE1_EPOCHS, then unfreeze the backbone and
#   fine-tune everything at a lower backbone learning rate (see run_model.ipynb).
FREEZE = "two_phase"

# Data
DATASET = "disc_centred_r4.0_cl34_augmented_lowres14"  # subfolder under DATA_ROOT/, see load_data.get_dataloaders
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training
NUM_EPOCHS = 50       # phase 2 epoch budget when FREEZE == "two_phase", else the only phase
PATIENCE = 10         # early-stopping patience, in epochs of no val_loss improvement
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PHASE1_EPOCHS = 10    # head-only warmup epochs; only used when FREEZE == "two_phase"

# Reproducibility
SEED = 42