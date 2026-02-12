
EXPERIMENT_NOTE = (
    "Mobile Net V3 with disc centering, no augmentation, removed manually rated class 4, removed some datasets"
)

# Model
MODEL_NAME = "mobilenet_v3"
NUM_CLASSES = 3
FREEZE = "head"

# Data
DATASET = "disc_centred_r4.0_cl4_removed"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training
NUM_EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Augmentation

# Reproducibility
SEED = 42