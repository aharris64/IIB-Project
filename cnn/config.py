
EXPERIMENT_NOTE = (
    "First attempt at running"
)

# Model
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 3
FREEZE = "head"

# Data
DATA_ROOT = "/content/drive/MyDrive/datasets/"
DATASET = "basic_resize_224"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training
NUM_EPOCHS = 50
PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Augmentation

# Reproducibility
SEED = 42