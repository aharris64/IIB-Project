
EXPERIMENT_NOTE = (
    "First pass of models: efficient net lite 1"
)

# Model
MODEL_NAME = "efficientnet_lite1"
NUM_CLASSES = 3
FREEZE = "head"

# Data
DATASET = "basic_resize_224"
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