
EXPERIMENT_NOTE = (
    "First pass of models: res net"
)

# Model
MODEL_NAME = "resnet"
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