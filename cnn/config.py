
EXPERIMENT_NOTE = (
    "Efficient net b0 with disc centering, augmentation, removed manually rated class 3 and 4, removed some datasets, two-phase"
)

# Model
MODEL_NAME = "ghostnet"
NUM_CLASSES = 3
FREEZE = "two_phase"   # options: "none" | "head" | "two_phase"

# Data
DATASET = "disc_centred_r4.0_cl34_augmented"
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training
NUM_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PHASE1_EPOCHS = 10

# Augmentation

# Reproducibility
SEED = 42