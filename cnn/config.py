
EXPERIMENT_NOTE = (
    "Mobilenetv3 with disc centering, augmentation, removed manually rated class 3 and 4, removed some datasets, two-phase, lowres14"
)

# Model
MODEL_NAME = "mobilenet_v3_small"
NUM_CLASSES = 3
FREEZE = "two_phase"   # options: "none" | "head" | "two_phase"

# Data
DATASET = "disc_centred_r4.0_cl34_augmented_lowres28"
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