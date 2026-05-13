import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

# Datasets
LEAF_DATA_DIR   = os.path.join(BASE_DIR, "data",
                      "plantvillage dataset", "color")
FRUIT_DATA_DIR  = os.path.join(BASE_DIR, "data",
                      "fruit_dataset",
                      "Fruit And Vegetable Diseases Dataset")
COMBINED_DATA_DIR = os.path.join(BASE_DIR, "data",
                      "combined_dataset",
                      "dataset_clean_final")

# Active dataset for training
DATA_DIR        = COMBINED_DATA_DIR

# Checkpoints
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LEAF_MODEL_PATH = os.path.join(BASE_DIR, "models",
                      "leaf_model", "best_model.pth")
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "models",
                      "fruit_model", "best_model.pth")
COMBINED_MODEL_PATH = os.path.join(BASE_DIR, "models",
                          "combined_model", "best_model.pth")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# ── Image settings ─────────────────────────────────────
IMAGE_SIZE  = 224
BATCH_SIZE  = 16

# ── Training settings ──────────────────────────────────
NUM_EPOCHS      = 10
LEARNING_RATE   = 0.0001
NUM_WORKERS     = 0

# ── Dataset split ──────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ── Model ──────────────────────────────────────────────
NUM_CLASSES = 134
MODEL_NAME  = "NaanNee"

# ── Metrics ────────────────────────────────────────────
LEAF_MODEL_METRICS = {
    "accuracy": 99.74,
    "precision": 1.00,
    "recall": 1.00,
    "f1_score": 1.00,
    "total_images": 139980,
    "num_classes": 66,
    "best_epoch": 4
}

FRUIT_MODEL_METRICS = {
    "accuracy": 98.98,
    "precision": 0.99,
    "recall": 0.99,
    "f1_score": 0.99,
    "total_images": 29291,
    "num_classes": 28,
    "best_epoch": 7
}
