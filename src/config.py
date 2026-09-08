import torch

# ---- paths ----
# point this at wherever you download the Kaggle dataset locally
BASE_DIR = "data/cassava-leaf-disease-classification/"
IMAGE_DIR = BASE_DIR + "train_images/"

# ---- device ----
# uses Apple Silicon GPU (mps) if available, falls back to cuda, then cpu
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---- image / training config ----
DIM = (256, 256)
WIDTH, HEIGHT = DIM
NUM_CLASSES = 5
NUM_WORKERS = 4          # lower than the original 24 - tuned for a laptop, not a Kaggle instance
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
SEED = 1

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---- checkpoints ----
MODEL_SAVE_PATH = "saved_models/best_model.pt"
