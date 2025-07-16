import os
import torch

BASE_CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
# Paths
BASE_APP_DIR = os.path.abspath(os.path.join(BASE_CONFIG_DIR, "./app"))

BASE_PROJECT_DIR = os.path.abspath(os.path.join(BASE_CONFIG_DIR, "."))
BEST_CHECKPOINT_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'checkpoint_best.pth')
VOCAB_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'vocab.pkl')

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference Config
BEAM_SIZE = 3
