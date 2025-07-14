import os
import torch

# Paths
BASE_APP_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_APP_DIR, 'static', 'uploads')
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'audio')

BASE_PROJECT_DIR = os.path.abspath(os.path.join(BASE_APP_DIR, "."))
BEST_CHECKPOINT_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'checkpoint_best.pth')
VOCAB_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'vocab.pkl')

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference Config
BEAM_SIZE = 3
