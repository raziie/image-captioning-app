import os

BASE_CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
# Paths
BASE_APP_DIR = os.path.abspath(os.path.join(BASE_CONFIG_DIR, "./app"))
AUDIO_DIR = os.path.join(BASE_APP_DIR, 'static', 'uploads', 'audio')

BASE_PROJECT_DIR = os.path.abspath(os.path.join(BASE_CONFIG_DIR, "."))
BEST_CHECKPOINT_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'checkpoint_best.pth')
SIMPLE_CHECKPOINT_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'checkpoint.pth')
VOCAB_PATH = os.path.join(BASE_PROJECT_DIR, 'checkpoints', 'vocab.pkl')

# Inference Config
BEAM_SIZE = 3
