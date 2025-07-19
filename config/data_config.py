import os

# Paths
BASE_CONFIG_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_DATA_DIR = os.path.abspath(os.path.join(BASE_CONFIG_DIR, "./data"))
KARPATHY_JSON_PATH = os.path.abspath(os.path.join(BASE_DATA_DIR, "input/karpathy-splits/dataset_flickr8k.json"))
IMAGE_FOLDER = os.path.abspath(os.path.join(BASE_DATA_DIR, "input/flickr8kimagescaptions/flickr8k/images/"))

# Hyperparameters
FREQ_THRESHOLD = 3
CAPTIONS_PER_IMAGE = 5
BATCH_SIZE = 32
WORKERS = 2
PIN_MEMORY = True
