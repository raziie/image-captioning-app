import torch
from torchvision.models import resnet101

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBED_DIM = 512
ATTENTION_DIM = 512
DECODER_DIM = 512
ENCODER_DIM = resnet101(pretrained=False).fc.in_features
DROPOUT = 0.5
FINE_TUNE_ENCODER = True
ENCODER_LR = 1e-4
DECODER_LR = 4e-4
