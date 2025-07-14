import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from .config import *
from .utils import *
from captioning.visualize import visualize_attention
from captioning.generator import CaptionGenerator


# Load the model
checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=str(DEVICE), weights_only=False)

decoder = checkpoint['decoder'].to(DEVICE).eval()
encoder = checkpoint['encoder'].to(DEVICE).eval()

# Load vocab
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained checkpoints
])

captioner = CaptionGenerator(encoder, decoder, vocab, device=DEVICE)


def generate_caption(image_path, visualize=False):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).to(DEVICE)  # (3, 256, 256)

    seq, alphas = captioner.generate_beam_search(img, beam_size=BEAM_SIZE)

    # Visualize caption and attention of best sequence
    if visualize:
        visualize_attention(image_path, seq, torch.FloatTensor(alphas), vocab, smooth=True)

    caption = ' '.join([vocab.itos[idx] for idx in seq if vocab.itos[idx] not in ['<start>', '<end>']])
    return caption
