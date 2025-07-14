import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from config.app_config import *
from .utils import *
from captioning.visualize import visualize_attention
from captioning.generator import CaptionGenerator
from models.encoder import Encoder
from models.decoder import Decoder
from config.base_config import *
# from data.vocab import Vocabulary


def get_encoder_dim(enc, image_shape, device='cpu'):
    with torch.no_grad():
        dummy_input = torch.randn(*image_shape).to(device)
        out = enc(dummy_input)
        return out.shape[-1]


# Load vocab
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained checkpoints
])

# Load the model
checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=str(DEVICE))

encoder = Encoder(FINE_TUNE_ENCODER).to(DEVICE)
encoder_dim = get_encoder_dim(encoder, (1, 3, 256, 256), DEVICE)

decoder = Decoder(encoder_dim, DECODER_DIM, ATTENTION_DIM, len(vocab), EMBED_DIM, DROPOUT).to(DEVICE)
encoder.load_state_dict(checkpoint['encoder_state_dict']).to(DEVICE).eval()
decoder.load_state_dict(checkpoint['decoder_state_dict']).to(DEVICE).eval()

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
