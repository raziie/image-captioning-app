import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch
from data.vocab import Vocabulary


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        image_folder,
        captions_dict,
        tokens,
        split,
        captions_per_image=5,
        transform=None,
        freq_threshold=5,
        vocab=None,
    ):
        self.image_folder = image_folder
        self.split = split
        self.captions_per_image = captions_per_image
        self.transform = transform

        self.image_names = list(captions_dict.keys())
        self.captions_dict = self.sample_captions(captions_dict)
        self.captions = [caption for caps in self.captions_dict.values() for caption in caps]
        self.tokens = tokens
        self.caption_lengths = [len(token) + 2 for token in tokens]  # +2 for <start> and <end>

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.tokens)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Select random caption
        captions = self.captions_dict[image_name]
        rand_idx = random.randrange(len(captions))
        selected_token = self.tokens[idx * self.captions_per_image + rand_idx]
        caption_tensor = self.vocab.tokenize_caption(selected_token)
        caption_length = self.caption_lengths[idx * self.captions_per_image + rand_idx]

        output = (image, caption_tensor, caption_length)

        # If val or test, return all captions
        if self.split != "train":
            all_captions = [
                self.vocab.tokenize_caption(
                    self.tokens[idx * self.captions_per_image + i]
                )
                for i in range(self.captions_per_image)
            ]
            output += (all_captions,)

        return output

    def sample_captions(self, captions_dict):
        """Ensure a fixed number of captions per image by sampling or duplicating."""
        sampled = {}
        for img, caps in captions_dict.items():
            if len(caps) < self.captions_per_image:
                caps = caps + random.choices(caps, k=self.captions_per_image - len(caps))
            else:
                caps = random.sample(caps, k=self.captions_per_image)
            sampled[img] = caps
        return sampled
