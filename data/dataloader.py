# data/dataloader.py

import json
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import ImageCaptionDataset
from data.vocab import Vocabulary
from data.collate import MyCollate
from config.data_config import *
from config.app_config import *


def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open(KARPATHY_JSON_PATH, "r") as f:
        data = json.load(f)

    split_data = {"train": {}, "val": {}, "test": {}}
    token_data = {"train": [], "val": [], "test": []}

    for img in data["images"]:
        image_name = img["filename"]
        split = img["split"]
        captions = [sent["raw"] for sent in img["sentences"]]
        tokens = [sent["tokens"] for sent in img["sentences"]]

        split_data[split][image_name] = captions
        token_data[split] += tokens

    # Training dataset builds the vocab
    train_dataset = ImageCaptionDataset(
        IMAGE_FOLDER, split_data["train"], token_data["train"],
        split="train", captions_per_image=CAPTIONS_PER_IMAGE, transform=transform,
        freq_threshold=FREQ_THRESHOLD
    )

    shared_vocab = train_dataset.vocab

    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump(shared_vocab, f)

    val_dataset = ImageCaptionDataset(
        IMAGE_FOLDER, split_data["val"], token_data["val"],
        split="val", captions_per_image=CAPTIONS_PER_IMAGE, transform=transform,
        freq_threshold=FREQ_THRESHOLD, vocab=shared_vocab
    )

    test_dataset = ImageCaptionDataset(
        IMAGE_FOLDER, split_data["test"], token_data["test"],
        split="test", captions_per_image=CAPTIONS_PER_IMAGE, transform=transform,
        freq_threshold=FREQ_THRESHOLD, vocab=shared_vocab
    )

    pad_idx = shared_vocab.stoi["<pad>"]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=MyCollate(pad_idx, split="train"), num_workers=WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=MyCollate(pad_idx, split="val"), num_workers=WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=MyCollate(pad_idx, split="test"), num_workers=WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, val_loader, test_loader, shared_vocab
