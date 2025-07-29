from torch.nn.utils.rnn import pad_sequence
import torch


class MyCollate:
    def __init__(self, pad_idx, split="train"):
        self.pad_idx = pad_idx
        self.split = split

    def __call__(self, batch):
        # Unpack components
        images = [item[0].unsqueeze(0) for item in batch]  # Shape: [1, C, H, W]
        captions = [item[1] for item in batch]
        lengths = [item[2] for item in batch]

        # Stack images into [B, C, H, W]
        images = torch.cat(images, dim=0)
        # Pad captions into [B, max_len]
        padded_captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        output = (images, padded_captions, torch.tensor(lengths))

        if self.split != "train":
            # Expect item[3] to be a list of multiple captions per image
            all_captions = [item[3] for item in batch]  # Shape: [B, num_caps, variable_len]

            flat = [cap.clone().detatch() for caps in all_captions for cap in caps]
            padded = pad_sequence(flat, batch_first=True, padding_value=self.pad_idx)

            B = len(batch)
            num_caps = len(all_captions[0])  # Assume consistent count
            reshaped = padded.view(B, num_caps, -1)

            output += (reshaped,)
        return output
