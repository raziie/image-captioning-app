import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class Encoder(nn.Module):
    def __init__(self, fine_tune=True):
        super().__init__()
        self.fine_tune = fine_tune

        # Load ResNet-101 with pretrained weights
        weights = ResNet101_Weights.DEFAULT
        resnet = resnet101(weights=weights)

        # Remove the final average pool and fully connected layers
        self.model = nn.Sequential(*list(resnet.children())[:-2])  # Output shape: [B, 2048, H/32, W/32]

        # Apply fine-tuning rules
        self._set_grad()

    def forward(self, images):
        """
        Forward pass: Extract visual features and permute for decoder
        Output shape: [B, H/32, W/32, 2048]
        """
        x = self.model(images)  # [B, 2048, H/32, W/32]
        x = x.permute(0, 2, 3, 1)  # => [B, H/32, W/32, 2048]
        return x

    def _set_grad(self):
        """
        Freeze all layers except the final residual block (layer4) if fine_tune=True.
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = self.fine_tune and "7" in name


if __name__ == '__main__':
    model = Encoder()
    import torch
    print(f'image feature vector shape: {model(torch.randn(1, 3, 256, 256)).shape}')