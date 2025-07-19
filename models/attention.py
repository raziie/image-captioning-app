import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()

        # Project encoder and decoder features to attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # For encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # For decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # Combine and score

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor):
        """
        encoder_out: [B, num_pixels, encoder_dim]
        decoder_hidden: [B, decoder_dim]
        Returns:
            attention_weighted_encoding: [B, encoder_dim]
            alpha: attention weights [B, num_pixels]
        """
        # Project encoder and decoder outputs
        enc_att = self.encoder_att(encoder_out)            # [B, num_pixels, attention_dim]
        dec_att = self.decoder_att(decoder_hidden)         # [B, attention_dim]
        dec_att = dec_att.unsqueeze(1)                     # [B, 1, attention_dim]

        # Combine and score
        att = self.full_att(self.relu(enc_att + dec_att))  # [B, num_pixels, 1]
        att = att.squeeze(2)                               # [B, num_pixels]

        # Normalize into a distribution
        alpha = self.softmax(att)                          # [B, num_pixels]

        # Weighted sum of encoder output
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [B, encoder_dim]

        return context, alpha


if __name__ == '__main__':
    model = Attention(encoder_dim=1280, decoder_dim=256, attention_dim=256)
    attention_weighted_encoding, alpha = model(torch.randn(1, 64, 1280), torch.randn(1, 256))
    print(attention_weighted_encoding.shape)  # (batch_size, encoder_dim)
    print(alpha.shape)  # (batch_size, num_pixels)
