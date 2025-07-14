import torch
import torch.nn as nn
from models.attention import Attention


class Decoder(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, vocab_size, embed_dim, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.gate_linear = nn.Linear(decoder_dim, encoder_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_states(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def sort_batch(self, encoder_out, embedded_captions, caption_lengths):
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        return (
            encoder_out[sort_ind],
            embedded_captions[sort_ind],
            caption_lengths,
            sort_ind,
        )

    def decode_step(self, embeddings, encoder_out, h, c, decode_lengths, device):
        batch_size, num_pixels = encoder_out.size(0), encoder_out.size(1)

        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_t = sum([l > t for l in decode_lengths])

            attn_context, alpha = self.attention(encoder_out[:batch_t], h[:batch_t])
            gate = self.sigmoid(self.gate_linear(h[:batch_t]))
            attn_context = gate * attn_context

            lstm_input = torch.cat([embeddings[:batch_t, t, :], attn_context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h[:batch_t], c[:batch_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_t, t, :] = preds
            alphas[:batch_t, t, :] = alpha

        return predictions, alphas

    def forward(self, encoder_out, embedded_captions, caption_lengths):
        device = encoder_out.device
        batch_size, encoder_dim = encoder_out.size(0), encoder_out.size(-1)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [B, num_pixels, enc_dim]
        encoder_out, embedded_captions, caption_lengths, sort_ind = self.sort_batch(
            encoder_out, embedded_captions, caption_lengths
        )
        embeddings = self.embedding(embedded_captions)

        h, c = self.init_states(encoder_out)
        decode_lengths = (caption_lengths - 1).tolist()

        predictions, alphas = self.decode_step(
            embeddings, encoder_out, h, c, decode_lengths, device
        )

        return predictions, alphas, sort_ind, embedded_captions, decode_lengths
