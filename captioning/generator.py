import torch
import torch.nn.functional as F
import numpy as np


class CaptionGenerator:
    def __init__(self, encoder, decoder, vocab, device=None):
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = self.encoder.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()

    def _encode_image(self, image):
        encoder_out = self.encoder(image.unsqueeze(0).to(self.device))
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # flatten
        return encoder_out, enc_image_size, encoder_dim

    def _apply_attention(self, encoder_out, h):
        attn_encoding, alpha = self.decoder.attention(encoder_out, h)
        gate = self.decoder.sigmoid(self.decoder.gate_linear(h))
        attn_encoding = gate * attn_encoding
        return attn_encoding, alpha

    def generate(self, image, max_len=50):
        encoder_out, enc_image_size, _ = self._encode_image(image)
        h, c = self.decoder.init_states(encoder_out)

        prev_word = torch.LongTensor([self.vocab.stoi['<start>']]).to(self.device)
        seq = []
        alphas = []

        step = 1
        while True:
            embeddings = self.decoder.embedding(prev_word).squeeze(1)
            attn_encoding, alpha = self._apply_attention(encoder_out, h)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)

            h, c = self.decoder.lstm_cell(torch.cat([embeddings, attn_encoding], dim=1), (h, c))
            scores = F.log_softmax(self.decoder.fc(h), dim=1)
            top_word_ind = torch.argmax(scores, dim=1)

            if top_word_ind.item() == self.vocab.stoi['<end>'] or step > max_len:
                break

            seq.append(top_word_ind.item())
            alphas.append(alpha.cpu().detach().numpy())
            prev_word = top_word_ind.unsqueeze(0)
            step += 1

        return seq, alphas

    def generate_beam_search(self, image, beam_size=5, max_len=50):
        k = beam_size
        vocab_size = len(self.vocab)

        encoder_out, enc_image_size, encoder_dim = self._encode_image(image)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[self.vocab.stoi['<start>']]] * k).to(self.device)
        seqs = k_prev_words.clone()
        top_k_scores = torch.zeros(k, 1).to(self.device)
        seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)

        complete_seqs, complete_alphas, complete_scores = [], [], []

        h, c = self.decoder.init_states(encoder_out)

        step = 1
        while True:
            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
            attn_encoding, alpha = self._apply_attention(encoder_out, h)
            alpha = alpha.view(-1, enc_image_size, enc_image_size)

            h, c = self.decoder.lstm_cell(torch.cat([embeddings, attn_encoding], dim=1), (h, c))
            scores = F.log_softmax(self.decoder.fc(h), dim=1)
            scores = top_k_scores.expand_as(scores) + scores

            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_inds = top_k_words // vocab_size
            next_inds = top_k_words % vocab_size

            seqs = torch.cat([seqs[prev_inds], next_inds.unsqueeze(1)], dim=1)
            seqs_alpha = torch.cat([seqs_alpha[prev_inds], alpha[prev_inds].unsqueeze(1)], dim=1)

            incomplete_inds = (next_inds != self.vocab.stoi['<end>']).nonzero(as_tuple=False).squeeze(1)
            complete_inds = (next_inds == self.vocab.stoi['<end>']).nonzero(as_tuple=False).squeeze(1)

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_alphas.extend(seqs_alpha[complete_inds].tolist())
                complete_scores.extend(top_k_scores[complete_inds])

            k -= len(complete_inds)
            if k == 0 or step > max_len:
                break

            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h = h[prev_inds[incomplete_inds]]
            c = c[prev_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_inds[incomplete_inds].unsqueeze(1)

            step += 1

        if len(complete_scores) == 0:
            complete_seqs = seqs.tolist()
            complete_alphas = seqs_alpha.tolist()
            complete_scores = top_k_scores

        else:
            complete_scores = torch.tensor(complete_scores)

        best_seq_idx = complete_scores.argmax().item()
        return complete_seqs[best_seq_idx], complete_alphas[best_seq_idx]

    # def generate_beam_search(self, image, beam_size=5, max_len=50):
    #     encoder_out, enc_image_size, encoder_dim = self._encode_image(image)
    #     encoder_out = encoder_out.expand(beam_size, -1, -1)
    #
    #     sequences = self._init_beam_sequences(beam_size, enc_image_size)
    #     h, c = self.decoder.init_states(encoder_out)
    #
    #     step = 1
    #     while True:
    #         sequences = self._beam_step(sequences, encoder_out, h, c, enc_image_size)
    #         h, c, encoder_out = sequences['h'], sequences['c'], sequences['encoder_out']
    #         if sequences['done'] or step > max_len:
    #             break
    #         step += 1
    #
    #     return self._finalize_beam(sequences)
    #
    # def _init_beam_sequences(self, k, enc_image_size):
    #     start_token = self.vocab.stoi['<start>']
    #     k_prev_words = torch.LongTensor([[start_token]] * k).to(self.device)
    #     top_k_scores = torch.zeros(k, 1).to(self.device)
    #     seqs = k_prev_words.clone()
    #     seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(self.device)
    #     return {
    #         'k': k, 'seqs': seqs, 'alphas': seqs_alpha,
    #         'top_k_scores': top_k_scores,
    #         'prev_words': k_prev_words,
    #         'complete_seqs': [], 'complete_alphas': [], 'complete_scores': [],
    #         'h': None, 'c': None, 'encoder_out': None, 'done': False
    #     }
    #
    # def _beam_step(self, s, encoder_out, h, c, enc_image_size):
    #     embeddings = self.decoder.embedding(s['prev_words']).squeeze(1)
    #     attn_encoding, alpha = self._apply_attention(encoder_out, h)
    #     alpha = alpha.view(-1, enc_image_size, enc_image_size)
    #
    #     h, c = self.decoder.lstm_cell(torch.cat([embeddings, attn_encoding], dim=1), (h, c))
    #     scores = F.log_softmax(self.decoder.fc(h), dim=1)
    #     scores = s['top_k_scores'].expand_as(scores) + scores
    #
    #     top_scores, top_words = scores.view(-1).topk(s['k'], 0, True, True)
    #     vocab_size = scores.size(1)
    #     prev_inds = top_words // vocab_size
    #     next_inds = top_words % vocab_size
    #
    #     seqs = torch.cat([s['seqs'][prev_inds], next_inds.unsqueeze(1)], dim=1)
    #     alphas = torch.cat([s['alphas'][prev_inds], alpha[prev_inds].unsqueeze(1)], dim=1)
    #
    #     complete_mask = (next_inds == self.vocab.stoi['<end>'])
    #     incomplete_mask = ~complete_mask
    #
    #     if complete_mask.any():
    #         s['complete_seqs'].extend(seqs[complete_mask].tolist())
    #         s['complete_alphas'].extend(alphas[complete_mask].tolist())
    #         s['complete_scores'].extend(top_scores[complete_mask])
    #
    #     s['k'] -= complete_mask.sum().item()
    #     if s['k'] == 0:
    #         s['done'] = True
    #         return s
    #
    #     # Prepare next step
    #     seqs = seqs[incomplete_mask]
    #     alphas = alphas[incomplete_mask]
    #     h = h[prev_inds[incomplete_mask]]
    #     c = c[prev_inds[incomplete_mask]]
    #     encoder_out = encoder_out[prev_inds[incomplete_mask]]
    #     top_k_scores = top_scores[incomplete_mask].unsqueeze(1)
    #     k_prev_words = next_inds[incomplete_mask].unsqueeze(1)
    #
    #     s.update({
    #         'seqs': seqs, 'alphas': alphas,
    #         'top_k_scores': top_k_scores,
    #         'prev_words': k_prev_words,
    #         'h': h, 'c': c, 'encoder_out': encoder_out
    #     })
    #     return s
    #
    # def _finalize_beam(self, s):
    #     if not s['complete_scores']:
    #         s['complete_seqs'] = s['seqs'].tolist()
    #         s['complete_alphas'] = s['alphas'].tolist()
    #         s['complete_scores'] = s['top_k_scores'].squeeze(1)
    #
    #     best_idx = torch.tensor(s['complete_scores']).argmax().item()
    #     return s['complete_seqs'][best_idx], s['complete_alphas'][best_idx]

