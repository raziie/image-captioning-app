import torch
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from config.train_config import *


def validate(val_loader, encoder, decoder, criterion, epoch, device, vocab):
    # Performs one epoch's validation.

    decoder.eval()
    encoder.eval()

    total_loss = 0

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

            # Forward prop.
            imgs = encoder(imgs)
            scores, alphas, sort_ind, captions_sorted, decode_lengths = decoder(imgs, caps, caplens)

            targets = captions_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores, targets)
            loss += ALPHA_C * ((1. - alphas.sum(dim=1)) ** 2).mean()
            total_loss += loss.item()

            bleu4 = compute_bleu4(scores_copy, decode_lengths, sort_ind, allcaps, vocab)

            if i % PRINT_FREQ == 0:
                print(f'Validation -> Epoch: [{epoch}][{i}/{len(val_loader)}]\t ,loss: {loss}\t ,bleu4: {bleu4}')

    avg_loss = total_loss / len(val_loader)
    return avg_loss, bleu4


def compute_bleu4(scores_copy, decode_lengths, sort_ind, allcaps, vocab):
    _, preds = torch.max(scores_copy, dim=2)

    preds = preds.cpu().tolist()

    # Trim predictions
    hypotheses = [caption[:length] for caption, length in zip(preds, decode_lengths)]

    # Sort and filter references
    sorted_allcaps = allcaps[sort_ind]
    references = [
        [
            [token for token in cap.tolist()
             if token not in {vocab.stoi["<start>"], vocab.stoi["<pad>"]}]
            for cap in image_caps
        ]
        for image_caps in sorted_allcaps
    ]
    assert len(references) == len(hypotheses), "Mismatch between references and hypotheses"

    return corpus_bleu(references, hypotheses)
