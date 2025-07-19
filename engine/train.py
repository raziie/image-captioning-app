import torch
from torch.nn.utils.rnn import pack_padded_sequence
from engine.validate import validate
from config.train_config import *
from config.base_config import *


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, device):
    """
    Performs one epoch of training.

    Args:
        train_loader: DataLoader for training data
        encoder: Image encoder
        decoder: Caption decoder
        criterion: Loss function
        encoder_optimizer: Optimizer for encoder
        decoder_optimizer: Optimizer for decoder
        epoch: Current epoch number (int)
        device: Device to run on (e.g. "cuda" or "cpu")
        alpha_c: Attention regularization coefficient
        grad_clip: Max norm for gradient clipping (None to disable)
        print_freq: Print status every N batches
        fine_tune_encoder: Whether encoder is being fine-tuned
    """
    decoder.train()
    encoder.train()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward
        imgs = encoder(imgs)
        scores, alphas, sort_ind, captions_sorted, decode_lengths = decoder(imgs, caps, caplens)

        targets = captions_sorted[:, 1:]  # Remove <start>

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Loss
        loss = criterion(scores, targets)
        loss += ALPHA_C * ((1. - alphas.sum(dim=1)) ** 2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        if GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
            if FINE_TUNE_ENCODER:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)

        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        if i % PRINT_FREQ == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss: {loss.item():.4f}')

    return loss


def update_scheduler(scheduler, val_loss, name=""):
    old_lr = scheduler.optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = scheduler.optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"{name} learning rate decreased: {old_lr:.6f} => {new_lr:.6f}")


def train_model(train_loader, val_loader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion, encoder_scheduler, decoder_scheduler,
                shared_vocab, device):

    best_bleu4 = 0.0
    epochs_since_improvement = 0
    start_epoch = 0

    if CHECKPOINT_PATH:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder.eval()
        decoder.eval()

        if 'encoder_optimizer_state_dict' in checkpoint and FINE_TUNE_ENCODER:
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        if 'decoder_optimizer_state_dict' in checkpoint:
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_bleu4 = checkpoint.get('best_bleu4', 0.0)

        print(f"Resumed training from epoch {start_epoch}, best BLEU-4: {best_bleu4:.4f}")

    for epoch in range(start_epoch, EPOCHS):
        if epochs_since_improvement == 20:
            print("Early stopping: No improvement in 20 epochs.")
            break

        loss = train(
            train_loader=train_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            epoch=epoch,
            device=device
        )

        val_loss, last_bleu4 = validate(
            val_loader=val_loader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            epoch=epoch,
            device=device,
            vocab=shared_vocab
        )

        if fine_tune_encoder:
            update_scheduler(encoder_scheduler, val_loss, "Encoder")

        update_scheduler(decoder_scheduler, val_loss, "Decoder")

        is_best = last_bleu4 > best_bleu4
        best_bleu4 = max(last_bleu4, best_bleu4)

        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict() if encoder_optimizer else None,
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
            'best_bleu4': best_bleu4,
        }

        torch.save(checkpoint, 'checkpoint.pth')
        if is_best:
            torch.save(checkpoint, 'checkpoint_best.pth')


