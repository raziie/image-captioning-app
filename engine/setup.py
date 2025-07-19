import torch
import torch.nn as nn


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=2,
        verbose=True
    )


def get_optimizers(encoder, decoder, decoder_lr, encoder_lr, fine_tune_encoder):
    decoder_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=decoder_lr
    )

    encoder_optimizer = None
    if fine_tune_encoder:
        encoder_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=encoder_lr
        )

    return encoder_optimizer, decoder_optimizer


def get_schedulers(encoder_optimizer, decoder_optimizer, fine_tune_encoder):
    encoder_scheduler = get_scheduler(encoder_optimizer) if fine_tune_encoder else None
    decoder_scheduler = get_scheduler(decoder_optimizer)
    return encoder_scheduler, decoder_scheduler


def get_criterion(device):
    return nn.CrossEntropyLoss().to(device)
