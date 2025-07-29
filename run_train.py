from data.dataloader import get_dataloaders
from config.base_config import *
from config.app_config import *
from models.encoder import Encoder
from models.decoder import Decoder
from engine.setup import *
from engine.train import train_model
from engine.evaluate import *
from data.data_analysis import *
import random
import nltk


def analyze_data(train_dataset, train_loader, shared_vocab):
    # Call these as needed
    nltk.download('punkt_tab')
    show_batch(train_loader, shared_vocab)
    show_sample(train_dataset, random.randint(0, len(train_dataset)))
    analyze_caption_lengths(train_dataset)
    analyze_word_freq(train_dataset)


def main():
    train_dataset, train_loader, val_loader, test_loader, vocab = get_dataloaders()

    # Optional
    analyze_data(train_dataset, train_loader, vocab)

    # encoder = Encoder(FINE_TUNE_ENCODER).to(DEVICE)
    # decoder = Decoder(ENCODER_DIM, DECODER_DIM, ATTENTION_DIM, len(vocab), EMBED_DIM, DROPOUT).to(DEVICE)
    #
    # # Setup training components
    # encoder_optimizer, decoder_optimizer = get_optimizers(encoder, decoder, DECODER_LR, ENCODER_LR, FINE_TUNE_ENCODER)
    # encoder_scheduler, decoder_scheduler = get_schedulers(encoder_optimizer, decoder_optimizer, FINE_TUNE_ENCODER)
    # criterion = get_criterion(DEVICE)
    #
    # train_model(
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     encoder=encoder,
    #     decoder=decoder,
    #     encoder_optimizer=encoder_optimizer,
    #     decoder_optimizer=decoder_optimizer,
    #     criterion=criterion,
    #     encoder_scheduler=encoder_scheduler,
    #     decoder_scheduler=decoder_scheduler,
    #     shared_vocab=vocab,
    #     device=DEVICE
    # )
    #
    # evaluate(encoder, decoder, test_loader, vocab, beam_size=BEAM_SIZE, device=DEVICE)


if __name__ == '__main__':
    main()
