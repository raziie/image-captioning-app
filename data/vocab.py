import spacy
from collections import Counter
import torch


class Vocabulary:
    def __init__(self, freq_threshold, spacy_model="en_core_web_sm"):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.nlp = spacy.load(spacy_model)

    def __len__(self):
        return len(self.itos)

    def tokenizer_eng(self, text):
        return [tok.text.lower() for tok in self.nlp.tokenizer(text)]

    def build_vocabulary(self, tokenized_captions):
        frequencies = Counter()

        for tokens in tokenized_captions:
            frequencies.update(tokens)

        words = [w for w, freq in frequencies.items() if freq > self.freq_threshold]

        for idx, word in enumerate(words, start=len(self.itos)):
            self.stoi[word] = idx
            self.itos[idx] = word

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]

    def textualize(self, embedding):
        embedding_list = embedding.tolist()
        return " ".join([self.itos.get(idx, "<unk>") for idx in embedding_list])

    def tokenize_caption(self, tokens):
        caption_ids = [self.stoi["<start>"]]
        caption_ids += self.numericalize(tokens)
        caption_ids.append(self.stoi["<end>"])
        return torch.tensor(caption_ids)
