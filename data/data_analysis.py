import matplotlib.pyplot as plt
from collections import Counter
import nltk


def show_batch(train_loader, shared_vocab):
    images, captions, caplens = next(iter(train_loader))
    print(f"Batch Size: {len(images)}")
    print(f"Image Tensor Shape: {images.shape}")

    image = images[0].permute(1, 2, 0).numpy()
    image = image * 0.225 + 0.45
    plt.imshow(image)
    plt.title("First Image")
    plt.axis('off')
    plt.show()

    text_caption = shared_vocab.textualize(captions[0])
    print(f"First Caption Tensor: {captions[0]}")
    print(f"Converted: {text_caption}")
    print(f"Length: {caplens[0]}")


def show_sample(dataset, idx):
    image, caption, caption_length = dataset[idx]
    image = image.permute(1, 2, 0).numpy()
    image = image * 0.225 + 0.45
    text_caption = dataset.vocab.textualize(caption)
    all_caps = "\n".join(dataset.captions_dict[dataset.image_names[idx]])
    plt.imshow(image)
    plt.title(f'All captions:\n{all_caps}\n\nSampled: {text_caption} (len={caption_length})', fontsize=10)
    plt.axis('off')
    plt.show()


def analyze_caption_lengths(dataset):
    lengths = dataset.caption_lengths
    plt.hist(lengths, bins=20)
    plt.xlabel('Caption Length')
    plt.ylabel('Frequency')
    plt.title('Caption Length Distribution')
    plt.show()


def analyze_word_freq(dataset):
    nltk.download('punkt')
    all_captions = [caption for caption in dataset.captions]
    all_words = nltk.tokenize.word_tokenize(" ".join(all_captions).lower())
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    plt.bar(*zip(*top_words))
    plt.xticks(rotation=45)
    plt.title("Top 20 Words")
    plt.show()
