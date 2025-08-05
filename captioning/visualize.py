import matplotlib.pyplot as plt
import skimage.transform
import math
from PIL import Image
import os


def visualize_attention(image_path, seq, alphas, vocab, output_path=None, smooth=True, max_words=50):
    """
    Visualizes attention maps over image for each word in the caption.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [vocab.itos[ind] for ind in seq]
    num_words = min(len(words), max_words)
    fig, axes = plt.subplots(
        math.ceil(num_words / 5), 5,
        figsize=(15, math.ceil(num_words / 5) * 3)
    )
    axes = axes.flatten()

    for t, word in enumerate(words[:max_words]):
        ax = axes[t]
        ax.set_title(word, fontsize=10)
        ax.imshow(image)

        alpha = alphas[t].detach().cpu().numpy()
        alpha = skimage.transform.pyramid_expand(alpha, upscale=24, sigma=8) if smooth \
            else skimage.transform.resize(alpha, [14 * 24, 14 * 24])

        ax.imshow(alpha, cmap='Greys_r', alpha=0.8 if t != 0 else 0)
        ax.axis('off')

    for ax in axes[num_words:]:
        ax.axis('off')

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
