from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from captioning.generator import CaptionGenerator


def evaluate(encoder, decoder, data_loader, vocab, device, beam_size=5):
    caption_generator = CaptionGenerator(encoder, decoder, vocab, device=device)

    encoder.eval()
    decoder.eval()

    references = []
    hypotheses = []

    for i, (image, caps, caplens, allcaps) in enumerate(data_loader):
        image = image.to(device)
        allcaps = allcaps.to(device)

        seq, alphas = caption_generator.generate_beam_search(image.squeeze(0), beam_size=beam_size)

        pred = [vocab.itos[ind] for ind in seq if ind not in {vocab.stoi['<start>'], vocab.stoi['<pad>'], vocab.stoi['<end>']}]
        hypotheses.append(pred)

        img_refs = [
            [vocab.itos[ind] for ind in cap if ind not in {vocab.stoi['<start>'], vocab.stoi['<pad>'], vocab.stoi['<end>']}]
            for cap in allcaps[0].tolist()
        ]
        references.append(img_refs)

    assert len(references) == len(hypotheses), "Mismatch between references and hypotheses"

    bleu4 = corpus_bleu(references, hypotheses)
    print(f"\nBLEU-4: {bleu4:.4f}")

    joined_hypotheses = [' '.join(pred) for pred in hypotheses]
    joined_references = [[' '.join(ref) for ref in refs] for refs in references]

    coco_predictions = {idx: [cap] for idx, cap in enumerate(joined_hypotheses)}
    coco_references = {idx: refs for idx, refs in enumerate(joined_references)}

    cider_score, _ = Cider().compute_score(coco_references, coco_predictions)
    print(f"CIDEr: {cider_score:.4f}")

    meteor_score, _ = Meteor().compute_score(coco_references, coco_predictions)
    print(f"METEOR: {meteor_score:.4f}")

    spice_result = Spice().compute_score(coco_references, coco_predictions)
    spice_score = spice_result[0] if isinstance(spice_result[0], float) else spice_result[0].get("All", {}).get("f", 0.0)
    print(f"SPICE: {spice_score:.4f}")
