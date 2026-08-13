"""Toy InfoNCE example for Stage 1 multimodal alignment.

This script uses fake image/text embeddings to show the core CLIP-style idea:
matched image-text pairs should sit on the diagonal of the similarity matrix.
"""

import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)

    image_embeddings = F.normalize(torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]), dim=-1)

    text_embeddings = F.normalize(torch.tensor([
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.2, 0.9],
    ]), dim=-1)

    scale = 10.0
    logits_per_image = scale * image_embeddings @ text_embeddings.T
    logits_per_text = logits_per_image.T

    labels = torch.arange(logits_per_image.size(0))
    image_to_text_loss = F.cross_entropy(logits_per_image, labels)
    text_to_image_loss = F.cross_entropy(logits_per_text, labels)
    symmetric_loss = (image_to_text_loss + text_to_image_loss) / 2

    print("Image embeddings shape:", tuple(image_embeddings.shape))
    print("Text embeddings shape: ", tuple(text_embeddings.shape))
    print()

    print("Similarity matrix: rows are images, columns are reports")
    print(logits_per_image.round(decimals=3))
    print()

    print("Correct labels:", labels.tolist())
    print("Interpretation: image_i should match report_i, so the correct class is the diagonal.")
    print()

    print(f"Image-to-text loss: {image_to_text_loss.item():.4f}")
    print(f"Text-to-image loss: {text_to_image_loss.item():.4f}")
    print(f"Symmetric loss:     {symmetric_loss.item():.4f}")
    print()

    image_predictions = logits_per_image.argmax(dim=1)
    text_predictions = logits_per_text.argmax(dim=1)
    print("Image-to-text predictions:", image_predictions.tolist())
    print("Text-to-image predictions:", text_predictions.tolist())

    print()
    print("--- Bad alignment example ---")
    bad_text_embeddings = text_embeddings[[1, 0, 2]]
    bad_logits_per_image = scale * image_embeddings @ bad_text_embeddings.T
    bad_logits_per_text = bad_logits_per_image.T
    bad_image_to_text_loss = F.cross_entropy(bad_logits_per_image, labels)
    bad_text_to_image_loss = F.cross_entropy(bad_logits_per_text, labels)
    bad_symmetric_loss = (bad_image_to_text_loss + bad_text_to_image_loss) / 2

    print("Similarity matrix after swapping report_0 and report_1")
    print(bad_logits_per_image.round(decimals=3))
    print(f"Bad image-to-text loss: {bad_image_to_text_loss.item():.4f}")
    print(f"Bad text-to-image loss: {bad_text_to_image_loss.item():.4f}")
    print(f"Bad symmetric loss:     {bad_symmetric_loss.item():.4f}")
    print("Bad image-to-text predictions:", bad_logits_per_image.argmax(dim=1).tolist())
    print("Bad text-to-image predictions:", bad_logits_per_text.argmax(dim=1).tolist())


if __name__ == "__main__":
    main()
