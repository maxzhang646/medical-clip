"""BioMedCLIP wrapped in the MedCLIP interface so it can be fine-tuned on OpenI.

The point of this backbone is to fill the missing cell of the 2x2 comparison:

                        zero-shot        fine-tuned on OpenI
    CLIP + ClinicalBERT   done                 done
    BioMedCLIP            done                 this file

BioMedCLIP already ships aligned towers *and* trained projection heads, so this
wrapper deliberately adds no new projection layers -- a randomly initialised
Linear on top would destroy the pretrained alignment and invalidate the
comparison. `image_proj` / `text_proj` are Identity placeholders that exist only
to keep the parameter-group layout in `train.py` unchanged.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEFAULT_CONTEXT_LENGTH = 256


def _visual_blocks(visual) -> Optional[nn.ModuleList]:
    """Transformer blocks of the image tower (timm trunk or native open_clip)."""
    trunk = getattr(visual, "trunk", None)
    if trunk is not None and hasattr(trunk, "blocks"):
        return trunk.blocks
    transformer = getattr(visual, "transformer", None)
    if transformer is not None and hasattr(transformer, "resblocks"):
        return transformer.resblocks
    return None


def _text_layers(text) -> Optional[nn.ModuleList]:
    """Encoder layers of the HuggingFace text tower inside open_clip."""
    transformer = getattr(text, "transformer", None)
    encoder = getattr(transformer, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return encoder.layer
    return None


class BioMedCLIPFinetune(nn.Module):
    """open_clip BioMedCLIP exposing the same interface as `MedCLIP`."""

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 freeze_image_layers: int = 8, freeze_text_layers: int = 0):
        super().__init__()
        import open_clip

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.model = model
        self.model_name = model_name
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        # Placeholders: BioMedCLIP's own projections live inside the towers.
        self.image_proj = nn.Identity()
        self.text_proj = nn.Identity()

        self.frozen_image_layers = self._freeze(_visual_blocks(model.visual), freeze_image_layers,
                                                "image")
        self.frozen_text_layers = self._freeze(_text_layers(model.text), freeze_text_layers,
                                               "text")

    @staticmethod
    def _freeze(blocks, n_layers: int, tower: str) -> int:
        if n_layers <= 0:
            return 0
        if blocks is None:
            raise RuntimeError(
                f"Could not locate {tower} tower blocks to freeze. "
                "The open_clip model layout changed; update _visual_blocks/_text_layers."
            )
        n_frozen = min(n_layers, len(blocks))
        for block in list(blocks)[:n_frozen]:
            for p in block.parameters():
                p.requires_grad = False
        return n_frozen

    # `train.py` groups parameters by these names; expose them as properties so
    # the modules are not registered twice in the module tree / state_dict.
    @property
    def image_encoder(self) -> nn.Module:
        return self.model.visual

    @property
    def text_encoder(self) -> nn.Module:
        return self.model.text

    @property
    def logit_scale(self) -> nn.Parameter:
        return self.model.logit_scale

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(images), dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # open_clip's HF text tower derives its own attention mask from pad ids;
        # `attention_mask` is accepted only to match the MedCLIP signature.
        return F.normalize(self.model.encode_text(input_ids), dim=-1)

    def forward(self, images, input_ids, attention_mask=None):
        img_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * img_emb @ text_emb.T
        return logits_per_image, logits_per_image.T


def build_tokenize_fn(model_name: str = DEFAULT_MODEL,
                      context_length: int = DEFAULT_CONTEXT_LENGTH):
    """Return a `tokenize_fn` for OpenIDataset backed by BioMedCLIP's tokenizer."""
    import open_clip

    tokenizer = open_clip.get_tokenizer(model_name)
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    pad_id = getattr(hf_tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    def tokenize(caption) -> dict[str, torch.Tensor]:
        """A single caption returns 1-D tensors (dataset use); a list keeps the batch
        dimension (prompt encoding for zero-shot)."""
        is_batch = not isinstance(caption, str)
        texts = list(caption) if is_batch else caption
        try:
            ids = tokenizer(texts, context_length=context_length)
        except TypeError:  # older open_clip signatures
            ids = tokenizer(texts)
        if not is_batch and ids.dim() == 2:
            ids = ids[0]
        return {"input_ids": ids, "attention_mask": (ids != pad_id).long()}

    return tokenize
