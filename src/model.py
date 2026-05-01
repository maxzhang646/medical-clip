import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedCLIP(nn.Module):
    def __init__(self, embed_dim: int = 512, freeze_image_layers: int = 8):
        super().__init__()
        self.image_encoder = self._build_image_encoder(freeze_image_layers)
        self.text_encoder  = self._build_text_encoder()
        self.image_proj    = nn.Linear(512, embed_dim)
        self.text_proj     = nn.Linear(768, embed_dim)
        self.logit_scale   = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def _build_image_encoder(self, freeze_layers: int):
        import clip
        model, _ = clip.load("ViT-B/32", device="cpu")
        encoder = model.visual
        # Freeze the first `freeze_layers` transformer blocks
        for i, block in enumerate(encoder.transformer.resblocks):
            if i < freeze_layers:
                for p in block.parameters():
                    p.requires_grad = False
        return encoder

    def _build_text_encoder(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained("medicalai/ClinicalBERT")

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(images).float()
        return F.normalize(self.image_proj(feats), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out  = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        feats = out.last_hidden_state[:, 0]  # [CLS] token
        return F.normalize(self.text_proj(feats), dim=-1)

    def forward(self, images, input_ids, attention_mask):
        img_emb  = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        scale    = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * img_emb @ text_emb.T
        logits_per_text  = logits_per_image.T
        return logits_per_image, logits_per_text
