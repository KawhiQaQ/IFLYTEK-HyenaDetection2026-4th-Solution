from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


BIOCLIP2_MODEL_NAME = "imageomics_bioclip2_vitl14"
BIOCLIP2_PRETRAINING_SOURCE = "https://huggingface.co/imageomics/bioclip-2"
BIOCLIP2_PRETRAINING_REVISION = "2957b322090f9cb17ae72c71981c7218a28d81e0"
BIOCLIP2_PRETRAINING_SHA256 = (
    "b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef"
)


class BioClip2DenseVision(nn.Module):
    """Visual-only BioCLIP 2 ViT-L/14 in its projected token space.

    OpenCLIP performs the strict released-checkpoint load and positional
    interpolation.  The text tower is discarded immediately.  Applying the
    released visual projection token-wise retains the contrastive visual space
    for both CLS and local patch observations.
    """

    embed_dim = 768
    transformer_width = 1024
    num_prefix_tokens = 1

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        image_size: int,
        grad_checkpointing: bool,
    ) -> None:
        super().__init__()
        if int(image_size) % 14:
            raise ValueError("BioCLIP 2 input must be divisible by patch size 14")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("V10.6 BioCLIP 2 requires open_clip") from exc

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        complete = open_clip.create_model(
            "ViT-L-14",
            pretrained=os.fspath(checkpoint_path),
            force_image_size=int(image_size),
            device="cpu",
        )
        visual = complete.visual
        grid = int(image_size) // 14
        if (
            tuple(visual.image_size) != (image_size, image_size)
            or tuple(visual.grid_size) != (grid, grid)
            or len(visual.transformer.resblocks) != 24
            or tuple(visual.positional_embedding.shape)
            != (1 + grid * grid, self.transformer_width)
            or tuple(visual.proj.shape) != (self.transformer_width, self.embed_dim)
        ):
            raise AssertionError("BioCLIP 2 visual geometry changed")

        complete_state = complete.state_dict()
        self.patch_embed = visual.conv1
        self.cls_token = visual.class_embedding
        self.pos_embed = visual.positional_embedding
        self.patch_drop = visual.patch_dropout
        self.norm_pre = visual.ln_pre
        self.blocks = visual.transformer.resblocks
        self.norm = visual.ln_post
        self.proj = visual.proj
        self.grad_checkpointing = bool(grad_checkpointing)
        self.image_size = int(image_size)
        self.grid_size = (grid, grid)
        self.visual_tensor_count = len(visual.state_dict())
        self.discarded_nonvisual_tensor_count = sum(
            not key.startswith("visual.") for key in complete_state
        )
        self.external_text_tower_retained = False
        self.projected_token_space = True
        del complete_state, complete, visual

    def set_grad_checkpointing(self, enabled: bool) -> None:
        self.grad_checkpointing = bool(enabled)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        tokens = tokens.permute(0, 2, 1)
        prefix = self.cls_token.to(tokens.dtype).reshape(1, 1, -1)
        tokens = torch.cat([prefix.expand(len(tokens), -1, -1), tokens], dim=1)
        if tokens.shape[1:] != self.pos_embed.shape:
            raise AssertionError(
                f"BioCLIP 2 token/position mismatch: {tokens.shape} / "
                f"{self.pos_embed.shape}"
            )
        tokens = self.norm_pre(tokens + self.pos_embed.to(tokens.dtype))
        tokens = self.patch_drop(tokens)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens @ self.proj.to(tokens.dtype)
