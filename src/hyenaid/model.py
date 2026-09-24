from __future__ import annotations

import copy
import functools
import math
import os
import sys

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50

from bioclip2_vision import (
    BIOCLIP2_MODEL_NAME,
    BIOCLIP2_PRETRAINING_SOURCE,
    BioClip2DenseVision,
)


MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
PRETRAINING_SOURCE = "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m"
LARGE_MODEL_NAME = "vit_large_patch16_dinov3.lvd1689m"
LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/vit_large_patch16_dinov3.lvd1689m"
)
HUGE_PLUS_MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
HUGE_PLUS_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "vit_huge_plus_patch16_dinov3.lvd1689m"
)


def configure_suffix_stochastic_depth(
    model: nn.Module,
    first_block: int,
    drop_probability: float,
) -> dict[str, object]:
    """Apply training-only stochastic depth to a ViT suffix.

    DINOv3-H+/16 is published with identity residual drop paths.  Replacing
    only those stateless modules does not add checkpoint tensors and is an
    exact identity in evaluation mode.
    """
    probability = float(drop_probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("Suffix stochastic-depth probability must be in (0, 1)")
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise TypeError("Suffix stochastic depth requires a transformer block list")
    if not 0 <= int(first_block) < len(blocks):
        raise ValueError("Suffix stochastic-depth boundary is outside the backbone")

    replaced: list[str] = []
    for block_index, block in enumerate(blocks):
        for residual_name in ("drop_path1", "drop_path2"):
            residual = getattr(block, residual_name, None)
            if not isinstance(residual, nn.Identity):
                raise AssertionError(
                    f"Published block {block_index}.{residual_name} is not Identity"
                )
            if block_index >= int(first_block):
                setattr(block, residual_name, timm.layers.DropPath(probability))
                replaced.append(f"backbone.blocks.{block_index}.{residual_name}")

    metadata: dict[str, object] = {
        "enabled": True,
        "first_block": int(first_block),
        "last_block": len(blocks) - 1,
        "drop_probability": probability,
        "residual_modules": len(replaced),
        "module_names": replaced,
        "checkpoint_state_keys": 0,
        "evaluation_identity": True,
    }
    setattr(model, "suffix_stochastic_depth_metadata", metadata)
    return metadata
EVA02_LARGE_MODEL_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
EVA02_LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
)
CONVNEXTV2_LARGE_MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"
CONVNEXTV2_LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "convnextv2_large.fcmae_ft_in22k_in1k_384"
)
SWINV2_LARGE_MODEL_NAME = (
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"
)
SWINV2_LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"
)
SWINV2_BASE_MODEL_NAME = (
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
)
SWINV2_BASE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k"
)
SIGLIP2_LARGE_MODEL_NAME = "vit_large_patch16_siglip_384.v2_webli"
SIGLIP2_LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "vit_large_patch16_siglip_384.v2_webli"
)
DINOV2_LARGE_REGISTER_MODEL_NAME = "vit_large_patch14_reg4_dinov2.lvd142m"
DINOV2_LARGE_REGISTER_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/"
    "vit_large_patch14_reg4_dinov2.lvd142m"
)
DINOV2_GIANT_REGISTER_MODEL_NAME = "vit_giant_patch14_reg4_dinov2"
DINOV2_GIANT_REGISTER_PRETRAINING_SOURCE = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/"
    "dinov2_vitg14_reg4_pretrain.pth"
)
DINOV2_GIANT_REGISTER_PRETRAINING_SHA256 = (
    "746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283"
)
LINGBOT_LARGE_MODEL_NAME = "robbyant/lingbot-vision-vit-large"
LINGBOT_LARGE_REVISION = "5e0370623d4fa5db945d00bc47a8545eed407d6b"
LINGBOT_LARGE_PRETRAINING_SOURCE = (
    "https://huggingface.co/robbyant/lingbot-vision-vit-large/tree/"
    f"{LINGBOT_LARGE_REVISION}"
)
CONVNEXT_MODEL_NAME = "convnext_base.dinov3_lvd1689m"
CONVNEXT_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/convnext_base.dinov3_lvd1689m"
)
CONVNEXT_LARGE_DINOV3_MODEL_NAME = "convnext_large.dinov3_lvd1689m"
CONVNEXT_LARGE_DINOV3_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/convnext_large.dinov3_lvd1689m"
)
CONVNEXT_LARGE_DINOV3_PRETRAINING_SHA256 = (
    "77f63fee584d2c38416b865ee4b1bf9d4416df93a44ef641c2675de08d538a7b"
)
EFFICIENTNET_MODEL_NAME = "efficientnetv2_rw_m.agc_in1k"
EFFICIENTNET_PRETRAINING_SOURCE = (
    "https://huggingface.co/timm/efficientnetv2_rw_m.agc_in1k"
)
ARBASE_MODEL_NAME = "resnet50_ibn_a_mgn"
ARBASE_PRETRAINING_SOURCE = (
    "https://github.com/XingangPan/IBN-Net/releases/download/v1.0/"
    "resnet50_ibn_a-d9d0bb7b.pth"
)
PETFACE_R50_PRETRAINING_SOURCE = (
    "https://drive.google.com/file/d/"
    "1AtJDV8cuB9IRKPL9nYooCA2PcTWacI6p/view"
)
PETFACE_R50_PRETRAINING_SHA256 = (
    "b836d6a15e596069594940d9e3ebe6c50870e67295a64ccf2db4b413d29b83bb"
)
PETFACE_R50_EXTERNAL_CLASS_COUNT = 175_081
BIOCLIP_MODEL_NAME = "imageomics_bioclip_vitb16"
BIOCLIP_PRETRAINING_SOURCE = "https://huggingface.co/imageomics/bioclip"
BIOCLIP_PRETRAINING_SHA256 = (
    "e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4"
)
RADIO_V25_B_MODEL_NAME = "radio_v2.5_b_multiteacher"
RADIO_V25_B_PRETRAINING_SOURCE = (
    "https://github.com/NVlabs/RADIO/tree/main#radio-v2.5-b"
)
RADIO_V25_B_PRETRAINING_SHA256 = (
    "6bff4bd732d815136652d454598e8fe6c6f4e658716d5e6a697f0e6b60bd8a98"
)
SAM2_HIERA_SMALL_MODEL_NAME = "sam2_hiera_small"
SAM2_HIERA_SMALL_PRETRAINING_SOURCE = (
    "https://github.com/facebookresearch/sam2#model-description"
)
SAM2_HIERA_SMALL_PRETRAINING_SHA256 = (
    "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
)
TIPS_L14_HR_MODEL_NAME = "tips_v1_l14_highres_distilled_vision"
TIPS_L14_HR_PRETRAINING_SOURCE = (
    "https://storage.googleapis.com/tips_data/v1_0/checkpoints/pytorch/"
    "tips_oss_l14_highres_distilled_vision.npz"
)
TIPS_L14_HR_PRETRAINING_SHA256 = (
    "3421b4e0d473cafe5e9cdbd2bf05b5c346270b6be05a27b0af7eca0734639c80"
)


class TipsL14HighResDenseVision(nn.Module):
    """Exact TIPS-v1 L/14-HR visual state exposed as dense ViT tokens."""

    embed_dim = 1024
    num_prefix_tokens = 2
    checkpoint_tensor_count = 344
    checkpoint_value_count = 304_016_384

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        image_size: int,
        grad_checkpointing: bool,
    ) -> None:
        super().__init__()
        if image_size != 448:
            raise ValueError("V12.15 TIPS L/14-HR is locked to 448 input")
        # This timm graph is state-dictionary equivalent to DeepMind's public
        # PyTorch implementation. TIPS positions CLS+patches first, then
        # inserts the second CLS/register token; custom forward_features below
        # preserves that published ordering exactly.
        graph = timm.models.vision_transformer.VisionTransformer(
            img_size=image_size,
            patch_size=14,
            num_classes=0,
            global_pool="",
            embed_dim=self.embed_dim,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            proj_bias=True,
            init_values=1.0,
            class_token=True,
            reg_tokens=0,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        )
        self.patch_embed = graph.patch_embed
        self.cls_token = graph.cls_token
        self.pos_embed = graph.pos_embed
        self.blocks = graph.blocks
        self.norm = graph.norm
        self.register_tokens = nn.Parameter(torch.empty(1, 1, self.embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, self.embed_dim))
        # An explicit stateless hook point allows the established fold-train
        # crop-geometry conditioner without changing a public checkpoint key.
        self.norm_pre = nn.Identity()
        del graph

        source = np.load(os.fspath(checkpoint_path), allow_pickle=False)
        source_keys = set(source.files)
        target_state = self.state_dict()
        if (
            len(source_keys) != self.checkpoint_tensor_count
            or sum(int(source[name].size) for name in source.files)
            != self.checkpoint_value_count
            or source_keys != set(target_state)
        ):
            raise AssertionError("TIPS L/14-HR checkpoint allowlist changed")
        shape_mismatches = {
            name
            for name in source.files
            if tuple(source[name].shape) != tuple(target_state[name].shape)
        }
        if shape_mismatches:
            raise AssertionError(
                f"TIPS L/14-HR state geometry changed: {sorted(shape_mismatches)}"
            )
        state = {
            name: torch.from_numpy(np.array(source[name], copy=True))
            for name in source.files
        }
        self.load_state_dict(state, strict=True)
        source.close()
        del source, state, target_state

        if (
            len(self.blocks) != 24
            or tuple(self.patch_embed.grid_size) != (32, 32)
            or tuple(self.pos_embed.shape) != (1, 1025, self.embed_dim)
            or self.norm.eps != 1e-6
        ):
            raise AssertionError("TIPS L/14-HR public graph changed")
        self.image_size = int(image_size)
        self.grid_size = (32, 32)
        self.pretraining_sha256 = TIPS_L14_HR_PRETRAINING_SHA256
        self.external_text_tower_retained = False
        self.external_class_state_loaded = False
        self.grad_checkpointing = bool(grad_checkpointing)
        self.last_token_shape: tuple[int, ...] | None = None

    def set_grad_checkpointing(self, enabled: bool) -> None:
        self.grad_checkpointing = bool(enabled)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[-2:]) != (448, 448):
            raise AssertionError("TIPS L/14-HR input geometry changed")
        if not torch.isfinite(images).all():
            raise FloatingPointError("TIPS input is non-finite")
        if float(images.detach().amin()) < -1e-6 or float(
            images.detach().amax()
        ) > 1.000001:
            raise AssertionError("TIPS requires its published [0,1] RGB input")
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(len(images), -1, -1)
        tokens = torch.cat((cls, patches), dim=1)
        if tokens.shape[1:] != self.pos_embed.shape[1:]:
            raise AssertionError("TIPS patch/position geometry changed")
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype)
        register = self.register_tokens.expand(len(images), -1, -1)
        tokens = torch.cat((tokens[:, :1], register, tokens[:, 1:]), dim=1)
        tokens = self.norm_pre(tokens)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        self.last_token_shape = tuple(tokens.shape)
        # Both published CLS embeddings are official inference outputs. Their
        # fixed mean is exposed as the semantic global token while token 1 is
        # retained only as a prefix, so all 1024 dense patches remain aligned.
        global_token = 0.5 * (tokens[:, 0] + tokens[:, 1])
        return torch.cat((global_token[:, None], tokens[:, 1:]), dim=1)


class RadioV25DenseVision(nn.Module):
    """Official RADIO-v2.5-B backbone exposed as dense timm-like tokens.

    The official wrapper owns its published input conditioner and receives
    RGB tensors in [0, 1].  RADIO provides three 768-D summary slots (CLIP,
    SigLIP and DINOv2 teachers) plus the shared dense patch map.  Their fixed
    arithmetic mean supplies the one global token expected by the mature MGN
    graph; no target-trained projection or external taxonomy head is retained.
    """

    embed_dim = 768
    num_prefix_tokens = 1

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        image_size: int,
        grad_checkpointing: bool,
    ) -> None:
        super().__init__()
        if image_size % 16:
            raise ValueError("RADIO-v2.5-B image size must be divisible by 16")
        repo = os.environ.get("RADIO_REPO_PATH")
        if not repo or not os.path.isdir(repo):
            raise RuntimeError("RADIO_REPO_PATH must point to the pinned official repo")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from hubconf import radio_model
        except ImportError as exc:
            raise ImportError("V12.1 requires the pinned official RADIO source") from exc
        radio, checkpoint = radio_model(
            version=os.fspath(checkpoint_path),
            progress=False,
            return_checkpoint=True,
        )
        if (
            checkpoint.get("arch") != "vit_base_patch16_224"
            or int(checkpoint.get("version", -1)) != 1
            or int(radio.embed_dim) != self.embed_dim
            or int(radio.patch_size) != 16
            or int(radio.num_cls_tokens) != 4
            or tuple(radio.summary_idxs.tolist()) != (0, 1, 2)
            or len(radio.blocks) != 12
        ):
            raise AssertionError("RADIO-v2.5-B checkpoint/interface changed")
        self.radio = radio
        # The mature geometry side information is injected after RADIO's
        # published final representation through this explicit hook point.
        self.norm_pre = nn.Identity()
        self.image_size = int(image_size)
        self.grid_size = (image_size // 16, image_size // 16)
        self.pretraining_sha256 = RADIO_V25_B_PRETRAINING_SHA256
        self.external_teacher_heads_retained = False
        self.set_grad_checkpointing(grad_checkpointing)

    @property
    def blocks(self):
        return self.radio.blocks

    def set_grad_checkpointing(self, enabled: bool) -> None:
        setter = getattr(self.radio.model, "set_grad_checkpointing", None)
        if callable(setter):
            setter(bool(enabled))
        else:
            self.radio.model.grad_checkpointing = bool(enabled)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[-2:] != (
            self.image_size,
            self.image_size,
        ):
            raise AssertionError("RADIO input geometry changed")
        if not torch.isfinite(images).all():
            raise FloatingPointError("RADIO input is non-finite")
        # The loader contract, not a clamp, guarantees [0, 1].
        if float(images.detach().amin()) < -1e-6 or float(images.detach().amax()) > 1.000001:
            raise AssertionError("RADIO received an already-normalized image")
        output = self.radio(images, feature_fmt="NLC")
        summary = output.summary.reshape(len(images), 3, self.embed_dim).mean(dim=1)
        features = output.features
        expected_patches = self.grid_size[0] * self.grid_size[1]
        if tuple(features.shape[1:]) != (expected_patches, self.embed_dim):
            raise AssertionError("RADIO dense feature geometry changed")
        return self.norm_pre(torch.cat((summary[:, None, :], features), dim=1))


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        power = self.p.clamp(1.0, 6.0)
        return (
            features.clamp_min(self.eps)
            .pow(power)
            .mean(dim=(-2, -1))
            .pow(power.reciprocal())
        )


class BioClipDenseVision(nn.Module):
    """Visual-only BioCLIP ViT-B/16 with dense final tokens.

    OpenCLIP performs the published checkpoint load and positional-embedding
    interpolation.  This wrapper deliberately retains no text tower or
    external output head and exposes timm-like names used by the existing
    MGN optimizer/freeze machinery.
    """

    embed_dim = 768
    num_prefix_tokens = 1

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        image_size: int,
        grad_checkpointing: bool,
    ) -> None:
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("V10.4 BioCLIP requires the open_clip package") from exc
        complete = open_clip.create_model(
            "ViT-B-16",
            pretrained=os.fspath(checkpoint_path),
            force_image_size=int(image_size),
            device="cpu",
        )
        visual = complete.visual
        if (
            tuple(visual.image_size) != (image_size, image_size)
            or tuple(visual.grid_size) != (image_size // 16, image_size // 16)
            or len(visual.transformer.resblocks) != 12
            or tuple(visual.positional_embedding.shape)
            != (1 + (image_size // 16) ** 2, self.embed_dim)
        ):
            raise AssertionError("BioCLIP visual geometry changed")
        self.patch_embed = visual.conv1
        self.cls_token = visual.class_embedding
        self.pos_embed = visual.positional_embedding
        self.patch_drop = visual.patch_dropout
        self.norm_pre = visual.ln_pre
        self.blocks = visual.transformer.resblocks
        self.norm = visual.ln_post
        self.grad_checkpointing = bool(grad_checkpointing)
        self.image_size = int(image_size)
        self.grid_size = (image_size // 16, image_size // 16)
        self.visual_tensor_count = len(visual.state_dict())
        self.external_text_tower_retained = False
        del complete, visual

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
                f"BioCLIP token/position mismatch: {tokens.shape} / "
                f"{self.pos_embed.shape}"
            )
        tokens = self.norm_pre(tokens + self.pos_embed.to(tokens.dtype))
        tokens = self.patch_drop(tokens)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.norm(tokens)

class ResidualPartAdapter(nn.Module):
    """Zero-initialized low-rank residual that specializes one crop domain."""

    def __init__(self, feature_dim: int, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_dim, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(self.norm(features))))


class ParallelAdaptMlp(nn.Module):
    """AdaptFormer branch parallel to one frozen transformer MLP."""

    def __init__(
        self,
        base_mlp: nn.Module,
        feature_dim: int,
        bottleneck_dim: int = 64,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.adapter_down = nn.Linear(feature_dim, bottleneck_dim, bias=True)
        self.adapter_activation = nn.ReLU()
        self.adapter_up = nn.Linear(bottleneck_dim, feature_dim, bias=True)
        self.scale = float(scale)
        nn.init.kaiming_normal_(self.adapter_down.weight, nonlinearity="relu")
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter_up(
            self.adapter_activation(self.adapter_down(features))
        )
        return self.base_mlp(features) + self.scale * adapted

    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.adapter_down.parameters()) + list(
            self.adapter_up.parameters()
        )


class AdaptFormerBranch(nn.Module):
    """Standalone AdaptFormer residual for an alternate representation path."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 64,
        scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter_down = nn.Linear(feature_dim, bottleneck_dim, bias=True)
        self.adapter_activation = nn.ReLU()
        self.adapter_up = nn.Linear(bottleneck_dim, feature_dim, bias=True)
        self.scale = float(scale)
        nn.init.kaiming_normal_(self.adapter_down.weight, nonlinearity="relu")
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scale * self.adapter_up(
            self.adapter_activation(self.adapter_down(features))
        )


class QvLowRankLinear(nn.Module):
    """Frozen fused QKV projection with zero-start query/value LoRA paths."""

    def __init__(
        self,
        base_qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError("Q/V LoRA requires a fused nn.Linear QKV projection")
        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError("Q/V LoRA requires fused [Q, K, V] output slices")
        if rank < 1 or alpha <= 0.0:
            raise ValueError("Q/V LoRA rank and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Q/V LoRA dropout must be in [0, 1)")
        self.base_qkv = base_qkv
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout_probability = float(dropout)
        self.adapter_dropout = nn.Dropout(self.dropout_probability)
        width = int(base_qkv.in_features)
        self.query_down = nn.Linear(width, self.rank, bias=False)
        self.query_up = nn.Linear(self.rank, width, bias=False)
        self.value_down = nn.Linear(width, self.rank, bias=False)
        self.value_up = nn.Linear(self.rank, width, bias=False)
        nn.init.kaiming_uniform_(self.query_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.value_up.weight)
        self.audit_residual = False
        self.last_residual_max_abs: torch.Tensor | None = None

    def adapter_forward(self, features: torch.Tensor) -> torch.Tensor:
        reduced = self.adapter_dropout(features)
        query = self.query_up(self.query_down(reduced)) * self.scale
        value = self.value_up(self.value_down(reduced)) * self.scale
        key = torch.zeros_like(query)
        return torch.cat((query, key, value), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.adapter_forward(features)
        if self.audit_residual:
            self.last_residual_max_abs = residual.detach().abs().amax()
        return self.base_qkv(features) + residual

    def adapter_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.query_down.parameters())
            + list(self.query_up.parameters())
            + list(self.value_down.parameters())
            + list(self.value_up.parameters())
        )


class HeadRoutedQvLowRankLinear(nn.Module):
    """Fused QKV plus a zero-start Q/V LoRA active only on head rows."""

    def __init__(
        self,
        base_qkv: nn.Linear,
        routing_context: PartRoutingContext,
        rank: int = 8,
        alpha: float = 8.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError("Head-routed Q/V LoRA requires fused nn.Linear QKV")
        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError("Head-routed Q/V LoRA requires fused [Q, K, V]")
        if rank < 1 or alpha <= 0.0:
            raise ValueError("Head-routed Q/V LoRA rank and alpha must be positive")
        self.base_qkv = base_qkv
        self.routing_context = routing_context
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        width = int(base_qkv.in_features)
        self.query_down = nn.Linear(width, self.rank, bias=False)
        self.query_up = nn.Linear(self.rank, width, bias=False)
        self.value_down = nn.Linear(width, self.rank, bias=False)
        self.value_up = nn.Linear(self.rank, width, bias=False)
        nn.init.normal_(self.query_down.weight, std=0.02)
        nn.init.normal_(self.value_down.weight, std=0.02)
        nn.init.zeros_(self.query_up.weight)
        nn.init.zeros_(self.value_up.weight)
        self.enabled = True
        self.audit_residual = False
        self.last_residual_max_abs: torch.Tensor | None = None
        self.last_body_residual_max_abs: torch.Tensor | None = None

    def adapter_forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return features.new_zeros(
                (*features.shape[:-1], 3 * self.base_qkv.in_features)
            )
        parts = self.routing_context.parts
        if parts is None or parts.ndim != 1 or len(parts) != len(features):
            raise RuntimeError("Head-routed Q/V LoRA has no aligned part metadata")
        query = self.query_up(self.query_down(features)) * self.scale
        value = self.value_up(self.value_down(features)) * self.scale
        head = parts.eq(0).to(dtype=query.dtype).view(-1, 1, 1)
        query = query * head
        value = value * head
        key = torch.zeros_like(query)
        residual = torch.cat((query, key, value), dim=-1)
        if self.audit_residual:
            self.last_residual_max_abs = residual.detach().abs().amax()
            body = parts.ne(0)
            self.last_body_residual_max_abs = (
                residual[body].detach().abs().amax()
                if body.any()
                else residual.new_zeros(())
            )
        return residual

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_qkv(features) + self.adapter_forward(features)

    def adapter_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.query_down.parameters())
            + list(self.query_up.parameters())
            + list(self.value_down.parameters())
            + list(self.value_up.parameters())
        )


class PartRoutingContext:
    """Transient official-part metadata shared by routed prefix adapters."""

    def __init__(self) -> None:
        self.parts: torch.Tensor | None = None


class PartRoutedQvLowRankLinear(nn.Module):
    """Frozen fused QKV plus zero-start shared and official-part LoRA paths."""

    def __init__(
        self,
        base_qkv: nn.Linear,
        routing_context: PartRoutingContext,
        shared_rank: int = 4,
        part_rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.05,
        num_parts: int = 3,
    ) -> None:
        super().__init__()
        if not isinstance(base_qkv, nn.Linear):
            raise TypeError("Part-routed Q/V LoRA requires a fused nn.Linear QKV")
        if base_qkv.out_features != 3 * base_qkv.in_features:
            raise ValueError("Part-routed Q/V LoRA requires fused [Q, K, V]")
        if shared_rank < 1 or part_rank < 1 or alpha <= 0.0:
            raise ValueError("Part-routed Q/V LoRA ranks and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Part-routed Q/V LoRA dropout must be in [0, 1)")
        if num_parts < 1:
            raise ValueError("Part-routed Q/V LoRA requires at least one part")
        self.base_qkv = base_qkv
        self.routing_context = routing_context
        self.shared_rank = int(shared_rank)
        self.part_rank = int(part_rank)
        self.alpha = float(alpha)
        self.shared_scale = self.alpha / self.shared_rank
        self.part_scale = self.alpha / self.part_rank
        self.dropout_probability = float(dropout)
        self.num_parts = int(num_parts)
        self.adapter_dropout = nn.Dropout(self.dropout_probability)
        width = int(base_qkv.in_features)

        self.query_down = nn.Linear(width, self.shared_rank, bias=False)
        self.query_up = nn.Linear(self.shared_rank, width, bias=False)
        self.value_down = nn.Linear(width, self.shared_rank, bias=False)
        self.value_up = nn.Linear(self.shared_rank, width, bias=False)
        self.query_part_down = nn.ModuleList(
            nn.Linear(width, self.part_rank, bias=False)
            for _ in range(self.num_parts)
        )
        self.query_part_up = nn.ModuleList(
            nn.Linear(self.part_rank, width, bias=False)
            for _ in range(self.num_parts)
        )
        self.value_part_down = nn.ModuleList(
            nn.Linear(width, self.part_rank, bias=False)
            for _ in range(self.num_parts)
        )
        self.value_part_up = nn.ModuleList(
            nn.Linear(self.part_rank, width, bias=False)
            for _ in range(self.num_parts)
        )

        for down in (
            self.query_down,
            self.value_down,
            *self.query_part_down,
            *self.value_part_down,
        ):
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
        for up in (
            self.query_up,
            self.value_up,
            *self.query_part_up,
            *self.value_part_up,
        ):
            nn.init.zeros_(up.weight)
        self.audit_residual = False
        self.last_residual_max_abs: torch.Tensor | None = None

    def _parts_for(
        self, features: torch.Tensor, parts: torch.Tensor | None
    ) -> torch.Tensor:
        selected = self.routing_context.parts if parts is None else parts
        if selected is None:
            raise RuntimeError("Part-routed Q/V LoRA has no active part metadata")
        if selected.ndim != 1 or len(selected) != len(features):
            raise ValueError("Part-routed Q/V LoRA part batch is misaligned")
        return selected

    def adapter_forward(
        self, features: torch.Tensor, parts: torch.Tensor | None = None
    ) -> torch.Tensor:
        selected_parts = self._parts_for(features, parts)
        reduced = self.adapter_dropout(features)
        query = self.query_up(self.query_down(reduced)) * self.shared_scale
        value = self.value_up(self.value_down(reduced)) * self.shared_scale
        for part_index in range(self.num_parts):
            mask = selected_parts.eq(part_index)
            query[mask] = query[mask] + self.query_part_up[part_index](
                self.query_part_down[part_index](reduced[mask])
            ) * self.part_scale
            value[mask] = value[mask] + self.value_part_up[part_index](
                self.value_part_down[part_index](reduced[mask])
            ) * self.part_scale
        key = torch.zeros_like(query)
        return torch.cat((query, key, value), dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.adapter_forward(features)
        if self.audit_residual:
            self.last_residual_max_abs = residual.detach().abs().amax()
        return self.base_qkv(features) + residual

    def adapter_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.query_down,
            self.query_up,
            self.value_down,
            self.value_up,
            *self.query_part_down,
            *self.query_part_up,
            *self.value_part_down,
            *self.value_part_up,
        )
        return [parameter for module in modules for parameter in module.parameters()]


class PartRoutedMlpExperts(nn.Module):
    """Full-width deterministic MLP experts routed by the official crop part."""

    def __init__(
        self,
        base_mlp: nn.Module,
        routing_context: PartRoutingContext,
        num_parts: int = 3,
    ) -> None:
        super().__init__()
        self.routing_context = routing_context
        self.experts = nn.ModuleList(
            [base_mlp]
            + [copy.deepcopy(base_mlp) for _ in range(int(num_parts) - 1)]
        )
        if len(self.experts) != 3:
            raise ValueError("Hyena crop routing requires exactly three experts")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = self.routing_context.parts
        if parts is None or len(parts) != len(features):
            raise AssertionError("MLP expert part-routing context is missing")
        if torch.any((parts < 0) | (parts >= len(self.experts))):
            raise AssertionError("MLP expert received an invalid part index")

        routed_indices: list[torch.Tensor] = []
        routed_outputs: list[torch.Tensor] = []
        for part_index, expert in enumerate(self.experts):
            indices = torch.nonzero(parts.eq(part_index), as_tuple=False).flatten()
            if len(indices) == 0:
                continue
            routed_indices.append(indices)
            routed_outputs.append(expert(features.index_select(0, indices)))
        if not routed_outputs:
            raise AssertionError("MLP expert router received an empty batch")

        grouped_indices = torch.cat(routed_indices, dim=0)
        grouped_outputs = torch.cat(routed_outputs, dim=0)
        restore_order = torch.argsort(grouped_indices)
        return grouped_outputs.index_select(0, restore_order)

    def extra_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for expert in self.experts[1:]
            for parameter in expert.parameters()
        )


class CropAwareA2GCAggregator(nn.Module):
    """Crop-aware asymmetric OT aggregation of dense identity patterns."""

    def __init__(
        self,
        feature_dim: int,
        num_parts: int = 3,
        num_clusters: int = 64,
        local_dim: int = 128,
        global_dim: int = 256,
        geometry_dim: int = 16,
        dropout: float = 0.3,
        transport_iterations: int = 3,
        transport_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_parts = int(num_parts)
        self.num_clusters = int(num_clusters)
        self.local_dim = int(local_dim)
        self.global_dim = int(global_dim)
        self.geometry_dim = int(geometry_dim)
        self.transport_iterations = int(transport_iterations)
        self.transport_temperature = float(transport_temperature)
        if self.num_parts != 3:
            raise ValueError("Hyena A2GC aggregation requires exactly three parts")
        if self.num_clusters < 1 or self.local_dim < 1 or self.geometry_dim < 1:
            raise ValueError("A2GC dimensions must be positive")
        if self.transport_iterations < 1 or self.transport_temperature <= 0.0:
            raise ValueError("A2GC transport constants must be positive")

        drop = nn.Dropout(float(dropout))
        self.local_projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            drop,
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.local_dim, kernel_size=1),
        )
        self.assignment_hidden = nn.Sequential(
            nn.Conv2d(self.feature_dim, 512, kernel_size=1),
            nn.Dropout(float(dropout)),
            nn.ReLU(inplace=True),
        )
        self.assignment_output = nn.Conv2d(
            512,
            self.num_parts * self.num_clusters,
            kernel_size=1,
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.global_dim),
        )
        self.coordinate_projection = nn.Conv2d(
            2, self.geometry_dim, kernel_size=1
        )
        self.cluster_geometry = nn.Parameter(
            torch.empty(self.num_parts, self.num_clusters, self.geometry_dim)
        )
        nn.init.normal_(self.cluster_geometry, std=0.02)
        self.geometry_weight = nn.Parameter(
            torch.full((self.num_parts,), 0.15)
        )
        self.dustbin_score = nn.Parameter(torch.ones(self.num_parts))
        self.output_dim = self.num_clusters * self.local_dim + self.global_dim

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mirror_x: bool = False,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        if mirror_x:
            x = -x
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0)

    def geometry_scores(
        self,
        parts: torch.Tensor,
        height: int,
        width: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        mirror_x: bool = False,
    ) -> torch.Tensor:
        """Return part-routed cluster/patch geometry compatibility scores."""
        if torch.any((parts < 0) | (parts >= self.num_parts)):
            raise AssertionError("A2GC received an invalid part index")
        grid = self._coordinate_grid(
            height,
            width,
            device=device,
            dtype=dtype,
            mirror_x=mirror_x,
        ).expand(len(parts), -1, -1, -1).clone()
        body = parts.ne(0)
        if body.any():
            grid[body, 0] = grid[body, 0].abs()
        embedded = self.coordinate_projection(grid).flatten(2)
        cluster_geometry = self.cluster_geometry[parts].to(dtype=embedded.dtype)
        return torch.einsum("bgn,bkg->bkn", embedded, cluster_geometry)

    def asymmetric_transport(
        self,
        scores: torch.Tensor,
        dustbin_score: torch.Tensor,
    ) -> torch.Tensor:
        """Published row/column-average OT with separate marginal calibration."""
        batch, clusters, tokens = scores.shape
        if clusters != self.num_clusters or dustbin_score.shape != (batch,):
            raise AssertionError("A2GC transport input shape mismatch")
        # The transport normalization is deliberately FP32 even under AMP.
        scores = scores.float() / self.transport_temperature
        dustbin = dustbin_score.float().view(batch, 1, 1).expand(-1, 1, tokens)
        log_transport = torch.cat((scores, dustbin), dim=1)
        for _ in range(self.transport_iterations):
            row_normalized = log_transport - torch.logsumexp(
                log_transport, dim=2, keepdim=True
            )
            column_normalized = log_transport - torch.logsumexp(
                log_transport, dim=1, keepdim=True
            )
            log_transport = 0.5 * (row_normalized + column_normalized)

        total_mass = float(tokens + clusters)
        log_source = log_transport.new_full(
            (batch, clusters + 1), -math.log(total_mass)
        )
        dustbin_mass = max(tokens - clusters, 1)
        log_source[:, -1] = math.log(dustbin_mass) - math.log(total_mass)
        log_target = log_transport.new_full(
            (batch, tokens), -math.log(total_mass)
        )
        source_correction = log_source - torch.logsumexp(
            log_transport, dim=2
        )
        log_transport = log_transport + source_correction.unsqueeze(2)
        target_correction = log_target - torch.logsumexp(
            log_transport, dim=1
        )
        log_transport = log_transport + target_correction.unsqueeze(1)
        return log_transport.exp()

    def forward(
        self,
        patch_map: torch.Tensor,
        global_token: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = patch_map.shape
        if channels != self.feature_dim or len(parts) != batch:
            raise AssertionError("A2GC feature or part shape mismatch")
        local = self.local_projection(patch_map).flatten(2)
        assignment = self.assignment_output(
            self.assignment_hidden(patch_map)
        ).reshape(
            batch,
            self.num_parts,
            self.num_clusters,
            height * width,
        )
        row = torch.arange(batch, device=parts.device)
        assignment = assignment[row, parts]
        geometry = self.geometry_scores(
            parts,
            height,
            width,
            dtype=patch_map.dtype,
            device=patch_map.device,
        )
        assignment = assignment + (
            self.geometry_weight[parts, None, None].to(assignment.dtype)
            * geometry.to(assignment.dtype)
        )
        transport = self.asymmetric_transport(
            assignment, self.dustbin_score[parts]
        )
        real_transport = transport[:, :-1]
        aggregated = torch.einsum(
            "bdn,bkn->bdk", local.float(), real_transport
        )
        aggregated = F.normalize(aggregated, p=2, dim=1).flatten(1)
        global_descriptor = F.normalize(
            self.global_projection(global_token).float(), p=2, dim=-1
        )
        return F.normalize(
            torch.cat((global_descriptor, aggregated), dim=-1),
            p=2,
            dim=-1,
        )


class HierarchicalPartSlotAggregator(nn.Module):
    """Part-conditioned competitive slots over intermediate and final patches."""

    def __init__(
        self,
        feature_dim: int,
        num_parts: int = 3,
        num_identity_slots: int = 4,
        slot_dim: int = 512,
        iterations: int = 3,
        epsilon: float = 1e-8,
        reconstruction_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_parts = int(num_parts)
        self.num_identity_slots = int(num_identity_slots)
        self.num_slots = self.num_identity_slots + 1
        self.slot_dim = int(slot_dim)
        self.iterations = int(iterations)
        self.epsilon = float(epsilon)
        self.reconstruction_weight = float(reconstruction_weight)
        if self.num_parts != 3:
            raise ValueError("Hyena semantic slots require exactly three parts")
        if self.num_identity_slots != 4 or self.num_slots != 5:
            raise ValueError("V2.47 requires four identity and one background slot")
        if self.iterations != 3:
            raise ValueError("V2.47 requires exactly three hierarchical refinements")
        if self.slot_dim <= 0 or self.epsilon <= 0.0:
            raise ValueError("Slot dimensions and epsilon must be positive")
        if self.reconstruction_weight <= 0.0:
            raise ValueError("Slot reconstruction weight must be positive")

        self.intermediate_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.slot_dim),
        )
        self.final_projection = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.slot_dim),
        )
        self.coordinate_projection = nn.Linear(2, self.slot_dim)
        self.input_norm = nn.LayerNorm(self.slot_dim)
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.query_projection = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.key_projection = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.value_projection = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.slot_update = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.slot_mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 4 * self.slot_dim),
            nn.GELU(),
            nn.Linear(4 * self.slot_dim, self.slot_dim),
        )
        self.slot_initialization = nn.Parameter(
            torch.empty(self.num_parts, self.num_slots, self.slot_dim)
        )
        self.aggregation_query = nn.Parameter(
            torch.empty(self.num_parts, 1, self.slot_dim)
        )
        nn.init.trunc_normal_(self.slot_initialization, std=0.02)
        nn.init.trunc_normal_(self.aggregation_query, std=0.02)

        self.aggregation_attention = nn.MultiheadAttention(
            self.slot_dim,
            num_heads=8,
            dropout=0.10,
            batch_first=True,
        )
        self.aggregation_norm = nn.LayerNorm(self.slot_dim)
        self.aggregation_mlp = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 4 * self.slot_dim),
            nn.GELU(),
            nn.Linear(4 * self.slot_dim, self.slot_dim),
        )
        self.reconstruction_decoder = nn.Linear(self.slot_dim, self.slot_dim)
        self.last_attention: torch.Tensor | None = None

    @staticmethod
    def _coordinate_grid(
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mirror_x: bool = False,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        if mirror_x:
            x = -x
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def coordinate_features(
        self,
        parts: torch.Tensor,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        mirror_x: bool = False,
    ) -> torch.Tensor:
        if torch.any((parts < 0) | (parts >= self.num_parts)):
            raise AssertionError("Semantic slots received an invalid part index")
        grid = self._coordinate_grid(
            height,
            width,
            device=device,
            dtype=dtype,
            mirror_x=mirror_x,
        ).expand(len(parts), -1, -1).clone()
        body = parts.ne(0)
        if body.any():
            grid[body, :, 0] = grid[body, :, 0].abs()
        return self.coordinate_projection(grid)

    def _refine(
        self,
        slots: torch.Tensor,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.query_projection(self.slot_norm(slots))
        normalized_inputs = self.input_norm(inputs)
        keys = self.key_projection(normalized_inputs)
        values = self.value_projection(normalized_inputs)
        logits = torch.einsum("bkd,bnd->bkn", queries, keys)
        logits = logits / math.sqrt(self.slot_dim)
        competition = torch.softmax(logits, dim=1)
        weights = competition + self.epsilon
        weights = weights / weights.sum(dim=2, keepdim=True)
        updates = torch.einsum("bkn,bnd->bkd", weights, values)
        updated = self.slot_update(
            updates.flatten(0, 1), slots.flatten(0, 1)
        ).reshape_as(slots)
        return updated + self.slot_mlp(updated), competition

    def forward(
        self,
        intermediate_patches: torch.Tensor,
        final_patches: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if intermediate_patches.shape != final_patches.shape:
            raise AssertionError("Slot feature levels must have identical grids")
        batch, patch_count, channels = final_patches.shape
        if channels != self.feature_dim or len(parts) != batch:
            raise AssertionError("Slot feature or part shape mismatch")
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise AssertionError(f"Semantic slots require a square grid: {patch_count}")

        intermediate = self.intermediate_projection(intermediate_patches)
        final = self.final_projection(final_patches)
        coordinates = self.coordinate_features(
            parts,
            side,
            side,
            device=final.device,
            dtype=final.dtype,
        )
        intermediate_inputs = intermediate + coordinates
        final_inputs = final + coordinates
        slots = self.slot_initialization[parts].to(dtype=final.dtype)
        competition: torch.Tensor | None = None
        for inputs in (intermediate_inputs, final_inputs, final_inputs):
            slots, competition = self._refine(slots, inputs)
        if competition is None:
            raise AssertionError("Semantic slots produced no competition map")
        self.last_attention = competition.detach()

        decoded_slots = self.reconstruction_decoder(slots)
        reconstruction = torch.einsum(
            "bkn,bkd->bnd", competition, decoded_slots
        )
        reconstruction = F.normalize(reconstruction.float(), dim=-1)
        target = F.normalize(final.float().detach(), dim=-1)
        reconstruction_loss = (
            1.0 - (reconstruction * target).sum(dim=-1)
        ).mean()

        identity_slots = slots[:, : self.num_identity_slots]
        query = self.aggregation_query[parts].to(dtype=identity_slots.dtype)
        aggregated, _ = self.aggregation_attention(
            query,
            identity_slots,
            identity_slots,
            need_weights=False,
        )
        aggregated = self.aggregation_norm(query + aggregated)
        aggregated = aggregated + self.aggregation_mlp(aggregated)
        return aggregated[:, 0], identity_slots, reconstruction_loss

    def weighted_reconstruction_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_weight * loss


class ParallelConvPass(nn.Module):
    """ConvPass residual parallel to one frozen attention or MLP module."""

    def __init__(
        self,
        base_module: nn.Module,
        feature_dim: int,
        prefix_tokens: int,
        bottleneck_dim: int = 64,
        scale: float = 0.1,
        dropout: float = 0.1,
        routing_context: PartRoutingContext | None = None,
        active_part: int | None = None,
    ) -> None:
        super().__init__()
        self.base_module = base_module
        self.adapter_down = nn.Linear(feature_dim, bottleneck_dim, bias=True)
        self.adapter_conv = nn.Conv2d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.adapter_up = nn.Linear(bottleneck_dim, feature_dim, bias=True)
        self.adapter_dropout = nn.Dropout(float(dropout))
        self.prefix_tokens = int(prefix_tokens)
        self.bottleneck_dim = int(bottleneck_dim)
        self.scale = float(scale)
        self.routing_context = routing_context
        self.active_part = active_part
        if self.active_part is not None and self.routing_context is None:
            raise ValueError("A routed ConvPass requires a part context")

        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_conv.weight)
        with torch.no_grad():
            identity = torch.eye(self.bottleneck_dim)
            self.adapter_conv.weight[:, :, 1, 1].copy_(identity)
        nn.init.zeros_(self.adapter_conv.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    @staticmethod
    def _quick_gelu(features: torch.Tensor) -> torch.Tensor:
        return features * torch.sigmoid(1.702 * features)

    def adapter_forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = features.shape
        patch_count = token_count - self.prefix_tokens
        side = math.isqrt(patch_count)
        if patch_count <= 0 or side * side != patch_count:
            raise AssertionError(
                f"ConvPass requires a square patch grid, got {token_count} "
                f"tokens with {self.prefix_tokens} prefixes"
            )
        reduced = self._quick_gelu(self.adapter_down(features))
        patches = reduced[:, self.prefix_tokens :].reshape(
            batch, side, side, self.bottleneck_dim
        ).permute(0, 3, 1, 2)
        patches = self.adapter_conv(patches).permute(0, 2, 3, 1).reshape(
            batch, patch_count, self.bottleneck_dim
        )
        if self.prefix_tokens:
            prefixes = reduced[:, : self.prefix_tokens].reshape(
                batch * self.prefix_tokens,
                self.bottleneck_dim,
                1,
                1,
            )
            prefixes = self.adapter_conv(prefixes).reshape(
                batch, self.prefix_tokens, self.bottleneck_dim
            )
            reduced = torch.cat([prefixes, patches], dim=1)
        else:
            reduced = patches
        reduced = self.adapter_dropout(self._quick_gelu(reduced))
        return self.adapter_up(reduced)

    def routed_adapter_forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter_forward(features)
        if self.active_part is None:
            return adapted
        parts = self.routing_context.parts
        if parts is None or len(parts) != len(features):
            raise AssertionError("ConvPass part-routing context is missing")
        mask = parts.eq(self.active_part).to(dtype=adapted.dtype)
        return adapted * mask[:, None, None]

    def forward(
        self,
        features: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return self.base_module(features, *args, **kwargs) + (
            self.scale * self.routed_adapter_forward(features)
        )

    def adapter_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.adapter_down.parameters())
            + list(self.adapter_conv.parameters())
            + list(self.adapter_up.parameters())
        )


class DualEmbeddingDinoV3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        local_queries: int = 4,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_blocks: int = 6,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        feature_dim = int(self.backbone.embed_dim)
        self.patch_norm = nn.LayerNorm(feature_dim)
        self.part_queries = nn.Parameter(torch.empty(3, local_queries, feature_dim))
        nn.init.trunc_normal_(self.part_queries, std=0.02)
        self.local_attention = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            dropout=0.05,
            batch_first=True,
        )
        self.local_norm = nn.LayerNorm(feature_dim)
        self.shared_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.10),
        )
        self.part_projection = nn.Sequential(
            nn.Linear(feature_dim * (2 + local_queries), embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.10),
        )
        self.shared_class_weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.part_class_delta = nn.Parameter(
            torch.zeros(3, num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.shared_class_weight)
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        self.num_classes = num_classes
        self.local_queries_count = local_queries
        self.arc_scale = arc_scale
        self.arc_margin = arc_margin
        self.part_delta_scale = part_delta_scale
        self.cos_m = math.cos(arc_margin)
        self.sin_m = math.sin(arc_margin)
        self.threshold = math.cos(math.pi - arc_margin)
        self.margin_correction = math.sin(math.pi - arc_margin) * arc_margin
        self.freeze_backbone_prefix(freeze_blocks)
        self.model_name = MODEL_NAME
        self.pretraining_source = PRETRAINING_SOURCE

    def optimizer_layer_count(self) -> int:
        return len(self.backbone.blocks) + 1

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith("blocks."):
            return int(inner_name.split(".")[1]) + 1
        if inner_name.startswith(("patch_embed", "cls_token", "reg_token")):
            return 0
        return self.optimizer_layer_count()

    def freeze_backbone_prefix(self, block_count: int) -> None:
        for name, parameter in self.backbone.named_parameters():
            frozen = name.startswith(
                ("patch_embed", "cls_token", "reg_token", "pos_embed", "mask_token")
            )
            if name.startswith("blocks."):
                block_index = int(name.split(".")[1])
                frozen = block_index < block_count
            parameter.requires_grad = not frozen

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != {self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    def _part_weights(self, parts: torch.Tensor) -> torch.Tensor:
        shared = self.shared_class_weight.unsqueeze(0)
        weights = shared + self.part_delta_scale * self.part_class_delta[parts]
        return F.normalize(weights, dim=-1)

    def _cosines(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_cosine = F.linear(
            shared_embedding, F.normalize(self.shared_class_weight, dim=-1)
        )
        part_cosine = torch.einsum(
            "bd,bcd->bc", part_embedding, self._part_weights(parts)
        )
        return shared_cosine, part_cosine

    def _margin(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        safe = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
        phi = safe * self.cos_m - sine * self.sin_m
        phi = torch.where(
            safe > self.threshold, phi, safe - self.margin_correction
        )
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(safe.dtype)
        return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        tokens = self.backbone.forward_features(images)
        prefix_count = int(self.backbone.num_prefix_tokens)
        cls = tokens[:, 0]
        patches = self.patch_norm(tokens[:, prefix_count:])
        mean_patch = patches.mean(dim=1)
        shared_embedding = F.normalize(
            self.shared_projection(torch.cat([cls, mean_patch], dim=-1)), dim=-1
        )
        queries = self.part_queries[parts]
        local_tokens, _ = self.local_attention(
            queries, patches, patches, need_weights=False
        )
        local_tokens = self.local_norm(local_tokens + queries)
        part_input = torch.cat(
            [cls, mean_patch, local_tokens.flatten(start_dim=1)], dim=-1
        )
        part_embedding = F.normalize(self.part_projection(part_input), dim=-1)
        local_descriptor = None
        if return_local:
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1]:
                raise AssertionError(f"Non-square patch grid: {patches.shape}")
            patch_grid = patches.reshape(
                patches.shape[0], side, side, patches.shape[-1]
            ).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(
                patch_grid, (local_grid, local_grid)
            ).permute(0, 2, 3, 1)
            local_descriptor = F.normalize(pooled, dim=-1).flatten(1, 2)
        return shared_embedding, part_embedding, local_descriptor

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        available = self.part_class_available[parts]
        fused = 0.45 * shared_cosine + 0.55 * part_cosine
        return torch.where(available, fused, shared_cosine)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        shared_embedding, part_embedding, _ = self.encode(images, parts)
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        if labels is None:
            return (
                self.inference_scores(shared_embedding, part_embedding, parts),
                shared_embedding,
                part_embedding,
            )
        return (
            self._margin(shared_cosine, labels),
            self._margin(part_cosine, labels),
            shared_embedding,
            part_embedding,
        )


class DinoV3ConvNeXtDOLG(nn.Module):
    """Multi-scale global/local ConvNeXt with part-conditioned local attention."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_stages: int = 2,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            CONVNEXT_MODEL_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        local_dim = 512
        global_dim = 1024
        self.local_projection = nn.Sequential(
            nn.Conv2d(local_dim, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, embedding_dim),
            nn.GELU(),
        )
        self.part_attention = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(embedding_dim, 128, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(128, 1, kernel_size=1),
            )
            for _ in range(3)
        )
        self.shared_projection = nn.Sequential(
            nn.Linear(global_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.10),
        )
        self.part_projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.10),
        )
        self.shared_class_weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        self.part_class_delta = nn.Parameter(
            torch.zeros(3, num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.shared_class_weight)
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        self.num_classes = num_classes
        self.arc_scale = arc_scale
        self.arc_margin = arc_margin
        self.part_delta_scale = part_delta_scale
        self.cos_m = math.cos(arc_margin)
        self.sin_m = math.sin(arc_margin)
        self.threshold = math.cos(math.pi - arc_margin)
        self.margin_correction = math.sin(math.pi - arc_margin) * arc_margin
        self.model_name = CONVNEXT_MODEL_NAME
        self.pretraining_source = CONVNEXT_PRETRAINING_SOURCE
        self.freeze_backbone_stages(freeze_stages)

    def optimizer_layer_count(self) -> int:
        return 4

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith("stem_"):
            return 0
        if inner_name.startswith("stages_"):
            return int(inner_name.split("_")[1].split(".")[0]) + 1
        return self.optimizer_layer_count()

    def freeze_backbone_stages(self, stage_count: int) -> None:
        for name, parameter in self.backbone.named_parameters():
            frozen = name.startswith("stem_") and stage_count > 0
            if name.startswith("stages_"):
                stage = int(name.split("_")[1].split(".")[0])
                frozen = stage < stage_count
            parameter.requires_grad = not frozen

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != {self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    def _part_weights(self, parts: torch.Tensor) -> torch.Tensor:
        shared = self.shared_class_weight.unsqueeze(0)
        weights = shared + self.part_delta_scale * self.part_class_delta[parts]
        return F.normalize(weights, dim=-1)

    def _cosines(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_cosine = F.linear(
            shared_embedding, F.normalize(self.shared_class_weight, dim=-1)
        )
        part_cosine = torch.einsum(
            "bd,bcd->bc", part_embedding, self._part_weights(parts)
        )
        return shared_cosine, part_cosine

    def _margin(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        safe = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
        phi = safe * self.cos_m - sine * self.sin_m
        phi = torch.where(
            safe > self.threshold, phi, safe - self.margin_correction
        )
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(safe.dtype)
        return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        _, local_map, global_map = self.backbone(images)
        global_vector = F.adaptive_avg_pool2d(global_map.float(), 1).flatten(1)
        shared_embedding = F.normalize(
            self.shared_projection(global_vector), dim=-1
        )
        local_feature = self.local_projection(local_map.float())
        all_attention = torch.cat(
            [attention(local_feature) for attention in self.part_attention], dim=1
        )
        selected_attention = all_attention[
            torch.arange(len(parts), device=parts.device), parts
        ].flatten(1)
        selected_attention = torch.softmax(selected_attention, dim=1).reshape(
            len(parts), 1, local_feature.shape[2], local_feature.shape[3]
        )
        local_vector = (local_feature * selected_attention).sum(dim=(2, 3))
        projection = (
            (local_vector * shared_embedding).sum(dim=1, keepdim=True)
            * shared_embedding
        )
        orthogonal_local = local_vector - projection
        part_embedding = F.normalize(
            self.part_projection(
                torch.cat([shared_embedding, orthogonal_local], dim=-1)
            ),
            dim=-1,
        )
        local_descriptor = None
        if return_local:
            pooled = F.adaptive_avg_pool2d(
                local_feature, (local_grid, local_grid)
            ).permute(0, 2, 3, 1)
            local_descriptor = F.normalize(pooled, dim=-1).flatten(1, 2)
        return shared_embedding, part_embedding, local_descriptor

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        available = self.part_class_available[parts]
        fused = 0.45 * shared_cosine + 0.55 * part_cosine
        return torch.where(available, fused, shared_cosine)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        shared_embedding, part_embedding, _ = self.encode(images, parts)
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        if labels is None:
            return (
                self.inference_scores(shared_embedding, part_embedding, parts),
                shared_embedding,
                part_embedding,
            )
        return (
            self._margin(shared_cosine, labels),
            self._margin(part_cosine, labels),
            shared_embedding,
            part_embedding,
        )


class EfficientNetV2MSubCenter(nn.Module):
    """ImageNet-only MiewID-style embedding with shared part-aware prototypes."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.50,
        part_delta_scale: float = 0.20,
        freeze_stages: int = 2,
        subcenters: int = 3,
        adapter_scale: float = 0.15,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            EFFICIENTNET_MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        feature_dim = int(self.backbone.num_features)
        self.pooling = GeM(p=3.0)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.bn.bias.requires_grad_(False)
        self.part_adapters = nn.ModuleList(
            ResidualPartAdapter(feature_dim) for _ in range(3)
        )
        self.shared_class_weight = nn.Parameter(
            torch.empty(num_classes, subcenters, feature_dim)
        )
        self.part_class_delta = nn.Parameter(
            torch.zeros(3, num_classes, subcenters, feature_dim)
        )
        nn.init.uniform_(
            self.shared_class_weight,
            -1.0 / math.sqrt(feature_dim),
            1.0 / math.sqrt(feature_dim),
        )
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        self.register_buffer(
            "class_margins", torch.full((num_classes,), float(arc_margin))
        )
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.subcenters = subcenters
        self.arc_scale = arc_scale
        self.arc_margin = arc_margin
        self.part_delta_scale = part_delta_scale
        self.adapter_scale = adapter_scale
        self.model_name = EFFICIENTNET_MODEL_NAME
        self.pretraining_source = EFFICIENTNET_PRETRAINING_SOURCE
        self.freeze_backbone_stages(freeze_stages)

    def optimizer_layer_count(self) -> int:
        return len(self.backbone.blocks) + 1

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith("blocks."):
            return int(inner_name.split(".")[1]) + 1
        if inner_name.startswith(("conv_stem", "bn1")):
            return 0
        return self.optimizer_layer_count()

    def freeze_backbone_stages(self, stage_count: int) -> None:
        for name, parameter in self.backbone.named_parameters():
            frozen = name.startswith(("conv_stem", "bn1")) and stage_count > 0
            if name.startswith("blocks."):
                stage = int(name.split(".")[1])
                frozen = stage < stage_count
            parameter.requires_grad = not frozen

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != {self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    def set_class_counts(self, counts: torch.Tensor) -> None:
        """Compute adaptive margins exclusively from this fold's training rows."""
        if counts.shape != self.class_margins.shape:
            raise ValueError(f"Counts {counts.shape} != {self.class_margins.shape}")
        counts = counts.float().clamp_min(1.0)
        rarity = counts.pow(-0.25)
        span = rarity.max() - rarity.min()
        if float(span) < 1e-8:
            margins = torch.full_like(rarity, self.arc_margin)
        else:
            margins = 0.20 + (rarity - rarity.min()) / span * (
                self.arc_margin - 0.20
            )
        self.class_margins.copy_(margins)

    @staticmethod
    def _subcenter_cosine(
        embeddings: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        weights = F.normalize(weights, dim=-1)
        all_cosines = torch.einsum("bd,ckd->bck", embeddings, weights)
        return all_cosines.max(dim=-1).values

    def _cosines(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_cosine = self._subcenter_cosine(
            shared_embedding, self.shared_class_weight
        )
        part_weights = self.shared_class_weight.unsqueeze(0) + (
            self.part_delta_scale * self.part_class_delta[parts]
        )
        normalized = F.normalize(part_weights, dim=-1)
        all_part_cosines = torch.einsum(
            "bd,bckd->bck", part_embedding, normalized
        )
        return shared_cosine, all_part_cosines.max(dim=-1).values

    def _margin(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        safe = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        margins = self.class_margins[labels].to(safe.dtype)
        cosine_margin = torch.cos(margins).unsqueeze(1)
        sine_margin = torch.sin(margins).unsqueeze(1)
        threshold = torch.cos(math.pi - margins).unsqueeze(1)
        correction = (torch.sin(math.pi - margins) * margins).unsqueeze(1)
        sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
        phi = safe * cosine_margin - sine * sine_margin
        phi = torch.where(safe > threshold, phi, safe - correction)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(safe.dtype)
        return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feature_map = self.backbone.forward_features(images)
        pooled = self.pooling(feature_map.float())
        base = self.bn(pooled)
        shared_embedding = F.normalize(base, dim=-1)
        part_base = base.clone()
        for part_index, adapter in enumerate(self.part_adapters):
            mask = parts == part_index
            if mask.any():
                part_base[mask] = part_base[mask] + self.adapter_scale * adapter(
                    base[mask]
                )
        part_embedding = F.normalize(part_base, dim=-1)
        local_descriptor = None
        if return_local:
            pooled_grid = F.adaptive_avg_pool2d(
                feature_map.float(), (local_grid, local_grid)
            ).permute(0, 2, 3, 1)
            local_descriptor = F.normalize(pooled_grid, dim=-1).flatten(1, 2)
        return shared_embedding, part_embedding, local_descriptor

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        available = self.part_class_available[parts]
        fused = 0.45 * shared_cosine + 0.55 * part_cosine
        return torch.where(available, fused, shared_cosine)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        shared_embedding, part_embedding, _ = self.encode(images, parts)
        shared_cosine, part_cosine = self._cosines(
            shared_embedding, part_embedding, parts
        )
        if labels is None:
            return (
                self.inference_scores(shared_embedding, part_embedding, parts),
                shared_embedding,
                part_embedding,
            )
        return (
            self._margin(shared_cosine, labels),
            self._margin(part_cosine, labels),
            shared_embedding,
            part_embedding,
        )


class PetFaceR50HeadSpecialist(nn.Module):
    """PetFace-domain ResNet-50 with a fresh competition classifier.

    Only the official PetFace visual backbone is loaded.  The 175,081-way
    external identity classifier is shape-audited and deliberately discarded.
    """

    branch_count = 1

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        pretrained_path: str | os.PathLike[str] | None = None,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_stages: int = 2,
    ) -> None:
        super().__init__()
        if embedding_dim != 512:
            raise ValueError("The audited PetFace checkpoint requires 512 dimensions")
        if not 0 <= freeze_stages <= 5:
            raise ValueError("PetFace freeze_stages must be in [0, 5]")
        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self.petface_pretrained_loaded = False
        self.petface_backbone_tensor_count = 0
        self.petface_external_classifier_shape: tuple[int, ...] | None = None
        if pretrained:
            if pretrained_path is None:
                raise ValueError("PetFace training requires its audited checkpoint path")
            payload = torch.load(
                os.fspath(pretrained_path), map_location="cpu", weights_only=True
            )
            if set(payload) != {"state_dict_backbone", "state_dict_softmax_fc"}:
                raise AssertionError("Unexpected PetFace checkpoint payload")
            backbone_state = payload["state_dict_backbone"]
            external_classifier = payload["state_dict_softmax_fc"]
            if len(backbone_state) != 325:
                raise AssertionError("PetFace backbone tensor inventory changed")
            if set(external_classifier) != {"weight"}:
                raise AssertionError("Unexpected PetFace external classifier state")
            classifier_shape = tuple(external_classifier["weight"].shape)
            if classifier_shape != (
                PETFACE_R50_EXTERNAL_CLASS_COUNT,
                embedding_dim,
            ):
                raise AssertionError(
                    f"Unexpected PetFace external classifier shape: {classifier_shape}"
                )
            self.backbone.load_state_dict(backbone_state, strict=True)
            self.petface_pretrained_loaded = True
            self.petface_backbone_tensor_count = len(backbone_state)
            self.petface_external_classifier_shape = classifier_shape
            del payload, backbone_state, external_classifier

        self.shared_class_weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        self.part_class_delta = nn.Parameter(
            torch.zeros(3, num_classes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.shared_class_weight)
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.arc_scale = float(arc_scale)
        self.arc_margin = float(arc_margin)
        self.part_delta_scale = float(part_delta_scale)
        self.cos_m = math.cos(self.arc_margin)
        self.sin_m = math.sin(self.arc_margin)
        self.threshold = math.cos(math.pi - self.arc_margin)
        self.margin_correction = math.sin(math.pi - self.arc_margin) * self.arc_margin
        self.freeze_stages = int(freeze_stages)
        self.freeze_backbone_prefix(self.freeze_stages)
        self.model_name = "petface_unified_arcface_resnet50_head_specialist"
        self.pretraining_source = PETFACE_R50_PRETRAINING_SOURCE
        self.pretraining_sha256 = PETFACE_R50_PRETRAINING_SHA256

    def optimizer_layer_count(self) -> int:
        return 5

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith(("conv1.", "bn1.")):
            return 0
        for layer_index in range(1, 5):
            if inner_name.startswith(f"layer{layer_index}."):
                return layer_index
        return 5

    def freeze_backbone_prefix(self, stage_count: int) -> None:
        prefixes: list[str] = []
        if stage_count >= 1:
            prefixes.extend(("conv1.", "bn1."))
        for layer_index in range(1, min(stage_count, 5)):
            prefixes.append(f"layer{layer_index}.")
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(not any(name.startswith(p) for p in prefixes))

    def train(self, mode: bool = True) -> "PetFaceR50HeadSpecialist":
        super().train(mode)
        if mode and self.freeze_stages >= 1:
            self.backbone.bn1.eval()
        if mode:
            for layer_index in range(1, min(self.freeze_stages, 5)):
                getattr(self.backbone, f"layer{layer_index}").eval()
        return self

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != "
                f"{self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    def _part_weights(self, parts: torch.Tensor) -> torch.Tensor:
        weights = self.shared_class_weight.unsqueeze(0) + (
            self.part_delta_scale * self.part_class_delta[parts]
        )
        return F.normalize(weights, dim=-1)

    def _cosines(
        self, embeddings: torch.Tensor, parts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = F.linear(
            embeddings, F.normalize(self.shared_class_weight, dim=-1)
        )
        part = torch.einsum(
            "bd,bcd->bc", embeddings, self._part_weights(parts)
        )
        return shared, part

    def _margin(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        safe = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
        phi = safe * self.cos_m - sine * self.sin_m
        phi = torch.where(
            safe > self.threshold, phi, safe - self.margin_correction
        )
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(safe.dtype)
        return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del parts, local_grid
        embedding = F.normalize(self.backbone(images).float(), dim=-1)
        local = embedding[:, None] if return_local else None
        return embedding, embedding, local

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        del part_embedding
        shared, part = self._cosines(shared_embedding, parts)
        fused = 0.45 * shared + 0.55 * part
        return torch.where(self.part_class_available[parts], fused, shared)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        shared_embedding, part_embedding, _ = self.encode(images, parts)
        shared_cosine, part_cosine = self._cosines(shared_embedding, parts)
        if labels is None:
            return (
                self.inference_scores(shared_embedding, part_embedding, parts),
                shared_embedding,
                part_embedding,
            )
        return (
            self._margin(shared_cosine, labels),
            self._margin(part_cosine, labels),
            shared_embedding,
            part_embedding,
        )


class ARBaseMGN(nn.Module):
    """ARBase-inspired IBN multi-granularity network for the three crop domains."""

    branch_count = 8
    feature_dim = 2048

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        part_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        base = torch.hub.load(
            "XingangPan/IBN-Net",
            "resnet50_ibn_a",
            pretrained=pretrained,
            trust_repo=True,
            verbose=False,
        )
        # ARBase keeps the final feature map at stride 16 for subtle markings.
        base.layer4[0].conv2.stride = (1, 1)
        base.layer4[0].downsample[0].stride = (1, 1)
        layer3_tail = nn.Sequential(*list(base.layer3.children())[1:])
        layer4 = base.layer4
        self.backbone = nn.ModuleDict(
            {
                "shared": nn.Sequential(
                    base.conv1,
                    base.bn1,
                    base.relu,
                    base.maxpool,
                    base.layer1,
                    base.layer2,
                    base.layer3[0],
                ),
                "branch1": nn.Sequential(layer3_tail, layer4),
                "branch2": nn.Sequential(
                    copy.deepcopy(layer3_tail), copy.deepcopy(layer4)
                ),
                "branch3": nn.Sequential(
                    copy.deepcopy(layer3_tail), copy.deepcopy(layer4)
                ),
            }
        )
        self.bn_necks = nn.ModuleList(
            nn.BatchNorm1d(self.feature_dim) for _ in range(self.branch_count)
        )
        for neck in self.bn_necks:
            nn.init.ones_(neck.weight)
            nn.init.zeros_(neck.bias)
            neck.bias.requires_grad_(False)
        self.shared_class_weight = nn.Parameter(
            torch.empty(self.branch_count, num_classes, self.feature_dim)
        )
        self.part_class_delta = nn.Parameter(
            torch.zeros(3, self.branch_count, num_classes, self.feature_dim)
        )
        nn.init.normal_(self.shared_class_weight, std=0.01)
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        self.num_classes = num_classes
        self.part_delta_scale = part_delta_scale
        self.model_name = ARBASE_MODEL_NAME
        self.pretraining_source = ARBASE_PRETRAINING_SOURCE

    def optimizer_layer_count(self) -> int:
        return 4

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith("shared"):
            return 0
        if inner_name.startswith("branch1"):
            return 1
        if inner_name.startswith("branch2"):
            return 2
        return 3

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != {self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    @staticmethod
    def _global_pool(feature_map: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)

    def _branch_descriptors(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.backbone["shared"](images)
        branch1 = self.backbone["branch1"](shared)
        branch2 = self.backbone["branch2"](shared)
        branch3 = self.backbone["branch3"](shared)
        branch2_parts = F.adaptive_avg_pool2d(branch2, (2, 1)).unbind(dim=2)
        branch3_parts = F.adaptive_avg_pool2d(branch3, (3, 1)).unbind(dim=2)
        pooled = [
            self._global_pool(branch1),
            self._global_pool(branch2),
            self._global_pool(branch3),
            *(part.squeeze(-1) for part in branch2_parts),
            *(part.squeeze(-1) for part in branch3_parts),
        ]
        raw = torch.stack(pooled, dim=1)
        neck = torch.stack(
            [bn(raw[:, index]) for index, bn in enumerate(self.bn_necks)], dim=1
        )
        return raw, neck

    def _logits(
        self, neck_features: torch.Tensor, parts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_logits = torch.einsum(
            "bmd,mcd->bmc", neck_features, self.shared_class_weight
        )
        part_weights = self.shared_class_weight.unsqueeze(0) + (
            self.part_delta_scale * self.part_class_delta[parts]
        )
        part_logits = torch.einsum(
            "bmd,bmcd->bmc", neck_features, part_weights
        )
        return shared_logits, part_logits

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        raw, neck = self._branch_descriptors(images)
        shared_embedding = neck.flatten(1)
        part_embedding = raw.flatten(1)
        # MGN's pooled stripe descriptors already are the local representation.
        local_descriptor = None
        if return_local:
            local_descriptor = F.normalize(raw, dim=-1)
        return shared_embedding, part_embedding, local_descriptor

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        del part_embedding
        neck = shared_embedding.reshape(
            -1, self.branch_count, self.feature_dim
        )
        shared_logits, part_logits = self._logits(neck, parts)
        shared_score = shared_logits.mean(dim=1)
        part_score = part_logits.mean(dim=1)
        available = self.part_class_available[parts]
        fused = 0.45 * shared_score + 0.55 * part_score
        return torch.where(available, fused, shared_score)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        del labels
        raw, neck = self._branch_descriptors(images)
        shared_logits, part_logits = self._logits(neck, parts)
        shared_embedding = neck.flatten(1)
        part_embedding = raw.flatten(1)
        return (
            shared_logits,
            part_logits,
            shared_embedding,
            part_embedding,
        )


class DepthwiseSeparableTextureBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class MultiScaleImageTextureEncoder(nn.Module):
    """Compact image-frequency side path aligned to a 28 x 28 DINO grid."""

    output_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "luminance",
            torch.tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
            ).view(1, 1, 3, 3)
            / 8.0,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                ((-1.0, -2.0, -1.0), (0.0, 0.0, 0.0), (1.0, 2.0, 1.0))
            ).view(1, 1, 3, 3)
            / 8.0,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.downsample = nn.Sequential(
            DepthwiseSeparableTextureBlock(
                32, 64, kernel_size=5, stride=2
            ),
            DepthwiseSeparableTextureBlock(
                64, 96, kernel_size=3, stride=2
            ),
            DepthwiseSeparableTextureBlock(
                96, self.output_dim, kernel_size=3, stride=2
            ),
        )
        self.refinement = DepthwiseSeparableTextureBlock(
            self.output_dim,
            self.output_dim,
            kernel_size=3,
            stride=1,
        )

    @staticmethod
    def _local_mean(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
        return F.avg_pool2d(
            value,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rgb = images * self.imagenet_std.to(images.dtype) + self.imagenet_mean.to(
            images.dtype
        )
        gray = (rgb * self.luminance.to(images.dtype)).sum(
            dim=1, keepdim=True
        )
        rgb_contrast = rgb - self._local_mean(rgb, 9)
        gray_contrasts = [
            gray - self._local_mean(gray, kernel_size)
            for kernel_size in (3, 7, 15)
        ]
        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        gradient_x = F.conv2d(padded, self.sobel_x.to(images.dtype))
        gradient_y = F.conv2d(padded, self.sobel_y.to(images.dtype))
        gradient = torch.sqrt(
            gradient_x.square() + gradient_y.square() + 1e-6
        )
        texture = torch.cat(
            [rgb_contrast, *gray_contrasts, gradient], dim=1
        )
        features = self.downsample(self.stem(texture))
        return features + self.refinement(features)


class DinoV3PatchMGN(nn.Module):
    """Dense DINO patch descriptors with global, 2-region and 3-region heads."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_blocks: int = 0,
        grad_checkpointing: bool = True,
        bidirectional: bool = False,
        hierarchical_part_heads: bool = False,
        subcenters: int = 1,
        part_adapter_bottleneck: int = 0,
        quality_adaptive_margin: bool = False,
        class_adaptive_margin: bool = False,
        head_class_adaptive_margin: bool = False,
        learned_branch_gates: bool = False,
        part_side_embedding: bool = False,
        continuous_geometry_conditioning: bool = False,
        foreground_auxiliary: bool = False,
        foreground_token_conditioning: bool = False,
        foreground_scale_normalization: bool = False,
        covariance_branch: bool = False,
        covariance_dim: int = 64,
        covariance_standardize: bool = False,
        covariance_matrix_sqrt: bool = False,
        intermediate_covariance_branch: bool = False,
        intermediate_covariance_block: int = 17,
        gradient_covariance_branch: bool = False,
        semantic_topk_branch: bool = False,
        semantic_topk_fraction: float = 0.25,
        simpool_branch: bool = False,
        vertical_branches: bool = False,
        part_aligned_axis: bool = False,
        domain_specific_bn: bool = False,
        modality_specific_bn: bool = False,
        token_mixstyle: bool = False,
        mixstyle_probability: float = 0.5,
        mixstyle_alpha: float = 0.1,
        frozen_prefix_adaptformer: bool = False,
        adaptformer_dim: int = 64,
        adaptformer_scale: float = 0.1,
        frozen_prefix_qv_lora: bool = False,
        qv_lora_rank: int = 8,
        qv_lora_part_rank: int = 0,
        qv_lora_alpha: float = 8.0,
        qv_lora_dropout: float = 0.05,
        suffix_head_qv_lora: bool = False,
        frozen_prefix_convpass: bool = False,
        convpass_dim: int = 64,
        convpass_scale: float = 0.1,
        convpass_dropout: float = 0.1,
        head_convpass_attention: bool = False,
        part_mlp_expert_blocks: int = 0,
        pattern_a2gc_branch: bool = False,
        hierarchical_slot_architecture: bool = False,
        dense_correspondence_training: bool = False,
        identity_query_pooling: bool = False,
        cross_level_texture_pyramid: bool = False,
        part_routed_cross_level_texture: bool = False,
        head_routed_cross_level_texture: bool = False,
        image_frequency_texture_side: bool = False,
        body_to_head_distillation: bool = False,
        head_tail_expert_blocks: int = 0,
        jpm_local_branches: int = 0,
        prototype_memory: bool = False,
        prototype_momentum: float = 0.9,
        prototype_mix: float = 0.5,
        prototype_transport_rank: int = 0,
        prototype_transport_scale: float = 0.20,
        part_topology_supcon: bool = False,
        training_instance_queue: bool = False,
        instance_queue_capacity: int = 2048,
        decision_aligned_classification: bool = False,
        source_paired_logit_distillation: bool = False,
        head_two_view_part_balanced_supcon: bool = False,
        head_identity_expert: bool = False,
        head_identity_expert_detach: bool = True,
        head_identity_expert_inference: bool = True,
        head_identity_expert_loss_weight: float = 0.50,
        head_tail_other_expert: bool = False,
        external_head_representation: bool = False,
        ordered_head_grid_expert: bool = False,
        quality_h: float = 0.333,
        quality_t_alpha: float = 0.01,
        backbone_model_name: str = MODEL_NAME,
        pretraining_source: str = PRETRAINING_SOURCE,
        lingbot_backbone: bool = False,
        spatial_backbone: bool = False,
        spatial_layout: str = "nchw",
        spatial_image_size: bool = False,
        spatial_strict_image_size: bool = True,
        backbone_output_stride: int | None = 32,
        bioclip_backbone: bool = False,
        bioclip2_backbone: bool = False,
        radio_backbone: bool = False,
        tips_backbone: bool = False,
        external_pretrained_path: str | os.PathLike[str] | None = None,
    ) -> None:
        super().__init__()
        self.lingbot_backbone = bool(lingbot_backbone)
        self.spatial_backbone = bool(spatial_backbone)
        self.bioclip_backbone = bool(bioclip_backbone)
        self.bioclip2_backbone = bool(bioclip2_backbone)
        self.radio_backbone = bool(radio_backbone)
        self.tips_backbone = bool(tips_backbone)
        self.spatial_layout = spatial_layout
        self.grad_checkpointing = bool(grad_checkpointing)
        if sum(
            (
                self.lingbot_backbone,
                self.spatial_backbone,
                self.bioclip_backbone,
                self.bioclip2_backbone,
                self.radio_backbone,
                self.tips_backbone,
            )
        ) > 1:
            raise ValueError("Special backbone modes are mutually exclusive")
        if self.spatial_layout not in {"nchw", "nhwc"}:
            raise ValueError("spatial_layout must be nchw or nhwc")
        if self.lingbot_backbone:
            if part_side_embedding:
                raise ValueError("Part SIE is currently implemented only for timm DINOv3")
            try:
                from lingbot_vision import load_config, load_pretrained_backbone
                from lingbot_vision.build import build_backbone_from_cfg
            except ImportError as exc:
                raise ImportError(
                    "Install the pinned official LingBot-Vision package before "
                    "using the lingbot_large_patch_mgn variant"
                ) from exc
            if pretrained:
                self.backbone, backbone_dim = load_pretrained_backbone(
                    repo_id_or_path=LINGBOT_LARGE_MODEL_NAME,
                    variant="large",
                    device="cpu",
                    dtype=torch.float32,
                    cache_dir=os.environ.get("LINGBOT_HF_CACHE"),
                    revision=LINGBOT_LARGE_REVISION,
                    verbose=True,
                )
            else:
                config = load_config("configs/lbot_vision_vitl.yaml")
                self.backbone, backbone_dim = build_backbone_from_cfg(config)
            # The official loader intentionally returns a frozen feature
            # extractor. This downstream ReID model fine-tunes it, with the
            # requested prefix freeze applied below.
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(True)
        elif self.radio_backbone:
            if not pretrained or external_pretrained_path is None:
                raise ValueError("V12.1 requires its exact RADIO checkpoint")
            self.backbone = RadioV25DenseVision(
                external_pretrained_path,
                image_size=image_size,
                grad_checkpointing=grad_checkpointing,
            )
            backbone_dim = int(self.backbone.embed_dim)
            self.pretraining_sha256 = self.backbone.pretraining_sha256
        elif self.tips_backbone:
            if not pretrained or external_pretrained_path is None:
                raise ValueError("V12.15 requires its exact public TIPS checkpoint")
            self.backbone = TipsL14HighResDenseVision(
                external_pretrained_path,
                image_size=image_size,
                grad_checkpointing=grad_checkpointing,
            )
            backbone_dim = int(self.backbone.embed_dim)
            self.pretraining_sha256 = self.backbone.pretraining_sha256
        elif self.bioclip_backbone:
            if not pretrained or external_pretrained_path is None:
                raise ValueError("V10.4 requires its exact BioCLIP checkpoint")
            self.backbone = BioClipDenseVision(
                external_pretrained_path,
                image_size=image_size,
                grad_checkpointing=grad_checkpointing,
            )
            backbone_dim = int(self.backbone.embed_dim)
        elif self.bioclip2_backbone:
            if not pretrained or external_pretrained_path is None:
                raise ValueError("V10.6 requires its exact BioCLIP 2 checkpoint")
            self.backbone = BioClip2DenseVision(
                external_pretrained_path,
                image_size=image_size,
                grad_checkpointing=grad_checkpointing,
            )
            backbone_dim = int(self.backbone.embed_dim)
        elif self.spatial_backbone:
            spatial_model_kwargs: dict[str, object] = {}
            if backbone_output_stride is not None:
                spatial_model_kwargs["output_stride"] = backbone_output_stride
            if spatial_image_size:
                spatial_model_kwargs["img_size"] = image_size
                spatial_model_kwargs["strict_img_size"] = spatial_strict_image_size
            self.backbone = timm.create_model(
                backbone_model_name,
                pretrained=pretrained,
                num_classes=0,
                **spatial_model_kwargs,
            )
            if grad_checkpointing:
                self.backbone.set_grad_checkpointing(True)
            backbone_dim = int(self.backbone.num_features)
        else:
            pretrained_overlay = (
                {"file": str(external_pretrained_path)}
                if external_pretrained_path is not None
                else None
            )
            self.backbone = timm.create_model(
                backbone_model_name,
                pretrained=pretrained,
                pretrained_cfg_overlay=pretrained_overlay,
                num_classes=0,
                img_size=image_size,
            )
            if backbone_model_name == DINOV2_GIANT_REGISTER_MODEL_NAME:
                if not pretrained or external_pretrained_path is None:
                    raise ValueError(
                        "V12.16 requires its exact public DINOv2-G checkpoint"
                    )
                self.pretraining_sha256 = (
                    DINOV2_GIANT_REGISTER_PRETRAINING_SHA256
                )
            if grad_checkpointing:
                self.backbone.set_grad_checkpointing(True)
            backbone_dim = int(self.backbone.embed_dim)
        self.bidirectional = bidirectional
        self.hierarchical_part_heads = hierarchical_part_heads
        self.subcenters = int(subcenters)
        self.part_adapter_bottleneck = int(part_adapter_bottleneck)
        self.quality_adaptive_margin = bool(quality_adaptive_margin)
        self.class_adaptive_margin = bool(class_adaptive_margin)
        self.head_class_adaptive_margin = bool(head_class_adaptive_margin)
        self.learned_branch_gates = bool(learned_branch_gates)
        self.part_side_embedding_enabled = bool(part_side_embedding)
        self.continuous_geometry_conditioning = bool(
            continuous_geometry_conditioning
        )
        self.foreground_auxiliary = bool(foreground_auxiliary)
        self.foreground_token_conditioning = bool(
            foreground_token_conditioning
        )
        self.foreground_scale_normalization = bool(
            foreground_scale_normalization
        )
        self.foreground_scale_canvas_fraction = 0.75
        self.foreground_scale_blend = 0.25
        self._foreground_scale_last_indices: torch.Tensor | None = None
        self._foreground_scale_last_blend_error: torch.Tensor | None = None
        self._foreground_scale_last_base_path_error: torch.Tensor | None = None
        self._foreground_scale_last_canvas_shapes: tuple[tuple[int, int], ...] = ()
        self._foreground_scale_debug = False
        self._foreground_scale_last_canonical: torch.Tensor | None = None
        if self.foreground_scale_normalization and self.foreground_token_conditioning:
            raise ValueError(
                "Foreground scale normalization and token conditioning are exclusive"
            )
        self.covariance_branch = bool(covariance_branch)
        self.covariance_dim = int(covariance_dim)
        self.covariance_standardize = bool(covariance_standardize)
        self.covariance_matrix_sqrt = bool(covariance_matrix_sqrt)
        self.intermediate_covariance_branch = bool(
            intermediate_covariance_branch
        )
        self.intermediate_covariance_block = int(intermediate_covariance_block)
        self.gradient_covariance_branch = bool(gradient_covariance_branch)
        self.semantic_topk_branch = bool(semantic_topk_branch)
        self.semantic_topk_fraction = float(semantic_topk_fraction)
        self.simpool_branch = bool(simpool_branch)
        self.vertical_branches = bool(vertical_branches)
        self.part_aligned_axis = bool(part_aligned_axis)
        self.domain_specific_bn = bool(domain_specific_bn)
        self.modality_specific_bn = bool(modality_specific_bn)
        self.token_mixstyle = bool(token_mixstyle)
        self.mixstyle_probability = float(mixstyle_probability)
        self.mixstyle_alpha = float(mixstyle_alpha)
        self.frozen_prefix_adaptformer = bool(frozen_prefix_adaptformer)
        self.adaptformer_dim = int(adaptformer_dim)
        self.adaptformer_scale = float(adaptformer_scale)
        self.frozen_prefix_qv_lora = bool(frozen_prefix_qv_lora)
        self.qv_lora_rank = int(qv_lora_rank)
        self.qv_lora_part_rank = int(qv_lora_part_rank)
        self.part_routed_qv_lora = self.qv_lora_part_rank > 0
        self.qv_lora_alpha = float(qv_lora_alpha)
        self.qv_lora_dropout = float(qv_lora_dropout)
        self.suffix_head_qv_lora = bool(suffix_head_qv_lora)
        self.frozen_prefix_convpass = bool(frozen_prefix_convpass)
        self.convpass_dim = int(convpass_dim)
        self.convpass_scale = float(convpass_scale)
        self.convpass_dropout = float(convpass_dropout)
        self.head_convpass_attention = bool(head_convpass_attention)
        self.part_mlp_expert_blocks = int(part_mlp_expert_blocks)
        self.pattern_a2gc_branch = bool(pattern_a2gc_branch)
        self.hierarchical_slot_architecture = bool(
            hierarchical_slot_architecture
        )
        self.dense_correspondence_training = bool(
            dense_correspondence_training
        )
        self.identity_query_pooling = bool(identity_query_pooling)
        self.part_routed_cross_level_texture = bool(
            part_routed_cross_level_texture
        )
        self.head_routed_cross_level_texture = bool(
            head_routed_cross_level_texture
        )
        if (
            self.part_routed_cross_level_texture
            and self.head_routed_cross_level_texture
        ):
            raise ValueError(
                "Part-routed and head-routed cross-level texture are exclusive"
            )
        self.head_routed_cross_level_texture_enabled = (
            self.head_routed_cross_level_texture
        )
        self.cross_level_texture_pyramid = bool(
            cross_level_texture_pyramid
            or self.part_routed_cross_level_texture
            or self.head_routed_cross_level_texture
        )
        self.image_frequency_texture_side = bool(
            image_frequency_texture_side
        )
        self.body_to_head_distillation = bool(body_to_head_distillation)
        self.head_tail_expert_blocks = int(head_tail_expert_blocks)
        self.jpm_local_branches = int(jpm_local_branches)
        self.jpm_shift = 5
        self.jpm_shuffle_groups = 2
        self.prototype_memory = bool(prototype_memory)
        self.prototype_momentum = float(prototype_momentum)
        self.prototype_mix = float(prototype_mix)
        self.prototype_transport_rank = int(prototype_transport_rank)
        self.prototype_transport_scale = float(prototype_transport_scale)
        self.part_topology_supcon = bool(part_topology_supcon)
        self.training_instance_queue = bool(training_instance_queue)
        self.instance_queue_capacity = int(instance_queue_capacity)
        self.decision_aligned_classification = bool(
            decision_aligned_classification
        )
        self.source_paired_logit_distillation = bool(
            source_paired_logit_distillation
        )
        self.source_paired_logit_temperature = 0.10
        self.source_paired_logit_weight = 0.10
        self._instance_queue_last_current_valid = 0
        self._instance_queue_last_augmented_valid = 0
        self.head_two_view_part_balanced_supcon = bool(
            head_two_view_part_balanced_supcon
        )
        self.external_head_representation = bool(external_head_representation)
        self.ordered_head_grid_expert = bool(ordered_head_grid_expert)
        if self.ordered_head_grid_expert and (
            head_identity_expert
            or head_tail_other_expert
            or self.external_head_representation
        ):
            raise ValueError("Ordered head grid expert is an exclusive expert mode")
        if head_identity_expert and head_tail_other_expert:
            raise ValueError("Head expert modes are mutually exclusive")
        self.head_tail_other_expert = bool(head_tail_other_expert)
        self.head_identity_expert = bool(
            head_identity_expert
            or self.head_tail_other_expert
            or self.ordered_head_grid_expert
        )
        self.head_identity_expert_branch_count = (
            10 if self.ordered_head_grid_expert else 0
        )
        self.head_identity_expert_output_classes = num_classes + int(
            self.head_tail_other_expert
        )
        self.head_identity_expert_detach = bool(head_identity_expert_detach)
        self.head_identity_expert_inference = bool(
            head_identity_expert_inference
        )
        self.head_identity_expert_loss_weight = float(
            head_identity_expert_loss_weight
        )
        self.head_identity_expert_mix = (
            0.10
            if self.head_tail_other_expert
            else 0.20
            if self.ordered_head_grid_expert
            else 0.25
        )
        self.head_identity_expert_routing_enabled = (
            self.head_identity_expert_inference
        )
        self._head_identity_expert_branch_cosine: torch.Tensor | None = None
        self._head_identity_expert_score: torch.Tensor | None = None
        self.quality_h = float(quality_h)
        self.quality_t_alpha = float(quality_t_alpha)
        if self.subcenters < 1:
            raise ValueError("subcenters must be positive")
        if self.head_identity_expert and (
            self.bidirectional
            or self.subcenters > 1
            or self.domain_specific_bn
            or self.modality_specific_bn
            or self.identity_query_pooling
            or self.prototype_memory
        ):
            raise ValueError(
                "V6.1 head expert requires standard single-center V4.3 heads"
            )
        if self.head_identity_expert_loss_weight < 0.0:
            raise ValueError("Head expert loss weight must be non-negative")
        if self.external_head_representation and (
            not self.head_identity_expert
            or self.head_identity_expert_detach
            or not self.head_identity_expert_inference
            or self.head_tail_other_expert
            or not self.frozen_prefix_adaptformer
            or int(freeze_blocks) != 24
            or not self.covariance_branch
            or self.lingbot_backbone
            or self.spatial_backbone
        ):
            raise ValueError(
                "External head representation requires the fixed V4.3 head expert path"
            )
        if not self.head_identity_expert and (
            not self.head_identity_expert_detach
            or not self.head_identity_expert_inference
            or abs(self.head_identity_expert_loss_weight - 0.50) > 1e-12
        ):
            raise ValueError("Head expert controls require a head expert")
        if self.prototype_transport_rank < 0:
            raise ValueError("prototype transport rank must be non-negative")
        if self.prototype_transport_rank > 0 and (
            self.bidirectional or self.subcenters > 1
        ):
            raise ValueError(
                "prototype transport requires shared-plus-delta single-center heads"
            )
        if self.prototype_transport_rank > 0 and not (
            0.0 < self.prototype_transport_scale <= 1.0
        ):
            raise ValueError("prototype transport scale must be in (0, 1]")
        if self.training_instance_queue and not self.part_topology_supcon:
            raise ValueError("Training instance queue requires topology SupCon")
        if self.source_paired_logit_distillation and not (
            self.part_topology_supcon and self.training_instance_queue
        ):
            raise ValueError(
                "Source-paired logit distillation requires Q topology and queue"
            )
        if self.instance_queue_capacity <= 0:
            raise ValueError("Instance queue capacity must be positive")
        if self.part_adapter_bottleneck < 0:
            raise ValueError("part_adapter_bottleneck must be non-negative")
        if self.class_adaptive_margin and self.head_class_adaptive_margin:
            raise ValueError("full and head-only class-adaptive margins are mutually exclusive")
        if self.quality_adaptive_margin and (
            self.class_adaptive_margin or self.head_class_adaptive_margin
        ):
            raise ValueError("quality- and class-adaptive margins are mutually exclusive")
        if self.subcenters > 1 and bidirectional:
            raise ValueError("subcenters are currently supported for MGN only")
        if self.covariance_branch and bidirectional:
            raise ValueError("covariance and bidirectional branches are mutually exclusive")
        if (
            self.covariance_branch or self.gradient_covariance_branch
        ) and self.covariance_dim < 2:
            raise ValueError("covariance_dim must be at least two")
        if self.covariance_standardize and not self.covariance_branch:
            raise ValueError("covariance standardization requires covariance branch")
        if self.covariance_matrix_sqrt and not self.covariance_branch:
            raise ValueError("matrix square-root normalization requires covariance branch")
        if self.intermediate_covariance_branch and not self.covariance_branch:
            raise ValueError("cross-layer covariance requires the final covariance branch")
        if self.intermediate_covariance_branch and self.lingbot_backbone:
            raise ValueError("cross-layer covariance is currently implemented for timm DINOv3")
        if not 0.0 < self.semantic_topk_fraction <= 1.0:
            raise ValueError("semantic top-k fraction must be in (0, 1]")
        if self.domain_specific_bn and self.modality_specific_bn:
            raise ValueError("part and modality BN modes are mutually exclusive")
        if self.part_aligned_axis and (self.bidirectional or self.vertical_branches):
            raise ValueError("aligned-axis and dual-axis MGN modes are mutually exclusive")
        if self.continuous_geometry_conditioning and (
            self.lingbot_backbone or self.spatial_backbone
        ):
            raise ValueError(
                "Continuous crop geometry requires a timm Vision Transformer"
            )
        if self.foreground_auxiliary and not self.continuous_geometry_conditioning:
            raise ValueError(
                "Foreground auxiliary supervision is isolated to V2.61 geometry"
            )
        if self.foreground_token_conditioning and (
            not self.continuous_geometry_conditioning
            or self.lingbot_backbone
            or self.spatial_backbone
        ):
            raise ValueError(
                "Foreground token conditioning requires the geometry-aware timm ViT"
            )
        if self.token_mixstyle and self.lingbot_backbone:
            raise ValueError("token MixStyle is currently implemented for timm backbones")
        if not 0.0 <= self.mixstyle_probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0, 1]")
        if self.mixstyle_alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive")
        if self.frozen_prefix_adaptformer and (
            self.lingbot_backbone or self.spatial_backbone
        ):
            raise ValueError("AdaptFormer prefix requires a timm Vision Transformer")
        if self.frozen_prefix_convpass and (
            self.lingbot_backbone or self.spatial_backbone
        ):
            raise ValueError("ConvPass prefix requires a timm Vision Transformer")
        if self.frozen_prefix_adaptformer and self.frozen_prefix_convpass:
            raise ValueError("AdaptFormer and ConvPass prefixes are mutually exclusive")
        if self.head_convpass_attention and not self.frozen_prefix_adaptformer:
            raise ValueError("Head ConvPass attention requires shared AdaptFormer")
        if self.head_convpass_attention and self.frozen_prefix_convpass:
            raise ValueError("Head-routed and full ConvPass modes are mutually exclusive")
        if self.frozen_prefix_adaptformer and freeze_blocks <= 0:
            raise ValueError("AdaptFormer prefix requires at least one frozen block")
        if self.frozen_prefix_qv_lora and (
            self.lingbot_backbone or self.spatial_backbone
        ):
            raise ValueError("Q/V LoRA prefix requires a timm Vision Transformer")
        if self.frozen_prefix_qv_lora and freeze_blocks <= 0:
            raise ValueError("Q/V LoRA prefix requires at least one frozen block")
        if self.qv_lora_rank < 1 or self.qv_lora_alpha <= 0.0:
            raise ValueError("Q/V LoRA rank and alpha must be positive")
        if self.qv_lora_part_rank < 0:
            raise ValueError("Q/V LoRA part rank cannot be negative")
        if self.part_routed_qv_lora and not self.frozen_prefix_qv_lora:
            raise ValueError("Part-routed Q/V LoRA requires prefix Q/V LoRA")
        if not 0.0 <= self.qv_lora_dropout < 1.0:
            raise ValueError("Q/V LoRA dropout must be in [0, 1)")
        if self.frozen_prefix_qv_lora and self.frozen_prefix_convpass:
            raise ValueError("Q/V LoRA and ConvPass attention are mutually exclusive")
        if self.frozen_prefix_qv_lora and self.head_convpass_attention:
            raise ValueError("Q/V LoRA and head ConvPass attention are isolated")
        if self.suffix_head_qv_lora and (
            self.lingbot_backbone
            or self.spatial_backbone
            or self.frozen_prefix_qv_lora
            or self.head_convpass_attention
            or self.frozen_prefix_convpass
        ):
            raise ValueError("Suffix head Q/V LoRA requires an otherwise unwrapped timm attention")
        if self.suffix_head_qv_lora and (
            len(self.backbone.blocks) != 32 or int(freeze_blocks) != 24
        ):
            raise ValueError("Suffix head Q/V LoRA requires the fixed H+ 24/8 boundary")
        if self.frozen_prefix_convpass and freeze_blocks <= 0:
            raise ValueError("ConvPass prefix requires at least one frozen block")
        if self.head_convpass_attention and freeze_blocks <= 0:
            raise ValueError("Head ConvPass attention requires frozen prefix blocks")
        if self.part_mlp_expert_blocks < 0:
            raise ValueError("Part MLP expert block count cannot be negative")
        if self.part_mlp_expert_blocks:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("Part MLP experts require a timm Vision Transformer")
            if self.part_mlp_expert_blocks > len(self.backbone.blocks) - freeze_blocks:
                raise ValueError("Part MLP experts must stay inside the trainable suffix")
        if self.pattern_a2gc_branch and (
            self.lingbot_backbone or self.spatial_backbone
        ):
            raise ValueError("Pattern A2GC requires a timm Vision Transformer")
        if self.hierarchical_slot_architecture:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("Hierarchical slots require a timm Vision Transformer")
            if not self.covariance_branch:
                raise ValueError("Hierarchical slots retain the covariance branch")
            incompatible_slot_flags = (
                self.bidirectional
                or self.vertical_branches
                or self.part_aligned_axis
                or self.intermediate_covariance_branch
                or self.gradient_covariance_branch
                or self.semantic_topk_branch
                or self.simpool_branch
                or self.pattern_a2gc_branch
            )
            if incompatible_slot_flags:
                raise ValueError(
                    "Hierarchical slots replace fixed/extra local descriptor branches"
                )
        if self.dense_correspondence_training:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("Dense correspondence requires a timm Vision Transformer")
            if self.hierarchical_slot_architecture or self.intermediate_covariance_branch:
                raise ValueError("Dense correspondence requires its own block-24 capture")
        if self.identity_query_pooling:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("Identity-query pooling requires a timm Vision Transformer")
            if self.hierarchical_slot_architecture:
                raise ValueError("Identity-query pooling retains the fixed MGN branches")
            if self.subcenters > 1 or self.bidirectional or self.learned_branch_gates:
                raise ValueError(
                    "Identity-query pooling requires the standard shared/delta classifier"
                )
            if self.prototype_memory:
                raise ValueError("Identity-query pooling and prototype memory are separate hypotheses")
        if self.cross_level_texture_pyramid:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError(
                    "Cross-level texture pyramid requires a timm Vision Transformer"
                )
            incompatible_texture_flags = (
                self.hierarchical_slot_architecture
                or self.dense_correspondence_training
                or self.identity_query_pooling
                or self.intermediate_covariance_branch
                or self.pattern_a2gc_branch
            )
            if incompatible_texture_flags:
                raise ValueError(
                    "Cross-level texture pyramid is an isolated V2.42 extension"
                )
        if (
            self.image_frequency_texture_side
            and not self.part_routed_cross_level_texture
        ):
            raise ValueError(
                "Image-frequency side network requires the V2.51 routed mother"
            )
        if (
            self.body_to_head_distillation
            and not self.part_routed_cross_level_texture
        ):
            raise ValueError(
                "Body-to-head distillation requires the V2.51 routed mother"
            )
        if self.body_to_head_distillation and self.image_frequency_texture_side:
            raise ValueError(
                "Body-to-head and image-frequency stages are isolated hypotheses"
            )
        if self.head_tail_expert_blocks < 0:
            raise ValueError("Head-tail expert block count cannot be negative")
        if self.head_tail_expert_blocks:
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("Head-tail experts require a timm Vision Transformer")
            if not self.part_routed_cross_level_texture:
                raise ValueError("Head-tail experts require the V2.51 routed mother")
            if self.head_tail_expert_blocks > len(self.backbone.blocks):
                raise ValueError("Head-tail expert count exceeds backbone depth")
            if self.part_mlp_expert_blocks:
                raise ValueError("Head-tail and three-part MLP experts are isolated")
            if self.body_to_head_distillation or self.image_frequency_texture_side:
                raise ValueError("Only one V2.51 extension may be active")
        if self.jpm_local_branches < 0:
            raise ValueError("JPM local branch count cannot be negative")
        if self.jpm_local_branches:
            if self.jpm_local_branches != 4:
                raise ValueError("The declared JPM design requires exactly four branches")
            if self.lingbot_backbone or self.spatial_backbone:
                raise ValueError("JPM requires the timm DINOv3 Transformer backbone")
            if self.hierarchical_slot_architecture:
                raise ValueError("JPM retains the fixed MGN/covariance branches")
            if self.part_mlp_expert_blocks or self.head_tail_expert_blocks:
                raise ValueError("JPM and tail-expert hypotheses are isolated")
            if getattr(self.backbone, "rope", None) is None:
                raise ValueError("JPM requires the backbone's published rotary embedding")
            if int(self.backbone.num_prefix_tokens) != 5:
                raise ValueError("DINOv3-H+ JPM expects CLS plus four register tokens")
        if self.adaptformer_dim < 1:
            raise ValueError("AdaptFormer bottleneck must be positive")
        if self.adaptformer_scale <= 0.0:
            raise ValueError("AdaptFormer scale must be positive")
        if self.convpass_dim < 1:
            raise ValueError("ConvPass bottleneck must be positive")
        if self.convpass_scale <= 0.0:
            raise ValueError("ConvPass scale must be positive")
        if not 0.0 <= self.convpass_dropout < 1.0:
            raise ValueError("ConvPass dropout must be in [0, 1)")
        if not 0.0 <= self.prototype_momentum < 1.0:
            raise ValueError("prototype momentum must be in [0, 1)")
        if not 0.0 <= self.prototype_mix <= 1.0:
            raise ValueError("prototype mix must be in [0, 1]")
        self._active_parts: torch.Tensor | None = None
        self._active_image_geometry: torch.Tensor | None = None
        self._active_foreground_mask: torch.Tensor | None = None
        self._geometry_last_residual: torch.Tensor | None = None
        self._foreground_token_last_residual: torch.Tensor | None = None
        self._foreground_auxiliary_logits: torch.Tensor | None = None
        self._intermediate_tokens: torch.Tensor | None = None
        self.cross_level_blocks = (7, 15, 23)
        self.cross_level_dim = 128
        self.cross_level_scale = 0.1
        self._cross_level_tokens: dict[int, torch.Tensor] = {}
        self._cross_level_last_residual: torch.Tensor | None = None
        self._cross_level_hooks: list[object] = []
        self._cross_level_texture_stage_only = False
        self._image_texture_stage_only = False
        self._body_head_stage_only = False
        self._head_tail_expert_stage_only = False
        self._image_texture_last_residual: torch.Tensor | None = None
        self._body_head_last_residual: torch.Tensor | None = None
        self._body_head_alignment_loss: torch.Tensor | None = None
        self._jpm_capture_enabled = False
        self._jpm_last_input_tokens: torch.Tensor | None = None
        self._jpm_last_rope: torch.Tensor | None = None
        self._jpm_last_patch_order: torch.Tensor | None = None
        self._jpm_last_group_patch_counts: tuple[int, ...] = ()
        self._jpm_last_group_rope_counts: tuple[int, ...] = ()
        capture_block: int | None = None
        if self.intermediate_covariance_branch:
            capture_block = self.intermediate_covariance_block
        elif self.hierarchical_slot_architecture:
            capture_block = 23
        elif self.dense_correspondence_training:
            capture_block = 23
        if capture_block is not None:
            if not 0 <= capture_block < len(self.backbone.blocks):
                raise ValueError("intermediate feature block is out of range")
            self._intermediate_hook = self.backbone.blocks[
                capture_block
            ].register_forward_hook(self._capture_intermediate_tokens)
        if self.cross_level_texture_pyramid:
            if any(
                block_index >= len(self.backbone.blocks)
                for block_index in self.cross_level_blocks
            ):
                raise ValueError("Cross-level texture block is out of range")
            for block_index in self.cross_level_blocks:
                self._cross_level_hooks.append(
                    self.backbone.blocks[block_index].register_forward_hook(
                        self._make_cross_level_capture_hook(block_index)
                    )
                )
        if self.part_side_embedding_enabled:
            # TransReID-style side information, adapted to the three official
            # crop domains.  Zero initialization keeps the initial network
            # exactly identical to the V2.9 backbone.
            self.part_side_embedding = nn.Parameter(
                torch.zeros(3, backbone_dim)
            )
            self._part_side_hook = self.backbone.norm_pre.register_forward_hook(
                self._inject_part_side_embedding
            )
        if self.continuous_geometry_conditioning:
            # The released crop's original area/aspect are standardized using
            # fold-train rows only. Part is a public task input, not a learned
            # or identity-dependent route. The final zero layer makes the
            # complete initial function exactly equal to the generic baseline.
            self.geometry_conditioner = nn.Sequential(
                nn.Linear(5, 128),
                nn.GELU(),
                nn.Linear(128, 3 * backbone_dim),
            )
            nn.init.zeros_(self.geometry_conditioner[-1].weight)
            nn.init.zeros_(self.geometry_conditioner[-1].bias)
            self.geometry_condition_scale = 0.1
            self._geometry_hook = self.backbone.norm_pre.register_forward_hook(
                self._inject_continuous_geometry
            )
        if self.foreground_token_conditioning:
            # One zero-initialized foreground/background contrast vector per
            # released part is injected before all Transformer blocks.
            self.foreground_token_embedding = nn.Parameter(
                torch.zeros(3, backbone_dim)
            )
            self.foreground_token_scale = 0.1
            self._foreground_token_hook = (
                self.backbone.norm_pre.register_forward_hook(
                    self._inject_foreground_token_embedding
                )
            )
        if self.token_mixstyle:
            # Feature-statistics mixing happens immediately before the ViT
            # blocks, matching MixStyle's early-feature placement while
            # remaining outside activation-checkpoint recomputation.
            self._mixstyle_hook = self.backbone.norm_pre.register_forward_hook(
                self._mix_token_style
            )
        self._part_routing_context = PartRoutingContext()
        self.prefix_adaptformer_modules: list[ParallelAdaptMlp] = []
        if self.frozen_prefix_adaptformer:
            prefix_count = min(int(freeze_blocks), len(self.backbone.blocks))
            for block_index in range(prefix_count):
                block = self.backbone.blocks[block_index]
                adapted_mlp = ParallelAdaptMlp(
                    block.mlp,
                    feature_dim=backbone_dim,
                    bottleneck_dim=self.adaptformer_dim,
                    scale=self.adaptformer_scale,
                )
                block.mlp = adapted_mlp
                self.prefix_adaptformer_modules.append(adapted_mlp)
        self.external_head_prefix_adapters = nn.ModuleList()
        self.external_head_tail_blocks = nn.ModuleList()
        self.external_head_norm: nn.Module | None = None
        self._external_head_capture_enabled = False
        self._external_head_initial_tokens: torch.Tensor | None = None
        self._external_head_rope: torch.Tensor | None = None
        self._external_head_capture_hook: object | None = None
        if self.external_head_representation:
            if len(self.backbone.blocks) != 32 or len(
                self.prefix_adaptformer_modules
            ) != 24:
                raise ValueError("External head representation expects DINOv3-H+ 32/24")
            # The external tensors are overwritten by the strict loader. Keep
            # their construction from perturbing the fresh V4.3 target-head draw.
            cpu_rng_state = torch.get_rng_state()
            self.external_head_prefix_adapters.extend(
                AdaptFormerBranch(
                    feature_dim=backbone_dim,
                    bottleneck_dim=self.adaptformer_dim,
                    scale=self.adaptformer_scale,
                )
                for _ in range(24)
            )
            self.external_head_tail_blocks.extend(
                copy.deepcopy(self.backbone.blocks[index]) for index in (30, 31)
            )
            self.external_head_norm = copy.deepcopy(self.backbone.norm)
            torch.set_rng_state(cpu_rng_state)
            self._external_head_capture_hook = self.backbone.blocks[
                0
            ].register_forward_pre_hook(
                self._capture_external_head_input,
                with_kwargs=True,
            )
        self.prefix_qv_lora_modules: list[nn.Module] = []
        if self.frozen_prefix_qv_lora:
            prefix_count = min(int(freeze_blocks), len(self.backbone.blocks))
            for block_index in range(prefix_count):
                attention = self.backbone.blocks[block_index].attn
                if getattr(attention, "q_bias", None) is not None:
                    raise ValueError(
                        "Q/V LoRA cannot wrap an attention path that bypasses qkv.forward"
                    )
                qkv = getattr(attention, "qkv", None)
                adapted_qkv = (
                    PartRoutedQvLowRankLinear(
                        qkv,
                        routing_context=self._part_routing_context,
                        shared_rank=self.qv_lora_rank,
                        part_rank=self.qv_lora_part_rank,
                        alpha=self.qv_lora_alpha,
                        dropout=self.qv_lora_dropout,
                    )
                    if self.part_routed_qv_lora
                    else QvLowRankLinear(
                        qkv,
                        rank=self.qv_lora_rank,
                        alpha=self.qv_lora_alpha,
                        dropout=self.qv_lora_dropout,
                    )
                )
                attention.qkv = adapted_qkv
                self.prefix_qv_lora_modules.append(adapted_qkv)
        self.suffix_head_qv_lora_modules: list[HeadRoutedQvLowRankLinear] = []
        if self.suffix_head_qv_lora:
            for block_index in range(24, 32):
                attention = self.backbone.blocks[block_index].attn
                if getattr(attention, "q_bias", None) is not None:
                    raise ValueError(
                        "Suffix head Q/V LoRA cannot wrap a path bypassing qkv.forward"
                    )
                adapted_qkv = HeadRoutedQvLowRankLinear(
                    attention.qkv,
                    routing_context=self._part_routing_context,
                    rank=self.qv_lora_rank,
                    alpha=self.qv_lora_alpha,
                )
                attention.qkv = adapted_qkv
                self.suffix_head_qv_lora_modules.append(adapted_qkv)
        self.prefix_convpass_modules: list[ParallelConvPass] = []
        if self.frozen_prefix_convpass:
            prefix_count = min(int(freeze_blocks), len(self.backbone.blocks))
            token_prefix_count = int(self.backbone.num_prefix_tokens)
            for block_index in range(prefix_count):
                block = self.backbone.blocks[block_index]
                adapted_attention = ParallelConvPass(
                    block.attn,
                    feature_dim=backbone_dim,
                    prefix_tokens=token_prefix_count,
                    bottleneck_dim=self.convpass_dim,
                    scale=self.convpass_scale,
                    dropout=self.convpass_dropout,
                )
                adapted_mlp = ParallelConvPass(
                    block.mlp,
                    feature_dim=backbone_dim,
                    prefix_tokens=token_prefix_count,
                    bottleneck_dim=self.convpass_dim,
                    scale=self.convpass_scale,
                    dropout=self.convpass_dropout,
                )
                block.attn = adapted_attention
                block.mlp = adapted_mlp
                self.prefix_convpass_modules.extend(
                    [adapted_attention, adapted_mlp]
                )
        if self.head_convpass_attention:
            prefix_count = min(int(freeze_blocks), len(self.backbone.blocks))
            token_prefix_count = int(self.backbone.num_prefix_tokens)
            for block_index in range(prefix_count):
                block = self.backbone.blocks[block_index]
                adapted_attention = ParallelConvPass(
                    block.attn,
                    feature_dim=backbone_dim,
                    prefix_tokens=token_prefix_count,
                    bottleneck_dim=self.convpass_dim,
                    scale=self.convpass_scale,
                    dropout=self.convpass_dropout,
                    routing_context=self._part_routing_context,
                    active_part=0,
                )
                block.attn = adapted_attention
                self.prefix_convpass_modules.append(adapted_attention)
        self.part_mlp_expert_modules: list[PartRoutedMlpExperts] = []
        if self.part_mlp_expert_blocks:
            start_index = len(self.backbone.blocks) - self.part_mlp_expert_blocks
            for block_index in range(start_index, len(self.backbone.blocks)):
                block = self.backbone.blocks[block_index]
                routed_mlp = PartRoutedMlpExperts(
                    block.mlp,
                    routing_context=self._part_routing_context,
                    num_parts=3,
                )
                block.mlp = routed_mlp
                self.part_mlp_expert_modules.append(routed_mlp)
        self.head_tail_expert_modules = nn.ModuleList()
        self.head_tail_expert_block_indices: tuple[int, ...] = ()
        self._head_tail_expert_hooks: list[object] = []
        if self.head_tail_expert_blocks:
            self.register_buffer(
                "head_tail_expert_enabled",
                torch.tensor(False, dtype=torch.bool),
            )
            start_index = len(self.backbone.blocks) - self.head_tail_expert_blocks
            self.head_tail_expert_block_indices = tuple(
                range(start_index, len(self.backbone.blocks))
            )
            for block_index in self.head_tail_expert_block_indices:
                expert = copy.deepcopy(self.backbone.blocks[block_index])
                self.head_tail_expert_modules.append(expert)
                self._head_tail_expert_hooks.append(
                    self.backbone.blocks[block_index].register_forward_hook(
                        self._make_head_tail_expert_hook(expert)
                    )
                )
        self.jpm_refinement_block: nn.Module | None = None
        self.jpm_refinement_norm: nn.Module | None = None
        self._jpm_capture_hook: object | None = None
        if self.jpm_local_branches:
            # Copy before registering the source hook so the local block cannot
            # inherit a hook that would overwrite the captured base sequence.
            self.jpm_refinement_block = copy.deepcopy(self.backbone.blocks[-1])
            self.jpm_refinement_norm = copy.deepcopy(self.backbone.norm)
            if not hasattr(self.jpm_refinement_block, "attn"):
                raise ValueError("JPM refinement block has no attention module")
            self.jpm_refinement_block.attn.num_prefix_tokens = 1
            self._jpm_capture_hook = self.backbone.blocks[-1].register_forward_pre_hook(
                self._capture_jpm_last_input,
                with_kwargs=True,
            )
        self.pattern_aggregator = (
            CropAwareA2GCAggregator(feature_dim=backbone_dim)
            if self.pattern_a2gc_branch
            else None
        )
        self.slot_aggregator = (
            HierarchicalPartSlotAggregator(feature_dim=backbone_dim)
            if self.hierarchical_slot_architecture
            else None
        )
        self._slot_reconstruction_loss: torch.Tensor | None = None
        self.dense_correspondence_topk = 16
        self.dense_correspondence_temperature = 0.07
        self.dense_correspondence_weight = 0.20
        self._dense_correspondence_tokens: torch.Tensor | None = None
        if self.dense_correspondence_training:
            self.dense_intermediate_projection = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, 64),
            )
            self.dense_final_projection = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, 64),
            )
        self.identity_query_dim = 128
        self.identity_query_temperature = 0.07
        self._identity_query_shared_score: torch.Tensor | None = None
        self._identity_query_part_score: torch.Tensor | None = None
        self._identity_query_attention_mass: tuple[
            torch.Tensor, torch.Tensor
        ] | None = None
        if self.identity_query_pooling:
            self.identity_query_patch_projection = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, self.identity_query_dim, bias=False),
            )
            self.identity_query_weight_projection = nn.Linear(
                embedding_dim, self.identity_query_dim, bias=False
            )
        if self.hierarchical_slot_architecture:
            self.branch_count = 7
            input_dims = [backbone_dim * 2] + [512] * 5
        else:
            spatial_branch_count = (
                11 if (bidirectional or self.vertical_branches) else 6
            )
            self.branch_count = (
                spatial_branch_count
                + int(self.covariance_branch)
                + int(self.intermediate_covariance_branch)
                + int(self.gradient_covariance_branch)
                + int(self.semantic_topk_branch)
                + int(self.simpool_branch)
                + int(self.pattern_a2gc_branch)
                + self.jpm_local_branches
            )
            input_dims = [backbone_dim * 2] + [backbone_dim] * (
                spatial_branch_count - 1
            )
        if self.head_identity_expert and not self.ordered_head_grid_expert:
            self.head_identity_expert_branch_count = self.branch_count
        if self.covariance_branch or self.gradient_covariance_branch:
            self.covariance_reduction = nn.Sequential(
                nn.Linear(backbone_dim, self.covariance_dim, bias=False),
                nn.LayerNorm(self.covariance_dim),
            )
        if self.covariance_branch:
            input_dims.append(self.covariance_dim * self.covariance_dim)
        if self.intermediate_covariance_branch:
            self.intermediate_covariance_reduction = nn.Sequential(
                nn.Linear(backbone_dim, self.covariance_dim, bias=False),
                nn.LayerNorm(self.covariance_dim),
            )
            input_dims.append(self.covariance_dim * self.covariance_dim)
        if self.gradient_covariance_branch:
            input_dims.append(self.covariance_dim * self.covariance_dim)
        if self.semantic_topk_branch:
            input_dims.append(backbone_dim)
        if self.simpool_branch:
            self.simpool_norm = nn.LayerNorm(backbone_dim, eps=1e-6)
            self.simpool_query = nn.Linear(backbone_dim, backbone_dim, bias=False)
            self.simpool_key = nn.Linear(backbone_dim, backbone_dim, bias=False)
            # Identity initialization begins from the pretrained patch
            # geometry instead of imposing a random attention map.
            nn.init.eye_(self.simpool_query.weight)
            nn.init.eye_(self.simpool_key.weight)
            input_dims.append(backbone_dim)
        if self.pattern_a2gc_branch:
            input_dims.append(self.pattern_aggregator.output_dim)
        if self.jpm_local_branches:
            input_dims.extend([backbone_dim] * self.jpm_local_branches)
        self.jpm_branch_start = self.branch_count - self.jpm_local_branches
        self.branch_projections = nn.ModuleList()
        self.branch_necks = nn.ModuleList()

        def make_bn_neck() -> nn.BatchNorm1d:
            neck = nn.BatchNorm1d(embedding_dim)
            nn.init.ones_(neck.weight)
            nn.init.zeros_(neck.bias)
            neck.bias.requires_grad_(False)
            return neck

        for input_dim in input_dims:
            self.branch_projections.append(
                nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.GELU(),
                )
            )
            normalization_domains = (
                3 if self.domain_specific_bn else 2
            )
            neck = (
                nn.ModuleList(
                    make_bn_neck() for _ in range(normalization_domains)
                )
                if self.domain_specific_bn or self.modality_specific_bn
                else make_bn_neck()
            )
            self.branch_necks.append(neck)
        self.part_adapters = (
            nn.ModuleList(
                [
                    ResidualPartAdapter(
                        embedding_dim,
                        bottleneck_dim=self.part_adapter_bottleneck,
                    )
                    for _ in range(3)
                ]
            )
            if self.part_adapter_bottleneck > 0
            else None
        )
        classifier_shape = (
            (self.branch_count, num_classes, self.subcenters, embedding_dim)
            if self.subcenters > 1
            else (self.branch_count, num_classes, embedding_dim)
        )
        self.shared_class_weight = nn.Parameter(torch.empty(*classifier_shape))
        if bidirectional and hierarchical_part_heads:
            self.part_class_delta = nn.Parameter(
                torch.zeros(3, self.branch_count, num_classes, embedding_dim)
            )
            self.register_buffer(
                "part_class_shrinkage", torch.zeros(3, num_classes)
            )
            self.shared_branch_gate = nn.Parameter(
                torch.zeros(self.branch_count)
            )
            self.part_branch_gate = nn.Parameter(
                torch.zeros(3, self.branch_count)
            )
        elif bidirectional:
            # Head and flank crops are different visual domains.  Keep the
            # generic visual trunk shared, but do not force their closed-set
            # identity prototypes to occupy one common classifier geometry.
            self.part_class_weight = nn.Parameter(
                torch.empty(3, self.branch_count, num_classes, embedding_dim)
            )
            nn.init.normal_(self.part_class_weight, std=0.01)
            self.shared_branch_gate = nn.Parameter(
                torch.zeros(self.branch_count)
            )
            self.part_branch_gate = nn.Parameter(
                torch.zeros(3, self.branch_count)
            )
        else:
            self.part_class_delta = nn.Parameter(
                torch.zeros(3, *classifier_shape)
            )
            if self.learned_branch_gates:
                # Zero logits reproduce V2.9's exact equal branch average at
                # initialization. Training can then learn which fixed spatial
                # granularities are reliable for each crop domain.
                self.shared_branch_gate = nn.Parameter(
                    torch.zeros(self.branch_count)
                )
                self.part_branch_gate = nn.Parameter(
                    torch.zeros(3, self.branch_count)
                )
        if self.prototype_transport_rank > 0:
            self.part_prototype_transport_down = nn.Parameter(
                torch.empty(
                    3,
                    self.branch_count,
                    self.prototype_transport_rank,
                    embedding_dim,
                )
            )
            self.part_prototype_transport_up = nn.Parameter(
                torch.zeros(
                    3,
                    self.branch_count,
                    embedding_dim,
                    self.prototype_transport_rank,
                )
            )
            nn.init.kaiming_uniform_(
                self.part_prototype_transport_down, a=math.sqrt(5)
            )
        else:
            self.register_parameter("part_prototype_transport_down", None)
            self.register_parameter("part_prototype_transport_up", None)
        nn.init.normal_(self.shared_class_weight, std=0.01)
        self.register_buffer(
            "part_class_available", torch.ones(3, num_classes, dtype=torch.bool)
        )
        if self.head_tail_other_expert:
            self.register_buffer(
                "head_tail_identity_mask",
                torch.zeros(num_classes, dtype=torch.bool),
            )
        if self.prototype_memory:
            # Non-parametric class centres are stored class-first so a whole
            # fold-train batch can update them with one index_add operation.
            # They are persistent buffers: checkpoint selection and later
            # inference use exactly the train-only state that produced each
            # validation score.
            self.register_buffer(
                "shared_prototype_memory",
                torch.zeros(num_classes, self.branch_count, embedding_dim),
            )
            self.register_buffer(
                "part_prototype_memory",
                torch.zeros(3, num_classes, self.branch_count, embedding_dim),
            )
            self.register_buffer(
                "shared_prototype_seen",
                torch.zeros(num_classes, dtype=torch.bool),
            )
            self.register_buffer(
                "part_prototype_seen",
                torch.zeros(3, num_classes, dtype=torch.bool),
            )
        if self.training_instance_queue:
            queue_width = self.branch_count * embedding_dim
            self.register_buffer(
                "instance_queue_embeddings",
                torch.zeros(
                    self.instance_queue_capacity,
                    queue_width,
                    dtype=torch.float16,
                ),
            )
            self.register_buffer(
                "instance_queue_labels",
                torch.full(
                    (self.instance_queue_capacity,), -1, dtype=torch.long
                ),
            )
            self.register_buffer(
                "instance_queue_parts",
                torch.full(
                    (self.instance_queue_capacity,), -1, dtype=torch.long
                ),
            )
            self.register_buffer(
                "instance_queue_sources",
                torch.full(
                    (self.instance_queue_capacity,), -1, dtype=torch.long
                ),
            )
            self.register_buffer(
                "instance_queue_pointer", torch.zeros((), dtype=torch.long)
            )
            self.register_buffer(
                "instance_queue_size", torch.zeros((), dtype=torch.long)
            )
        if self.quality_adaptive_margin:
            self.register_buffer("quality_batch_mean", torch.tensor(20.0))
            self.register_buffer("quality_batch_std", torch.tensor(100.0))
        if self.class_adaptive_margin or self.head_class_adaptive_margin:
            self.register_buffer(
                "shared_class_margins",
                torch.full((num_classes,), float(arc_margin)),
            )
            self.register_buffer(
                "part_class_margins",
                torch.full((3, num_classes), float(arc_margin)),
            )
        # Construct the optional zero-start texture path only after every
        # incumbent V2.42 parameter has been initialized.  This preserves the
        # exact seeded initialization of all seven existing heads/classifiers.
        if self.cross_level_texture_pyramid:
            self.cross_level_lateral_projections = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(
                        backbone_dim,
                        self.cross_level_dim,
                        bias=False,
                    ),
                    nn.GELU(),
                )
                for _ in self.cross_level_blocks
            )
            self.cross_level_depthwise = nn.ModuleList(
                nn.Conv2d(
                    self.cross_level_dim,
                    self.cross_level_dim,
                    kernel_size=3,
                    padding=1,
                    groups=self.cross_level_dim,
                )
                for _ in self.cross_level_blocks
            )
            if self.part_routed_cross_level_texture:
                self.cross_level_part_fusions = nn.ModuleList(
                    nn.Sequential(
                        nn.Conv2d(
                            len(self.cross_level_blocks)
                            * self.cross_level_dim,
                            self.cross_level_dim,
                            kernel_size=1,
                        ),
                        nn.GELU(),
                    )
                    for _ in range(3)
                )
                self.cross_level_part_expansions = nn.ModuleList(
                    nn.Conv2d(
                        self.cross_level_dim,
                        backbone_dim,
                        kernel_size=1,
                        bias=False,
                    )
                    for _ in range(3)
                )
                for expansion in self.cross_level_part_expansions:
                    nn.init.zeros_(expansion.weight)
            else:
                self.cross_level_fusion = nn.Sequential(
                    nn.Conv2d(
                        len(self.cross_level_blocks) * self.cross_level_dim,
                        self.cross_level_dim,
                        kernel_size=1,
                    ),
                    nn.GELU(),
                )
                self.cross_level_expansion = nn.Conv2d(
                    self.cross_level_dim,
                    backbone_dim,
                    kernel_size=1,
                    bias=False,
                )
                nn.init.zeros_(self.cross_level_expansion.weight)
        if self.image_frequency_texture_side:
            self.image_texture_encoder = MultiScaleImageTextureEncoder()
            self.image_texture_part_fusions = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=1),
                    nn.GELU(),
                )
                for _ in range(3)
            )
            self.image_texture_part_expansions = nn.ModuleList(
                nn.Conv2d(
                    128,
                    backbone_dim,
                    kernel_size=1,
                    bias=False,
                )
                for _ in range(3)
            )
            for expansion in self.image_texture_part_expansions:
                nn.init.zeros_(expansion.weight)
        if self.body_to_head_distillation:
            self.body_head_adapter_scale = 0.1
            self.body_head_alignment_weight = 0.20
            self.head_identity_adapters = nn.ModuleList(
                ResidualPartAdapter(embedding_dim, bottleneck_dim=64)
                for _ in range(self.branch_count)
            )
            self.register_buffer(
                "body_teacher_prototypes",
                torch.zeros(
                    num_classes,
                    self.branch_count,
                    embedding_dim,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "body_teacher_available",
                torch.zeros(num_classes, dtype=torch.bool),
            )
        if self.foreground_auxiliary:
            # This head is supervised only on reliable fold-train SAM masks.
            # It is created after all incumbent V2.61 parameters so their
            # seeded generic initialization remains exactly unchanged.
            self.foreground_auxiliary_head = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Linear(backbone_dim, 1),
            )
        if self.head_identity_expert:
            if self.ordered_head_grid_expert:
                self.head_identity_expert_grid_dim = 256
                self.head_identity_expert_grid_projection = nn.Sequential(
                    nn.LayerNorm(backbone_dim),
                    nn.Linear(
                        backbone_dim,
                        self.head_identity_expert_grid_dim,
                        bias=False,
                    ),
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.head_identity_expert_grid_dim,
                    nhead=8,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.head_identity_expert_grid_transformer = (
                    nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=2,
                        norm=nn.LayerNorm(self.head_identity_expert_grid_dim),
                        enable_nested_tensor=False,
                    )
                )
                frequency = torch.exp(
                    -math.log(10000.0)
                    * torch.arange(64, dtype=torch.float32)
                    / 64.0
                )
                positions = torch.zeros(
                    1,
                    self.head_identity_expert_branch_count,
                    self.head_identity_expert_grid_dim,
                    dtype=torch.float32,
                )
                for token_index in range(1, 10):
                    row = (token_index - 1) // 3
                    column = (token_index - 1) % 3
                    positions[0, token_index, :64] = torch.sin(row * frequency)
                    positions[0, token_index, 64:128] = torch.cos(row * frequency)
                    positions[0, token_index, 128:192] = torch.sin(
                        column * frequency
                    )
                    positions[0, token_index, 192:] = torch.cos(
                        column * frequency
                    )
                self.register_buffer(
                    "head_identity_expert_grid_position",
                    positions,
                    persistent=True,
                )
                self.head_identity_expert_projections = nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(
                            self.head_identity_expert_grid_dim,
                            embedding_dim,
                        ),
                        nn.GELU(),
                    )
                    for _ in range(self.head_identity_expert_branch_count)
                )
                self.head_identity_expert_necks = nn.ModuleList(
                    make_bn_neck()
                    for _ in range(self.head_identity_expert_branch_count)
                )
                expert_weight = torch.empty(
                    self.head_identity_expert_branch_count,
                    num_classes,
                    embedding_dim,
                )
                nn.init.xavier_uniform_(expert_weight)
                self._head_grid_last_head_count = 0
                self._head_grid_last_valid_mask_count = 0
                self._head_grid_last_empty_cell_fallbacks = 0
            else:
                # V6.1 is initialized only after every base head/classifier
                # draw. deepcopy/clone consumes no RNG and never reads a
                # competition checkpoint.
                self.head_identity_expert_projections = copy.deepcopy(
                    self.branch_projections
                )
                self.head_identity_expert_necks = copy.deepcopy(
                    self.branch_necks
                )
                expert_weight = self.shared_class_weight.detach().clone()
            if self.head_tail_other_expert:
                expert_weight = torch.cat(
                    [expert_weight, expert_weight.mean(dim=1, keepdim=True)],
                    dim=1,
                )
            self.head_identity_expert_class_weight = nn.Parameter(expert_weight)
            self.external_head_covariance_reduction = (
                copy.deepcopy(self.covariance_reduction)
                if self.external_head_representation
                else None
            )
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.arc_scale = arc_scale
        self.arc_margin = arc_margin
        self.part_delta_scale = part_delta_scale
        self.cos_m = math.cos(arc_margin)
        self.sin_m = math.sin(arc_margin)
        self.threshold = math.cos(math.pi - arc_margin)
        self.margin_correction = math.sin(math.pi - arc_margin) * arc_margin
        suffix = f"_sc{self.subcenters}" if self.subcenters > 1 else ""
        if self.part_adapter_bottleneck > 0:
            suffix += f"_pa{self.part_adapter_bottleneck}"
        if self.quality_adaptive_margin:
            suffix += "_ada"
        if self.class_adaptive_margin:
            suffix += "_ldam"
        if self.head_class_adaptive_margin:
            suffix += "_headldam"
        if self.learned_branch_gates:
            suffix += "_gate"
        if self.part_side_embedding_enabled:
            suffix += "_sie"
        if self.continuous_geometry_conditioning:
            suffix += "_geosie2"
        if self.foreground_auxiliary:
            suffix += "_fgaux"
        if self.covariance_branch:
            statistic = (
                "mpncov"
                if self.covariance_matrix_sqrt
                else ("corr" if self.covariance_standardize else "cov")
            )
            suffix += f"_{statistic}{self.covariance_dim}"
        if self.intermediate_covariance_branch:
            suffix += f"_xcovb{self.intermediate_covariance_block + 1}"
        if self.gradient_covariance_branch:
            suffix += f"_gradcov{self.covariance_dim}"
        if self.semantic_topk_branch:
            suffix += f"_semk{int(round(100 * self.semantic_topk_fraction))}"
        if self.simpool_branch:
            suffix += "_simpool"
        if self.vertical_branches:
            suffix += "_orthogonal"
        if self.part_aligned_axis:
            suffix += "_partaxis"
        if self.domain_specific_bn:
            suffix += "_dsbn"
        if self.modality_specific_bn:
            suffix += "_mbn"
        if self.token_mixstyle:
            suffix += "_mixstyle"
        if self.prototype_memory:
            suffix += "_protomem"
        if self.prototype_transport_rank:
            suffix += (
                f"_ptransport{self.prototype_transport_rank}"
                f"s{int(round(100 * self.prototype_transport_scale))}"
            )
        if self.foreground_token_conditioning:
            suffix += "_fgtoken"
        if self.part_topology_supcon:
            suffix += "_ptoposupcon"
        if self.training_instance_queue:
            suffix += f"_queue{self.instance_queue_capacity}"
        if self.source_paired_logit_distillation:
            suffix += "_sourcepairlogitkd"
        if self.head_two_view_part_balanced_supcon:
            suffix += "_head2vpbsupcon"
        if self.frozen_prefix_adaptformer:
            suffix += (
                f"_adapt{self.adaptformer_dim}"
                f"s{int(round(100 * self.adaptformer_scale))}"
            )
        if self.part_routed_qv_lora:
            suffix += (
                f"_partqvlora{self.qv_lora_rank}x{self.qv_lora_part_rank}"
                f"a{int(round(self.qv_lora_alpha))}"
                f"d{int(round(100 * self.qv_lora_dropout))}"
            )
        elif self.frozen_prefix_qv_lora:
            suffix += (
                f"_qvlora{self.qv_lora_rank}"
                f"a{int(round(self.qv_lora_alpha))}"
                f"d{int(round(100 * self.qv_lora_dropout))}"
            )
        if self.suffix_head_qv_lora:
            suffix += (
                f"_suffixheadqvlora{self.qv_lora_rank}"
                f"a{int(round(self.qv_lora_alpha))}d0"
            )
        if self.frozen_prefix_convpass:
            suffix += (
                f"_convpass{self.convpass_dim}"
                f"s{int(round(100 * self.convpass_scale))}"
            )
        if self.head_convpass_attention:
            suffix += (
                f"_headconvattn{self.convpass_dim}"
                f"s{int(round(100 * self.convpass_scale))}"
            )
        if self.part_mlp_expert_blocks:
            suffix += f"_partmoe{self.part_mlp_expert_blocks}x3"
        if self.pattern_a2gc_branch:
            suffix += "_a2gc64"
        if self.hierarchical_slot_architecture:
            suffix += "_hslot4bg1r10"
        if self.dense_correspondence_training:
            suffix += "_dmatch16"
        if self.identity_query_pooling:
            suffix += "_pqmil128"
        if self.cross_level_texture_pyramid:
            suffix += (
                "_prcltp8_16_24d128"
                if self.part_routed_cross_level_texture
                else "_headcltp8_16_24d128"
                if self.head_routed_cross_level_texture
                else "_cltp8_16_24d128"
            )
        if self.image_frequency_texture_side:
            suffix += "_imgfreq128"
        if self.body_to_head_distillation:
            suffix += "_b2hproto64w20"
        if self.head_tail_expert_blocks:
            suffix += f"_headtail{self.head_tail_expert_blocks}"
        if self.external_head_representation:
            suffix += "_extheadrep24tail2"
        if self.ordered_head_grid_expert:
            suffix += "_headgrid10d256x2"
        if self.jpm_local_branches:
            suffix += (
                f"_jpm{self.jpm_local_branches}"
                f"s{self.jpm_shift}g{self.jpm_shuffle_groups}"
            )
        self.model_name = f"{backbone_model_name}_patch_mgn{suffix}"
        self.pretraining_source = pretraining_source
        if self.tips_backbone:
            self.external_pretraining_metadata = {
                "kind": "public_generic_spatial_vision_language_pretraining",
                "license": "Apache-2.0/CC-BY-4.0",
                "checkpoint_sha256": TIPS_L14_HR_PRETRAINING_SHA256,
                "loaded_tensors": TipsL14HighResDenseVision.checkpoint_tensor_count,
                "loaded_values": TipsL14HighResDenseVision.checkpoint_value_count,
                "official_commit": (
                    "4db271d3b9622b901cf4a1a821e4cb7a5e6b2490"
                ),
                "external_text_tower_retained": False,
                "external_class_state_loaded": False,
                "external_identity_taxonomy_loaded": False,
                "competition_checkpoint_loaded": False,
            }
        elif backbone_model_name == DINOV2_GIANT_REGISTER_MODEL_NAME:
            self.external_pretraining_metadata = {
                "kind": "public_generic_self_supervised_visual_pretraining",
                "checkpoint_sha256": DINOV2_GIANT_REGISTER_PRETRAINING_SHA256,
                "source_tensors": 568,
                "source_values": 1_136_486_912,
                "implementation": "timm.vit_giant_patch14_reg4_dinov2",
                "mask_token_retained": False,
                "external_class_state_loaded": False,
                "external_identity_taxonomy_loaded": False,
                "competition_checkpoint_loaded": False,
            }
        self.freeze_backbone_prefix(freeze_blocks)
        for adapted_mlp in self.prefix_adaptformer_modules:
            for parameter in adapted_mlp.adapter_parameters():
                parameter.requires_grad_(True)
        for adapted_qkv in self.prefix_qv_lora_modules:
            for parameter in adapted_qkv.adapter_parameters():
                parameter.requires_grad_(True)
        for adapted_qkv in self.suffix_head_qv_lora_modules:
            for parameter in adapted_qkv.adapter_parameters():
                parameter.requires_grad_(True)
        for adapted_module in self.prefix_convpass_modules:
            for parameter in adapted_module.adapter_parameters():
                parameter.requires_grad_(True)

    def prefix_adaptformer_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapted_mlp in self.prefix_adaptformer_modules
            for parameter in adapted_mlp.adapter_parameters()
        ]

    def prefix_qv_lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapted_qkv in self.prefix_qv_lora_modules
            for parameter in adapted_qkv.adapter_parameters()
        ]

    def suffix_head_qv_lora_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapted_qkv in self.suffix_head_qv_lora_modules
            for parameter in adapted_qkv.adapter_parameters()
        ]

    def set_suffix_head_qv_lora_routing(self, enabled: bool) -> None:
        """Expose the exact base graph for V13.3 routing-isolation audits."""
        if enabled and not self.suffix_head_qv_lora:
            raise ValueError("Suffix head Q/V LoRA is disabled")
        for adapted_qkv in self.suffix_head_qv_lora_modules:
            adapted_qkv.enabled = bool(enabled)

    def prefix_convpass_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for adapted_module in self.prefix_convpass_modules
            for parameter in adapted_module.adapter_parameters()
        ]

    def jpm_refinement_parameters(self) -> list[nn.Parameter]:
        if not self.jpm_local_branches:
            return []
        if self.jpm_refinement_block is None or self.jpm_refinement_norm is None:
            raise AssertionError("JPM refinement modules are missing")
        return list(self.jpm_refinement_block.parameters()) + list(
            self.jpm_refinement_norm.parameters()
        )

    def cross_level_texture_parameters(self) -> list[nn.Parameter]:
        if not self.cross_level_texture_pyramid:
            return []
        modules: list[nn.Module] = [
            self.cross_level_lateral_projections,
            self.cross_level_depthwise,
        ]
        if self.part_routed_cross_level_texture:
            modules.extend(
                [
                    self.cross_level_part_fusions,
                    self.cross_level_part_expansions,
                ]
            )
        else:
            modules.extend(
                [self.cross_level_fusion, self.cross_level_expansion]
            )
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]

    def set_head_routed_cross_level_texture(self, enabled: bool) -> None:
        """Expose the exact Q base graph for V14.1 route-isolation audits."""
        if enabled and not self.head_routed_cross_level_texture:
            raise ValueError("Head-routed cross-level texture is disabled")
        self.head_routed_cross_level_texture_enabled = bool(enabled)

    def activate_cross_level_texture_stage(self) -> None:
        if not self.part_routed_cross_level_texture:
            raise RuntimeError(
                "Frozen texture stage requires part-routed cross-level texture"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.cross_level_texture_parameters():
            parameter.requires_grad_(True)
        self._cross_level_texture_stage_only = True

    def image_texture_parameters(self) -> list[nn.Parameter]:
        if not self.image_frequency_texture_side:
            return []
        return [
            parameter
            for module in (
                self.image_texture_encoder,
                self.image_texture_part_fusions,
                self.image_texture_part_expansions,
            )
            for parameter in module.parameters()
        ]

    def activate_image_texture_stage(self) -> None:
        if not self.image_frequency_texture_side:
            raise RuntimeError("This model has no image-frequency side network")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.image_texture_parameters():
            parameter.requires_grad_(True)
        self._image_texture_stage_only = True

    def body_head_adapter_parameters(self) -> list[nn.Parameter]:
        if not self.body_to_head_distillation:
            return []
        return [
            parameter
            for adapter in self.head_identity_adapters
            for parameter in adapter.parameters()
        ]

    def activate_body_head_stage(self) -> None:
        if not self.body_to_head_distillation:
            raise RuntimeError("This model has no body-to-head distillation stage")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.body_head_adapter_parameters():
            parameter.requires_grad_(True)
        self._body_head_stage_only = True

    def head_tail_expert_parameters(self) -> list[nn.Parameter]:
        return list(self.head_tail_expert_modules.parameters())

    @torch.no_grad()
    def initialize_head_tail_experts_from_base(self) -> None:
        if not self.head_tail_expert_blocks:
            raise RuntimeError("This model has no head-tail experts")
        for block_index, expert in zip(
            self.head_tail_expert_block_indices,
            self.head_tail_expert_modules,
            strict=True,
        ):
            expert.load_state_dict(
                self.backbone.blocks[block_index].state_dict(), strict=True
            )

    def activate_head_tail_expert_stage(self) -> None:
        if not self.head_tail_expert_blocks:
            raise RuntimeError("This model has no head-tail expert stage")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.head_tail_expert_parameters():
            parameter.requires_grad_(True)
        # The expert is the first trainable operation after a fully frozen
        # prefix. Disabling timm's block checkpoint wrapper ensures gradients
        # do not depend on a checkpoint input that itself requires no grad.
        self.backbone.set_grad_checkpointing(False)
        self._head_tail_expert_stage_only = True

    @torch.no_grad()
    def enable_head_tail_experts(self) -> None:
        if not self.head_tail_expert_blocks:
            raise RuntimeError("This model has no head-tail expert route")
        self.head_tail_expert_enabled.fill_(True)

    @torch.no_grad()
    def set_body_teacher_prototypes(
        self,
        prototypes: torch.Tensor,
        available: torch.Tensor,
    ) -> None:
        expected = (
            self.num_classes,
            self.branch_count,
            self.embedding_dim,
        )
        if tuple(prototypes.shape) != expected:
            raise ValueError(
                f"Body teacher shape {tuple(prototypes.shape)} != {expected}"
            )
        if tuple(available.shape) != (self.num_classes,):
            raise ValueError("Body teacher availability shape is invalid")
        if available.any():
            norms = prototypes[available].float().norm(dim=-1)
            if not torch.isfinite(norms).all() or not torch.allclose(
                norms,
                torch.ones_like(norms),
                atol=1e-5,
                rtol=1e-5,
            ):
                raise ValueError("Available body teacher prototypes are not normalized")
        self.body_teacher_prototypes.copy_(
            prototypes.to(
                device=self.body_teacher_prototypes.device,
                dtype=self.body_teacher_prototypes.dtype,
            )
        )
        self.body_teacher_available.copy_(
            available.to(device=self.body_teacher_available.device)
        )

    def stage_initializer_missing_prefixes(self) -> tuple[str, ...]:
        if self.head_tail_expert_blocks:
            return (
                "head_tail_expert_modules.",
                "head_tail_expert_enabled",
            )
        if self.body_to_head_distillation:
            return (
                "head_identity_adapters.",
                "body_teacher_prototypes",
                "body_teacher_available",
            )
        if self.image_frequency_texture_side:
            return ("image_texture_",)
        if self.part_routed_cross_level_texture:
            return ("cross_level_",)
        return ()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and (
            self._cross_level_texture_stage_only
            or self._image_texture_stage_only
            or self._body_head_stage_only
            or self._head_tail_expert_stage_only
        ):
            # The incumbent remains a deterministic inference network while
            # the additive side path learns; in particular, BN running
            # statistics in the seven mature heads cannot drift.
            self.backbone.eval()
            self.branch_projections.eval()
            self.branch_necks.eval()
            if self.part_adapters is not None:
                self.part_adapters.eval()
            if self._cross_level_texture_stage_only:
                self.cross_level_lateral_projections.train()
                self.cross_level_depthwise.train()
                self.cross_level_part_fusions.train()
                self.cross_level_part_expansions.train()
            if self._image_texture_stage_only:
                self.image_texture_encoder.train()
                self.image_texture_part_fusions.train()
                self.image_texture_part_expansions.train()
            if self._body_head_stage_only:
                self.head_identity_adapters.train()
            if self._head_tail_expert_stage_only:
                self.head_tail_expert_modules.train()
        return self

    def auxiliary_training_loss(self) -> torch.Tensor:
        if self.hierarchical_slot_architecture:
            if (
                self.slot_aggregator is None
                or self._slot_reconstruction_loss is None
            ):
                raise RuntimeError("Slot reconstruction loss is unavailable")
            return self.slot_aggregator.weighted_reconstruction_loss(
                self._slot_reconstruction_loss
            )
        if self.body_to_head_distillation:
            if self._body_head_alignment_loss is None:
                raise RuntimeError("Body-to-head alignment loss is unavailable")
            return self._body_head_alignment_loss
        raise RuntimeError("This model has no auxiliary training objective")

    def foreground_training_logits(self) -> torch.Tensor:
        if not self.foreground_auxiliary:
            raise RuntimeError("Foreground auxiliary supervision is disabled")
        if self._foreground_auxiliary_logits is None:
            raise RuntimeError("Foreground auxiliary logits are unavailable")
        return self._foreground_auxiliary_logits

    def dense_correspondence_tokens(self) -> torch.Tensor:
        if not self.dense_correspondence_training:
            raise RuntimeError("Dense correspondence training is disabled")
        if self._dense_correspondence_tokens is None:
            raise RuntimeError("Dense correspondence tokens are unavailable")
        return self._dense_correspondence_tokens

    def optimizer_layer_count(self) -> int:
        if self.spatial_backbone:
            stages = getattr(self.backbone, "stages", None)
            layers = getattr(self.backbone, "layers", None)
            if stages is not None:
                # ConvNeXt's official layer-decay recipe groups its long
                # 27-block third stage in threes, yielding twelve semantic
                # depth levels rather than treating four stages as four
                # equally shallow layers.
                return 12
            return sum(len(layer.blocks) for layer in layers)
        return len(self.backbone.blocks) + 1

    def optimizer_layer_id(self, inner_name: str) -> int:
        if self.spatial_backbone:
            if inner_name.startswith(("stem.", "patch_embed.")):
                return 0
            if inner_name.startswith("stages."):
                components = inner_name.split(".")
                stage_index = int(components[1])
                if stage_index == 0:
                    return 1
                if stage_index == 1:
                    return 2
                if stage_index == 2:
                    if len(components) > 3 and components[2] == "blocks":
                        return 3 + int(components[3]) // 3
                    return 3
                return 12
            if inner_name.startswith("layers."):
                components = inner_name.split(".")
                stage_index = int(components[1])
                prior_blocks = sum(
                    len(self.backbone.layers[index].blocks)
                    for index in range(stage_index)
                )
                if len(components) > 3 and components[2] == "blocks":
                    return prior_blocks + int(components[3]) + 1
                return prior_blocks + 1
            return self.optimizer_layer_count()
        if self.radio_backbone and inner_name.startswith("radio.model."):
            inner_name = inner_name[len("radio.model.") :]
        if inner_name.startswith("blocks."):
            return int(inner_name.split(".")[1]) + 1
        if inner_name.startswith(
            (
                "patch_embed",
                "patch_generator",
                "cls_token",
                "reg_token",
                "register_tokens",
            )
        ):
            return 0
        return self.optimizer_layer_count()

    def freeze_backbone_prefix(self, block_count: int) -> None:
        if self.spatial_backbone:
            stages = getattr(self.backbone, "stages", None)
            layers = getattr(self.backbone, "layers", None)
            stage_modules = stages if stages is not None else layers
            stage_count = min(max(int(block_count), 0), len(stage_modules))
            for name, parameter in self.backbone.named_parameters():
                frozen = name.startswith(("stem.", "patch_embed.")) and stage_count > 0
                if name.startswith(("stages.", "layers.")):
                    stage_index = int(name.split(".")[1])
                    frozen = stage_index < stage_count
                parameter.requires_grad = not frozen
            return
        for name, parameter in self.backbone.named_parameters():
            backbone_name = (
                name[len("radio.model.") :]
                if self.radio_backbone and name.startswith("radio.model.")
                else name
            )
            frozen = backbone_name.startswith(
                (
                    "patch_embed",
                    "patch_generator",
                    "cls_token",
                    "reg_token",
                    "register_tokens",
                    "storage_tokens",
                    "pos_embed",
                    "mask_token",
                )
            ) and block_count > 0
            if backbone_name.startswith("blocks."):
                block_index = int(backbone_name.split(".")[1])
                frozen = block_index < block_count
            parameter.requires_grad = not frozen

    def set_part_availability(self, availability: torch.Tensor) -> None:
        if availability.shape != self.part_class_available.shape:
            raise ValueError(
                f"Availability {availability.shape} != {self.part_class_available.shape}"
            )
        self.part_class_available.copy_(availability.bool())

    def set_head_identity_expert_routing(self, enabled: bool) -> None:
        """Enable the fixed V6.1 head route or expose its base-only audit."""
        if not self.head_identity_expert and enabled:
            raise ValueError("Head identity expert is disabled")
        if enabled and not self.head_identity_expert_inference:
            raise ValueError("This training-only head auxiliary cannot route scores")
        self.head_identity_expert_routing_enabled = bool(enabled)

    def head_identity_expert_parameters(self) -> list[nn.Parameter]:
        """Return the isolated V6.1 parameter set for independent clipping."""
        if not self.head_identity_expert:
            return []
        grid_parameters: tuple[nn.Parameter, ...] = ()
        if self.ordered_head_grid_expert:
            grid_parameters = tuple(
                self.head_identity_expert_grid_projection.parameters()
            ) + tuple(self.head_identity_expert_grid_transformer.parameters())
        return [
            parameter
            for parameter in (
                *grid_parameters,
                *self.head_identity_expert_projections.parameters(),
                *self.head_identity_expert_necks.parameters(),
                self.head_identity_expert_class_weight,
            )
            if parameter.requires_grad
        ]

    def head_identity_expert_targets(
        self, head_labels: torch.Tensor
    ) -> torch.Tensor:
        if not self.head_identity_expert:
            raise RuntimeError("Head identity expert is disabled")
        if not self.head_tail_other_expert:
            return head_labels
        other = torch.full_like(head_labels, self.num_classes)
        return torch.where(
            self.head_tail_identity_mask[head_labels], head_labels, other
        )

    def head_identity_expert_loss_space(
        self,
        expert_logits: torch.Tensor,
        head_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return expert logits/targets in the preregistered active class space."""
        targets = self.head_identity_expert_targets(head_labels)
        if not self.head_tail_other_expert:
            return expert_logits, targets
        allowed = torch.cat(
            [
                self.head_tail_identity_mask,
                self.head_tail_identity_mask.new_ones(1),
            ]
        )
        if not allowed[-1] or int(allowed.sum()) < 2:
            raise AssertionError("Tail-ID + OTHER expert class space is invalid")
        compact_logits = expert_logits[..., allowed]
        compact_lookup = allowed.long().cumsum(dim=0) - 1
        compact_targets = compact_lookup[targets]
        return compact_logits, compact_targets

    def set_part_counts(self, counts: torch.Tensor) -> None:
        expected_shape = (3, self.num_classes)
        if counts.shape != expected_shape:
            raise ValueError(
                f"Part counts {counts.shape} != {expected_shape}"
            )
        if self.head_tail_other_expert:
            head_counts = counts[0]
            self.head_tail_identity_mask.copy_(
                head_counts.gt(0) & head_counts.le(4)
            )
        if self.class_adaptive_margin:
            margins = torch.stack(
                [self._count_adaptive_margins(row) for row in counts], dim=0
            )
            self.part_class_margins.copy_(margins)
        elif self.head_class_adaptive_margin:
            self.part_class_margins.fill_(self.arc_margin)
            self.part_class_margins[0].copy_(
                self._count_adaptive_margins(counts[0])
            )
        if not self.hierarchical_part_heads:
            return
        # A train-only empirical-Bayes shrinkage rule: four examples provide
        # equal weight to the shared identity prior and the crop-specific
        # residual. Few-shot classes stay closer to the shared prototype.
        counts = counts.to(dtype=self.part_class_shrinkage.dtype)
        self.part_class_shrinkage.copy_(counts / (counts + 4.0))

    def set_class_counts(self, counts: torch.Tensor) -> None:
        if not self.class_adaptive_margin:
            return
        if counts.shape != self.shared_class_margins.shape:
            raise ValueError(
                f"Class counts {counts.shape} != {self.shared_class_margins.shape}"
            )
        self.shared_class_margins.copy_(self._count_adaptive_margins(counts))

    def _count_adaptive_margins(self, counts: torch.Tensor) -> torch.Tensor:
        # LDAM's n^(-1/4) rarity rule, transplanted into ArcFace. The incumbent
        # 0.2 margin remains the frequent-class floor; the rarest class gets
        # exactly 1.5x that margin. Counts come only from the active fold-train
        # rows and no class prior is applied during inference.
        counts = counts.float().clamp_min(1.0)
        rarity = counts.pow(-0.25)
        span = rarity.max() - rarity.min()
        if float(span) < 1e-8:
            return torch.full_like(rarity, self.arc_margin)
        normalized = (rarity - rarity.min()) / span
        return self.arc_margin * (1.0 + 0.5 * normalized)

    def _inject_part_side_embedding(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        if self._active_parts is None:
            return output
        side = self.part_side_embedding[self._active_parts]
        return output + side.to(dtype=output.dtype).unsqueeze(1)

    def _inject_continuous_geometry(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        self._geometry_last_residual = None
        if self._active_image_geometry is None or self._active_parts is None:
            return output
        if self._active_image_geometry.shape != (len(output), 2):
            raise ValueError("Expected standardized crop geometry [B, 2]")
        if len(self._active_parts) != len(output):
            raise ValueError("Crop geometry part labels do not match tokens")
        condition = torch.cat(
            [
                self._active_image_geometry.float(),
                F.one_hot(self._active_parts, num_classes=3).float(),
            ],
            dim=1,
        )
        vectors = self.geometry_conditioner(condition).reshape(
            len(output), 3, output.shape[-1]
        )
        constant, horizontal, vertical = vectors.unbind(dim=1)
        prefix_count = int(self.backbone.num_prefix_tokens)
        patch_count = output.shape[1] - prefix_count
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise AssertionError("Geometry conditioning requires a square patch grid")
        coordinates = torch.linspace(
            -1.0,
            1.0,
            side,
            device=output.device,
            dtype=vectors.dtype,
        )
        y_coordinate, x_coordinate = torch.meshgrid(
            coordinates, coordinates, indexing="ij"
        )
        x_coordinate = x_coordinate.flatten()[None, :, None]
        y_coordinate = y_coordinate.flatten()[None, :, None]
        patch_residual = (
            constant[:, None, :]
            + x_coordinate * horizontal[:, None, :]
            + y_coordinate * vertical[:, None, :]
        )
        prefix_residual = constant[:, None, :].expand(
            -1, prefix_count, -1
        )
        residual = self.geometry_condition_scale * torch.cat(
            [prefix_residual, patch_residual], dim=1
        )
        self._geometry_last_residual = residual
        return output + residual.to(dtype=output.dtype)

    def _inject_foreground_token_embedding(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        self._foreground_token_last_residual = None
        if self._active_foreground_mask is None or self._active_parts is None:
            return output
        mask = self._active_foreground_mask
        if mask.ndim != 3 or mask.shape[0] != len(output):
            raise ValueError("Foreground token mask must be [B, H, W]")
        if len(self._active_parts) != len(output):
            raise ValueError("Foreground token parts do not match tokens")
        prefix_count = int(self.backbone.num_prefix_tokens)
        patch_count = output.shape[1] - prefix_count
        side = math.isqrt(patch_count)
        if side * side != patch_count:
            raise AssertionError(
                "Foreground token conditioning requires a square patch grid"
            )
        patch_mask = F.interpolate(
            mask.float().unsqueeze(1),
            size=(side, side),
            mode="area",
        ).squeeze(1)
        valid = patch_mask.amax(dim=(1, 2), keepdim=True).gt(0.0)
        centered = patch_mask - patch_mask.mean(dim=(1, 2), keepdim=True)
        centered = torch.where(valid, centered, torch.zeros_like(centered))
        direction = self.foreground_token_embedding[self._active_parts]
        patch_residual = centered.unsqueeze(-1) * direction[:, None, None, :]
        prefix_residual = torch.zeros(
            len(output),
            prefix_count,
            output.shape[-1],
            device=output.device,
            dtype=patch_residual.dtype,
        )
        residual = self.foreground_token_scale * torch.cat(
            [prefix_residual, patch_residual.flatten(1, 2)], dim=1
        )
        self._foreground_token_last_residual = residual
        return output + residual.to(dtype=output.dtype)

    def geometry_conditioning_parameters(self) -> list[nn.Parameter]:
        if not self.continuous_geometry_conditioning:
            return []
        return list(self.geometry_conditioner.parameters())

    def _mix_token_style(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Mix per-image early-token statistics during training only."""
        del module, inputs
        if (
            not self.training
            or len(output) < 2
            or torch.rand((), device=output.device) > self.mixstyle_probability
        ):
            return output
        prefix_count = int(self.backbone.num_prefix_tokens)
        prefix, patches = output[:, :prefix_count], output[:, prefix_count:]
        patches_float = patches.float()
        mean = patches_float.mean(dim=1, keepdim=True)
        std = patches_float.var(
            dim=1, keepdim=True, unbiased=False
        ).add(1e-6).sqrt()
        normalized = (patches_float - mean) / std
        # Detaching statistics follows MixStyle: gradients optimize content,
        # not a shortcut that cancels the synthesized feature domain.
        mean = mean.detach()
        std = std.detach()
        permutation = torch.randperm(len(patches), device=patches.device)
        concentration = torch.full(
            (len(patches),),
            self.mixstyle_alpha,
            dtype=torch.float32,
            device=patches.device,
        )
        mixing = torch.distributions.Beta(concentration, concentration).sample()
        mixing = mixing[:, None, None]
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_std = mixing * std + (1.0 - mixing) * std[permutation]
        mixed_patches = (normalized * mixed_std + mixed_mean).to(patches.dtype)
        return torch.cat([prefix, mixed_patches], dim=1)

    def _capture_intermediate_tokens(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        self._intermediate_tokens = output

    def _capture_external_head_input(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
    ) -> None:
        del module
        if not self._external_head_capture_enabled:
            return
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise AssertionError("Unexpected external-head source block input")
        rope = kwargs.get("rope")
        if rope is None and len(inputs) > 1:
            rope = inputs[1]
        if not isinstance(rope, torch.Tensor):
            raise AssertionError("External-head source block did not receive RoPE")
        self._external_head_initial_tokens = inputs[0]
        self._external_head_rope = rope

    def _external_head_block_forward(
        self,
        tokens: torch.Tensor,
        rope: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        """Run a shared frozen block with its external-only adapter residual."""
        block = self.backbone.blocks[block_index]
        mlp = block.mlp
        if not isinstance(mlp, ParallelAdaptMlp):
            raise AssertionError("External head prefix lost its AdaptFormer wrapper")
        attention = block.attn(block.norm1(tokens), rope=rope)
        attention_residual = block.drop_path1(
            attention if block.gamma_1 is None else block.gamma_1 * attention
        )
        tokens = tokens + attention_residual
        normalized = block.norm2(tokens)
        feed_forward = mlp.base_mlp(normalized) + self.external_head_prefix_adapters[
            block_index
        ](normalized)
        return tokens + block.drop_path2(
            feed_forward
            if block.gamma_2 is None
            else block.gamma_2 * feed_forward
        )

    def _external_head_representation_inputs(
        self,
        parts: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Build seven MGN/covariance inputs on the legal DogFace head route."""
        if not self.external_head_representation:
            raise RuntimeError("External head representation is disabled")
        initial = self._external_head_initial_tokens
        rope = self._external_head_rope
        if initial is None or rope is None or self.external_head_norm is None:
            raise AssertionError("External head input/RoPE capture is unavailable")
        head_indices = torch.nonzero(parts.eq(0), as_tuple=False).flatten()
        input_dims = [2560, 1280, 1280, 1280, 1280, 1280, 4096]
        if head_indices.numel() == 0:
            return [
                initial.new_zeros((len(parts), width)) for width in input_dims
            ]
        tokens = initial.index_select(0, head_indices)
        for block_index in range(30):
            if block_index < 24:
                def run_prefix(
                    value: torch.Tensor,
                    position: torch.Tensor,
                    index: int = block_index,
                ) -> torch.Tensor:
                    return self._external_head_block_forward(value, position, index)

                tokens = (
                    checkpoint(run_prefix, tokens, rope, use_reentrant=False)
                    if self.grad_checkpointing and self.training
                    else run_prefix(tokens, rope)
                )
            else:
                block = self.backbone.blocks[block_index]

                def run_shared(
                    value: torch.Tensor,
                    position: torch.Tensor,
                    module: nn.Module = block,
                ) -> torch.Tensor:
                    return module(value, rope=position)

                tokens = (
                    checkpoint(run_shared, tokens, rope, use_reentrant=False)
                    if self.grad_checkpointing and self.training
                    else run_shared(tokens, rope)
                )
        for block in self.external_head_tail_blocks:
            def run_tail(
                value: torch.Tensor,
                position: torch.Tensor,
                module: nn.Module = block,
            ) -> torch.Tensor:
                return module(value, rope=position)

            tokens = (
                checkpoint(run_tail, tokens, rope, use_reentrant=False)
                if self.grad_checkpointing and self.training
                else run_tail(tokens, rope)
            )
        tokens = self.external_head_norm(tokens)
        prefix_count = int(self.backbone.num_prefix_tokens)
        cls = tokens[:, 0]
        patches = tokens[:, prefix_count:]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise AssertionError("External head patch grid is not square")
        patch_map = patches.reshape(
            len(patches), side, side, patches.shape[-1]
        ).permute(0, 3, 1, 2)
        half = F.adaptive_avg_pool2d(patch_map, (2, 1)).squeeze(-1)
        thirds = F.adaptive_avg_pool2d(patch_map, (3, 1)).squeeze(-1)
        expert_inputs = [
            torch.cat([cls, patches.mean(dim=1)], dim=-1),
            half[:, :, 0],
            half[:, :, 1],
            thirds[:, :, 0],
            thirds[:, :, 1],
            thirds[:, :, 2],
        ]
        reduction = self.external_head_covariance_reduction
        if reduction is None:
            raise AssertionError("External head covariance reduction is unavailable")
        reduced = reduction(patches).float()
        centered = reduced - reduced.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bni,bnj->bij", centered, centered) / max(
            1, centered.shape[1] - 1
        )
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1e-6
        )
        expert_inputs.append(
            F.normalize(covariance.flatten(1), dim=-1).to(dtype=patches.dtype)
        )
        if len(expert_inputs) != 7:
            raise AssertionError("External head branch count changed")
        return [
            value.new_zeros((len(parts), value.shape[-1])).index_copy(
                0, head_indices, value
            )
            for value in expert_inputs
        ]

    def _make_cross_level_capture_hook(self, block_index: int):
        def capture(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            self._cross_level_tokens[block_index] = output

        return capture

    def _capture_jpm_last_input(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
    ) -> None:
        del module
        if not self._jpm_capture_enabled:
            return
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise AssertionError("Unexpected JPM source block input")
        rope = kwargs.get("rope")
        if rope is None and len(inputs) > 1:
            rope = inputs[1]
        if not isinstance(rope, torch.Tensor):
            raise AssertionError("DINOv3 JPM source block did not receive RoPE")
        self._jpm_last_input_tokens = inputs[0]
        self._jpm_last_rope = rope

    def jpm_patch_order(
        self,
        patch_count: int,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if not self.jpm_local_branches:
            raise RuntimeError("This model has no JPM branches")
        if patch_count % self.jpm_shuffle_groups:
            raise AssertionError("Patch count is not divisible by the JPM shuffle groups")
        if patch_count % self.jpm_local_branches:
            raise AssertionError("Patch count is not divisible by the JPM local branches")
        order = torch.arange(patch_count, device=device)
        order = torch.roll(order, shifts=-self.jpm_shift, dims=0)
        return (
            order.reshape(self.jpm_shuffle_groups, -1)
            .transpose(0, 1)
            .contiguous()
            .reshape(-1)
        )

    @staticmethod
    def _jpm_select_rope(
        rope: torch.Tensor,
        order: torch.Tensor,
        patch_count: int,
    ) -> torch.Tensor:
        if rope.ndim == 2 and rope.shape[0] == patch_count:
            return rope.index_select(0, order)
        if rope.ndim >= 3 and rope.shape[-2] == patch_count:
            return rope.index_select(-2, order)
        raise AssertionError(f"Unexpected JPM RoPE geometry: {tuple(rope.shape)}")

    def _jpm_local_cls_features(self, prefix_count: int) -> list[torch.Tensor]:
        if not self.jpm_local_branches:
            raise RuntimeError("This model has no JPM branches")
        if self.jpm_refinement_block is None or self.jpm_refinement_norm is None:
            raise AssertionError("JPM refinement modules are missing")
        tokens = self._jpm_last_input_tokens
        rope = self._jpm_last_rope
        if tokens is None or rope is None:
            raise AssertionError("JPM penultimate tokens or RoPE were not captured")
        if prefix_count != 5 or tokens.shape[1] <= prefix_count:
            raise AssertionError(f"Unexpected JPM prefix/token geometry: {tokens.shape}")
        cls = tokens[:, :1]
        patches = tokens[:, prefix_count:]
        patch_count = int(patches.shape[1])
        order = self.jpm_patch_order(patch_count, device=patches.device)
        shuffled_patches = patches.index_select(1, order)
        shuffled_rope = self._jpm_select_rope(rope, order, patch_count)
        group_length = patch_count // self.jpm_local_branches
        local_features: list[torch.Tensor] = []
        patch_counts: list[int] = []
        rope_counts: list[int] = []
        for branch_index in range(self.jpm_local_branches):
            start = branch_index * group_length
            stop = start + group_length
            local_patches = shuffled_patches[:, start:stop]
            local_rope = shuffled_rope[..., start:stop, :]
            local_sequence = torch.cat([cls, local_patches], dim=1)

            if self.grad_checkpointing and self.training:
                def run_local_block(
                    value: torch.Tensor,
                    position: torch.Tensor,
                ) -> torch.Tensor:
                    return self.jpm_refinement_block(value, rope=position)

                refined = checkpoint(
                    run_local_block,
                    local_sequence,
                    local_rope,
                    use_reentrant=False,
                )
            else:
                refined = self.jpm_refinement_block(
                    local_sequence,
                    rope=local_rope,
                )
            normalized = self.jpm_refinement_norm(refined)
            local_features.append(normalized[:, 0])
            patch_counts.append(int(local_patches.shape[1]))
            rope_counts.append(int(local_rope.shape[-2]))

        self._jpm_last_patch_order = order.detach()
        self._jpm_last_group_patch_counts = tuple(patch_counts)
        self._jpm_last_group_rope_counts = tuple(rope_counts)
        self._jpm_last_input_tokens = None
        self._jpm_last_rope = None
        return local_features

    def _make_head_tail_expert_hook(self, expert: nn.Module):
        def route_head(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            del module
            parts = self._part_routing_context.parts
            if parts is None or len(parts) != len(output):
                raise AssertionError("Head-tail expert routing context is missing")
            if len(inputs) != 1 or not isinstance(output, torch.Tensor):
                raise AssertionError("Unexpected Transformer block interface")
            if not bool(self.head_tail_expert_enabled):
                return output
            head_indices = torch.nonzero(
                parts.eq(0), as_tuple=False
            ).flatten()
            if head_indices.numel() == 0:
                return output
            # Preserve the mother's exact batched GEMM geometry at
            # initialization. Only selected head rows enter the returned
            # graph, so discarded body rows contribute no expert gradient.
            expert_output = expert(inputs[0])
            return output.index_copy(
                0,
                head_indices,
                expert_output.index_select(0, head_indices),
            )

        return route_head

    def _cross_level_texture_residual(
        self,
        *,
        prefix_count: int,
        patches: torch.Tensor,
        side: int,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.cross_level_texture_pyramid:
            raise RuntimeError("This model has no cross-level texture pyramid")
        if set(self._cross_level_tokens) != set(self.cross_level_blocks):
            raise AssertionError("Cross-level texture features were not all captured")
        lateral_maps: list[torch.Tensor] = []
        for block_index, projection, depthwise in zip(
            self.cross_level_blocks,
            self.cross_level_lateral_projections,
            self.cross_level_depthwise,
            strict=True,
        ):
            normalized = self.backbone.norm(
                self._cross_level_tokens[block_index]
            )
            level_patches = normalized[:, prefix_count:]
            if level_patches.shape != patches.shape:
                raise AssertionError(
                    "Cross-level and final DINO patch grids do not match"
                )
            lateral = projection(level_patches)
            lateral = lateral.reshape(
                lateral.shape[0], side, side, self.cross_level_dim
            ).permute(0, 3, 1, 2)
            lateral_maps.append(depthwise(lateral))
        concatenated = torch.cat(lateral_maps, dim=1)
        if self.part_routed_cross_level_texture:
            residual = torch.zeros(
                patches.shape[0],
                patches.shape[-1],
                side,
                side,
                dtype=concatenated.dtype,
                device=concatenated.device,
            )
            for part_index, (fusion, expansion) in enumerate(
                zip(
                    self.cross_level_part_fusions,
                    self.cross_level_part_expansions,
                    strict=True,
                )
            ):
                indices = torch.nonzero(
                    parts == part_index, as_tuple=False
                ).flatten()
                if indices.numel() == 0:
                    continue
                routed = concatenated.index_select(0, indices)
                part_residual = self.cross_level_scale * expansion(
                    fusion(routed)
                )
                residual = residual.index_copy(0, indices, part_residual)
        else:
            fused = self.cross_level_fusion(concatenated)
            residual = self.cross_level_scale * self.cross_level_expansion(fused)
            if self.head_routed_cross_level_texture:
                active = parts.eq(0) & bool(
                    self.head_routed_cross_level_texture_enabled
                )
                residual = residual * active.to(residual.dtype)[:, None, None, None]
        self._cross_level_last_residual = residual
        return residual

    def _image_texture_residual(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        if not self.image_frequency_texture_side:
            raise RuntimeError("This model has no image-frequency side network")
        features = self.image_texture_encoder(images)
        if features.shape[1:] != (128, side, side):
            raise AssertionError(
                f"Image-frequency grid {features.shape} does not match DINO {side}"
            )
        residual = torch.zeros(
            features.shape[0],
            1280,
            side,
            side,
            dtype=features.dtype,
            device=features.device,
        )
        for part_index, (fusion, expansion) in enumerate(
            zip(
                self.image_texture_part_fusions,
                self.image_texture_part_expansions,
                strict=True,
            )
        ):
            indices = torch.nonzero(
                parts == part_index, as_tuple=False
            ).flatten()
            if indices.numel() == 0:
                continue
            routed = features.index_select(0, indices)
            part_residual = 0.1 * expansion(fusion(routed))
            residual = residual.index_copy(
                0, indices, part_residual.to(dtype=residual.dtype)
            )
        self._image_texture_last_residual = residual
        return residual

    @staticmethod
    def _matrix_square_root(
        covariance: torch.Tensor, iterations: int = 5
    ) -> torch.Tensor:
        """Differentiable iSQRT-COV Newton--Schulz matrix square root."""
        batch, dimension, _ = covariance.shape
        identity = torch.eye(
            dimension, dtype=covariance.dtype, device=covariance.device
        ).expand(batch, -1, -1)
        # Trace normalization places every positive semi-definite covariance
        # in Newton--Schulz's stable convergence region. A tiny diagonal also
        # keeps rank-deficient few-shot batches differentiable.
        covariance = covariance + 1e-5 * identity
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-6)
        normalized = covariance / trace[:, None, None]
        estimate = normalized
        inverse_estimate = identity
        for _ in range(iterations):
            correction = 0.5 * (
                3.0 * identity - inverse_estimate @ estimate
            )
            estimate = estimate @ correction
            inverse_estimate = correction @ inverse_estimate
        return estimate * trace.sqrt()[:, None, None]

    def _backbone_tokens(self, images: torch.Tensor) -> torch.Tensor:
        self._intermediate_tokens = None
        self._cross_level_tokens = {}
        self._cross_level_last_residual = None
        self._jpm_last_input_tokens = None
        self._jpm_last_rope = None
        self._jpm_last_patch_order = None
        self._jpm_last_group_patch_counts = ()
        self._jpm_last_group_rope_counts = ()
        self._external_head_initial_tokens = None
        self._external_head_rope = None
        if not self.lingbot_backbone:
            self._jpm_capture_enabled = bool(self.jpm_local_branches)
            self._external_head_capture_enabled = bool(
                self.external_head_representation
            )
            try:
                features = self.backbone.forward_features(images)
            finally:
                self._jpm_capture_enabled = False
                self._external_head_capture_enabled = False
            # Some timm ViTs trained with average pooling (notably EVA-02)
            # keep the pretrained final LayerNorm in `fc_norm`, so
            # `forward_features` deliberately returns unnormalized tokens.
            # Apply that published representation norm token-wise before the
            # dense MGN heads; DINOv3 already normalizes inside the backbone
            # and therefore takes the no-op path.
            backbone_norm = getattr(self.backbone, "norm", None)
            feature_norm = getattr(self.backbone, "fc_norm", None)
            if (
                features.ndim == 3
                and isinstance(backbone_norm, nn.Identity)
                and isinstance(feature_norm, nn.Module)
                and not isinstance(feature_norm, nn.Identity)
            ):
                features = feature_norm(features)
            if features.ndim == 4 and self.spatial_backbone:
                spatial_norm = getattr(
                    getattr(self.backbone, "head", None), "norm", None
                )
                if isinstance(spatial_norm, nn.Module) and not isinstance(
                    spatial_norm, nn.Identity
                ):
                    features = spatial_norm(features)
            return features

        tokens, (height, width) = self.backbone.prepare_tokens_with_masks(images)
        rope_module = self.backbone.rope_embed
        dynamic_rope = rope_module.training and any(
            value is not None
            for value in (
                rope_module.shift_coords,
                rope_module.jitter_coords,
                rope_module.rescale_coords,
            )
        )
        static_rope = (
            None if dynamic_rope else rope_module(H=height, W=width)
        )
        for block in self.backbone.blocks:
            rope = (
                rope_module(H=height, W=width)
                if dynamic_rope
                else static_rope
            )
            if self.grad_checkpointing and self.training:
                def run_block(
                    value: torch.Tensor,
                    module: nn.Module = block,
                    position=rope,
                ) -> torch.Tensor:
                    return module(value, position)

                tokens = checkpoint(run_block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, rope)

        prefix_count = int(self.backbone.n_storage_tokens) + 1
        if self.backbone.untie_cls_and_patch_norms:
            prefix = self.backbone.cls_norm(tokens[:, :prefix_count])
            patches = self.backbone.norm(tokens[:, prefix_count:])
            return torch.cat([prefix, patches], dim=1)
        return self.backbone.norm(tokens)

    def _ordered_head_grid_inputs(
        self,
        patches: torch.Tensor,
        cls: torch.Tensor,
        parts: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return CLS plus row-major 3x3 detached foreground-grid tokens."""
        if not self.ordered_head_grid_expert:
            raise RuntimeError("Ordered head grid expert is disabled")
        if foreground_mask.shape != (len(parts), 448, 448):
            raise ValueError("Ordered head expert requires aligned 448x448 masks")
        side = math.isqrt(patches.shape[1])
        if side != 28 or side * side != patches.shape[1]:
            raise AssertionError("Ordered head expert requires a 28x28 patch grid")
        head_indices = torch.where(parts.eq(0))[0]
        self._head_grid_last_head_count = int(len(head_indices))
        if not len(head_indices):
            self._head_grid_last_valid_mask_count = 0
            self._head_grid_last_empty_cell_fallbacks = 0
            return [
                patches.new_zeros(
                    (len(parts), self.head_identity_expert_grid_dim)
                )
                for _ in range(self.head_identity_expert_branch_count)
            ]

        head_patches = patches.index_select(0, head_indices).detach()
        head_cls = cls.index_select(0, head_indices).detach()
        projected_patches = self.head_identity_expert_grid_projection(
            head_patches
        ).reshape(
            len(head_indices),
            side,
            side,
            self.head_identity_expert_grid_dim,
        )
        projected_cls = self.head_identity_expert_grid_projection(head_cls)
        mask = foreground_mask.index_select(0, head_indices).float().unsqueeze(1)
        mask = F.avg_pool2d(mask, kernel_size=16, stride=16).squeeze(1)
        if mask.shape[1:] != (side, side):
            raise AssertionError("Foreground mask did not align to DINO patches")
        valid = mask.flatten(1).sum(dim=1) > 0.0
        self._head_grid_last_valid_mask_count = int(valid.sum())
        mask = torch.where(valid[:, None, None], mask, torch.ones_like(mask))

        boundaries = (0, 9, 18, 28)
        grid_tokens: list[torch.Tensor] = []
        empty_fallbacks = 0
        for row in range(3):
            for column in range(3):
                top, bottom = boundaries[row], boundaries[row + 1]
                left, right = boundaries[column], boundaries[column + 1]
                cell = projected_patches[:, top:bottom, left:right]
                weights = mask[:, top:bottom, left:right]
                mass = weights.sum(dim=(1, 2))
                empty = mass <= 1e-6
                empty_fallbacks += int(empty.sum())
                effective = torch.where(
                    empty[:, None, None], torch.ones_like(weights), weights
                )
                pooled = (
                    cell * effective.unsqueeze(-1).to(dtype=cell.dtype)
                ).sum(dim=(1, 2)) / effective.sum(dim=(1, 2)).clamp_min(
                    1e-6
                ).unsqueeze(-1)
                grid_tokens.append(pooled)
        self._head_grid_last_empty_cell_fallbacks = empty_fallbacks
        ordered = torch.stack([projected_cls, *grid_tokens], dim=1)
        position = self.head_identity_expert_grid_position.to(
            device=ordered.device, dtype=ordered.dtype
        )
        refined = self.head_identity_expert_grid_transformer(ordered + position)
        return [
            refined[:, index].new_zeros(
                (len(parts), self.head_identity_expert_grid_dim)
            ).index_copy(0, head_indices, refined[:, index])
            for index in range(self.head_identity_expert_branch_count)
        ]

    def _head_identity_expert_cosines(
        self,
        inputs: list[torch.Tensor],
        parts: torch.Tensor,
    ) -> None:
        """Run the detached all-ID expert only for official head rows."""
        self._head_identity_expert_branch_cosine = None
        self._head_identity_expert_score = None
        if not self.head_identity_expert:
            return
        if len(inputs) != self.head_identity_expert_branch_count:
            raise AssertionError("Head expert branch count changed")
        head_indices = torch.where(parts.eq(0))[0]
        if len(head_indices) == 0:
            self._head_identity_expert_score = inputs[0].new_zeros(
                (len(parts), self.head_identity_expert_output_classes)
            )
            return
        expert_neck_values: list[torch.Tensor] = []
        for projection, neck, value in zip(
            self.head_identity_expert_projections,
            self.head_identity_expert_necks,
            inputs,
        ):
            expert_input = value.index_select(0, head_indices)
            if self.head_identity_expert_detach:
                expert_input = expert_input.detach()
            expert_raw = projection(expert_input)
            if self.training and len(expert_raw) == 1:
                expert_neck = F.batch_norm(
                    expert_raw,
                    neck.running_mean,
                    neck.running_var,
                    neck.weight,
                    neck.bias,
                    training=False,
                    momentum=neck.momentum,
                    eps=neck.eps,
                )
            else:
                expert_neck = neck(expert_raw)
            expert_neck_values.append(expert_neck)
        expert_neck = F.normalize(
            torch.stack(expert_neck_values, dim=1), dim=-1
        )
        expert_weights = F.normalize(
            self.head_identity_expert_class_weight, dim=-1
        )
        branch_cosine = torch.einsum(
            "bmd,mcd->bmc", expert_neck, expert_weights
        )
        score = branch_cosine.mean(dim=1)
        full_score = score.new_zeros(
            (len(parts), self.head_identity_expert_output_classes)
        ).index_copy(0, head_indices, score)
        self._head_identity_expert_branch_cosine = branch_cosine
        self._head_identity_expert_score = full_score

    def canonical_head_views(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the fixed mean-canvas head views declared for V13.1."""
        if images.ndim != 4 or foreground_mask.shape != (
            len(images), images.shape[-2], images.shape[-1]
        ):
            raise ValueError("Canonical head mask/image geometry differs")
        if parts.shape != (len(images),):
            raise ValueError("Canonical head parts batch differs")
        valid = parts.eq(0) & foreground_mask.flatten(1).any(dim=1)
        indices = torch.where(valid)[0]
        if not len(indices):
            return images.new_empty((0, *images.shape[1:])), indices
        height, width = images.shape[-2:]
        target = int(round(min(height, width) * self.foreground_scale_canvas_fraction))
        if (height, width, target) != (448, 448, 336):
            raise AssertionError("V13.1 canonical canvas constants changed")
        views: list[torch.Tensor] = []
        canvas_shapes: list[tuple[int, int]] = []
        for index in indices.tolist():
            mask = foreground_mask[index].bool()
            coordinates = torch.nonzero(mask, as_tuple=False)
            top = int(coordinates[:, 0].min().item())
            bottom = int(coordinates[:, 0].max().item()) + 1
            left = int(coordinates[:, 1].min().item())
            right = int(coordinates[:, 1].max().item()) + 1
            margin_y = max(1, int(round(0.05 * (bottom - top))))
            margin_x = max(1, int(round(0.05 * (right - left))))
            top = max(0, top - margin_y)
            bottom = min(height, bottom + margin_y)
            left = max(0, left - margin_x)
            right = min(width, right + margin_x)
            crop = images[index : index + 1, :, top:bottom, left:right]
            crop_mask = mask[None, None, top:bottom, left:right].to(crop.dtype)
            crop = crop * crop_mask
            crop_height, crop_width = crop.shape[-2:]
            scale = target / max(crop_height, crop_width)
            resized_height = max(1, min(target, int(round(crop_height * scale))))
            resized_width = max(1, min(target, int(round(crop_width * scale))))
            canvas_shapes.append((resized_height, resized_width))
            resized = F.interpolate(
                crop,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            resized_mask = F.interpolate(
                crop_mask,
                size=(resized_height, resized_width),
                mode="nearest",
            )
            resized = resized * resized_mask
            pad_top = (height - resized_height) // 2
            pad_bottom = height - resized_height - pad_top
            pad_left = (width - resized_width) // 2
            pad_right = width - resized_width - pad_left
            views.append(
                F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom))
            )
        self._foreground_scale_last_canvas_shapes = tuple(canvas_shapes)
        return torch.cat(views, dim=0), indices

    def _foreground_scale_descriptors(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        image_geometry: torch.Tensor | None,
        foreground_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        canonical, indices = self.canonical_head_views(
            images, parts, foreground_mask
        )
        self._foreground_scale_last_canonical = None
        if self._foreground_scale_debug:
            canonical.requires_grad_(True)
            canonical.retain_grad()
            self._foreground_scale_last_canonical = canonical
        self.foreground_scale_normalization = False
        try:
            primary_raw, primary_neck = self._descriptors(
                images, parts, image_geometry=image_geometry
            )
            primary_geometry_residual = self._geometry_last_residual
            if not len(indices):
                self._foreground_scale_last_indices = indices.detach()
                self._foreground_scale_last_blend_error = images.new_zeros(())
                self._foreground_scale_last_base_path_error = images.new_zeros(())
                return primary_raw, primary_neck
            canonical_geometry = (
                image_geometry.index_select(0, indices)
                if image_geometry is not None
                else None
            )
            canonical_raw, canonical_neck = self._descriptors(
                canonical,
                parts.index_select(0, indices),
                image_geometry=canonical_geometry,
            )
            blend = self.foreground_scale_blend
            blended_raw = (1.0 - blend) * primary_raw.index_select(
                0, indices
            ) + blend * canonical_raw
            blended_neck = (1.0 - blend) * primary_neck.index_select(
                0, indices
            ) + blend * canonical_neck
            output_raw = primary_raw.index_copy(0, indices, blended_raw)
            output_neck = primary_neck.index_copy(0, indices, blended_neck)
            self._geometry_last_residual = primary_geometry_residual
            self._foreground_scale_last_indices = indices.detach()
            check = output_neck.index_select(0, indices) - blended_neck
            self._foreground_scale_last_blend_error = check.detach().abs().max()
            untouched = torch.ones(
                len(images), dtype=torch.bool, device=images.device
            )
            untouched[indices] = False
            if untouched.any():
                raw_error = (
                    output_raw[untouched] - primary_raw[untouched]
                ).detach().abs().max()
                neck_error = (
                    output_neck[untouched] - primary_neck[untouched]
                ).detach().abs().max()
                self._foreground_scale_last_base_path_error = torch.maximum(
                    raw_error, neck_error
                )
            else:
                self._foreground_scale_last_base_path_error = images.new_zeros(())
            return output_raw, output_neck
        finally:
            self.foreground_scale_normalization = True

    def _descriptors(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        image_geometry: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.foreground_scale_normalization:
            if foreground_mask is None:
                raise ValueError("Foreground scale normalization requires a mask")
            return self._foreground_scale_descriptors(
                images, parts, image_geometry, foreground_mask
            )
        self._slot_reconstruction_loss = None
        self._dense_correspondence_tokens = None
        self._identity_query_shared_score = None
        self._identity_query_part_score = None
        self._identity_query_attention_mass = None
        self._image_texture_last_residual = None
        self._body_head_last_residual = None
        self._body_head_alignment_loss = None
        self._geometry_last_residual = None
        self._foreground_auxiliary_logits = None
        self._head_identity_expert_branch_cosine = None
        self._head_identity_expert_score = None
        if (
            self.part_side_embedding_enabled
            or self.continuous_geometry_conditioning
            or self.foreground_token_conditioning
            or self.ordered_head_grid_expert
        ):
            if len(parts) != len(images):
                raise ValueError("parts batch does not match images batch")
            self._active_parts = parts
        if self.continuous_geometry_conditioning:
            if image_geometry is None:
                raise ValueError("Continuous geometry model requires image geometry")
            if image_geometry.shape != (len(images), 2):
                raise ValueError("Expected image geometry [B, 2]")
            self._active_image_geometry = image_geometry
        elif image_geometry is not None:
            raise ValueError("Image geometry supplied to a model without conditioning")
        if self.foreground_token_conditioning or self.ordered_head_grid_expert:
            if foreground_mask is None:
                raise ValueError("Foreground-aware model requires a mask")
            if foreground_mask.shape != images.shape[:1] + images.shape[2:]:
                raise ValueError("Foreground mask/image geometry differs")
            if self.foreground_token_conditioning:
                self._active_foreground_mask = foreground_mask
        elif foreground_mask is not None:
            raise ValueError(
                "Foreground mask supplied to a model without token conditioning"
            )
        if (
            self.head_convpass_attention
            or self.part_routed_qv_lora
            or self.suffix_head_qv_lora
            or self.part_mlp_expert_blocks
            or self.head_tail_expert_blocks
        ):
            if len(parts) != len(images):
                raise ValueError("parts batch does not match images batch")
            self._part_routing_context.parts = parts
        try:
            tokens = self._backbone_tokens(images)
        finally:
            self._active_parts = None
            self._active_image_geometry = None
            self._active_foreground_mask = None
            # Keep the no-grad part indices alive through activation-checkpoint
            # recomputation in backward. The next forward atomically replaces
            # them before any routed module runs.
        if self.spatial_backbone:
            if tokens.ndim != 4:
                raise AssertionError(f"Expected NCHW backbone map, got {tokens.shape}")
            patch_map = (
                tokens.permute(0, 3, 1, 2)
                if self.spatial_layout == "nhwc"
                else tokens
            )
            patches = patch_map.flatten(2).transpose(1, 2)
            # A convolutional backbone has no class token. Global max pooling
            # complements the patch mean with the strongest localized marking,
            # preserving the incumbent two-vector global branch width.
            cls = patches.amax(dim=1)
            side = int(patch_map.shape[-1])
            if patch_map.shape[-2] != side:
                raise AssertionError(f"Non-square feature map: {patch_map.shape}")
            prefix_count = 0
        else:
            prefix_count = (
                int(self.backbone.n_storage_tokens) + 1
                if self.lingbot_backbone
                else int(self.backbone.num_prefix_tokens)
            )
            if prefix_count:
                cls = tokens[:, 0]
                patches = tokens[:, prefix_count:]
            else:
                # SigLIP's vision encoder has no class token. Reuse its
                # published attention-map pool as the semantic global summary
                # while keeping every dense token for stripes/covariance.
                patches = tokens
                pool = getattr(self.backbone, "pool", None)
                cls = pool(tokens) if callable(pool) else patches.mean(dim=1)
                if cls.ndim == 3 and cls.shape[1] == 1:
                    cls = cls[:, 0]
                if cls.ndim != 2:
                    raise AssertionError(
                        f"Expected pooled global descriptor, got {cls.shape}"
                    )
                feature_norm = getattr(self.backbone, "fc_norm", None)
                if isinstance(feature_norm, nn.Module) and not isinstance(
                    feature_norm, nn.Identity
                ):
                    cls = feature_norm(cls)
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1]:
                raise AssertionError(f"Non-square patch grid: {patches.shape}")
            patch_map = patches.reshape(
                patches.shape[0], side, side, patches.shape[-1]
            ).permute(0, 3, 1, 2)
        if self.cross_level_texture_pyramid:
            residual = self._cross_level_texture_residual(
                prefix_count=prefix_count,
                patches=patches,
                side=side,
                parts=parts,
            )
            patch_map = patch_map + residual.to(dtype=patch_map.dtype)
            patches = patch_map.flatten(2).transpose(1, 2)
        if self.image_frequency_texture_side:
            image_residual = self._image_texture_residual(
                images, parts, side
            )
            patch_map = patch_map + image_residual.to(dtype=patch_map.dtype)
            patches = patch_map.flatten(2).transpose(1, 2)
        if self.foreground_auxiliary and self.training:
            logits = self.foreground_auxiliary_head(patches).squeeze(-1)
            self._foreground_auxiliary_logits = logits.reshape(
                len(images), side, side
            )
        global_descriptor = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        if self.dense_correspondence_training and self.training:
            if self._intermediate_tokens is None:
                raise AssertionError("Dense correspondence features were not captured")
            intermediate = self.backbone.norm(self._intermediate_tokens)
            intermediate_patches = intermediate[:, prefix_count:]
            if intermediate_patches.shape != patches.shape:
                raise AssertionError(
                    "Dense correspondence feature levels have different grids"
                )
            relevance = torch.einsum(
                "bnd,bd->bn",
                F.normalize(patches.float(), dim=-1),
                F.normalize(cls.float(), dim=-1),
            )
            if patches.shape[1] < self.dense_correspondence_topk:
                raise AssertionError("Dense correspondence patch grid is too small")
            indices = relevance.topk(
                self.dense_correspondence_topk, dim=1
            ).indices
            gather_index = indices.unsqueeze(-1).expand(
                -1, -1, patches.shape[-1]
            )
            selected_intermediate = intermediate_patches.gather(
                1, gather_index
            )
            selected_final = patches.gather(1, gather_index)
            local_intermediate = self.dense_intermediate_projection(
                selected_intermediate
            )
            local_final = self.dense_final_projection(selected_final)
            self._dense_correspondence_tokens = F.normalize(
                torch.cat((local_intermediate, local_final), dim=-1).float(),
                dim=-1,
            )
        if self.hierarchical_slot_architecture:
            if self.slot_aggregator is None or self._intermediate_tokens is None:
                raise AssertionError("Hierarchical slot features were not captured")
            intermediate = self.backbone.norm(self._intermediate_tokens)
            intermediate_patches = intermediate[:, prefix_count:]
            aggregated, identity_slots, reconstruction_loss = (
                self.slot_aggregator(intermediate_patches, patches, parts)
            )
            if identity_slots.shape[1] != 4:
                raise AssertionError("Hierarchical slots lost an identity descriptor")
            self._slot_reconstruction_loss = reconstruction_loss
            inputs = [
                global_descriptor,
                aggregated,
                *identity_slots.unbind(dim=1),
            ]
        else:
            half = F.adaptive_avg_pool2d(patch_map, (2, 1)).squeeze(-1)
            thirds = F.adaptive_avg_pool2d(patch_map, (3, 1)).squeeze(-1)
            vertical_half: torch.Tensor | None = None
            vertical_thirds: torch.Tensor | None = None
            if self.part_aligned_axis or self.bidirectional or self.vertical_branches:
                vertical_half = F.adaptive_avg_pool2d(
                    patch_map, (1, 2)
                ).squeeze(-2)
                vertical_thirds = F.adaptive_avg_pool2d(
                    patch_map, (1, 3)
                ).squeeze(-2)
            if self.part_aligned_axis:
                # Head crops use top-to-bottom facial structure. Side-body crops
                # are predominantly horizontal, so their identity markings are
                # partitioned along the image long axis instead. Channel-wise
                # symmetric mean/absolute-difference makes the two outer locations
                # an unordered pair: the descriptor survives unknown facing
                # direction and horizontal flip while retaining how different the
                # front/rear regions are.
                body = parts.ne(0)[:, None, None]
                body_half = torch.stack(
                    [
                        0.5 * (vertical_half[:, :, 0] + vertical_half[:, :, 1]),
                        (vertical_half[:, :, 0] - vertical_half[:, :, 1]).abs(),
                    ],
                    dim=2,
                )
                body_thirds = torch.stack(
                    [
                        0.5
                        * (vertical_thirds[:, :, 0] + vertical_thirds[:, :, 2]),
                        vertical_thirds[:, :, 1],
                        (
                            vertical_thirds[:, :, 0] - vertical_thirds[:, :, 2]
                        ).abs(),
                    ],
                    dim=2,
                )
                half = torch.where(body, body_half, half)
                thirds = torch.where(body, body_thirds, thirds)
            inputs = [
                global_descriptor,
                half[:, :, 0],
                half[:, :, 1],
                thirds[:, :, 0],
                thirds[:, :, 1],
                thirds[:, :, 2],
            ]
            if self.bidirectional or self.vertical_branches:
                inputs.extend(
                    [
                        vertical_half[:, :, 0],
                        vertical_half[:, :, 1],
                        vertical_thirds[:, :, 0],
                        vertical_thirds[:, :, 1],
                        vertical_thirds[:, :, 2],
                    ]
                )
        if self.covariance_branch or self.gradient_covariance_branch:
            # A compact bilinear descriptor: class evidence can depend on
            # channel co-occurrences over all patches, rather than only their
            # first-order means. Signed square-root and L2 normalization are
            # the standard stabilizers used by bilinear fine-grained models.
            reduced = self.covariance_reduction(patches).float()
        if self.covariance_branch:
            centered = reduced - reduced.mean(dim=1, keepdim=True)
            if self.covariance_standardize:
                # Correlation pooling is the second-order analogue of
                # instance normalization: remove per-image channel scale so
                # illumination/sensor contrast cannot dominate spot and edge
                # co-occurrences.  Epsilon is numerical only, not fitted.
                channel_scale = centered.square().mean(
                    dim=1, keepdim=True
                ).add(1e-4).rsqrt()
                centered = centered * channel_scale
            covariance = torch.einsum(
                "bni,bnj->bij", centered, centered
            ) / max(1, centered.shape[1] - 1)
            if self.covariance_matrix_sqrt:
                covariance = self._matrix_square_root(covariance)
            else:
                covariance = torch.sign(covariance) * torch.sqrt(
                    covariance.abs() + 1e-6
                )
            covariance = F.normalize(covariance.flatten(1), dim=-1)
            inputs.append(covariance.to(dtype=patches.dtype))
        if self.intermediate_covariance_branch:
            if self._intermediate_tokens is None:
                raise AssertionError("intermediate DINO tokens were not captured")
            intermediate = self.backbone.norm(self._intermediate_tokens)
            intermediate_patches = intermediate[:, prefix_count:]
            if intermediate_patches.shape != patches.shape:
                raise AssertionError(
                    "intermediate and final DINO patch grids do not match"
                )
            intermediate_reduced = self.intermediate_covariance_reduction(
                intermediate_patches
            ).float()
            intermediate_centered = intermediate_reduced - (
                intermediate_reduced.mean(dim=1, keepdim=True)
            )
            final_centered = reduced - reduced.mean(dim=1, keepdim=True)
            # Cross-layer covariance binds texture-rich block-18 evidence to
            # final semantic evidence at the same patch positions. It retains
            # local markings that survive into the identity-aware abstraction
            # while suppressing content present in only one representation.
            cross_covariance = torch.einsum(
                "bni,bnj->bij", intermediate_centered, final_centered
            ) / max(1, final_centered.shape[1] - 1)
            cross_covariance = torch.sign(cross_covariance) * torch.sqrt(
                cross_covariance.abs() + 1e-6
            )
            cross_covariance = F.normalize(
                cross_covariance.flatten(1), dim=-1
            )
            inputs.append(cross_covariance.to(dtype=patches.dtype))
        if self.gradient_covariance_branch:
            # DINO patch differences are a semantic high-pass signal.  Their
            # second-order statistics retain adjacent spot/fur/contour changes
            # without a second image-space FFT/backbone pass or its background
            # amplification. Horizontal and vertical changes share one basis.
            reduced_map = reduced.reshape(
                reduced.shape[0], side, side, self.covariance_dim
            )
            horizontal = reduced_map[:, :, 1:] - reduced_map[:, :, :-1]
            vertical = reduced_map[:, 1:, :] - reduced_map[:, :-1, :]
            gradients = torch.cat(
                [horizontal.flatten(1, 2), vertical.flatten(1, 2)], dim=1
            )
            gradients = gradients - gradients.mean(dim=1, keepdim=True)
            gradient_covariance = torch.einsum(
                "bni,bnj->bij", gradients, gradients
            ) / max(1, gradients.shape[1] - 1)
            gradient_covariance = torch.sign(gradient_covariance) * torch.sqrt(
                gradient_covariance.abs() + 1e-6
            )
            gradient_covariance = F.normalize(
                gradient_covariance.flatten(1), dim=-1
            )
            inputs.append(gradient_covariance.to(dtype=patches.dtype))
        if self.semantic_topk_branch:
            # DINO's global token supplies an identity-agnostic semantic
            # foreground cue. Pooling the most aligned quarter of dense patch
            # tokens reduces background dilution while keeping selection
            # independent of class labels and validation/test statistics.
            relevance = torch.einsum(
                "bnd,bd->bn",
                F.normalize(patches.float(), dim=-1),
                F.normalize(cls.float(), dim=-1),
            )
            keep = max(
                1,
                int(math.ceil(patches.shape[1] * self.semantic_topk_fraction)),
            )
            indices = relevance.topk(keep, dim=1).indices
            selected = patches.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
            )
            inputs.append(selected.mean(dim=1))
        if self.simpool_branch:
            # SimPool-style one-step attention pooling learns a clean
            # object-focused summary from the final DINO patch set. Keeping it
            # as an additional supervised branch preserves the incumbent CLS,
            # stripe, covariance and gradient evidence.
            values = self.simpool_norm(patches)
            query = self.simpool_query(patches.mean(dim=1, keepdim=True))
            keys = self.simpool_key(values)
            attention = torch.matmul(query, keys.transpose(1, 2))
            attention = (attention / math.sqrt(patches.shape[-1])).softmax(dim=-1)
            inputs.append(torch.matmul(attention, values).squeeze(1))
        if self.pattern_a2gc_branch:
            # The normalized OT descriptor sends large scaled gradients back
            # through its assignment logits. Keep this compact new branch in
            # FP32 so AMP's loss scale cannot overflow an FP16 conv backward;
            # the much larger H+ trunk and all incumbent heads remain under
            # the caller's autocast context.
            with torch.autocast(device_type=patch_map.device.type, enabled=False):
                inputs.append(
                    self.pattern_aggregator(
                        patch_map.float(), cls.float(), parts
                    )
                )
        if self.jpm_local_branches:
            inputs.extend(self._jpm_local_cls_features(prefix_count))
        if len(inputs) != self.branch_count:
            raise AssertionError(
                f"Descriptor branches {len(inputs)} != configured {self.branch_count}"
            )
        expert_inputs = (
            self._external_head_representation_inputs(parts)
            if self.external_head_representation
            else self._ordered_head_grid_inputs(
                patches, cls, parts, foreground_mask
            )
            if self.ordered_head_grid_expert
            else inputs
        )
        self._head_identity_expert_cosines(expert_inputs, parts)
        raw = torch.stack(
            [projection(value) for projection, value in zip(self.branch_projections, inputs)],
            dim=1,
        )
        if self.body_to_head_distillation:
            head_delta = torch.stack(
                [
                    adapter(raw[:, branch_index])
                    for branch_index, adapter in enumerate(
                        self.head_identity_adapters
                    )
                ],
                dim=1,
            )
            head_mask = parts.eq(0).to(dtype=raw.dtype).view(-1, 1, 1)
            self._body_head_last_residual = (
                self.body_head_adapter_scale
                * head_mask
                * head_delta.to(dtype=raw.dtype)
            )
            raw = raw + self._body_head_last_residual
        normalization_domains = parts
        if self.modality_specific_bn:
            # ARBase augmentation has no colour transform, so exact grayscale
            # remains observable after ImageNet normalization.  A sparse-grid
            # median ignores the small mean-colour translation padding.  The
            # fixed threshold separates the zero-chroma train cluster and is
            # not estimated from validation or test images.
            mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
            std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
            rgb = images[:, :, ::16, ::16] * std + mean
            chroma = (
                (rgb[:, 0] - rgb[:, 1]).abs()
                + (rgb[:, 1] - rgb[:, 2]).abs()
            ).flatten(1).median(dim=1).values
            normalization_domains = (chroma < 0.005).long()
        neck_values: list[torch.Tensor] = []
        for branch_index, branch_neck in enumerate(self.branch_necks):
            if not (self.domain_specific_bn or self.modality_specific_bn):
                neck_values.append(branch_neck(raw[:, branch_index]))
                continue
            normalized = torch.empty_like(raw[:, branch_index])
            for domain_index, domain_neck in enumerate(branch_neck):
                mask = normalization_domains == domain_index
                if not mask.any():
                    continue
                values = raw[mask, branch_index]
                if self.training and len(values) == 1:
                    # Avoid BatchNorm's singleton-batch error without leaking
                    # statistics across crop domains. Normal batches contain
                    # many examples of every part; this is a safe rare fallback.
                    values = F.batch_norm(
                        values,
                        domain_neck.running_mean,
                        domain_neck.running_var,
                        domain_neck.weight,
                        domain_neck.bias,
                        training=False,
                        momentum=domain_neck.momentum,
                        eps=domain_neck.eps,
                    )
                else:
                    values = domain_neck(values)
                normalized[mask] = values
            neck_values.append(normalized)
        neck = torch.stack(neck_values, dim=1)
        if self.identity_query_pooling:
            (
                self._identity_query_shared_score,
                self._identity_query_part_score,
            ) = self._prototype_query_scores(patches, parts)
        return raw, neck

    def _part_neck(
        self, neck: torch.Tensor, parts: torch.Tensor
    ) -> torch.Tensor:
        if self.part_adapters is None:
            return neck
        all_deltas = torch.stack(
            [adapter(neck) for adapter in self.part_adapters], dim=1
        )
        row = torch.arange(len(parts), device=parts.device)
        return neck + all_deltas[row, parts]

    def _prototype_query_scores(
        self,
        patches: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool identity-conditioned local evidence from the final patch grid."""
        if not self.identity_query_pooling:
            raise RuntimeError("Identity-query pooling is disabled")
        if self.shared_class_weight.ndim != 3:
            raise AssertionError("Identity queries require one prototype per branch/class")
        keys = F.normalize(
            self.identity_query_patch_projection(patches).float(), dim=-1
        )
        shared_weights = self.shared_class_weight[0]
        part_weights = shared_weights.unsqueeze(0) + (
            self.part_delta_scale * self.part_class_delta[parts, 0]
        )
        shared_queries = F.normalize(
            self.identity_query_weight_projection(shared_weights).float(),
            dim=-1,
        )
        part_queries = F.normalize(
            self.identity_query_weight_projection(part_weights).float(),
            dim=-1,
        )
        shared_similarity = torch.einsum("bnd,cd->bnc", keys, shared_queries)
        part_similarity = torch.einsum("bnd,bcd->bnc", keys, part_queries)
        shared_attention = torch.softmax(
            shared_similarity / self.identity_query_temperature, dim=1
        )
        part_attention = torch.softmax(
            part_similarity / self.identity_query_temperature, dim=1
        )
        self._identity_query_attention_mass = (
            shared_attention.sum(dim=1).detach(),
            part_attention.sum(dim=1).detach(),
        )
        shared_score = (shared_attention * shared_similarity).sum(dim=1)
        part_score = (part_attention * part_similarity).sum(dim=1)
        return shared_score, part_score

    def identity_query_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._identity_query_shared_score is None
            or self._identity_query_part_score is None
        ):
            raise RuntimeError("Identity-query scores are unavailable before forward")
        return self._identity_query_shared_score, self._identity_query_part_score

    def identity_query_attention_mass(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._identity_query_attention_mass is None:
            raise RuntimeError("Identity-query attention is unavailable before forward")
        return self._identity_query_attention_mass

    def _cosines(
        self,
        neck: torch.Tensor,
        parts: torch.Tensor,
        part_neck: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_neck = F.normalize(neck, dim=-1)
        if part_neck is None:
            part_neck = self._part_neck(neck, parts)
        normalized_part_neck = F.normalize(part_neck, dim=-1)
        shared_weights = F.normalize(self.shared_class_weight, dim=-1)
        if self.subcenters > 1:
            shared_cosine = torch.einsum(
                "bmd,mckd->bmck", normalized_neck, shared_weights
            ).amax(dim=-1)
        else:
            shared_cosine = torch.einsum(
                "bmd,mcd->bmc", normalized_neck, shared_weights
            )
        if self.hierarchical_part_heads:
            shrinkage = self.part_class_shrinkage[parts, None, :, None]
            part_weights = self.shared_class_weight.unsqueeze(0) + (
                self.part_delta_scale
                * shrinkage
                * self.part_class_delta[parts]
            )
        elif self.bidirectional:
            part_weights = self.part_class_weight[parts]
        else:
            part_weights = self.shared_class_weight.unsqueeze(0) + (
                self.part_delta_scale * self.part_class_delta[parts]
            )
            if self.prototype_transport_rank > 0:
                transport_hidden = torch.einsum(
                    "pmrd,mcd->pmcr",
                    self.part_prototype_transport_down,
                    self.shared_class_weight,
                )
                transported = torch.einsum(
                    "pmdr,pmcr->pmcd",
                    self.part_prototype_transport_up,
                    transport_hidden,
                )
                part_weights = part_weights + (
                    self.prototype_transport_scale * transported[parts]
                )
        normalized_part_weights = F.normalize(part_weights, dim=-1)
        if self.subcenters > 1:
            part_cosine = torch.einsum(
                "bmd,bmckd->bmck", normalized_part_neck, normalized_part_weights
            ).amax(dim=-1)
        else:
            part_cosine = torch.einsum(
                "bmd,bmcd->bmc", normalized_part_neck, normalized_part_weights
            )
        return shared_cosine, part_cosine

    def _fuse_cosines(
        self,
        shared_cosine: torch.Tensor,
        part_cosine: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.bidirectional or self.learned_branch_gates):
            return shared_cosine.mean(dim=1), part_cosine.mean(dim=1)
        shared_gate = torch.softmax(self.shared_branch_gate, dim=0)
        part_gate = torch.softmax(self.part_branch_gate[parts], dim=1)
        shared_score = torch.einsum(
            "bmc,m->bc", shared_cosine, shared_gate
        )
        part_score = torch.einsum("bmc,bm->bc", part_cosine, part_gate)
        return shared_score, part_score

    def _prototype_branch_cosines(
        self,
        neck: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        if not self.prototype_memory:
            raise RuntimeError("prototype memory is disabled")
        normalized_neck = F.normalize(neck.float(), dim=-1)
        shared_memory = F.normalize(
            self.shared_prototype_memory.float(), dim=-1
        )
        shared_cosine = torch.einsum(
            "bmd,cmd->bmc", normalized_neck, shared_memory
        )
        # Avoid materializing a [B, C, M, D] copy of the memory bank.  Each
        # crop belongs to exactly one of three known domains, so three compact
        # masked products retain the same gradients with much lower memory.
        part_cosine = torch.empty_like(shared_cosine)
        for part_index in range(3):
            mask = parts == part_index
            if not mask.any():
                continue
            part_memory = F.normalize(
                self.part_prototype_memory[part_index].float(), dim=-1
            )
            part_cosine[mask] = torch.einsum(
                "bmd,cmd->bmc", normalized_neck[mask], part_memory
            )
        part_seen = self.part_prototype_seen[parts].unsqueeze(1)
        fused = 0.45 * shared_cosine + 0.55 * part_cosine
        branch_cosine = torch.where(part_seen, fused, shared_cosine)
        shared_seen = self.shared_prototype_seen[None, None, :]
        return branch_cosine.masked_fill(~shared_seen, -1.0)

    def prototype_logits_from_embedding(
        self,
        shared_embedding: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return train-only memory logits and targets seen before this batch."""
        neck = shared_embedding.reshape(
            -1, self.branch_count, self.embedding_dim
        )
        branch_cosine = self._prototype_branch_cosines(neck, parts)
        logits = self._margin(branch_cosine, labels)
        logits = logits.masked_fill(
            ~self.shared_prototype_seen[None, None, :], -1e4
        )
        valid_targets = self.shared_prototype_seen[labels]
        return logits, valid_targets

    @torch.no_grad()
    def update_prototype_memory(
        self,
        shared_embedding: torch.Tensor,
        labels: torch.Tensor,
        parts: torch.Tensor,
    ) -> None:
        """Update class centres from a fold-train batch after its optimizer step."""
        if not self.prototype_memory:
            return
        if not self.training:
            raise RuntimeError("prototype memory may only update in train mode")
        features = F.normalize(
            shared_embedding.detach().float().reshape(
                -1, self.branch_count, self.embedding_dim
            ),
            dim=-1,
        )

        def momentum_update(
            memory: torch.Tensor,
            seen: torch.Tensor,
            indices: torch.Tensor,
        ) -> None:
            slot_count = memory.shape[0]
            sums = torch.zeros_like(memory)
            sums.index_add_(0, indices, features)
            counts = torch.bincount(indices, minlength=slot_count)
            active = counts > 0
            means = F.normalize(
                sums[active] / counts[active, None, None], dim=-1
            )
            previous = memory[active]
            blended = F.normalize(
                self.prototype_momentum * previous
                + (1.0 - self.prototype_momentum) * means,
                dim=-1,
            )
            values = torch.where(
                seen[active, None, None], blended, means
            )
            memory[active] = values.to(memory.dtype)
            seen[active] = True

        momentum_update(
            self.shared_prototype_memory,
            self.shared_prototype_seen,
            labels,
        )
        flat_part_memory = self.part_prototype_memory.view(
            3 * self.num_classes, self.branch_count, self.embedding_dim
        )
        flat_part_seen = self.part_prototype_seen.view(-1)
        part_labels = parts * self.num_classes + labels
        momentum_update(flat_part_memory, flat_part_seen, part_labels)

    def instance_queue_contents(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the filled training-only FIFO keys and provenance metadata."""
        if not self.training_instance_queue:
            raise RuntimeError("Training instance queue is disabled")
        size = int(self.instance_queue_size.item())
        if not 0 <= size <= self.instance_queue_capacity:
            raise AssertionError("Instance queue size is outside capacity")
        return (
            self.instance_queue_embeddings[:size].detach(),
            self.instance_queue_labels[:size],
            self.instance_queue_parts[:size],
            self.instance_queue_sources[:size],
        )

    @torch.no_grad()
    def update_instance_queue(
        self,
        shared_embedding: torch.Tensor,
        labels: torch.Tensor,
        parts: torch.Tensor,
        source_codes: torch.Tensor,
    ) -> None:
        """Enqueue one successful fold-train optimizer batch as detached keys."""
        if not self.training_instance_queue:
            return
        if not self.training:
            raise RuntimeError("Instance queue may only update in train mode")
        rows = len(shared_embedding)
        expected_width = self.branch_count * self.embedding_dim
        if shared_embedding.shape != (rows, expected_width):
            raise ValueError("Instance queue embedding geometry changed")
        if any(value.shape != (rows,) for value in (labels, parts, source_codes)):
            raise ValueError("Instance queue metadata geometry changed")
        if torch.any((parts < 0) | (parts >= 3)):
            raise ValueError("Instance queue received an invalid released part")
        if torch.any(labels < 0) or torch.any(source_codes < 0):
            raise ValueError("Instance queue received invalid train provenance")

        features = F.normalize(
            shared_embedding.detach().float(), dim=-1
        ).to(self.instance_queue_embeddings.dtype)
        labels = labels.detach().to(dtype=torch.long)
        parts = parts.detach().to(dtype=torch.long)
        source_codes = source_codes.detach().to(dtype=torch.long)
        capacity = self.instance_queue_capacity
        if rows >= capacity:
            self.instance_queue_embeddings.copy_(features[-capacity:])
            self.instance_queue_labels.copy_(labels[-capacity:])
            self.instance_queue_parts.copy_(parts[-capacity:])
            self.instance_queue_sources.copy_(source_codes[-capacity:])
            self.instance_queue_pointer.zero_()
            self.instance_queue_size.fill_(capacity)
            return

        pointer = int(self.instance_queue_pointer.item())
        first = min(rows, capacity - pointer)
        second = rows - first
        self.instance_queue_embeddings[pointer : pointer + first].copy_(
            features[:first]
        )
        self.instance_queue_labels[pointer : pointer + first].copy_(labels[:first])
        self.instance_queue_parts[pointer : pointer + first].copy_(parts[:first])
        self.instance_queue_sources[pointer : pointer + first].copy_(
            source_codes[:first]
        )
        if second:
            self.instance_queue_embeddings[:second].copy_(features[first:])
            self.instance_queue_labels[:second].copy_(labels[first:])
            self.instance_queue_parts[:second].copy_(parts[first:])
            self.instance_queue_sources[:second].copy_(source_codes[first:])
        self.instance_queue_pointer.fill_((pointer + rows) % capacity)
        self.instance_queue_size.fill_(
            min(capacity, int(self.instance_queue_size.item()) + rows)
        )

    def _quality_scaler(self, raw: torch.Tensor) -> torch.Tensor:
        safe_norms = raw.norm(dim=-1).clamp(0.001, 100.0).detach()
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std().clamp_min(1e-3)
            alpha = self.quality_t_alpha
            self.quality_batch_mean.mul_(1.0 - alpha).add_(alpha * mean)
            self.quality_batch_std.mul_(1.0 - alpha).add_(alpha * std)
        return (
            (safe_norms - self.quality_batch_mean)
            / (self.quality_batch_std + 1e-3)
            * self.quality_h
        ).clamp(-1.0, 1.0)

    def _margin(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        quality_scaler: torch.Tensor | None = None,
        target_margin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        safe = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[-1]).to(safe.dtype)
        one_hot = one_hot.unsqueeze(1)
        if quality_scaler is not None:
            scaler = quality_scaler.to(safe.dtype).unsqueeze(-1)
            angular = -self.arc_margin * scaler
            theta = safe.acos()
            theta = (theta + one_hot * angular).clamp(1e-3, math.pi - 1e-3)
            adaptive = theta.cos()
            additive = self.arc_margin + self.arc_margin * scaler
            return (adaptive - one_hot * additive) * self.arc_scale
        if target_margin is not None:
            margin = target_margin.to(safe.dtype).view(-1, 1, 1)
            cosine_margin = margin.cos()
            sine_margin = margin.sin()
            threshold = torch.cos(math.pi - margin)
            correction = torch.sin(math.pi - margin) * margin
            sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
            phi = safe * cosine_margin - sine * sine_margin
            phi = torch.where(safe > threshold, phi, safe - correction)
            return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale
        sine = torch.sqrt((1.0 - safe.square()).clamp_min(1e-7))
        phi = safe * self.cos_m - sine * self.sin_m
        phi = torch.where(
            safe > self.threshold, phi, safe - self.margin_correction
        )
        return (one_hot * phi + (1.0 - one_hot) * safe) * self.arc_scale

    def encode(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        return_local: bool = False,
        local_grid: int = 6,
        image_geometry: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del local_grid
        raw, neck = self._descriptors(
            images,
            parts,
            image_geometry=image_geometry,
            foreground_mask=foreground_mask,
        )
        local_descriptor = F.normalize(neck, dim=-1) if return_local else None
        part_embedding = (
            raw if self.part_adapters is None else self._part_neck(neck, parts)
        )
        shared_flat = neck.flatten(1)
        part_flat = part_embedding.flatten(1)
        if self.head_identity_expert:
            if self._head_identity_expert_score is None:
                raise AssertionError("Head expert score was not constructed")
            shared_flat = torch.cat(
                [shared_flat, self._head_identity_expert_score], dim=1
            )
            part_flat = torch.cat(
                [part_flat, self._head_identity_expert_score], dim=1
            )
        if self.identity_query_pooling:
            shared_query, part_query = self.identity_query_scores()
            shared_flat = torch.cat([shared_flat, shared_query], dim=-1)
            part_flat = torch.cat([part_flat, part_query], dim=-1)
        return shared_flat, part_flat, local_descriptor

    def combine_tta_embeddings(
        self,
        shared: torch.Tensor,
        part: torch.Tensor,
        flip_shared: torch.Tensor,
        flip_part: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge flip views while keeping direct class scores on their own scale."""
        if self.head_identity_expert:
            descriptor_width = self.branch_count * self.embedding_dim
            expert_width = self.head_identity_expert_output_classes
            expected_width = descriptor_width + expert_width
            if (
                shared.shape[1] != expected_width
                or part.shape[1] != expected_width
            ):
                raise AssertionError("Head-expert embedding width changed")
            shared_descriptor, shared_expert = shared.split(
                [descriptor_width, expert_width], dim=1
            )
            flip_shared_descriptor, flip_shared_expert = flip_shared.split(
                [descriptor_width, expert_width], dim=1
            )
            part_descriptor, part_expert = part.split(
                [descriptor_width, expert_width], dim=1
            )
            flip_part_descriptor, flip_part_expert = flip_part.split(
                [descriptor_width, expert_width], dim=1
            )
            return (
                torch.cat(
                    [
                        F.normalize(
                            shared_descriptor + flip_shared_descriptor, dim=-1
                        ),
                        0.5 * (shared_expert + flip_shared_expert),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        F.normalize(
                            part_descriptor + flip_part_descriptor, dim=-1
                        ),
                        0.5 * (part_expert + flip_part_expert),
                    ],
                    dim=1,
                ),
            )
        if not self.identity_query_pooling:
            return (
                F.normalize(shared + flip_shared, dim=-1),
                F.normalize(part + flip_part, dim=-1),
            )
        descriptor_width = self.branch_count * self.embedding_dim
        expected_width = descriptor_width + self.num_classes
        if shared.shape[1] != expected_width or part.shape[1] != expected_width:
            raise AssertionError("Identity-query embedding width changed")
        shared_descriptor, shared_query = shared.split(
            [descriptor_width, self.num_classes], dim=1
        )
        flip_shared_descriptor, flip_shared_query = flip_shared.split(
            [descriptor_width, self.num_classes], dim=1
        )
        part_descriptor, part_query = part.split(
            [descriptor_width, self.num_classes], dim=1
        )
        flip_part_descriptor, flip_part_query = flip_part.split(
            [descriptor_width, self.num_classes], dim=1
        )
        return (
            torch.cat(
                [
                    F.normalize(
                        shared_descriptor + flip_shared_descriptor, dim=-1
                    ),
                    0.5 * (shared_query + flip_shared_query),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    F.normalize(part_descriptor + flip_part_descriptor, dim=-1),
                    0.5 * (part_query + flip_part_query),
                ],
                dim=1,
            ),
        )

    def inference_scores(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        shared_query_score: torch.Tensor | None = None
        part_query_score: torch.Tensor | None = None
        head_expert_score: torch.Tensor | None = None
        if self.head_identity_expert:
            descriptor_width = self.branch_count * self.embedding_dim
            expert_width = self.head_identity_expert_output_classes
            expected_width = descriptor_width + expert_width
            if (
                shared_embedding.shape[1] != expected_width
                or part_embedding.shape[1] != expected_width
            ):
                raise AssertionError("Head-expert inference embedding width changed")
            shared_embedding, head_expert_score = shared_embedding.split(
                [descriptor_width, expert_width], dim=1
            )
            part_embedding, part_expert_score = part_embedding.split(
                [descriptor_width, expert_width], dim=1
            )
            if not torch.equal(head_expert_score, part_expert_score):
                raise AssertionError("Head-expert score copies diverged")
        if self.identity_query_pooling:
            descriptor_width = self.branch_count * self.embedding_dim
            expected_width = descriptor_width + self.num_classes
            if (
                shared_embedding.shape[1] != expected_width
                or part_embedding.shape[1] != expected_width
            ):
                raise AssertionError("Identity-query inference embedding width changed")
            shared_embedding, shared_query_score = shared_embedding.split(
                [descriptor_width, self.num_classes], dim=1
            )
            part_embedding, part_query_score = part_embedding.split(
                [descriptor_width, self.num_classes], dim=1
            )
        neck = shared_embedding.reshape(
            -1, self.branch_count, self.embedding_dim
        )
        part_neck = (
            part_embedding.reshape(
                -1, self.branch_count, self.embedding_dim
            )
            if self.part_adapters is not None
            else None
        )
        shared_cosine, part_cosine = self._cosines(
            neck, parts, part_neck=part_neck
        )
        shared_score, part_score = self._fuse_cosines(
            shared_cosine, part_cosine, parts
        )
        available = self.part_class_available[parts]
        if self.bidirectional:
            # Corresponding-part evidence is primary.  The shared classifier
            # is only a principled fallback for the few identities that have
            # no fold-train example of this crop type.
            parametric_score = torch.where(
                available, part_score, shared_score
            )
        else:
            fused = 0.45 * shared_score + 0.55 * part_score
            parametric_score = torch.where(available, fused, shared_score)
        if self.identity_query_pooling:
            if shared_query_score is None or part_query_score is None:
                raise AssertionError("Identity-query scores were not encoded")
            query_fused = 0.45 * shared_query_score + 0.55 * part_query_score
            query_score = torch.where(
                available, query_fused, shared_query_score
            )
            parametric_score = (
                self.branch_count * parametric_score + query_score
            ) / (self.branch_count + 1)
        if (
            head_expert_score is not None
            and self.head_identity_expert_routing_enabled
        ):
            head_mask = parts.eq(0)
            if head_mask.any():
                base_head_score = parametric_score[head_mask]
                expert_head_score = head_expert_score[head_mask]
                if self.head_tail_other_expert:
                    tail_evidence = (
                        expert_head_score[:, : self.num_classes]
                        - expert_head_score[:, self.num_classes :]
                    )
                    routed_head_score = base_head_score.clone()
                    tail_mask = self.head_tail_identity_mask
                    routed_head_score[:, tail_mask] = (
                        base_head_score[:, tail_mask]
                        + self.head_identity_expert_mix
                        * tail_evidence[:, tail_mask]
                    ).to(dtype=parametric_score.dtype)
                else:
                    expert_head_score = torch.where(
                        self.part_class_available[0][None, :],
                        expert_head_score,
                        base_head_score,
                    )
                    routed_head_score = (
                        (1.0 - self.head_identity_expert_mix) * base_head_score
                        + self.head_identity_expert_mix * expert_head_score
                    ).to(dtype=parametric_score.dtype)
                parametric_score = parametric_score.clone()
                parametric_score[head_mask] = routed_head_score
        if not self.prototype_memory:
            return parametric_score
        prototype_score = self._prototype_branch_cosines(
            neck, parts
        ).mean(dim=1)
        joint_score = (
            (1.0 - self.prototype_mix) * parametric_score
            + self.prototype_mix * prototype_score
        )
        return torch.where(
            self.shared_prototype_seen[None, :],
            joint_score,
            parametric_score,
        )

    def raw_scores_from_embeddings(
        self,
        shared_embedding: torch.Tensor,
        part_embedding: torch.Tensor,
        parts: torch.Tensor,
    ) -> torch.Tensor:
        """Return ordinary margin-free classifier scores for training KD."""
        if self.head_identity_expert or self.identity_query_pooling:
            raise RuntimeError("Source-paired KD requires the plain Q score space")
        neck = shared_embedding.reshape(
            -1, self.branch_count, self.embedding_dim
        )
        part_neck = (
            part_embedding.reshape(-1, self.branch_count, self.embedding_dim)
            if self.part_adapters is not None
            else None
        )
        shared_cosine, part_cosine = self._cosines(
            neck, parts, part_neck=part_neck
        )
        shared_score, part_score = self._fuse_cosines(
            shared_cosine, part_cosine, parts
        )
        available = self.part_class_available[parts]
        if self.bidirectional:
            return torch.where(available, part_score, shared_score)
        fused = 0.45 * shared_score + 0.55 * part_score
        return torch.where(available, fused, shared_score)

    def forward(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_geometry: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        raw, neck = self._descriptors(
            images,
            parts,
            image_geometry=image_geometry,
            foreground_mask=foreground_mask,
        )
        if self.body_to_head_distillation:
            self._body_head_alignment_loss = neck.sum() * 0.0
            if labels is not None and self.training:
                valid_head = parts.eq(0) & self.body_teacher_available[labels]
                if valid_head.any():
                    student = F.normalize(
                        neck[valid_head].float(), dim=-1
                    )
                    teacher = self.body_teacher_prototypes[
                        labels[valid_head]
                    ].float()
                    alignment = 1.0 - (student * teacher).sum(dim=-1)
                    self._body_head_alignment_loss = (
                        self.body_head_alignment_weight * alignment.mean()
                    )
        part_neck = self._part_neck(neck, parts)
        shared_cosine, part_cosine = self._cosines(
            neck, parts, part_neck=part_neck
        )
        shared_embedding = neck.flatten(1)
        part_embedding = (
            raw if self.part_adapters is None else part_neck
        ).flatten(1)
        if labels is None:
            if self.head_identity_expert:
                if self._head_identity_expert_score is None:
                    raise AssertionError("Head expert score was not constructed")
                shared_embedding = torch.cat(
                    [shared_embedding, self._head_identity_expert_score], dim=1
                )
                part_embedding = torch.cat(
                    [part_embedding, self._head_identity_expert_score], dim=1
                )
            if self.identity_query_pooling:
                shared_query, part_query = self.identity_query_scores()
                shared_embedding = torch.cat(
                    [shared_embedding, shared_query], dim=-1
                )
                part_embedding = torch.cat(
                    [part_embedding, part_query], dim=-1
                )
            return (
                self.inference_scores(shared_embedding, part_embedding, parts),
                shared_embedding,
                part_embedding,
            )
        if self.bidirectional or self.learned_branch_gates:
            shared_fused, part_fused = self._fuse_cosines(
                shared_cosine, part_cosine, parts
            )
            shared_cosine = torch.cat(
                [shared_cosine, shared_fused.unsqueeze(1)], dim=1
            )
            part_cosine = torch.cat(
                [part_cosine, part_fused.unsqueeze(1)], dim=1
            )
        if self.identity_query_pooling:
            shared_query, part_query = self.identity_query_scores()
            shared_cosine = torch.cat(
                [shared_cosine, shared_query.unsqueeze(1)], dim=1
            )
            part_cosine = torch.cat(
                [part_cosine, part_query.unsqueeze(1)], dim=1
            )
        quality_scaler = (
            self._quality_scaler(raw)
            if self.quality_adaptive_margin
            else None
        )
        shared_target_margin = (
            self.shared_class_margins[labels]
            if self.class_adaptive_margin
            else None
        )
        part_target_margin = (
            self.part_class_margins[parts, labels]
            if self.class_adaptive_margin or self.head_class_adaptive_margin
            else None
        )
        shared_logits = self._margin(
            shared_cosine,
            labels,
            quality_scaler=quality_scaler,
            target_margin=shared_target_margin,
        )
        part_logits = self._margin(
            part_cosine,
            labels,
            quality_scaler=quality_scaler,
            target_margin=part_target_margin,
        )
        if self.head_identity_expert:
            head_mask = parts.eq(0)
            branch_cosine = self._head_identity_expert_branch_cosine
            if branch_cosine is None:
                expert_logits = shared_logits.new_empty(
                    (
                        0,
                        self.head_identity_expert_branch_count,
                        self.head_identity_expert_output_classes,
                    )
                )
            else:
                expert_targets = self.head_identity_expert_targets(
                    labels[head_mask]
                )
                expert_logits = self._margin(
                    branch_cosine, expert_targets
                )
                if self.head_tail_other_expert:
                    allowed = torch.cat(
                        [
                            self.head_tail_identity_mask,
                            self.head_tail_identity_mask.new_ones(1),
                        ]
                    )
                    expert_logits = expert_logits.masked_fill(
                        ~allowed[None, None, :], -1e4
                    )
            return (
                shared_logits,
                part_logits,
                shared_embedding,
                part_embedding,
                expert_logits,
            )
        if self.prototype_memory:
            prototype_logits, prototype_valid = (
                self.prototype_logits_from_embedding(
                    shared_embedding, parts, labels
                )
            )
            return (
                shared_logits,
                part_logits,
                shared_embedding,
                part_embedding,
                prototype_logits,
                prototype_valid,
            )
        return (
            shared_logits,
            part_logits,
            shared_embedding,
            part_embedding,
        )


class PartRoutedSpatialAdapterBank(nn.Module):
    """Small spatial residuals hard-routed by the released crop part."""

    def __init__(self, channels: int, rank: int = 64) -> None:
        super().__init__()
        self.channels = int(channels)
        self.rank = int(rank)
        self.routes = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, rank, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(
                        rank,
                        rank,
                        kernel_size=3,
                        padding=1,
                        groups=rank,
                        bias=False,
                    ),
                    nn.GELU(),
                    nn.Conv2d(rank, channels, kernel_size=1, bias=False),
                )
                for _ in range(3)
            ]
        )
        for route in self.routes:
            nn.init.kaiming_uniform_(route[0].weight, a=math.sqrt(5))
            nn.init.dirac_(route[2].weight, groups=rank)
            nn.init.normal_(route[4].weight, mean=0.0, std=1e-5)
        self.last_route_counts: tuple[int, int, int] | None = None

    def forward(
        self, features: torch.Tensor, parts: torch.Tensor
    ) -> torch.Tensor:
        if parts.ndim != 1 or len(parts) != len(features):
            raise ValueError("Part-routed adapter received invalid part indices")
        if torch.any((parts < 0) | (parts >= len(self.routes))):
            raise ValueError("Part-routed adapter received an unknown part")
        residual = torch.zeros_like(features)
        counts: list[int] = []
        for part_index, route in enumerate(self.routes):
            indices = torch.where(parts.eq(part_index))[0]
            counts.append(int(len(indices)))
            if len(indices):
                routed = route(features.index_select(0, indices))
                residual = residual.index_add(
                    0, indices, routed.to(dtype=residual.dtype)
                )
        self.last_route_counts = tuple(counts)  # type: ignore[assignment]
        return features + residual


class DinoV3ConvNeXtLargeDualLevelMGNCov(DinoV3PatchMGN):
    """DINOv3 ConvNeXt-L with semantic globals and high-resolution locals."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_stages: int = 2,
        grad_checkpointing: bool = True,
        part_topology_supcon: bool = False,
        training_instance_queue: bool = False,
    ) -> None:
        if image_size != 448:
            raise ValueError("V3.1 dual-level ConvNeXt-L is locked to 448 input")
        if embedding_dim != 512:
            raise ValueError("V3.1 dual-level ConvNeXt-L requires 512-d heads")
        if freeze_stages != 2:
            raise ValueError("V3.1 freezes exactly the stem and stages 0--1")
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_stages,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            covariance_dim=64,
            part_topology_supcon=part_topology_supcon,
            training_instance_queue=training_instance_queue,
            instance_queue_capacity=2048,
            backbone_model_name=CONVNEXT_LARGE_DINOV3_MODEL_NAME,
            pretraining_source=CONVNEXT_LARGE_DINOV3_PRETRAINING_SOURCE,
            spatial_backbone=True,
            spatial_layout="nchw",
            backbone_output_stride=32,
        )
        self.local_feature_dim = 768
        self.global_feature_dim = 1536
        self.dual_level_convnext = True
        self._dual_level_last_shapes: tuple[
            tuple[int, ...], tuple[int, ...]
        ] | None = None
        # The parent creates final-map MGN heads. V3.1 deliberately sources
        # five stripe descriptors and covariance from the 28x28 stage-2 map.
        for branch_index in range(1, 6):
            self.branch_projections[branch_index] = nn.Sequential(
                nn.Linear(self.local_feature_dim, embedding_dim),
                nn.GELU(),
            )
        self.covariance_reduction = nn.Sequential(
            nn.Linear(self.local_feature_dim, self.covariance_dim, bias=False),
            nn.LayerNorm(self.covariance_dim),
        )
        self.model_name = (
            f"{CONVNEXT_LARGE_DINOV3_MODEL_NAME}_dual_level_mgn_cov64"
        )
        self.pretraining_source = CONVNEXT_LARGE_DINOV3_PRETRAINING_SOURCE
        self.pretraining_sha256 = CONVNEXT_LARGE_DINOV3_PRETRAINING_SHA256

    def _adapt_feature_maps(
        self,
        local_map: torch.Tensor,
        final_map: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del parts
        return local_map, final_map

    def _descriptors(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        image_geometry: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_geometry is not None:
            raise ValueError("V3 ConvNeXt does not consume crop-geometry side input")
        if foreground_mask is not None:
            raise ValueError("V3 ConvNeXt does not consume foreground-mask side input")
        final_map, intermediates = self.backbone.forward_intermediates(
            images,
            indices=(2,),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=False,
        )
        if len(intermediates) != 1:
            raise AssertionError("V3.1 did not capture exactly one local stage")
        local_map = intermediates[0]
        final_norm = getattr(getattr(self.backbone, "head", None), "norm", None)
        if not isinstance(final_norm, nn.Module):
            raise AssertionError("ConvNeXt-L published final feature norm is missing")
        final_map = final_norm(final_map)
        local_map, final_map = self._adapt_feature_maps(
            local_map, final_map, parts
        )
        expected_local = (self.local_feature_dim, 28, 28)
        expected_final = (self.global_feature_dim, 14, 14)
        if tuple(local_map.shape[1:]) != expected_local:
            raise AssertionError(
                f"V3.1 local map {tuple(local_map.shape)} != [B,{expected_local}]"
            )
        if tuple(final_map.shape[1:]) != expected_final:
            raise AssertionError(
                f"V3.1 final map {tuple(final_map.shape)} != [B,{expected_final}]"
            )
        self._dual_level_last_shapes = (
            tuple(local_map.shape),
            tuple(final_map.shape),
        )

        global_descriptor = torch.cat(
            [final_map.amax(dim=(2, 3)), final_map.mean(dim=(2, 3))], dim=-1
        )
        half = F.adaptive_avg_pool2d(local_map, (2, 1)).squeeze(-1)
        thirds = F.adaptive_avg_pool2d(local_map, (3, 1)).squeeze(-1)
        inputs: list[torch.Tensor] = [
            global_descriptor,
            half[:, :, 0],
            half[:, :, 1],
            thirds[:, :, 0],
            thirds[:, :, 1],
            thirds[:, :, 2],
        ]
        patches = local_map.flatten(2).transpose(1, 2)
        reduced = self.covariance_reduction(patches).float()
        centered = reduced - reduced.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bni,bnj->bij", centered, centered) / max(
            1, centered.shape[1] - 1
        )
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1e-6
        )
        covariance = F.normalize(covariance.flatten(1), dim=-1)
        inputs.append(covariance.to(dtype=local_map.dtype))
        if len(inputs) != self.branch_count or self.branch_count != 7:
            raise AssertionError("V3.1 descriptor inventory changed")
        raw = torch.stack(
            [
                projection(value)
                for projection, value in zip(
                    self.branch_projections, inputs, strict=True
                )
            ],
            dim=1,
        )
        neck = torch.stack(
            [
                branch_neck(raw[:, branch_index])
                for branch_index, branch_neck in enumerate(self.branch_necks)
            ],
            dim=1,
        )
        return raw, neck


class Sam2HieraSmallFpnMGNCov(DinoV3PatchMGN):
    """SAM2.1 Hiera-Small with its published FPN and mature seven heads."""

    trunk_tensor_count = 202
    trunk_value_count = 33_947_328
    fpn_tensor_count = 4
    fpn_value_count = 295_424

    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_blocks: int = 3,
        grad_checkpointing: bool = True,
        external_pretrained_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if image_size != 448:
            raise ValueError("V12.14 SAM2 Hiera-Small is locked to 448 input")
        if embedding_dim != 512:
            raise ValueError("V12.14 SAM2 Hiera-Small requires 512-d heads")
        if freeze_blocks != 3:
            raise ValueError("V12.14 freezes exactly Hiera blocks 0--2")
        if not pretrained or external_pretrained_path is None:
            raise ValueError("V12.14 requires the exact public SAM2.1 checkpoint")

        # timm supplies the exact Hiera graph but its hub entry is deliberately
        # not used.  The official Meta checkpoint is loaded below through a
        # strict, auditable allowlist; every target classifier remains fresh.
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=False,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            covariance_dim=64,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=SAM2_HIERA_SMALL_MODEL_NAME,
            pretraining_source=SAM2_HIERA_SMALL_PRETRAINING_SOURCE,
            spatial_backbone=True,
            spatial_layout="nchw",
            backbone_output_stride=None,
        )

        payload = torch.load(
            external_pretrained_path, map_location="cpu", weights_only=False
        )
        source_state = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(source_state, dict):
            raise AssertionError("SAM2.1 checkpoint has no model state")
        trunk_state: dict[str, torch.Tensor] = {}
        for source_name, value in source_state.items():
            if not source_name.startswith("image_encoder.trunk."):
                continue
            target_name = source_name[len("image_encoder.trunk.") :]
            target_name = target_name.replace(".mlp.layers.0.", ".mlp.fc1.")
            target_name = target_name.replace(".mlp.layers.1.", ".mlp.fc2.")
            trunk_state[target_name] = value
        if len(trunk_state) != self.trunk_tensor_count or sum(
            value.numel() for value in trunk_state.values()
        ) != self.trunk_value_count:
            raise AssertionError("SAM2.1 Hiera trunk allowlist changed")
        target_state = self.backbone.state_dict()
        unexpected_targets = set(trunk_state) - set(target_state)
        shape_mismatches = {
            name
            for name, value in trunk_state.items()
            if name in target_state and value.shape != target_state[name].shape
        }
        if unexpected_targets or shape_mismatches:
            raise AssertionError(
                "SAM2.1 Hiera/timm graph mismatch: "
                f"unexpected={sorted(unexpected_targets)}, "
                f"shape={sorted(shape_mismatches)}"
            )
        load_result = self.backbone.load_state_dict(trunk_state, strict=False)
        if set(load_result.missing_keys) != {
            "head.norm.weight",
            "head.norm.bias",
        } or load_result.unexpected_keys:
            raise AssertionError(
                "SAM2.1 Hiera allowlist load changed: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        # The random timm classification norm is not part of SAM2 and must not
        # survive as trainable target state.
        self.backbone.head = nn.Identity()

        self.backbone.sam2_fpn_laterals = nn.ModuleList(
            [
                nn.Conv2d(768, 256, kernel_size=1),
                nn.Conv2d(384, 256, kernel_size=1),
            ]
        )
        fpn_source_names = (
            "image_encoder.neck.convs.0.conv",
            "image_encoder.neck.convs.1.conv",
        )
        fpn_values = 0
        with torch.no_grad():
            for lateral, source_prefix in zip(
                self.backbone.sam2_fpn_laterals,
                fpn_source_names,
                strict=True,
            ):
                weight = source_state.get(f"{source_prefix}.weight")
                bias = source_state.get(f"{source_prefix}.bias")
                if not isinstance(weight, torch.Tensor) or not isinstance(
                    bias, torch.Tensor
                ):
                    raise AssertionError(f"Missing official SAM2 FPN {source_prefix}")
                if weight.shape != lateral.weight.shape or bias.shape != lateral.bias.shape:
                    raise AssertionError(f"SAM2 FPN shape changed for {source_prefix}")
                lateral.weight.copy_(weight)
                lateral.bias.copy_(bias)
                fpn_values += weight.numel() + bias.numel()
        if fpn_values != self.fpn_value_count:
            raise AssertionError("SAM2.1 FPN allowlist value count changed")

        # The retained FPN reduces both feature levels to 256 channels.  Replace
        # only the fresh projection geometry created by the generic parent;
        # all seven target classifiers and BN necks remain independently drawn.
        self.branch_projections[0] = nn.Sequential(
            nn.Linear(512, embedding_dim), nn.GELU()
        )
        for branch_index in range(1, 6):
            self.branch_projections[branch_index] = nn.Sequential(
                nn.Linear(256, embedding_dim), nn.GELU()
            )
        self.covariance_reduction = nn.Sequential(
            nn.Linear(256, self.covariance_dim, bias=False),
            nn.LayerNorm(self.covariance_dim),
        )
        self.model_name = "sam2.1_hiera_small_fpn_mgn_cov64_ptoposupcon_queue2048"
        self.pretraining_source = SAM2_HIERA_SMALL_PRETRAINING_SOURCE
        self.pretraining_sha256 = SAM2_HIERA_SMALL_PRETRAINING_SHA256
        self.sam2_hiera_backbone = True
        self.external_pretraining_metadata = {
            "kind": "public_generic_segmentation_tracking",
            "license": "Apache-2.0",
            "checkpoint_sha256": SAM2_HIERA_SMALL_PRETRAINING_SHA256,
            "trunk_tensors": self.trunk_tensor_count,
            "trunk_values": self.trunk_value_count,
            "fpn_tensors": self.fpn_tensor_count,
            "fpn_values": self.fpn_value_count,
            "loaded_tensors": self.trunk_tensor_count + self.fpn_tensor_count,
            "loaded_values": self.trunk_value_count + self.fpn_value_count,
            "external_class_state_loaded": False,
            "prompt_mask_memory_state_loaded": False,
            "competition_checkpoint_loaded": False,
        }
        self._sam2_last_shapes: tuple[
            tuple[int, ...], tuple[int, ...]
        ] | None = None
        del payload, source_state, trunk_state, target_state

    def optimizer_layer_count(self) -> int:
        return len(self.backbone.blocks) + 1

    def optimizer_layer_id(self, inner_name: str) -> int:
        if inner_name.startswith(("patch_embed.", "pos_embed", "pos_embed_window")):
            return 0
        if inner_name.startswith("blocks."):
            return int(inner_name.split(".")[1]) + 1
        # The two official FPN laterals sit after the final Hiera block and use
        # the full backbone LR, never the fresh-head LR.
        return self.optimizer_layer_count()

    def freeze_backbone_prefix(self, block_count: int) -> None:
        block_count = min(max(int(block_count), 0), len(self.backbone.blocks))
        for name, parameter in self.backbone.named_parameters():
            frozen = name.startswith(
                ("patch_embed.", "pos_embed", "pos_embed_window")
            ) and block_count > 0
            if name.startswith("blocks."):
                frozen = int(name.split(".")[1]) < block_count
            parameter.requires_grad_(not frozen)

    def _descriptors(
        self,
        images: torch.Tensor,
        parts: torch.Tensor,
        image_geometry: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(parts) != len(images):
            raise ValueError("SAM2 Hiera parts batch does not match images")
        if image_geometry is not None or foreground_mask is not None:
            raise ValueError("V12.14 consumes only the released RGB crop")
        features = self.backbone.forward_intermediates(
            images,
            indices=(2, 3),
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
            coarse=True,
        )
        if len(features) != 2:
            raise AssertionError("V12.14 did not capture two Hiera stages")
        stage2, stage3 = features
        if tuple(stage2.shape[1:]) != (384, 28, 28):
            raise AssertionError(f"Unexpected SAM2 stage2 shape {tuple(stage2.shape)}")
        if tuple(stage3.shape[1:]) != (768, 14, 14):
            raise AssertionError(f"Unexpected SAM2 stage3 shape {tuple(stage3.shape)}")
        final_map = self.backbone.sam2_fpn_laterals[0](stage3)
        local_lateral = self.backbone.sam2_fpn_laterals[1](stage2)
        local_map = local_lateral + F.interpolate(
            final_map.float(), scale_factor=2.0, mode="nearest"
        )
        if tuple(final_map.shape[1:]) != (256, 14, 14):
            raise AssertionError(f"Unexpected SAM2 final FPN shape {tuple(final_map.shape)}")
        if tuple(local_map.shape[1:]) != (256, 28, 28):
            raise AssertionError(f"Unexpected SAM2 local FPN shape {tuple(local_map.shape)}")
        self._sam2_last_shapes = (tuple(local_map.shape), tuple(final_map.shape))

        global_descriptor = torch.cat(
            [final_map.amax(dim=(2, 3)), final_map.mean(dim=(2, 3))], dim=-1
        )
        half = F.adaptive_avg_pool2d(local_map, (2, 1)).squeeze(-1)
        thirds = F.adaptive_avg_pool2d(local_map, (3, 1)).squeeze(-1)
        inputs: list[torch.Tensor] = [
            global_descriptor,
            half[:, :, 0],
            half[:, :, 1],
            thirds[:, :, 0],
            thirds[:, :, 1],
            thirds[:, :, 2],
        ]
        patches = local_map.flatten(2).transpose(1, 2)
        reduced = self.covariance_reduction(patches).float()
        centered = reduced - reduced.mean(dim=1, keepdim=True)
        covariance = torch.einsum("bni,bnj->bij", centered, centered) / max(
            1, centered.shape[1] - 1
        )
        covariance = torch.sign(covariance) * torch.sqrt(
            covariance.abs() + 1e-6
        )
        covariance = F.normalize(covariance.flatten(1), dim=-1)
        inputs.append(covariance.to(dtype=local_map.dtype))
        if len(inputs) != self.branch_count or self.branch_count != 7:
            raise AssertionError("V12.14 descriptor inventory changed")
        raw = torch.stack(
            [
                projection(value)
                for projection, value in zip(
                    self.branch_projections, inputs, strict=True
                )
            ],
            dim=1,
        )
        neck = torch.stack(
            [
                branch_neck(raw[:, branch_index])
                for branch_index, branch_neck in enumerate(self.branch_necks)
            ],
            dim=1,
        )
        return raw, neck


class DinoV3ConvNeXtLargeDualLevelPartAdapter(
    DinoV3ConvNeXtLargeDualLevelMGNCov
):
    """V3.2 dual-level ConvNeXt with part-routed spatial LoRA paths."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 448,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arc_scale: float = 30.0,
        arc_margin: float = 0.20,
        part_delta_scale: float = 0.20,
        freeze_stages: int = 2,
        grad_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
            grad_checkpointing=grad_checkpointing,
        )
        self.local_part_adapters = PartRoutedSpatialAdapterBank(
            self.local_feature_dim, rank=64
        )
        self.final_part_adapters = PartRoutedSpatialAdapterBank(
            self.global_feature_dim, rank=64
        )
        self.part_routed_spatial_adapters = True
        self.model_name = (
            f"{CONVNEXT_LARGE_DINOV3_MODEL_NAME}_dual_level_mgn_cov64_part_adapter64"
        )

    def _adapt_feature_maps(
        self,
        local_map: torch.Tensor,
        final_map: torch.Tensor,
        parts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.local_part_adapters(local_map, parts),
            self.final_part_adapters(final_map, parts),
        )


def build_model(
    variant: str,
    *,
    num_classes: int,
    image_size: int,
    embedding_dim: int,
    local_queries: int,
    pretrained: bool,
    arc_scale: float,
    arc_margin: float,
    part_delta_scale: float,
    freeze_blocks: int,
    freeze_stages: int,
    grad_checkpointing: bool,
    external_pretrained_path: str | os.PathLike[str] | None = None,
) -> nn.Module:
    if variant == "dinov3_vit":
        return DualEmbeddingDinoV3(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            local_queries=local_queries,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
        )
    if variant == "dinov3_convnext_dolg":
        return DinoV3ConvNeXtDOLG(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
            grad_checkpointing=grad_checkpointing,
        )
    if variant == "dinov3_convnext_large_dual_mgn_cov":
        return DinoV3ConvNeXtLargeDualLevelMGNCov(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
            grad_checkpointing=grad_checkpointing,
        )
    if (
        variant
        == "dinov3_convnext_large_dual_mgn_cov_ptoposupcon_queue2048"
    ):
        return DinoV3ConvNeXtLargeDualLevelMGNCov(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
            grad_checkpointing=grad_checkpointing,
            part_topology_supcon=True,
            training_instance_queue=True,
        )
    if variant == "dinov3_convnext_large_dual_mgn_cov_part_adapter":
        return DinoV3ConvNeXtLargeDualLevelPartAdapter(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
            grad_checkpointing=grad_checkpointing,
        )
    if variant == "sam2_hiera_small_fpn_mgn_cov_ptoposupcon_queue2048":
        return Sam2HieraSmallFpnMGNCov(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "efficientnetv2_m_subcenter":
        return EfficientNetV2MSubCenter(
            num_classes=num_classes,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
        )
    if variant == "petface_r50_head_specialist":
        return PetFaceR50HeadSpecialist(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            pretrained_path=external_pretrained_path,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_stages=freeze_stages,
        )
    if variant == "arbase_mgn":
        return ARBaseMGN(
            num_classes=num_classes,
            pretrained=pretrained,
            part_delta_scale=part_delta_scale,
        )
    if variant == "dinov3_patch_mgn":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
        )
    if variant == "dinov3_patch_bim":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            bidirectional=True,
        )
    if variant == "dinov3_patch_mgn_sc":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            subcenters=3,
        )
    if variant == "dinov3_large_patch_mgn":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_sc":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            subcenters=3,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_pa":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            part_adapter_bottleneck=128,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_ada":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            quality_adaptive_margin=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_gate":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            learned_branch_gates=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_sie":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            part_side_embedding=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "lingbot_large_patch_mgn":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            backbone_model_name=LINGBOT_LARGE_MODEL_NAME,
            pretraining_source=LINGBOT_LARGE_PRETRAINING_SOURCE,
            lingbot_backbone=True,
        )
    if variant == "dinov3_large_patch_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048"
    ):
        # V9.1 restores the exact V4.3 inference network. Its sole change is a
        # detached, train-only 2048-instance key queue for topology SupCon.
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_decisionalign"
    ):
        # V14.2 keeps Q's complete train/inference graph.  Its only switch is
        # consumed by the training objective; deployed scoring is unchanged.
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            decision_aligned_classification=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcltp"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            head_routed_cross_level_texture=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headcanon"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            foreground_scale_normalization=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headgrid10"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            head_identity_expert_detach=False,
            head_identity_expert_inference=True,
            head_identity_expert_loss_weight=0.50,
            ordered_head_grid_expert=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_headqvlora8"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            suffix_head_qv_lora=True,
            qv_lora_rank=8,
            qv_lora_alpha=8.0,
            qv_lora_dropout=0.0,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_queue2048_sourcepairlogitkd"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            source_paired_logit_distillation=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fulladapt32"
    ):
        # V8.2 uses the exact V4.3 decision network.  The launcher fixes
        # freeze_blocks=32, so every public Transformer block is frozen while
        # the existing AdaptFormer wrapper is installed in all 32 blocks and
        # its adapter parameters are re-enabled by DinoV3PatchMGN.
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headexpert"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            head_identity_expert=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxbase"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            head_identity_expert=True,
            head_identity_expert_detach=True,
            head_identity_expert_inference=False,
            head_identity_expert_loss_weight=0.50,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headauxext"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            head_identity_expert=True,
            head_identity_expert_detach=False,
            head_identity_expert_inference=False,
            head_identity_expert_loss_weight=0.25,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_extheadrep24tail2"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            head_identity_expert=True,
            head_identity_expert_detach=False,
            head_identity_expert_inference=True,
            head_identity_expert_loss_weight=0.50,
            external_head_representation=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_headtailother"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            head_tail_other_expert=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_ptransport32"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            prototype_transport_rank=32,
            prototype_transport_scale=0.20,
            part_topology_supcon=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if (
        variant
        == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_ptoposupcon_fgtoken"
    ):
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            foreground_token_conditioning=True,
            part_topology_supcon=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_head2vpbsupcon":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            head_two_view_part_balanced_supcon=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_geosie2_fgaux":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            continuous_geometry_conditioning=True,
            foreground_auxiliary=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_qvlora8":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            frozen_prefix_qv_lora=True,
            qv_lora_rank=8,
            qv_lora_alpha=8.0,
            qv_lora_dropout=0.05,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_partqvlora4x4":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            frozen_prefix_qv_lora=True,
            qv_lora_rank=4,
            qv_lora_part_rank=4,
            qv_lora_alpha=4.0,
            qv_lora_dropout=0.05,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_convpass":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_convpass=True,
            convpass_dim=64,
            convpass_scale=0.1,
            convpass_dropout=0.1,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_headconv":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            convpass_dim=64,
            convpass_scale=0.1,
            convpass_dropout=0.1,
            head_convpass_attention=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_partmoe":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            part_mlp_expert_blocks=4,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_a2gc":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            pattern_a2gc_branch=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_slots_cov_adapt":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            hierarchical_slot_architecture=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_dmatch":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            dense_correspondence_training=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_pqmil":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            identity_query_pooling=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_cltp":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            cross_level_texture_pyramid=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_cltp_jpm4":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            cross_level_texture_pyramid=True,
            jpm_local_branches=4,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            part_routed_cross_level_texture=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_headtail2":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            part_routed_cross_level_texture=True,
            head_tail_expert_blocks=2,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_b2hproto":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            part_routed_cross_level_texture=True,
            body_to_head_distillation=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_huge_plus_patch_mgn_cov_adapt_prcltp_imgfreq":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            frozen_prefix_adaptformer=True,
            adaptformer_dim=64,
            adaptformer_scale=0.1,
            part_routed_cross_level_texture=True,
            image_frequency_texture_side=True,
            backbone_model_name=HUGE_PLUS_MODEL_NAME,
            pretraining_source=HUGE_PLUS_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_proto":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            prototype_memory=True,
            prototype_momentum=0.9,
            prototype_mix=0.5,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant in {
        "eva02_large_patch_mgn_cov",
        "eva02_large_patch_mgn_cov_proto",
    }:
        use_prototype_memory = variant.endswith("_proto")
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            prototype_memory=use_prototype_memory,
            prototype_momentum=0.9,
            prototype_mix=0.5,
            backbone_model_name=EVA02_LARGE_MODEL_NAME,
            pretraining_source=EVA02_LARGE_PRETRAINING_SOURCE,
        )
    if variant == "convnextv2_large_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=CONVNEXTV2_LARGE_MODEL_NAME,
            pretraining_source=CONVNEXTV2_LARGE_PRETRAINING_SOURCE,
            spatial_backbone=True,
            backbone_output_stride=16,
        )
    if variant == "swinv2_large_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=SWINV2_LARGE_MODEL_NAME,
            pretraining_source=SWINV2_LARGE_PRETRAINING_SOURCE,
            spatial_backbone=True,
            spatial_layout="nhwc",
            spatial_image_size=True,
            spatial_strict_image_size=False,
            backbone_output_stride=None,
        )
    if variant == "swinv2_base_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=SWINV2_BASE_MODEL_NAME,
            pretraining_source=SWINV2_BASE_PRETRAINING_SOURCE,
            spatial_backbone=True,
            spatial_layout="nhwc",
            spatial_image_size=True,
            spatial_strict_image_size=False,
            backbone_output_stride=None,
        )
    if variant == "siglip2_large_patch_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=SIGLIP2_LARGE_MODEL_NAME,
            pretraining_source=SIGLIP2_LARGE_PRETRAINING_SOURCE,
        )
    if variant == "bioclip_vitb_patch_mgn_cov_ptoposupcon_queue2048":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=BIOCLIP_MODEL_NAME,
            pretraining_source=BIOCLIP_PRETRAINING_SOURCE,
            bioclip_backbone=True,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "radio_v25_b_patch_mgn_cov_ptoposupcon_queue2048":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=RADIO_V25_B_MODEL_NAME,
            pretraining_source=RADIO_V25_B_PRETRAINING_SOURCE,
            radio_backbone=True,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "tips_l14_hr_patch_mgn_cov_ptoposupcon_queue2048":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=TIPS_L14_HR_MODEL_NAME,
            pretraining_source=TIPS_L14_HR_PRETRAINING_SOURCE,
            tips_backbone=True,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "bioclip2_vitl14_projected_patch_mgn_cov_ptoposupcon_queue2048":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=BIOCLIP2_MODEL_NAME,
            pretraining_source=BIOCLIP2_PRETRAINING_SOURCE,
            bioclip2_backbone=True,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "dinov2_large_reg_patch_mgn_cov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            backbone_model_name=DINOV2_LARGE_REGISTER_MODEL_NAME,
            pretraining_source=DINOV2_LARGE_REGISTER_PRETRAINING_SOURCE,
        )
    if variant == "dinov2_giant_reg_patch_mgn_cov_geosie2_ptoposupcon_queue2048":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            continuous_geometry_conditioning=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=DINOV2_GIANT_REGISTER_MODEL_NAME,
            pretraining_source=DINOV2_GIANT_REGISTER_PRETRAINING_SOURCE,
            external_pretrained_path=external_pretrained_path,
        )
    if variant in {
        "dinov2_giant_reg_patch_mgn_cov_ptoposupcon_queue2048",
        "dinov2_giant_reg_patch_mgn_cov_suffix8_ptoposupcon_queue2048",
    }:
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            part_topology_supcon=True,
            training_instance_queue=True,
            instance_queue_capacity=2048,
            backbone_model_name=DINOV2_GIANT_REGISTER_MODEL_NAME,
            pretraining_source=DINOV2_GIANT_REGISTER_PRETRAINING_SOURCE,
            external_pretrained_path=external_pretrained_path,
        )
    if variant == "dinov3_large_patch_mgn_cov_ldam":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            class_adaptive_margin=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_headldam":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            head_class_adaptive_margin=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_mpncov":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            covariance_matrix_sqrt=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_sc":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            subcenters=3,
            covariance_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_corr":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            covariance_standardize=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_grad":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            gradient_covariance_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_xlayer":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            intermediate_covariance_branch=True,
            intermediate_covariance_block=17,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_orthogonal":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            vertical_branches=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_dsbn":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            domain_specific_bn=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_mbn":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            modality_specific_bn=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_mixstyle":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            token_mixstyle=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_axis":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            part_aligned_axis=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_grad_axis":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            gradient_covariance_branch=True,
            part_aligned_axis=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_grad_topk":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            gradient_covariance_branch=True,
            semantic_topk_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_grad_simpool":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            gradient_covariance_branch=True,
            simpool_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_mgn_cov_simpool":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            covariance_branch=True,
            simpool_branch=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_large_patch_bim":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            bidirectional=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    if variant == "dinov3_patch_hier":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            bidirectional=True,
            hierarchical_part_heads=True,
        )
    if variant == "dinov3_large_patch_hier":
        return DinoV3PatchMGN(
            num_classes=num_classes,
            image_size=image_size,
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            arc_scale=arc_scale,
            arc_margin=arc_margin,
            part_delta_scale=part_delta_scale,
            freeze_blocks=freeze_blocks,
            grad_checkpointing=grad_checkpointing,
            bidirectional=True,
            hierarchical_part_heads=True,
            backbone_model_name=LARGE_MODEL_NAME,
            pretraining_source=LARGE_PRETRAINING_SOURCE,
        )
    raise ValueError(f"Unknown model variant: {variant}")
