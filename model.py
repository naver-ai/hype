import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Callable
from collections import OrderedDict
from torchvision.transforms import Normalize, Compose, InterpolationMode, ToTensor, Resize, CenterCrop
from huggingface_hub import PyTorchModelHubMixin


def image_transform(image_size: int):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    normalize = Normalize(mean=mean, std=std)
    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend(
        [
            lambda x: x.convert("RGB"),
            ToTensor(),
            normalize,
        ]
    )
    return Compose(transforms)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        eps = torch.finfo(orig_type).eps
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps)
        return x.to(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(x=self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.scale = scale
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)

        self.ln_post = norm_layer(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        pooled = x[:, 0]
        pooled = self.ln_post(pooled)
        pooled = pooled @ self.proj
        return pooled


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width, layers=layers, heads=heads, act_layer=act_layer, norm_layer=norm_layer
        )
        self.ln_final = norm_layer(width)
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

    def build_attention_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)

        x = self.ln_final(x)
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return pooled


class CLIP(nn.Module):
    def __init__(self, embed_dim: int, vision_cfg: dict, text_cfg: dict):
        super().__init__()
        act_layer = nn.GELU
        norm_layer = LayerNorm

        self.visual = VisionTransformer(
            image_size=vision_cfg["image_size"],
            patch_size=vision_cfg["patch_size"],
            width=vision_cfg["width"],
            layers=vision_cfg["layers"],
            heads=vision_cfg["width"] // 64,
            mlp_ratio=4.0,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        text = TextTransformer(
            context_length=text_cfg["context_length"],
            vocab_size=text_cfg["vocab_size"],
            width=text_cfg["width"],
            heads=text_cfg["heads"],
            layers=text_cfg["layers"],
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.curvature = nn.Parameter(torch.ones([]) * np.log(1.0))
        self.alpha_img = nn.Parameter(torch.ones([]) * np.log(1 / np.sqrt(embed_dim)))
        self.alpha_txt = nn.Parameter(torch.ones([]) * np.log(1 / np.sqrt(embed_dim)))

    def encode_image(self, image):
        features = self.visual(image)
        return self.alpha_img.exp() * features

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return self.alpha_txt.exp() * x

    def forward(self, image: Optional[torch.Tensor] = None, text: Optional[torch.Tensor] = None):
        image_features = self.encode_image(image) if image is not None else None
        text_features = self.encode_text(text) if text is not None else None
        return (image_features, text_features)


def model_init(pretrained: str):
    cfg = {
        "embed_dim": 768,
        "vision_cfg": {"image_size": 224, "layers": 24, "width": 1024, "patch_size": 14},
        "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12},
    }
    model = CLIP(**cfg)

    state_dict = torch.load(pretrained)
    model.load_state_dict(state_dict, strict=False)

    model.visual.image_mean = (0.48145466, 0.4578275, 0.40821073)
    model.visual.image_std = (0.26862954, 0.26130258, 0.27577711)

    preprocess = image_transform(model.visual.image_size)

    return model, preprocess
