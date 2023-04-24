import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import einops
from typing import Tuple


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class LinearDecoder(nn.Module):
    def __init__(self, n_cls, patch_size, hidden_d):
        super(LinearDecoder, self).__init__()

        self.hidden_d = hidden_d
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(hidden_d, n_cls)

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x


class MaskTransformer(nn.Module):
    def __init__(self, n_cls, patch_size, n_layers, n_heads, d_model):
        super(MaskTransformer, self).__init__()
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model  # == d_encoder
        self.scale = d_model ** -0.5

        self.blocks = nn.ModuleList([ViTBlock(d_model, n_heads) for _ in range(n_layers)])

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_model, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

    def forward(self, x: torch.Tensor, im_size: Tuple[int, int]):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = einops.rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")

        return masks


# https://github.com/isaaccorley/segmenter-pytorch/blob/main/segmenter/segmenter.py
class MaskTransformerV2(nn.Module):

    def __init__(self, n_cls: int, patch_size: int, n_layers: int, n_heads: int, d_model: int):
        super(MaskTransformerV2, self).__init__()
        self.n_cls = n_cls
        self.patch_size = patch_size
        self.cls_tokens = nn.Parameter(torch.randn(1, n_cls, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                                           activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.upsample = nn.Upsample(scale_factor=patch_size, mode="bilinear")
        self.scale = d_model ** -0.5

    def forward(self, x: torch.Tensor, im_size: Tuple[int, int]) -> torch.Tensor:
        H, W = im_size
        b = x.shape[0]
        cls_tokens = self.cls_tokens.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        c = x[:, :self.n_cls]
        z = x[:, self.n_cls:]
        masks = z @ c.transpose(1, 2)
        masks = torch.softmax(masks / self.scale, dim=-1)

        masks = einops.rearrange(masks, "b (p1 p2) c -> b c p1 p2", p1=H // self.patch_size, p2=W // self.patch_size)

        masks = self.upsample(masks)

        return masks


class Segmenter(nn.Module):
    def __init__(self, chw, n_patches=7, n_layers=2, d_model=8, n_heads=2, n_cls=2, decoder_type='linear'):
        # Super constructor
        super(Segmenter, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model

        # Input and patches sizes
        assert(chw[1] == chw[2])
        assert(chw[1] % n_patches == 0)
        self.patch_size = chw[1] // n_patches

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size**2)
        self.linear_mapper = nn.Linear(self.input_d, d_model)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, d_model))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, d_model),
                             persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(d_model, n_heads) for _ in range(n_layers)])

        # 5) Decoding
        if decoder_type == 'linear':
            self.decoder = LinearDecoder(n_cls, self.patch_size, d_model)
        elif decoder_type == 'mask_transformer':
            self.decoder = MaskTransformer(n_cls, self.patch_size, n_layers, n_heads, d_model)
        elif decoder_type == 'mask_transformer_v2':
            self.decoder = MaskTransformerV2(n_cls, self.patch_size, n_layers, n_heads, d_model)
        else:
            raise Exception(f'Unkown decoder type ({decoder_type})')

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Removing the classification token
        out = out[:, :-1]

        out = self.decoder(out, (h, w))

        return out
