from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import AttentionBlock


class _FullAttentionFallback(nn.Module):
    """Drop-in replacement for xformers.components.attention.NystromAttention.

    Modern xformers removed the `components` module (the old experimental
    attention zoo, including NystromAttention). Nystrom attention is a
    linear-time approximation of full softmax attention, so plain
    scaled_dot_product_attention is a correct (if not asymptotically as
    cheap) substitute for inference on modest sequence lengths.
    """

    def __init__(self, num_landmarks: int, num_heads: int, dropout: float = 0.0):
        del num_landmarks, num_heads
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v, key_padding_mask=None):
        if key_padding_mask is not None:
            raise NotImplementedError(
                "key_padding_mask is not supported by the full-attention fallback"
            )
        # q, k, v: (b, n, h, d) -> (b, h, n, d)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return out.transpose(1, 2)


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        try:
            from xformers.components.attention import NystromAttention

            self.attention_fn = NystromAttention(
                num_landmarks=128, num_heads=num_heads, dropout=dropout
            )
        except ImportError:
            self.attention_fn = _FullAttentionFallback(
                num_landmarks=128, num_heads=num_heads, dropout=dropout
            )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context), "b n (kv h d) -> b n h d kv", h=self.num_heads, kv=2
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x
