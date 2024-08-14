#taken and modified from https://github.com/eloialonso/iris/blob/main/src/models/transformer.py
"""
Credits to https://github.com/karpathy/minGPT
"""


from dataclasses import dataclass
import math
from typing import Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class Cache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(n, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)


class KVCache:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None

@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class JuanTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: torch.Tensor = None, batched=True) -> torch.Tensor:
        if past_keys_values is not None:
            past_keys, past_values = past_keys_values
            assert past_keys_values is None or (len(past_keys) == len(self.blocks) and len(past_values) == len(self.blocks))
            if not batched:
                past_keys = past_keys.unsqueeze(1)
                past_values = past_values.unsqueeze(1)
        x = self.drop(sequences)
        src = x
        this_kvs = []
        for i, block in enumerate(self.blocks):
            x, this_kv = block(x, None if past_keys_values is None else (past_keys[i], past_values[i]))
            x = x + src  # skip connection, as for Juan et al.
            if not batched:
                this_k, this_v = this_kv['k'], this_kv['v']
                this_k = this_k.squeeze(0)
                this_v = this_v.squeeze(0)
                this_kv = {'k': this_k, 'v': this_v}

            this_kvs.append(this_kv)

        x = self.ln_f(x)
        this_kvs = {'k': torch.stack([kv['k'] for kv in this_kvs]), 'v': torch.stack([kv['v'] for kv in this_kvs])}
        return x, this_kvs


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        x_attn, this_kv = self.attn(self.ln1(x), past_keys_values)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x, this_kv


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        B, T, C = x.size()
        if past_kv is not None:
            past_k, past_v = past_kv
            b, nh, L, c = past_k.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        this_k, this_v = k, v
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2) # todo(milad): check if this is correct
            v = torch.cat((past_v, v), dim=2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

        y = self.resid_drop(self.proj(y))

        return y, {'k': this_k, 'v': this_v}


from typing import Tuple

import numpy as np
import torch


