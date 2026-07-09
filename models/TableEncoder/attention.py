import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def repeat_kv(hidden_states, repeats: int):
    if repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats, dim=1)


class FlashAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        num_key_value_heads=None,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        if num_heads % self.num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.num_key_value_heads == self.num_heads:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._attn = None
        self._attn_gradients = None
        self.attn_drop_prob = attn_drop

    def forward(self, x, mask=None, causal: bool = False):
        """
        :param x: Shape (batch_size, seq_len, hidden_dim)
        :param mask: 0 = ignored, 1 = attention enabled
        """
        self._attn = None
        self._attn_gradients = None

        batch_size, seq_len, hidden_dim = x.shape

        if self.num_key_value_heads == self.num_heads:
            qkv = self.qkv(x).contiguous().reshape(
                batch_size, seq_len, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = self.q_proj(x).contiguous().view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)
            k = self.k_proj(x).contiguous().view(
                batch_size, seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            v = self.v_proj(x).contiguous().view(
                batch_size, seq_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # Handle mask for SDPA
        # Use float mask (-inf for masked positions, 0.0 for valid) instead of bool,
        # which is compatible with all SDPA backends (flash, mem_efficient, math).
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.contiguous().view(batch_size, 1, 1, seq_len)  # [B, 1, 1, N]
            elif mask.dim() == 3:
                attn_mask = mask.contiguous().view(batch_size, 1, seq_len, seq_len)  # [B, 1, N, N]
            else:
                attn_mask = mask
            # Use a bool mask (True = attend, False = ignore).
            # SDPA accepts bool masks and can still select Flash / mem_efficient backends,
            # whereas a float additive mask (with -inf) forces the O(N²) Math backend.
            attn_mask = attn_mask.bool()

        if causal:
            causal_mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
            attn_mask = causal_mask if attn_mask is None else attn_mask & causal_mask

        # Flash Attention has a known backward bug for short sequences. Use mem_efficient backend which:
        #   - avoids Flash Attention's small-seq backward CUDA bug
        #   - uses O(N) memory unlike the math backend
        if seq_len < 64:
            ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
            
            with ctx:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop_prob if self.training else 0.0,
                    is_causal=False
                )
        else:
            # seq_len >= 64: let SDPA auto-select Flash Attention (O(N) memory)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # bool mask → Flash Attention can be selected
                dropout_p=self.attn_drop_prob if self.training else 0.0,
                is_causal=False
            )

        x = x.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _save_attn_grad(self, grad):
        self._attn_gradients = grad

    def get_attn(self):
        return self._attn

    def get_attn_grad(self):
        return self._attn_gradients


class CrossFlashAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context, context_mask=None):
        """
        :param query: Shape (batch_size, query_len, hidden_dim)
        :param context: Shape (batch_size, context_len, hidden_dim)
        :param context_mask: Shape (batch_size, context_len), 1 = attend, 0 = ignore
        """
        batch_size, query_len, hidden_dim = query.shape
        context_len = context.shape[1]
        head_dim = hidden_dim // self.num_heads

        q = self.q_proj(query).contiguous().view(batch_size, query_len, self.num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(context).contiguous().view(batch_size, context_len, self.num_heads, head_dim).transpose(1, 2)
        v = self.v_proj(context).contiguous().view(batch_size, context_len, self.num_heads, head_dim).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.contiguous().view(batch_size, 1, 1, context_len).bool()

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_prob if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).contiguous().reshape(batch_size, query_len, hidden_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # SwiGLU needs double the hidden dimension for the chunking
        inner_dim = int(hidden_dim * 2 / 3) 
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=False)
        self.swiglu = SwiGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(inner_dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.w12(x)
        x = self.swiglu(x)
        x = self.dropout1(x)
        x = self.w3(x)
        x = self.dropout2(x)
        return x

class StandardTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, kv_heads=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(
            dim,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None, causal: bool = False):
        x = x + self.attn(self.norm1(x.to(self.norm1.weight.dtype)), mask, causal=causal)
        x = x + self.ffn(self.norm2(x.to(self.norm2.weight.dtype)))
        return x
