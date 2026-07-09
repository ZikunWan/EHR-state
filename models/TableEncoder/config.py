from typing import List, Optional

from transformers import PretrainedConfig


class _BaseTableEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        # Core dimensions for input text embeddings and transformer hidden states.
        text_dim: int = 768,
        dim: int = 1536,
        depth: int = 40,
        heads: int = 24,
        kv_heads: Optional[int] = 8,
        dim_head: int = 64,
        mlp_dim: int = 6144,
        dropout: float = 0.0,

        # Output adapter: allocate at most max_queries, about one query per tokens_per_query rows.
        max_queries: int = 160,
        tokens_per_query: int = 32,
        dim_out: Optional[int] = None,
        # Keep only the most recent max_table_len rows before encoding.
        max_table_len: Optional[int] = 32768,
        # GPT-style table encoding: each row can only attend to current/past rows.
        is_causal: bool = True,

        # Table feature settings for numeric value Fourier features and type embeddings.
        fourier_scales: Optional[List[float]] = None,
        type_vocab_size: int = 11,

        **kwargs,
    ):
        super().__init__(**kwargs)

        if fourier_scales is None:
            fourier_scales = [0.1, 1.0, 10.0, 100.0]
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if tokens_per_query <= 0:
            raise ValueError("tokens_per_query must be positive")
        if max_table_len is not None and max_table_len <= 0:
            raise ValueError("max_table_len must be positive when set")
        if kv_heads is None:
            kv_heads = heads
        if heads <= 0:
            raise ValueError("heads must be positive")
        if kv_heads <= 0:
            raise ValueError("kv_heads must be positive")
        if heads % kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads for grouped-query attention")

        self.text_dim = text_dim
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.max_queries = max_queries
        self.tokens_per_query = tokens_per_query
        self.dim_out = dim_out
        self.max_table_len = max_table_len
        self.is_causal = is_causal

        self.fourier_scales = fourier_scales
        self.type_vocab_size = type_vocab_size


class LongTableEncoder1DConfig(_BaseTableEncoderConfig):
    model_type = "long_table_encoder_1d"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
