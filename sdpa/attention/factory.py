from sdpa.attention.base import BaseAttention
from sdpa.attention.multi_head import MultiHeadAttention
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention
from sdpa.utils.types import AttentionConfig


class AttentionFactory:
    @staticmethod
    def create(config: AttentionConfig) -> BaseAttention:
        if "num_heads" in config and config["num_heads"] is not None:
            if "d_model" not in config or config["d_model"] is None:
                raise ValueError(
                    "d_model must be specified when num_heads is provided."
                )
            return MultiHeadAttention(
                d_model=config["d_model"], num_heads=config["num_heads"]
            )
        elif "d_k" in config and config["d_k"] is not None:
            return ScaledDotProductAttention(d_k=config["d_k"])
        else:
            raise ValueError(
                "Invalid attention configuration: 'd_k' or 'num_heads' must be specified."  # noqa
            )
