from sdpa.attention.base import BaseAttention
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention
from sdpa.utils.types import AttentionConfig, AttentionType


class AttentionFactory:
    @staticmethod
    def create(
        attention_type: AttentionType, config: AttentionConfig
    ) -> BaseAttention:
        if attention_type == AttentionType.SCALED_DOT_PRODUCT:
            return ScaledDotProductAttention(config["d_model"])
        raise ValueError(f"Unsupported attention type: {attention_type}")
