from sdpa.attention.base import BaseAttention


class AttentionManager:
    _instance = None
    attention_modules: dict[str, BaseAttention]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.attention_modules = {}
        return cls._instance

    def register_attention(self, name: str, attention: BaseAttention) -> None:
        self.attention_modules[name] = attention

    def get_attention(self, name: str) -> BaseAttention:
        return self.attention_modules[name]
