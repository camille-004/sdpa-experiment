import unittest

from sdpa.attention.manager import AttentionManager
from sdpa.attention.scaled_dot_product import ScaledDotProductAttention


class TestAttentionManager(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AttentionManager()
        self.attention = ScaledDotProductAttention(d_model=4)

    def test_singleton(self) -> None:
        another_manager = AttentionManager()
        self.assertIs(
            self.manager,
            another_manager,
            "AttentionManager should be a singleton",
        )

    def test_register_and_get_attention(self) -> None:
        self.manager.register_attention("test", self.attention)
        retrieved_attention = self.manager.get_attention("test")
        self.assertIs(
            retrieved_attention,
            self.attention,
            "Retrieved attention should be the same as registered",
        )

    def test_get_nonexistent_attention(self) -> None:
        with self.assertRaises(KeyError):
            return self.manager.get_attention("nonexistent")

    def test_register_multiple_attentions(self) -> None:
        attention2 = ScaledDotProductAttention(d_model=8)
        self.manager.register_attention("test1", self.attention)
        self.manager.register_attention("test2", attention2)

        retrieved1 = self.manager.get_attention("test1")
        retrieved2 = self.manager.get_attention("test2")

        self.assertIs(retrieved1, self.attention)
        self.assertIs(retrieved2, attention2)


if __name__ == "__main__":
    unittest.main()
