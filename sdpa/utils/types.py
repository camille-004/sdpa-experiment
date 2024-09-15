from dataclasses import dataclass
from typing import NamedTuple, Protocol, TypedDict

import numpy as np


class AttentionProtocol(Protocol):
    def forward(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> "AttentionOutput": ...


class AttentionConfig(TypedDict, total=False):
    d_model: int
    d_k: int
    num_heads: int


AttentionOutput = NamedTuple(
    "AttentionOutput", [("output", np.ndarray), ("weights", np.ndarray)]
)


@dataclass(frozen=True, slots=True)
class AttentionParams:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
