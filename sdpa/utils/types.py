from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Protocol, TypedDict

import numpy as np


class AttentionProtocol(Protocol):
    def compute(
        self, query: np.ndarray, key: np.ndarray, value: np.ndarray
    ) -> np.ndarray: ...


class AttentionConfig(TypedDict):
    d_model: int
    num_heads: int


class AttentionType(Enum):
    SCALED_DOT_PRODUCT = auto()


AttentionOutput = NamedTuple(
    "AttentionOutput", [("output", np.ndarray), ("weights", np.ndarray)]
)


@dataclass(frozen=True, slots=True)
class AttentionParams:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
