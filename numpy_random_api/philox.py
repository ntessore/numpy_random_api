"""
Random implementation for the Philox bit generator.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Philox
from numpy.typing import NDArray

from .keys import register


def key(seed: NDArray[np.integer]) -> NDArray[np.uint64]:
    """
    Create a new Philox key from the given seed.
    """
    return Philox(key=seed).state["state"]["key"]


def split(key: NDArray[np.uint64], shape: tuple[int, ...]) -> NDArray[np.uint64]:
    """
    Split Philox key.
    """
    return bitgen(key).random_raw((*shape, 2))


def bitgen(key: NDArray[np.uint64]) -> Philox:
    """
    Return bit generator for key.
    """
    assert key.shape == (2,), "invalid key"
    return Philox(key=key)


register(name="philox", key=key, split=split, bitgen=bitgen)
