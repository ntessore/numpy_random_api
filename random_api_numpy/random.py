"""
Functional API for NumPy random number generators.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TypeAlias

import numpy as np
from numpy.random import Generator, Philox
from numpy.typing import NDArray

DTypeLikeFloat: TypeAlias = (
    np.dtype[np.float32]
    | np.dtype[np.float64]
    | type[np.float32]
    | type[np.float64]
    | type[float]
)

RandomKey: TypeAlias = NDArray[np.uint64]
Shape: TypeAlias = Sequence[int]


def key(seed: int | Sequence[int]) -> RandomKey:
    """
    Create a random key given an integer seed.
    """
    return Philox(key=seed).state["state"]["key"]


def split(key: RandomKey, num: int | tuple[int, ...] = 2) -> RandomKey:
    """
    Splits a random key into `num` new keys by adding a leading axis.
    """
    shape = (*num, 2) if isinstance(num, Sequence) else (num, 2)
    num = prod(shape)
    return np.apply_along_axis(
        lambda k: Philox(key=k).random_raw(num).reshape(shape),
        -1,
        key,
    )


def normal(
    key: RandomKey,
    shape: Shape = (),
    dtype: DTypeLikeFloat = float,
) -> NDArray[np.floating]:
    """
    Sample standard normal random values with given shape and float dtype.
    """
    return np.apply_along_axis(
        lambda k: Generator(Philox(key=k)).standard_normal(shape, dtype),
        -1,
        key,
    )
