"""
Implementation of the functional NumPy random API.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from math import prod
from typing import TYPE_CHECKING

from numpy import apply_along_axis
from numpy.random import Generator, Philox

if TYPE_CHECKING:
    from typing import TypeVar, TypeAlias, ParamSpec, Concatenate, Callable
    import numpy as np
    from numpy.typing import NDArray

    P = ParamSpec("P")
    R = TypeVar("R")

    RandomKey: TypeAlias = NDArray[np.uint64]
    Shape: TypeAlias = Sequence[int]

    DTypeLikeFloat: TypeAlias = (
        np.dtype[np.float32]
        | np.dtype[np.float64]
        | type[np.float32]
        | type[np.float64]
        | type[float]
    )


def _multikey(
    func: Callable[Concatenate[RandomKey, P], R],
) -> Callable[Concatenate[RandomKey, P], R]:
    """
    Decorator for functions accepting multi-dimensional random keys.
    """

    @wraps(func)
    def wrapper(key, *args, **kwargs):
        if key.ndim > 1:
            return apply_along_axis(func, -1, key, *args, **kwargs)
        return func(key, *args, **kwargs)

    return wrapper


def key(seed: int | Sequence[int]) -> RandomKey:
    """
    Create a random key given an integer seed.
    """
    return Philox(key=seed).state["state"]["key"]


@_multikey
def split(key: RandomKey, num: int | tuple[int, ...] = 2) -> RandomKey:
    """
    Splits a random key into `num` new keys by adding a leading axis.
    """
    shape = (*num, 2) if isinstance(num, Sequence) else (num, 2)
    num = prod(shape)
    return Philox(key=key).random_raw(num).reshape(shape)


@_multikey
def normal(
    key: RandomKey,
    shape: Shape = (),
    dtype: DTypeLikeFloat = float,
) -> NDArray[np.floating]:
    """
    Sample standard normal random values with given shape and float dtype.
    """
    return Generator(Philox(key=key)).standard_normal(shape, dtype)
