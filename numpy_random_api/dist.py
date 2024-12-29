"""
Implementation of stateless random distributions.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Callable, Concatenate, ParamSpec, TypeAlias, TypeVar
    from .keys import RandomKey

    P = ParamSpec("P")
    R = TypeVar("R")

    Shape: TypeAlias = tuple[int, ...]

    DTypeLikeFloat: TypeAlias = (
        np.dtype[np.float32]
        | np.dtype[np.float64]
        | type[np.float32]
        | type[np.float64]
        | type[float]
    )


def dist(
    func: Callable[Concatenate[RandomKey, P], R],
) -> Callable[Concatenate[RandomKey, P], R]:
    """
    Decorator for distributions accepting multi-dimensional random keys.
    """

    @functools.wraps(func)
    def wrapper(key, *args, **kwargs):
        if key.ndim > 1:
            return np.apply_along_axis(func, -1, key, *args, **kwargs)
        return func(key, *args, **kwargs)

    return wrapper


def rng(key: RandomKey) -> np.random.Generator:
    """
    Create a random number generator from a key.
    """
    return np.random.Generator(key.impl.bitgen(key))


@dist
def normal(
    key: RandomKey,
    shape: Shape = (),
    dtype: DTypeLikeFloat = float,
) -> np.typing.NDArray[np.floating]:
    """
    Sample standard normal random values with given shape and float dtype.
    """
    return rng(key).standard_normal(shape, dtype)
