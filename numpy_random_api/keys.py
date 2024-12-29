"""
Implementation of random keys.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, NamedTuple, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

Shape: TypeAlias = tuple[int, ...]


class RandomImpl(NamedTuple):
    """
    Description of a random implementation.
    """

    name: str
    key: Callable[[NDArray[np.integer]], NDArray[Any]]
    split: Callable[[NDArray[Any], Shape], NDArray[Any]]
    bitgen: Callable[[NDArray[Any]], np.random.BitGenerator]


impls = {}


def register(
    *,
    name: str,
    key: Callable[[NDArray[np.integer]], NDArray[Any]],
    split: Callable[[NDArray[Any], Shape], NDArray[Any]],
    bitgen: Callable[[NDArray[Any]], np.random.BitGenerator],
) -> None:
    """
    Register a random implementation.
    """
    if name in impls:
        raise ValueError(f"duplicate random implementation {name!r}")
    impls[name] = RandomImpl(
        name=name,
        key=key,
        split=split,
        bitgen=bitgen,
    )


class RandomKey(np.ndarray[tuple[int, ...], np.dtype[Any]]):
    """
    Array subclass for random arrays.
    """

    __slots__ = ("impl",)
    impl: RandomImpl

    def __new__(cls, impl: RandomImpl, key: NDArray[Any]) -> Self:
        """
        Create a new random key from the given seed.
        """
        obj = key.view(cls)
        obj.impl = impl
        return obj

    def __array_finalize__(self, obj: NDArray[Any] | None = None) -> None:
        """
        Finalise random key created from another array.
        """
        if isinstance(obj, RandomKey):
            self.impl = obj.impl

    def __repr__(self) -> str:
        """
        Representation of random key.
        """
        r = super().__repr__()
        if r[-1:] == ")":
            r = r[:-1] + f", impl={self.impl.name!r})"
        return r


def key(seed: int | Sequence[int], *, impl: str = "philox") -> RandomKey:
    """
    Create a random key given an integer seed.
    """

    try:
        _impl = impls[impl]
    except KeyError:
        raise ValueError(f"unknown implementation {impl!r}") from None

    key = _impl.key(np.asarray(seed))
    return RandomKey(_impl, key)


def split(key: RandomKey, num: int | tuple[int, ...] = 2) -> RandomKey:
    """
    Splits a random key into `num` new keys by adding leading axes.
    """

    impl = key.impl

    if isinstance(num, tuple):
        shape = num
    elif isinstance(num, int):
        shape = (num,)
    else:
        raise TypeError("num must be int or tuple of int")

    if key.ndim > 1:
        out = np.apply_along_axis(impl.split, -1, key, shape)
    else:
        out = impl.split(key, shape)

    return RandomKey(impl, out)
