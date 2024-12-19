# Based on the jax.random implementation.
#
# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functional API for NumPy random number generators.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike, NDArray as Array

from . import prng

DTypeLikeFloat: TypeAlias = (
    np.dtype[np.float32]
    | np.dtype[np.float64]
    | type[np.float32]
    | type[np.float64]
    | type[float]
)

Shape: TypeAlias = Sequence[int]

PRNGImpl = prng.PRNGImpl
PRNGKeyArray = prng.PRNGKeyArray


config_default_prng_impl = "philox"


def default_prng_impl() -> PRNGImpl:
    """
    Get the default PRNG implementation.
    """
    impl_name = config_default_prng_impl
    assert impl_name in prng.prngs, impl_name
    return prng.prngs[impl_name]


class PRNGSpec:
    """
    Specifies a PRNG key implementation.
    """

    __slots__ = ("_impl",)
    _impl: PRNGImpl

    def __init__(self, impl):
        self._impl = impl

    def __repr__(self) -> str:
        return f"PRNGSpec({self._impl.name!r})"

    def __str__(self) -> str:
        return str(self._impl)

    def __hash__(self) -> int:
        return hash(self._impl)

    def __eq__(self, other) -> bool:
        return isinstance(other, PRNGSpec) and self._impl == other._impl


PRNGSpecDesc = str | PRNGSpec | PRNGImpl


def resolve_prng_impl(impl_spec: PRNGSpecDesc | None) -> PRNGImpl:
    if impl_spec is None:
        return default_prng_impl()
    if type(impl_spec) is PRNGImpl:
        return impl_spec
    if type(impl_spec) is PRNGSpec:
        return impl_spec._impl
    if type(impl_spec) is str:
        if impl_spec in prng.prngs:
            return prng.prngs[impl_spec]

        keys_fmt = ", ".join(f'"{s}"' for s in prng.prngs.keys())
        raise ValueError(
            f'unrecognized PRNG implementation "{impl_spec}". '
            f"Did you mean one of: {keys_fmt}?"
        )

    t = type(impl_spec)
    raise TypeError(f"unrecognized type {t} for specifying PRNG implementation.")


def key(seed: int | ArrayLike, *, impl: PRNGSpecDesc | None = None) -> PRNGKeyArray:
    """
    Create a pseudo-random number generator (PRNG) key given an integer seed.
    """
    impl = resolve_prng_impl(impl)
    if isinstance(seed, PRNGKeyArray):
        raise TypeError("key() accepts a scalar seed, but was given a PRNG key")
    if np.ndim(seed):
        raise TypeError(
            f"key() accepts a scalar seed, but was given an array of shape "
            f"{np.shape(seed)} != ()"
        )
    return prng.random_seed(seed, impl=impl)


def split(key: PRNGKeyArray, num: int | tuple[int, ...] = 2) -> PRNGKeyArray:
    """
    Splits a PRNG key into `num` new keys by adding a leading axis.
    """
    if key.ndim > len(key._impl.key_shape):
        raise TypeError(
            "split accepts a single key, but was given a key array of "
            f"shape {key.shape} != ()."
        )
    shape = tuple(num) if isinstance(num, Sequence) else (num,)
    return prng.random_split(key, shape=shape)


def normal(
    key: PRNGKeyArray, shape: Shape = (), dtype: DTypeLikeFloat = float
) -> Array:
    """
    Sample standard normal random values with given shape and float dtype.
    """
    return Generator(prng.bit_generator(key)).standard_normal(shape, dtype)
