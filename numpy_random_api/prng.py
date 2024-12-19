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
JAX-like PRNG implementation around NumPy-like random number generators.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
from numpy.random import BitGenerator, Philox, SeedSequence
from numpy.typing import ArrayLike, NDArray

Shape = Sequence[int]


class PRNGImpl(NamedTuple):
    """
    Specifies PRNG key shape and operations.
    """

    key_shape: Shape
    seed: Callable[[NDArray], NDArray]
    split: Callable[[NDArray, Shape], NDArray]
    # random_bits: Callable
    # fold_in: Callable
    bit_generator: Callable[[NDArray], BitGenerator]
    name: str = "<unnamed>"
    tag: str = "?"

    def __hash__(self) -> int:
        return hash(self.tag)

    def __str__(self) -> str:
        return self.tag


prngs = {}


def register_prng(impl: PRNGImpl):
    if impl.name in prngs:
        raise ValueError(f"PRNG with name {impl.name} already registered: {impl}")
    prngs[impl.name] = impl


def _check_prng_key_data(impl, key_data: NDArray):
    ndim = len(impl.key_shape)
    if not all(hasattr(key_data, attr) for attr in ["ndim", "shape", "dtype"]):
        raise TypeError(
            f"invalid PRNG key data: expected key_data to have ndim, shape, "
            f"and dtype attributes. Got {key_data}"
        )
    if key_data.ndim < 1:
        raise TypeError(
            f"invalid PRNG key data: expected key_data.ndim >= 1; got "
            f"ndim={key_data.ndim}"
        )
    if key_data.shape[-ndim:] != impl.key_shape:
        raise TypeError(
            f"invalid PRNG key data: expected key_data.shape to end with "
            f"{impl.key_shape}; got shape={key_data.shape} for {impl=}"
        )
    if key_data.dtype != np.uint32:
        raise TypeError(
            f"invalid PRNG key data: expected key_data.dtype = uint32; got "
            f"dtype={key_data.dtype}"
        )


# TODO: implement array overlay as in jax
class PRNGKeyArray(np.ndarray):
    """
    An array of PRNG keys backed by an RNG implementation.
    """

    __slots__ = ("_impl",)
    _impl: PRNGImpl

    def __new__(cls, impl: PRNGImpl, key_data: NDArray) -> PRNGKeyArray:
        _check_prng_key_data(impl, key_data)
        obj = np.asarray(key_data).view(cls)
        obj._impl = impl
        return obj

    def __array_finalize__(self, obj: NDArray | None) -> None:
        if obj is None:
            return
        impl = getattr(obj, "_impl", None)
        if isinstance(impl, PRNGImpl):
            self._impl = impl

    def __repr__(self) -> str:
        r = super().__repr__()
        if r[-1] == ")":
            r = r[:-1] + f", impl={str(self._impl)!r})"
        return r


def random_seed(seeds: int | ArrayLike, impl: PRNGImpl) -> PRNGKeyArray:
    """
    Return key array for random seed.
    """
    seeds_arr = np.asarray(seeds)
    base_arr = _random_seed(seeds_arr, impl=impl)
    assert base_arr.shape[-1:] == impl.key_shape
    return PRNGKeyArray(impl, base_arr)


@np.vectorize(signature="()->(n)", excluded={"impl"})  # type: ignore[call-arg]
def _random_seed(seed: NDArray, *, impl: PRNGImpl) -> NDArray:
    """
    Vectorize seed over input array.
    """
    return impl.seed(seed)


def random_split(keys: PRNGKeyArray, shape: Shape) -> PRNGKeyArray:
    """
    Split key array.
    """
    impl = keys._impl
    keys_arr = keys.view(np.ndarray)
    base_arr = np.apply_along_axis(impl.split, -1, keys_arr, shape)
    return PRNGKeyArray(impl, base_arr)


def bit_generator(key: PRNGKeyArray) -> BitGenerator:
    """
    Return bit generator for random key.
    """
    return key._impl.bit_generator(key)


def philox_seed(seed: NDArray) -> NDArray:
    """
    Seed a Philox key.
    """
    if seed.shape:
        raise TypeError(f"PRNG key seed must be a scalar; got {seed!r}.")
    if not np.issubdtype(seed.dtype, np.integer):
        raise TypeError(f"PRNG key seed must be an integer; got {seed!r}")
    return SeedSequence(seed).generate_state(4, np.uint32)


def philox_split(key: NDArray, shape: Shape) -> NDArray:
    """
    Split a Philox key.
    """
    bit_gen = Philox(key=key.view(np.uint64))
    num = math.prod(shape)
    return bit_gen.random_raw(num * 2).view(np.uint32).reshape(*shape, 4)


def philox_bit_generator(key: NDArray) -> BitGenerator:
    """
    Create a Philox bit generator from key array.
    """
    return Philox(key=key.view(np.uint64))


philox_prng_impl = PRNGImpl(
    key_shape=(4,),
    seed=philox_seed,
    split=philox_split,
    bit_generator=philox_bit_generator,
    name="philox",
    tag="philox",
)

register_prng(philox_prng_impl)
