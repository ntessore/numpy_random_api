# Functional random number generation for NumPy

**Caveat emptor: this is only a proof of concept.**

Provides a functional API for NumPy's random number generation that is
compatible with [`jax.random`].

This is a very simple toy implementation, built around the existing NumPy bit
generators.

```py
>>> from numpy_random_api import random
>>> key = random.key(42)
>>> key
RandomKey([42,  0], dtype=uint64, impl='philox')
>>> key, subkey = random.split(key)
>>> key
RandomKey([15129985323320379406,  3490965594592278910], dtype=uint64, impl='philox')
>>> random.normal(subkey, 4)
array([-0.91505197, -0.72636576, -1.64833621, -0.33304836])
```

[`jax.random`]: https://jax.readthedocs.io/en/latest/jax.random.html
