# JAX-like random API for NumPy random number generators

Using a JAX-like interface to generate a random normal array:

```py
>>> from numpy_random_api import random
>>> key = random.key(42)
>>> key
PRNGKeyArray([3444837047, 2669555309, 2046530742, 3581440988],
             dtype=uint32, impl='philox')
>>> key, subkey = random.split(key)
>>> key
PRNGKeyArray([3973757322,  369700608,  604115056,  607984076],
             dtype=uint32, impl='philox')
>>> random.normal(subkey, 4)
array([-0.05883458,  0.6125753 , -1.29899843,  0.12702094])
```

Equivalent code using NumPy's random interface:

```py
>>> import numpy as np
>>> from numpy.random import Generator, Philox, SeedSequence
>>> key = SeedSequence(42).generate_state(4).view(np.uint32)
>>> key
array([3444837047, 2669555309, 2046530742, 3581440988], dtype=uint32)
>>> key, subkey = Philox(key=key.view(np.uint64)).random_raw(4).reshape(2, 2).view(np.uint32)
>>> key
array([3973757322,  369700608,  604115056,  607984076], dtype=uint32)
>>> Generator(Philox(key=subkey.view(np.uint64))).standard_normal(4)
array([-0.05883458,  0.6125753 , -1.29899843,  0.12702094])
```
