"""
Functional random API for NumPy.
"""

__all__ = [
    "key",
    "split",
    "normal",
]

from .keys import (
    key,
    split,
)

from .dist import (
    normal,
)

# load bit generator implementations
__import__("philox", globals(), locals(), (), 1)
