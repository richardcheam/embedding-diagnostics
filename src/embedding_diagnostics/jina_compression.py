"""Jina-CLIP-v2 learned prefixes (Koukounas et al., arXiv:2412.08802v2).

Both modalities use a leading prefix followed by L2 normalization. This does
not validate paper claims; analysis arithmetic uses the accepted FP64 policy.
"""

from __future__ import annotations

import numpy as np


def validate_jina(matrix, count):
    x = np.asarray(matrix)
    if count < 1 or x.dtype != np.float32 or x.shape != (count, 1024):
        raise ValueError("expected finite float32 [N,1024]")
    if not np.isfinite(x).all() or not np.allclose(
        np.linalg.norm(x, axis=1), 1, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("finite unit canonical vectors required")
    return x


def jina_matryoshka(canonical, dimension):
    x = validate_jina(canonical, len(canonical))
    if dimension not in (1024, 256, 128):
        raise ValueError("unsupported frozen learned dimension")
    if dimension == 1024:
        return x.copy()
    y = x[:, :dimension].astype(np.float64)
    n = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("zero prefix")
    return y / n
