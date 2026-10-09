import numpy as np
import pytest

from embedding_diagnostics.jina_compression import jina_matryoshka, validate_jina


def canonical(n=3):
    x = np.random.default_rng(4).normal(size=(n, 1024)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_native_identity_and_learned_prefix_normalization():
    x = canonical()
    assert np.array_equal(jina_matryoshka(x, 1024), x)
    for d in (256, 128):
        y = jina_matryoshka(x, d)
        assert y.shape == (3, d) and np.isfinite(y).all()
        assert np.allclose(np.linalg.norm(y, axis=1), 1)
        assert np.array_equal(y, jina_matryoshka(x, d))
        expected = x[:, :d].astype(np.float64)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert np.array_equal(y, expected)


@pytest.mark.parametrize("kind", ["half", "dimension", "nan", "zero", "norm"])
def test_invalid_canonical_rejected(kind):
    x = canonical()
    if kind == "half":
        x = x.astype(np.float16)
    elif kind == "dimension":
        x = x[:, :768]
    elif kind == "nan":
        x[0, 0] = np.nan
    elif kind == "zero":
        x[0] = 0
    else:
        x *= 2
    with pytest.raises(ValueError):
        validate_jina(x, 3)


def test_no_unsupported_or_zero_prefix():
    x = canonical()
    with pytest.raises(ValueError):
        jina_matryoshka(x, 512)
    x[0, :128] = 0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    with pytest.raises(ValueError):
        jina_matryoshka(x, 128)
