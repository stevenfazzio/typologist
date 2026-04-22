from __future__ import annotations

import numpy as np
import pandas as pd

from typologist._pipeline import _l2_normalize, _residualize_facet


def test_residualize_facet_equalizes_class_means():
    rng = np.random.default_rng(42)
    n = 80
    d = 16
    class_ids = rng.integers(0, 3, size=n)
    shifts = rng.standard_normal((3, d)) * 2.0
    x = rng.standard_normal((n, d)).astype(np.float32) + shifts[class_ids]

    labels = pd.Series(
        pd.Categorical(
            [["a", "b", "c"][k] for k in class_ids],
            categories=["a", "b", "c", "Unlabelled"],
        )
    )

    erased = _residualize_facet(x, labels, was_normalized=False)

    assert erased.shape == x.shape
    spread_before = np.stack([x[class_ids == k].mean(axis=0) for k in range(3)]).std(axis=0).mean()
    spread_after = (
        np.stack([erased[class_ids == k].mean(axis=0) for k in range(3)]).std(axis=0).mean()
    )
    assert spread_after < spread_before * 0.1


def test_residualize_facet_renormalizes_when_input_was_normalized():
    rng = np.random.default_rng(7)
    x = _l2_normalize(rng.standard_normal((50, 16)).astype(np.float32))
    labels = pd.Series(pd.Categorical(["a", "b"] * 25, categories=["a", "b", "Unlabelled"]))

    erased = _residualize_facet(x, labels, was_normalized=True)
    np.testing.assert_allclose(np.linalg.norm(erased, axis=1), 1.0, atol=1e-5)


def test_residualize_facet_handles_noise_rows():
    rng = np.random.default_rng(11)
    n = 60
    x = rng.standard_normal((n, 12)).astype(np.float32)
    values = ["a", "b", "Unlabelled"]
    labels = pd.Series(
        pd.Categorical(
            rng.choice(values, size=n, p=[0.4, 0.4, 0.2]),
            categories=values,
        )
    )

    erased = _residualize_facet(x, labels, was_normalized=False)
    assert erased.shape == x.shape


def test_residualize_facet_preserves_dtype():
    rng = np.random.default_rng(1)
    x32 = rng.standard_normal((30, 8)).astype(np.float32)
    x64 = rng.standard_normal((30, 8)).astype(np.float64)
    labels = pd.Series(pd.Categorical(["a", "b"] * 15, categories=["a", "b", "Unlabelled"]))

    assert _residualize_facet(x32, labels, was_normalized=False).dtype == np.float32
    assert _residualize_facet(x64, labels, was_normalized=False).dtype == np.float64
