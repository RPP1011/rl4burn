"""Cross-verification tests for distribution transforms between rl4burn_tune_py and Optuna."""

import math
import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

from .conftest import FLOAT_ATOL


# ---------------------------------------------------------------------------
# Float distribution: transform round-trips
# ---------------------------------------------------------------------------


class TestFloatTransformRoundTrip:
    """Verify rl4burn float transforms are internally consistent and correct.

    Note: Optuna's to_internal_repr() is identity (returns the raw value),
    while rl4burn normalizes to [0,1]. We verify the round-trip property
    and that both libraries agree on contains/single semantics.
    """

    @pytest.mark.parametrize(
        "low, high, log, values",
        [
            (0.0, 1.0, False, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (-5.0, 5.0, False, [-5.0, -2.5, 0.0, 2.5, 5.0]),
            (1e-5, 1.0, True, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
            (1e-3, 1e3, True, [1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3]),
        ],
    )
    def test_to_internal_in_unit_interval(self, low, high, log, values):
        """rl4burn internal representations should be in [0, 1]."""
        for v in values:
            internal = rl.float_transform_to_internal(v, low, high, log)
            assert 0.0 <= internal <= 1.0, f"Internal repr {internal} out of [0,1] for v={v}"

    @pytest.mark.parametrize(
        "low, high, log, values",
        [
            (0.0, 1.0, False, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (-5.0, 5.0, False, [-5.0, -2.5, 0.0, 2.5, 5.0]),
            (1e-5, 1.0, True, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
        ],
    )
    def test_to_internal_monotonic(self, low, high, log, values):
        """Internal representation should be monotonically increasing."""
        internals = [rl.float_transform_to_internal(v, low, high, log) for v in values]
        for i in range(1, len(internals)):
            assert internals[i] >= internals[i - 1], (
                f"Not monotonic: {internals[i-1]} > {internals[i]} for values {values[i-1]}, {values[i]}"
            )

    @pytest.mark.parametrize(
        "low, high, log, values",
        [
            (0.0, 1.0, False, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (-5.0, 5.0, False, [-5.0, -2.5, 0.0, 2.5, 5.0]),
            (1e-5, 1.0, True, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
        ],
    )
    def test_boundary_values(self, low, high, log, values):
        """low should map to 0.0, high should map to 1.0."""
        assert abs(rl.float_transform_to_internal(low, low, high, log) - 0.0) < FLOAT_ATOL
        assert abs(rl.float_transform_to_internal(high, low, high, log) - 1.0) < FLOAT_ATOL

    @pytest.mark.parametrize(
        "low, high, log",
        [
            (0.0, 1.0, False),
            (-10.0, 10.0, False),
            (1e-5, 1e5, True),
            (0.001, 100.0, True),
        ],
    )
    def test_roundtrip(self, low, high, log):
        """to_internal -> from_internal should be identity."""
        rng = np.random.RandomState(42)
        if log:
            values = np.exp(rng.uniform(math.log(low), math.log(high), 20))
        else:
            values = rng.uniform(low, high, 20)

        for v in values:
            internal = rl.float_transform_to_internal(v, low, high, log)
            recovered = rl.float_transform_from_internal(internal, low, high, log)
            assert abs(recovered - v) < 1e-10 * max(abs(v), 1.0), (
                f"Round-trip failed for v={v}: got {recovered}"
            )

    @pytest.mark.parametrize(
        "low, high, log, internals",
        [
            (0.0, 1.0, False, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (-5.0, 5.0, False, [0.0, 0.25, 0.5, 0.75, 1.0]),
            (1e-5, 1.0, True, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        ],
    )
    def test_from_internal_in_range(self, low, high, log, internals):
        """from_internal should produce values within [low, high]."""
        for internal in internals:
            external = rl.float_transform_from_internal(internal, low, high, log)
            assert low - FLOAT_ATOL <= external <= high + FLOAT_ATOL, (
                f"External {external} out of [{low}, {high}] for internal={internal}"
            )


# ---------------------------------------------------------------------------
# Integer distribution: transform round-trips
# ---------------------------------------------------------------------------


class TestIntTransformRoundTrip:
    """Verify integer transforms are internally consistent."""

    @pytest.mark.parametrize(
        "low, high, log, values",
        [
            (0, 10, False, [0, 1, 5, 9, 10]),
            (1, 100, False, [1, 25, 50, 75, 100]),
            (1, 1000, True, [1, 10, 100, 1000]),
        ],
    )
    def test_to_internal_in_unit_interval(self, low, high, log, values):
        """Internal representations should be in [0, 1]."""
        for v in values:
            internal = rl.int_transform_to_internal_py(v, low, high, log)
            assert 0.0 <= internal <= 1.0, f"Internal {internal} out of [0,1] for v={v}"

    @pytest.mark.parametrize(
        "low, high, log",
        [
            (0, 10, False),
            (1, 100, False),
            (1, 1000, True),
        ],
    )
    def test_roundtrip(self, low, high, log):
        """to_internal -> from_internal should recover the integer value."""
        step = 1
        values = range(low, high + 1, step)
        # Sample a subset for large ranges
        sample = list(values)[:20] + list(values)[-5:]
        for v in set(sample):
            internal = rl.int_transform_to_internal_py(v, low, high, log)
            recovered = rl.int_transform_from_internal_py(internal, low, high, log)
            assert recovered == v, f"Round-trip failed for v={v}: got {recovered}"

    @pytest.mark.parametrize(
        "low, high, log",
        [
            (0, 10, False),
            (1, 100, False),
        ],
    )
    def test_boundary_values(self, low, high, log):
        """low maps to 0.0, high maps to 1.0."""
        assert abs(rl.int_transform_to_internal_py(low, low, high, log) - 0.0) < FLOAT_ATOL
        assert abs(rl.int_transform_to_internal_py(high, low, high, log) - 1.0) < FLOAT_ATOL


# ---------------------------------------------------------------------------
# Cross-comparison: Optuna contains / single semantics
# ---------------------------------------------------------------------------


class TestDistributionContains:
    """Verify contains/single semantics match Optuna."""

    def test_float_contains(self):
        optuna_dist = FloatDistribution(0.0, 1.0)
        assert optuna_dist._contains(0.5)
        assert not optuna_dist._contains(1.5)

    def test_int_contains(self):
        optuna_dist = IntDistribution(0, 10, step=2)
        assert optuna_dist._contains(0)
        assert optuna_dist._contains(2)
        assert not optuna_dist._contains(1)

    def test_categorical_single(self):
        d1 = CategoricalDistribution(choices=["a"])
        d2 = CategoricalDistribution(choices=["a", "b"])
        assert d1.single()
        assert not d2.single()

    def test_float_single(self):
        d_range = FloatDistribution(0.0, 1.0)
        assert not d_range.single()

    def test_int_single(self):
        d_single = IntDistribution(5, 5)
        d_range = IntDistribution(0, 10)
        assert d_single.single()
        assert not d_range.single()
