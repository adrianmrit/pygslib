import gslib_rust_cdf as rust
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

import pygslib
from pygslib import nonlinear, utils

LINEAR = nonlinear.TailType.Linear


def make_cdf(values, probabilities, lower=None, upper=None):
    return nonlinear.EcdfContinous.new_from_ik_cdf(
        values, probabilities,
        min(values) if lower is None else lower, LINEAR, 1.,
        LINEAR, 1., max(values) if upper is None else upper,
        LINEAR, 1., "integration test",
    )


def test_namespace_uses_rust_implementation():
    assert set(nonlinear.__all__) == set(rust.__all__)
    for name in rust.__all__:
        assert getattr(nonlinear, name) is getattr(rust, name)
    assert not hasattr(nonlinear, "anamor")


def test_weighted_table_matches_fortran_and_scipy():
    values = np.array([6., 1., 4., 2., 5., 3.])
    weights = np.array([1., 2., 3., 4., 5., 6.])
    original_values, original_weights = values.copy(), weights.copy()
    cdf = nonlinear.EcdfContinous.new_from_data(
        values, weights, 0., LINEAR, 1., LINEAR, 1.,
        7., LINEAR, 1., 0., "weighted",
    )
    quantiles, probabilities, gaussians = cdf.get_transformation_table()
    order = np.argsort(values)
    sorted_weights = weights[order]
    expected = (np.cumsum(sorted_weights) - sorted_weights / 2) / weights.sum()
    assert_array_equal(quantiles, values[order])
    assert_allclose(probabilities, expected, atol=1e-15)
    assert_allclose(gaussians, norm.ppf(expected), atol=1e-12)

    legacy_z, legacy_y, error = pygslib.gslib.__dist_transf.ns_ttable(
        values, weights, despike=0,
    )
    assert error == 0
    assert_array_equal(quantiles, legacy_z)
    assert_allclose(gaussians, legacy_y, atol=2e-6)
    assert_allclose(cdf.get_quantiles_from_gaussians(gaussians), quantiles, atol=1e-12)
    assert_array_equal(values, original_values)
    assert_array_equal(weights, original_weights)


def test_endpoint_probabilities_and_outside_bounds():
    cdf = make_cdf([0., 1., 2.], [0.2, 0.5, 0.8])
    assert_allclose(cdf.get_probabilities([0., 1., 2.]), [0.2, 0.5, 0.8])
    assert_allclose(cdf.get_quantiles([0.2, 0.5, 0.8]), [0., 1., 2.])
    assert_array_equal(cdf.get_probabilities([-1., 3.]), [0., 1.])
    assert nonlinear.gslib_hyperbolic_cdf(10., 0.8, 2., 10.) == pytest.approx(0.8)


@pytest.mark.parametrize("tail", [LINEAR, nonlinear.TailType.Power, nonlinear.TailType.Hyperbolic])
def test_tail_roundtrip(tail):
    cdf = nonlinear.EcdfContinous.new_from_ik_cdf(
        [1., 2., 3.], [.2, .5, .8], 0., nonlinear.TailType.Power, 2.,
        nonlinear.TailType.Power, 2., 10., tail, 2., "tails",
    )
    probabilities = np.array([.01, .1, .2, .3, .5, .7, .8, .9, .98])
    assert_allclose(cdf.get_probabilities(cdf.get_quantiles(probabilities)), probabilities, atol=1e-12)
    if tail == nonlinear.TailType.Hyperbolic:
        assert_array_equal(cdf.get_quantiles([.99]), [10.])


def test_uniform_distribution_moments_and_grade_tonnage():
    cdf = make_cdf(np.linspace(0., 10., 11), np.linspace(0., 1., 11))
    assert_allclose(cdf.get_moments(10000), [5., 100. / 12], atol=1e-6)
    cutoffs = np.array([2.5, 5., 7.5])
    tonnage, above, below = cdf.grade_tonnage(cutoffs, 10000)
    assert_allclose(tonnage, 1 - cutoffs / 10, atol=1e-12)
    assert_allclose(above, (10 + cutoffs) / 2, atol=1e-6)
    assert_allclose(below, cutoffs / 2, atol=1e-6)
    assert_allclose(tonnage * above + (1 - tonnage) * below, 5., atol=1e-6)


@pytest.mark.parametrize("size,order", [(6, 5), (7, 5), (31, 5), (101, 5), (101, 30)])
def test_hermite_fitting_and_support_with_realistic_table_sizes(size, order):
    values = np.linspace(1., 10., size)
    probabilities = (np.arange(size) + .5) / size
    cdf = make_cdf(values, probabilities)
    hermite = nonlinear.HermitePolynomials.new_from_cdf(cdf, 1000, order, .9, .8, 1.)
    assert hermite.pci.shape == (order + 1,)
    assert np.isfinite(hermite.pci).all()
    variance, empirical = hermite.var_pci()
    assert variance > 0
    assert_allclose(empirical, cdf.get_moments(1000)[1])
    r = .7
    block_variance = np.sum(hermite.pci[1:]**2 * r**(2 * np.arange(1, order + 1)))
    assert hermite.get_r(block_variance) == pytest.approx(r, abs=1e-10)
    y = np.array([-.5, 0., .5])
    for method in (hermite.y2z_point, hermite.y2z_block):
        raw, corrected = method(y)
        assert raw.shape == corrected.shape == y.shape
        assert np.isfinite(raw).all() and np.isfinite(corrected).all()
    tonnage, metal = hermite.uc(y, y)
    assert tonnage.shape == metal.shape == (3, 3)
    assert np.isfinite(metal).all()
    assert np.all((0 <= tonnage) & (tonnage <= 1))
    assert np.all(np.diff(tonnage, axis=1) <= 0)
    assert np.all(np.diff(tonnage, axis=0) >= 0)


@pytest.mark.parametrize("tail", [LINEAR, nonlinear.TailType.Hyperbolic])
def test_gcos_uses_rust_and_preserves_inputs(monkeypatch, tail):
    monkeypatch.setattr(plt, "show", lambda: None)
    values = np.linspace(1., 10., 101)
    table = pd.DataFrame({"grade": values, "weight": np.ones(101)})
    blocks = pd.DataFrame({"estimate": np.r_[np.linspace(2., 8., 101), np.nan]})
    cutoffs = np.array([2., 4., 6.])
    original = cutoffs.copy()
    fig = utils.gcos(
        table, "grade", ["estimate"], cutoffs, blocks, 2.,
        decwt="weight", K=5, ndisc=1000, utail=tail,
    )
    try:
        assert len(fig.axes) == 2
        assert [line.get_label() for line in fig.axes[0].lines] == [
            "DGM grade", "Estimate estimate",
        ]
        for axis in fig.axes:
            for line in axis.lines:
                assert np.isfinite(line.get_ydata()).all()
                assert_array_equal(line.get_xdata(), cutoffs)
        assert_array_equal(cutoffs, original)
    finally:
        plt.close(fig)


def test_gcos_rejects_unsupported_interval():
    with pytest.raises(ValueError, match="ymin/ymax"):
        utils.gcos(None, "grade", [], [], None, 1., ymin=-3.)
