"""Execution checks for the local marimo SNR distribution sweeps."""

from __future__ import annotations

import runpy
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from repo import REPO_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def notebook_results(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Mapping[str, Any]]:
    """Run the real waveform on tiny ensembles, then require checked cache hits."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ASTROGWB_NOTEBOOK_SMOKE", "1")
        patch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("spectrum-snrs")))
        notebook = runpy.run_path(str(REPO_ROOT / "notebooks" / "spectrum_snrs.py"))
        app = notebook["app"]
        _, definitions = app.run()

        from astrogwb.catalog import SpectrumGenerator

        def forbid_generation(self: SpectrumGenerator, metadata: Any) -> Any:
            pytest.fail(f"repeat execution generated spectra for {metadata.key()}")

        patch.setattr(SpectrumGenerator, "__call__", forbid_generation)
        _, cached_definitions = app.run()
        for group in ("num_events_cases", "minimum_redshift_cases"):
            for key, case in definitions[group].items():
                np.testing.assert_array_equal(
                    case.snrs, cached_definitions[group][key].snrs
                )

        try:
            yield definitions
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")


def test_sweeps_keep_the_other_parameter_fixed(
    notebook_results: Mapping[str, Any],
) -> None:
    counts = notebook_results["num_events_cases"]
    cutoffs = notebook_results["minimum_redshift_cases"]
    assert list(counts) == [8, 16, 32]
    assert list(cutoffs) == [0.05, 0.15, 0.35]
    for count, case in counts.items():
        assert case.metadata.num_events == count
        assert case.metadata.population.model_kwargs["minimum_redshift"] == 0.35
    for cutoff, case in cutoffs.items():
        assert case.metadata.num_events == 8
        assert case.metadata.population.model_kwargs["minimum_redshift"] == cutoff
    assert counts[8].metadata.key() == cutoffs[0.35].metadata.key()
    assert len(list(Path(notebook_results["cache_dir"]).glob("*.h5"))) == 6
    data = notebook_results["data_catalog"]
    assert data.metadata.population.seed != counts[8].metadata.population.seed
    assert data.metadata.num_draws == 1
    assert data.metadata.fixed == counts[8].metadata.fixed
    assert notebook_results["write_figures"] is False

    for case in [*counts.values(), *cutoffs.values()]:
        assert case.metadata.count == "fixed"
        assert not case.metadata.sampled
        assert case.snrs.shape == (3,)
        assert np.all(np.isfinite(case.snrs))
        assert np.all(case.snrs > 0)
        np.testing.assert_allclose(case.sigma_h0, case.metadata.fixed["H0"] / case.snrs)
        assert case.summary["sd"] == pytest.approx(np.std(case.snrs, ddof=1))
        assert "mean_spectrum_snr" in case.summary


def test_four_overlays_have_ordered_matching_legends(
    notebook_results: Mapping[str, Any],
) -> None:
    count_labels = ["$N = 8$", "$N = 16$", "$N = 32$"]
    cutoff_labels = [r"$z_{\min} = 0.05$", r"$z_{\min} = 0.15$", r"$z_{\min} = 0.35$"]
    for group, labels in (
        ("num_events", count_labels),
        ("minimum_redshift", cutoff_labels),
    ):
        colors = []
        for quantity in ("snr", "sigma_h0"):
            figure = notebook_results[f"{group}_{quantity}_figure"]
            assert len(figure.axes) == 1
            axis = figure.axes[0]
            assert len(axis.get_lines()) == 3
            assert [text.get_text() for text in axis.get_legend().get_texts()] == labels
            assert axis.get_ylabel() == "Density"
            colors.append([line.get_color() for line in axis.get_lines()])
        assert colors[0] == colors[1]


def test_overlay_uses_common_ecdf_for_constant_cases(
    notebook_results: Mapping[str, Any],
) -> None:
    plot = notebook_results["plot_distribution_overlay"]
    figure = plot({"constant": [1.0], "varying": [1.0, 2.0, 3.0]}, xlabel="SNR")
    assert figure.axes[0].get_ylabel() == "Cumulative probability"
    assert len(figure.axes[0].get_lines()) == 2
    with pytest.raises(ValueError, match="non-empty finite"):
        plot({"invalid": [np.nan]}, xlabel="SNR")


def test_kde_shows_tails_and_interior_peak(notebook_results: Mapping[str, Any]) -> None:
    """Sample extrema must not act as reflecting physical boundaries."""
    values = np.random.default_rng(41).normal(loc=100.0, scale=2.0, size=1000)
    figure = notebook_results["plot_distribution_overlay"](
        {"normal": values}, xlabel="SNR"
    )
    axis = figure.axes[0]
    x, density = axis.get_lines()[0].get_data()
    assert axis.get_ylabel() == "Density"
    assert x[0] < values.min()
    assert x[-1] > values.max()
    assert density[0] < 0.01 * density.max()
    assert density[-1] < 0.01 * density.max()
    assert abs(x[np.argmax(density)] - 100.0) < 2.0
    assert np.trapezoid(density, x) == pytest.approx(1.0, abs=0.01)


def test_template_fisher_h0_prediction(notebook_results: Mapping[str, Any]) -> None:
    predict = notebook_results["fisher_h0_prediction"]
    prior = notebook_results["h0_prior"]
    prediction = predict(
        [100.0 / 80.0, 1.0, 100.0 / 120.0],
        [80.0, 100.0, 120.0],
        fiducial_h0=70.0,
        h0_prior=prior,
    )
    np.testing.assert_allclose(prediction.map_h0, [56.0, 70.0, 84.0])
    np.testing.assert_allclose(
        prediction.sigma_h0,
        np.array([56.0, 70.0, 84.0]) ** 2 / (70.0 * np.array([80.0, 100.0, 120.0])),
    )
    np.testing.assert_allclose(
        prediction.normalized_residuals(fiducial_h0=70.0), [-25.0, 0.0, 50.0 / 3.0]
    )
    constant = predict([1.0, 1.0], [100.0, 100.0], fiducial_h0=70.0, h0_prior=prior)
    np.testing.assert_array_equal(constant.map_h0, [70.0, 70.0])
    boundary = predict([10.0, 0.1], [100.0, 100.0], fiducial_h0=70.0, h0_prior=prior)
    np.testing.assert_array_equal(
        boundary.map_h0, [float(prior.low), float(prior.high)]
    )
    assert boundary.at_prior_boundary.all()
    with pytest.raises(ValueError, match="positive finite"):
        predict([0.0, 1.0], [100.0, 100.0], fiducial_h0=70.0, h0_prior=prior)


def test_model_statistics_fit_spectral_shapes(
    notebook_results: Mapping[str, Any],
) -> None:
    """A norm ratio cannot reproduce the fit when data and template shapes differ."""
    fit = notebook_results["template_amplitude_statistics"]
    template = np.array([[1.0, 2.0, 4.0, 100.0], [3.0, 1.0, 2.0, 200.0]])
    data = np.array([1.3, 1.7, 4.6, 500.0])
    scale = np.array([0.5, 0.4, 0.8, 1.0])
    band = np.array([True, True, True, False])
    amplitudes, snrs = fit(template, data, scale, band, fiducial_h0=70.0)
    weights = band / scale**2
    norms = np.sum(template**2 * weights, axis=-1)
    overlap = np.sum(data * template * weights, axis=-1)
    np.testing.assert_allclose(amplitudes, overlap / norms)
    np.testing.assert_allclose(snrs, np.sqrt(norms))
    data_snr = np.sqrt(np.sum(data**2 * weights))
    assert not np.allclose(amplitudes, data_snr / snrs)


def test_gaussian_mixture_combines_map_scatter_and_fisher_variance(
    notebook_results: Mapping[str, Any],
) -> None:
    density = notebook_results["gaussian_mixture_density"]
    grid = np.linspace(-7.0, 7.0, 10001)
    mixture = density(grid, [-1.0, 1.0], [0.3, 0.7])
    assert np.trapezoid(mixture, grid) == pytest.approx(1.0, abs=1e-8)
    mean = np.trapezoid(grid * mixture, grid)
    assert mean == pytest.approx(0.0, abs=1e-8)
    variance = np.trapezoid((grid - mean) ** 2 * mixture, grid)
    assert variance == pytest.approx(1.0 + (0.3**2 + 0.7**2) / 2, abs=1e-8)


def test_fisher_h0_overlay_uses_each_cached_case(
    notebook_results: Mapping[str, Any],
) -> None:
    predictions = notebook_results["num_events_h0_predictions"]
    cases = notebook_results["num_events_cases"]
    assert list(predictions) == list(cases)
    for count, prediction in predictions.items():
        case = cases[count]
        np.testing.assert_allclose(prediction.template_optimal_snr, case.snrs)
        np.testing.assert_allclose(
            prediction.sigma_h0,
            prediction.map_h0**2 / (case.metadata.fixed["H0"] * case.snrs),
        )

    from matplotlib.colors import to_rgba

    axis = notebook_results["num_events_h0_fisher_figure"].axes[0]
    lines = axis.get_lines()
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert labels == ["$N = 8$", "$N = 16$", "$N = 32$", r"$H_{0,\mathrm{fid}}$"]
    old_lines = notebook_results["num_events_snr_figure"].axes[0].get_lines()
    for line, old_line in zip(lines[:3], old_lines, strict=True):
        x, density = line.get_data()
        assert np.trapezoid(density, x) == pytest.approx(1.0, abs=1e-6)
        assert to_rgba(line.get_color()) == to_rgba(old_line.get_color())


def test_normalized_residuals_measure_template_scatter(
    notebook_results: Mapping[str, Any],
) -> None:
    """Normalization uses each draw's width, not mixture or ensemble SD."""
    residuals = notebook_results["num_events_h0_residuals"]
    summary = notebook_results["num_events_h0_residual_summary"].set_index("num_events")
    for count, prediction in notebook_results["num_events_h0_predictions"].items():
        case = notebook_results["num_events_cases"][count]
        h0 = case.metadata.fixed["H0"]
        np.testing.assert_allclose(
            residuals[count], (prediction.map_h0 - h0) / prediction.sigma_h0
        )
        assert summary.loc[count, "mean"] == pytest.approx(np.mean(residuals[count]))
        assert summary.loc[count, "sd"] == pytest.approx(
            np.std(residuals[count], ddof=1)
        )
        assert summary.loc[count, "rms"] == pytest.approx(
            np.sqrt(np.mean(residuals[count] ** 2))
        )
        assert summary.loc[count, "fraction_abs_gt_1"] == pytest.approx(
            np.mean(np.abs(residuals[count]) > 1)
        )

    axis = notebook_results["num_events_h0_residual_figure"].axes[0]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "$N = 8$",
        "$N = 16$",
        "$N = 32$",
        r"$\mathcal{N}(0,1)$",
    ]
    x, density = axis.get_lines()[3].get_data()
    np.testing.assert_allclose(density, np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi))
    assert axis.get_ylim()[1] > density.max()
