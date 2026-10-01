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
        default_cache_keys = {
            path.stem for path in Path(definitions["cache_dir"]).glob("*.h5")
        }
        _, poisson_definitions = app.run(defs={"data_reference": "poisson"})

        from astrogwb.catalog import SpectrumGenerator

        def forbid_generation(self: SpectrumGenerator, metadata: Any) -> Any:
            pytest.fail(f"repeat execution generated spectra for {metadata.key()}")

        patch.setattr(SpectrumGenerator, "__call__", forbid_generation)
        _, cached_definitions = app.run()
        _, cached_poisson = app.run(defs={"data_reference": "poisson"})
        np.testing.assert_array_equal(
            poisson_definitions["data_spectrum"], cached_poisson["data_spectrum"]
        )
        for group in ("num_events_cases", "minimum_redshift_cases"):
            for key, case in definitions[group].items():
                np.testing.assert_array_equal(
                    case.snrs, cached_definitions[group][key].snrs
                )

        try:
            yield {
                **dict(definitions),
                "poisson_results": poisson_definitions,
                "default_cache_keys": default_cache_keys,
            }
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
    assert len(notebook_results["default_cache_keys"]) == 5
    data = notebook_results["data_catalog"]
    assert notebook_results["data_reference"] == "largest_mean"
    assert data is counts[max(counts)].catalog
    np.testing.assert_allclose(
        notebook_results["data_spectrum"], np.mean(data.spectral_density, axis=0)
    )
    assert notebook_results["data_snr"] == pytest.approx(
        counts[max(counts)].summary["mean_spectrum_snr"]
    )
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


def test_poisson_reference_remains_available(
    notebook_results: Mapping[str, Any],
) -> None:
    results = notebook_results["poisson_results"]
    data = results["data_catalog"]
    assert results["data_reference"] == "poisson"
    assert data.metadata == results["data_metadata"]
    assert data.metadata.population.seed != results["base_metadata"].population.seed
    assert data.metadata.num_draws == 1
    np.testing.assert_array_equal(results["data_spectrum"], data.spectral_density[0])


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


def _example_catalog(
    notebook_results: Mapping[str, Any], spectra: Any, frequencies: Any
) -> Any:
    """Replace only the arrays in a valid fixed-hyperparameter smoke catalog."""
    from dataclasses import replace

    rows = np.asarray(spectra, dtype=np.float64)
    original = next(iter(notebook_results["num_events_cases"].values())).catalog
    metadata = original.metadata.model_copy(update={"num_draws": rows.shape[0]})
    return replace(
        original,
        spectral_density=rows,
        frequencies=np.asarray(frequencies, dtype=np.float64),
        n_events=np.full(rows.shape[0], metadata.num_events),
        total_merger_rate=np.full(rows.shape[0], original.total_merger_rate[0]),
        hyperparameters={
            name: np.full(rows.shape[0], value)
            for name, value in metadata.fixed.items()
        },
        _metadata=metadata,
    )


def test_pointwise_spectrum_statistics(notebook_results: Mapping[str, Any]) -> None:
    compute = notebook_results["compute_spectrum_statistics"]
    rows = np.array([[1, 2, 0, 5], [2, 4, 0, 5], [3, 8, 0, 5]], dtype=float)
    catalog = _example_catalog(notebook_results, rows, [2, 4, 8, 16])
    statistics = compute(catalog)
    np.testing.assert_allclose(statistics.mean, [2, 14 / 3, 0, 5])
    np.testing.assert_allclose(statistics.variance, [1, 28 / 3, 0, 0])
    np.testing.assert_allclose(
        statistics.standard_deviation, [1, np.sqrt(28 / 3), 0, 0]
    )
    residuals = rows[:, [0, 1, 3]] / rows[:, [0, 1, 3]].mean(axis=0) - 1
    np.testing.assert_allclose(
        statistics.relative_variance[[0, 1, 3]], np.var(residuals, axis=0, ddof=1)
    )
    assert np.isnan(statistics.relative_variance[2])
    for group in notebook_results["spectrum_statistics"].values():
        for case in group.values():
            positive = case.mean > 0
            np.testing.assert_allclose(
                case.relative_variance[positive],
                case.variance[positive] / case.mean[positive] ** 2,
                rtol=1e-12,
            )


def test_spectrum_statistics_reject_invalid_ensembles(
    notebook_results: Mapping[str, Any],
) -> None:
    from dataclasses import replace

    from astrogwb.metadata import PriorSpec

    compute = notebook_results["compute_spectrum_statistics"]
    for rows, message in (
        ([[1, 2]], "at least two"),
        ([[1, -2], [2, 3]], "finite and nonnegative"),
        ([[1, np.nan], [2, 3]], "finite and nonnegative"),
    ):
        with pytest.raises(ValueError, match=message):
            compute(_example_catalog(notebook_results, rows, [2, 4]))
    catalog = _example_catalog(notebook_results, [[1, 2], [2, 3]], [2, 4])
    metadata = catalog.metadata.model_copy(
        update={
            "hyperparameters": {
                **catalog.metadata.hyperparameters,
                "H0": PriorSpec(dist="Uniform", kwargs={"low": 50.0, "high": 90.0}),
            }
        }
    )
    with pytest.raises(ValueError, match="fixed hyperparameters"):
        compute(replace(catalog, _metadata=metadata))


def test_frequency_correlation_preserves_rows_and_masks_constant_bins(
    notebook_results: Mapping[str, Any],
) -> None:
    compute = notebook_results["compute_frequency_correlation"]
    rows = [[1, 2, 0, 5], [2, 4, 0, 5], [3, 8, 0, 5]]
    catalog = _example_catalog(notebook_results, rows, [2, 4, 8, 16])
    frequencies, correlation = compute(
        catalog, minimum_frequency=2, maximum_frequency=16
    )
    np.testing.assert_array_equal(frequencies, catalog.frequencies)
    np.testing.assert_allclose(
        correlation[:2, :2], np.corrcoef(np.array(rows)[:, :2].T)
    )
    assert np.isnan(correlation[2:, :]).all()
    assert np.isnan(correlation[:, 2:]).all()
    full_frequencies = np.geomspace(2, 2048, 150)
    rows = np.arange(1, 5)[:, None] * np.arange(1, 151)[None, :]
    catalog = _example_catalog(notebook_results, rows, full_frequencies)
    frequencies, correlation = compute(
        catalog, minimum_frequency=4, maximum_frequency=1024
    )
    assert 2 <= frequencies.size <= 64
    assert frequencies[0] >= 4 and frequencies[-1] <= 1024
    assert np.all(np.diff(frequencies) > 0)
    np.testing.assert_allclose(correlation, 1)
    with pytest.raises(ValueError, match="at least two frequency bins"):
        compute(catalog, minimum_frequency=3, maximum_frequency=3.01)


def test_sensitivity_uses_full_nonuniform_grid(
    notebook_results: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from astrogwb.frequency import bin_widths
    from astrogwb.utils import years_to_seconds

    frequencies = np.array([2.0, 3.0, 7.0, 20.0])
    catalog = _example_catalog(
        notebook_results, [[1, 2, 3, 4], [2, 3, 4, 5]], frequencies
    )
    compute = notebook_results["compute_network_sensitivity"]
    noise = np.array([10.0, 20.0, 30.0, np.inf])
    monkeypatch.setitem(compute.__globals__, "effective_psd", lambda *args: noise)
    sensitivity = compute(
        catalog,
        notebook_results["registry"],
        notebook_results["network"],
        minimum_frequency=3,
        maximum_frequency=7,
    )
    seconds = years_to_seconds(catalog.observation_time)
    np.testing.assert_array_equal(sensitivity.band, [False, True, True, False])
    np.testing.assert_allclose(
        sensitivity.per_bin,
        noise / np.sqrt(2 * seconds * np.asarray(bin_widths(frequencies))),
    )
    np.testing.assert_allclose(
        sensitivity.per_log_frequency, noise / np.sqrt(2 * seconds * frequencies)
    )
    assert not np.allclose(
        sensitivity.per_bin[1:3],
        noise[1:3] / np.sqrt(2 * seconds * np.asarray(bin_widths(frequencies[1:3]))),
        rtol=1e-3,
        atol=0,
    )
    with pytest.raises(ValueError, match="no spectrum bins"):
        compute(
            catalog,
            notebook_results["registry"],
            notebook_results["network"],
            minimum_frequency=30,
            maximum_frequency=40,
        )


def test_spectrum_comparison_figures_reuse_case_order_and_colors(
    notebook_results: Mapping[str, Any],
) -> None:
    from matplotlib.colors import to_rgba

    for panel, prefix in enumerate(("num_events", "minimum_redshift")):
        snr_axis = notebook_results[f"{prefix}_snr_figure"].axes[0]
        labels = [text.get_text() for text in snr_axis.get_legend().get_texts()]
        colors = [to_rgba(line.get_color()) for line in snr_axis.get_lines()]
        axes = [
            notebook_results["spectrum_mean_figure"].axes[panel],
            notebook_results["spectrum_relative_variance_figure"].axes[panel],
            *notebook_results[f"{prefix}_spectrum_sensitivity_figure"].axes,
        ]
        for axis in axes:
            assert axis.get_xscale() == axis.get_yscale() == "log"
            assert [text.get_text() for text in axis.get_legend().get_texts()][
                :3
            ] == labels
            assert [
                to_rgba(line.get_color()) for line in axis.get_lines()[:3]
            ] == colors
        sensitivity_axes = axes[2:]
        assert len(sensitivity_axes) == 2
        assert "per-bin" in sensitivity_axes[0].get_title()
        assert "Per-e-fold" in sensitivity_axes[1].get_title()
        for axis in sensitivity_axes:
            assert len(axis.get_lines()) == 4
            assert axis.get_lines()[-1].get_linestyle() == "--"
        correlation_figure = notebook_results[f"{prefix}_frequency_correlation_figure"]
        assert len(correlation_figure.axes) == 4  # Three cases plus shared colorbar.
        assert [axis.get_title() for axis in correlation_figure.axes[:3]] == labels
        for axis in correlation_figure.axes[:3]:
            assert axis.collections[0].get_clim() == (-1, 1)


def test_logarithmic_comparison_masks_undefined_and_zero_values(
    notebook_results: Mapping[str, Any],
) -> None:
    compute = notebook_results["compute_spectrum_statistics"]
    catalog = _example_catalog(notebook_results, [[1, 0, 5], [3, 0, 5]], [2, 4, 8])
    statistics = compute(catalog)
    plot = notebook_results["plot_relative_variance"]
    figure = plot({"Example": {"case": statistics}}, np.array([True, True, True]))
    x, y = figure.axes[0].get_lines()[0].get_data()
    np.testing.assert_array_equal(x, [2, 4, 8])
    assert y[0] == pytest.approx(0.5)
    assert np.isnan(y[1:]).all()
