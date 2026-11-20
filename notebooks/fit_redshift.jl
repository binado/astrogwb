### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 47421621-2f50-55b3-bed2-fc7b33f97b4c
md"""
# Madau-Dickinson redshift fit (Turing)

Julia mirror of `fit_redshift.py`. Fit the Madau-Dickinson shape
(`gamma`, `kappa`, `z_peak`) to every redshift of a complete, all-sky catalog
with no selection effects, at fixed ``H_0 = 67.66`` and ``\Omega_m = 0.3096``,
then infer the local merger rate from the catalog size and duration.

With the cosmology fixed,

```math
\log p(z_i \mid \theta) = \log\frac{\mathrm{d}V_c/\mathrm{d}z(z_i)}{1+z_i}
+ \log\psi(z_i;\theta) - \log Z(\theta), \qquad
\psi(z) = \frac{(1+z)^\gamma}{1 + \left(\frac{1+z}{1+z_p}\right)^{\gamma+\kappa}}.
```

The volume term does not depend on ``\theta`` and is dropped, ``\log\psi`` is
closed form per event, and ``Z(\theta)`` is a Gauss-Legendre quadrature in
``u = \log(1+z)``, where ``\mathrm{d}z/(1+z) = \mathrm{d}u``. The volume at the
nodes is computed once. Chains run on separate threads (`MCMCThreads()`), so
start Pluto or Julia with at least four threads.
"""

# ╔═╡ f94007ab-c06f-5b4d-ae9a-cae9ec53dd6d
begin
    # BackgroundCosmology.jl is not registered, so this notebook manages its own
    # environment instead of using Pluto's package manager.
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(url = "https://github.com/binado/BackgroundCosmology.jl"),
        Pkg.PackageSpec(name = "AdvancedHMC"),
        Pkg.PackageSpec(name = "CairoMakie"),
        Pkg.PackageSpec(name = "FastGaussQuadrature"),
        Pkg.PackageSpec(name = "ForwardDiff"),
        Pkg.PackageSpec(name = "HDF5"),
        Pkg.PackageSpec(name = "LogExpFunctions"),
        Pkg.PackageSpec(name = "MCMCChains"),
        Pkg.PackageSpec(name = "PairPlots"),
        Pkg.PackageSpec(name = "Turing"),
    ])
    using AdvancedHMC: DenseEuclideanMetric
    using BackgroundCosmology
    using CairoMakie
    using FastGaussQuadrature: gausslegendre
    using ForwardDiff
    using HDF5
    using LogExpFunctions: log1pexp, logsumexp
    using MCMCChains
    using Random
    using Statistics
    using PairPlots
    using Turing
end

# ╔═╡ eb882e8c-8a40-5dce-89b8-c91c99e7ef0c
Threads.nthreads()

# ╔═╡ 4d65d84f-d0ff-5e9b-b60a-438112d5d5a3
begin
    # The ET-0197A-23 description states a one-year catalog on z in [0, 15].
    catalog_path = joinpath(@__DIR__, "..", "data", "18321_1yrCatalogBNS.h5")
    minimum_redshift = 0.0
    maximum_redshift = 15.0
    observation_years = 1.0
    n_nodes = 64
    cosmo = FlatLCDM(67.66, 0.3096)
    parameter_names = (:gamma, :kappa, :z_peak)
end

# ╔═╡ 8a64be90-2f20-56ec-bf87-f42ad637ada1
"""
    log_shape(u, gamma, kappa, z_peak)

Log of the Madau-Dickinson shape at `u = log(1 + z)`, without its normalization.
"""
log_shape(u, gamma, kappa, z_peak) =
    gamma * u - log1pexp((gamma + kappa) * (u - log1p(z_peak)))

# ╔═╡ 3bc7c6ae-b26d-543e-8df6-401735e96b7d
"""
    volume_quadrature(cosmo, minimum, maximum, n)

Gauss-Legendre nodes in `u = log(1 + z)` and the log of the node weights times
the all-sky `dVc/dz` in Mpc^3, so that `Z = sum(exp.(log_shape.(u) .+ logw))`.
"""
function volume_quadrature(cosmo, minimum, maximum, n)
    x, w = gausslegendre(n)
    lo, hi = log1p(minimum), log1p(maximum)
    u = @. (hi + lo) / 2 + (hi - lo) / 2 * x
    volume = distance_and_volume_grid(cosmo, expm1.(u)).differential_comoving_volume
    return u, @. log((hi - lo) / 2 * w * volume)
end

# ╔═╡ 933108ed-d17d-516b-a383-570028a1e8fd
"""
    log_normalization(nodes, logw, gamma, kappa, z_peak)

Log of the quadrature normalization `Z` in Mpc^3.
"""
log_normalization(nodes, logw, gamma, kappa, z_peak) =
    logsumexp(log_shape.(nodes, gamma, kappa, z_peak) .+ logw)

# ╔═╡ 1b3f76b0-7502-5c4b-bfc8-10beec440d28
"""
    log_likelihood(gamma, kappa, z_peak, u, nodes, logw)

Full-catalog log likelihood, up to a theta-independent constant, for events
`u = log1p.(z)`.
"""
function log_likelihood(gamma, kappa, z_peak, u, nodes, logw)
    total = zero(gamma * kappa * z_peak)
    @inbounds @simd for x in u
        total += log_shape(x, gamma, kappa, z_peak)
    end
    return total - length(u) * log_normalization(nodes, logw, gamma, kappa, z_peak)
end

# ╔═╡ 1d1f95d3-3b65-57f4-84da-fecd21584ed6
"""
    unit_total_rate(gamma, kappa, z_peak, nodes, logw)

Total merger rate per observer-frame year for a local rate of one
`Gpc^-3 yr^-1`, which fixes `psi(0) = 1`.
"""
function unit_total_rate(gamma, kappa, z_peak, nodes, logw)
    log_local = log1pexp(-(gamma + kappa) * log1p(z_peak))
    # Mpc^3 to Gpc^3.
    return 1.0e-9 * exp(log_local + log_normalization(nodes, logw, gamma, kappa, z_peak))
end

# ╔═╡ cab79d66-4b68-51f5-a935-a33eae928b62
@model function redshift_model(u, nodes, logw)
    gamma ~ Uniform(0.0, 10.0)
    kappa ~ Uniform(0.0, 10.0)
    z_peak ~ Uniform(0.0, 10.0)
    @addlogprob! log_likelihood(gamma, kappa, z_peak, u, nodes, logw)
end

# ╔═╡ 52c25605-7b39-5566-aea6-a6de74d2dc11
observed = h5open(catalog_path, "r") do file
    read(file["z"])
end

# ╔═╡ 98bcc083-a584-53cb-9294-b3b856366045
log_one_plus_z = log1p.(observed)

# ╔═╡ 028bdbf9-3892-54a7-a9ff-6a7f62089e94
nodes, log_node_weights = volume_quadrature(cosmo, minimum_redshift, maximum_redshift, n_nodes)

# ╔═╡ b44f2fbc-68bb-58e4-9eae-8f5ddae62280
md"""
Loaded **$(length(observed))** redshifts in
[$(round(minimum(observed); sigdigits = 4)), $(round(maximum(observed); sigdigits = 4))].
"""

# ╔═╡ 8b4d3cd8-ba6b-5907-8e0d-4442bd84ef8c
let
    # Quadrature convergence and the cost of one ForwardDiff gradient.
    theta = [2.7, 2.9, 1.9]
    fine_u, fine_logw = volume_quadrature(cosmo, minimum_redshift, maximum_redshift, 256)
    relative_error =
        unit_total_rate(theta..., nodes, log_node_weights) /
        unit_total_rate(theta..., fine_u, fine_logw) - 1
    objective = t -> log_likelihood(t..., log_one_plus_z, nodes, log_node_weights)
    ForwardDiff.gradient(objective, theta)
    milliseconds = 1.0e3 * (@elapsed for _ in 1:20
        ForwardDiff.gradient(objective, theta)
    end) / 20
    md"""
    Relative change of the unit total rate from $(n_nodes) to 256 nodes:
    $(round(relative_error; sigdigits = 3)). Gradient cost:
    **$(round(milliseconds; sigdigits = 3)) ms**.
    """
end

# ╔═╡ 72ec11f1-fa03-5327-acaa-d8edc6a3c8c7
model = redshift_model(log_one_plus_z, nodes, log_node_weights)

# ╔═╡ f388fd26-e487-53d8-b978-d10b1edf9949
chain, sampling_seconds = let
    start = time()
    result = sample(
        Xoshiro(42),
        model,
        NUTS(200, 0.9; metricT = DenseEuclideanMetric),
        MCMCThreads(),
        1000,
        4;
        progress = true,
    )
    result, time() - start
end

# ╔═╡ 711d15cc-88a2-5ee9-9b29-0bfbc0ac8d4d
md"Sampled in **$(round(sampling_seconds; digits = 1)) s** on $(Threads.nthreads()) threads."

# ╔═╡ 12eab8c8-2f71-556c-a85e-4bbfdec8ca7a
summarystats(chain)

# ╔═╡ 65008ace-7de2-58f8-a239-4e56ee7aa0bd
quantile(chain; q = [0.05, 0.5, 0.95])

# ╔═╡ 319079bd-2342-5eda-9f04-4ec37d338578
md"Divergences: **$(Int(sum(chain[:numerical_error])))**."

# ╔═╡ 0e7c79d8-7706-5257-966f-14012418f54c
let
    figure = Figure(; size = (800, 600))
    for (row, name) in enumerate(parameter_names)
        axis = Axis(figure[row, 1]; ylabel = string(name), xlabel = "Draw")
        for draws in eachcol(chain[name])
            lines!(axis, draws; linewidth = 0.5)
        end
    end
    figure
end

# ╔═╡ 9420c2a4-efe1-5d8a-a4fe-f5d1243408ff
pairplot(chain[collect(parameter_names)])

# ╔═╡ ecb275da-33b9-5007-9e00-44e8689f1128
theta_samples = hcat((vec(chain[p]) for p in parameter_names)...)

# ╔═╡ bba07aa7-e7a1-5a94-bedb-2262787d2036
let
    log_likelihoods = [
        log_likelihood(t..., log_one_plus_z, nodes, log_node_weights)
        for t in eachrow(theta_samples)
    ]
    best = theta_samples[argmax(log_likelihoods), :]
    grid = collect(range(minimum_redshift, maximum_redshift; length = 500))
    log_volume =
        log.(distance_and_volume_grid(cosmo, grid).differential_comoving_volume ./ (1 .+ grid))
    density_on_grid(t) = exp.(
        log_shape.(log1p.(grid), t...) .+ log_volume .-
        log_normalization(nodes, log_node_weights, t...)
    )
    densities = reduce(hcat, density_on_grid(t) for t in eachrow(theta_samples))
    interval = mapslices(d -> quantile(d, [0.05, 0.95]), densities; dims = 2)
    figure = Figure()
    axis = Axis(
        figure[1, 1]; xlabel = "Redshift", ylabel = "Probability density",
        limits = (extrema(grid), nothing),
    )
    hist!(axis, observed; bins = 50, normalization = :pdf, color = (:gray, 0.35), label = "Catalog")
    band!(axis, grid, interval[:, 1], interval[:, 2]; alpha = 0.35, label = "90% band")
    lines!(axis, grid, density_on_grid(best); label = "Highest-likelihood draw")
    axislegend(axis)
    figure
end

# ╔═╡ 30858a89-a805-5d3f-9f0c-95e3bf6bb44e
let
    local_rates = [
        length(observed) / (observation_years * unit_total_rate(t..., nodes, log_node_weights))
        for t in eachrow(theta_samples)
    ]
    low, median, high = quantile(local_rates, [0.05, 0.5, 0.95])
    md"""
    ### Local merger rate

    ```math
    R_0(\theta) = \frac{N}{T\,\mathcal{R}_{\rm unit}(\theta)}
    ```

    Median **$(round(median; sigdigits = 4))**, equal-tail 90% interval
    **[$(round(low; sigdigits = 4)), $(round(high; sigdigits = 4))]**
    ``\mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}``. The interval reflects shape
    uncertainty only; the count ``N`` is fixed.
    """
end

# ╔═╡ Cell order:
# ╠═47421621-2f50-55b3-bed2-fc7b33f97b4c
# ╠═f94007ab-c06f-5b4d-ae9a-cae9ec53dd6d
# ╠═eb882e8c-8a40-5dce-89b8-c91c99e7ef0c
# ╠═4d65d84f-d0ff-5e9b-b60a-438112d5d5a3
# ╠═8a64be90-2f20-56ec-bf87-f42ad637ada1
# ╠═3bc7c6ae-b26d-543e-8df6-401735e96b7d
# ╠═933108ed-d17d-516b-a383-570028a1e8fd
# ╠═1b3f76b0-7502-5c4b-bfc8-10beec440d28
# ╠═1d1f95d3-3b65-57f4-84da-fecd21584ed6
# ╠═cab79d66-4b68-51f5-a935-a33eae928b62
# ╠═52c25605-7b39-5566-aea6-a6de74d2dc11
# ╠═98bcc083-a584-53cb-9294-b3b856366045
# ╠═028bdbf9-3892-54a7-a9ff-6a7f62089e94
# ╠═b44f2fbc-68bb-58e4-9eae-8f5ddae62280
# ╠═8b4d3cd8-ba6b-5907-8e0d-4442bd84ef8c
# ╠═72ec11f1-fa03-5327-acaa-d8edc6a3c8c7
# ╠═f388fd26-e487-53d8-b978-d10b1edf9949
# ╠═711d15cc-88a2-5ee9-9b29-0bfbc0ac8d4d
# ╠═12eab8c8-2f71-556c-a85e-4bbfdec8ca7a
# ╠═65008ace-7de2-58f8-a239-4e56ee7aa0bd
# ╠═319079bd-2342-5eda-9f04-4ec37d338578
# ╠═0e7c79d8-7706-5257-966f-14012418f54c
# ╠═9420c2a4-efe1-5d8a-a4fe-f5d1243408ff
# ╠═ecb275da-33b9-5007-9e00-44e8689f1128
# ╠═bba07aa7-e7a1-5a94-bedb-2262787d2036
# ╠═30858a89-a805-5d3f-9f0c-95e3bf6bb44e
