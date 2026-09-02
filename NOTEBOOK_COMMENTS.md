# Notebook feedback

## Notebook location

I decided to first put the notebooks in the root dir, but now I am not sure.
Should I move them to the paper dir along with the other notebooks? I guess it could be a good move,
because we could then factor out a bunch of common code and make the notebooks more concise. We lose the
self-contained nature (which is a plus), and we also lose the notebook-as-test thing. But I consider these
notebooks as experiments, so I guess they are part of the paper. What do you think?

## catalog_convergence

## Imports

Use the `warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")` for filtering the lal message

### Analytic vs sample mean

- I think that we should bump the frequency range for this section specifically.
The $\Omega_{GW}(f)$ spectrum has a maximum at $\mathcal{O}(1000)~\mathrm{Hz}$;
it would therefore be interesting to compare them at this larger frequency band.
- I think that we should compare more catalogs. We should have a base catalog
(which we should bump anyway, from 1024 to 8192) and subsample to 4096, 2048, 1024 and plot all of them,
including the residuals versus the analytic curve.
- The plot can be wrapped in a big function, which receives the analytic curve, frequency array, 
and catalog, and is responsible for the subsampling and plotting.

### Monte Carlo convergence

I don't quite understand what is being compared. It is the variance of Omega_gw for what? A fixed frequency bin?
I think that this plot should depend on frequency as well, to see which frequency bins get the higher variance.

## SNR

Your comment on the relative contribution of each frequency bin for the SNR is quite important.
I think that it deserves a separate notebook!

We should do the following:
- Plot the SNR integrand (e..g $S_h^2(f) / S^2_{eff}(f)$) per frequency bin,
as well as the cumulative SNR (up to a given frequency), to see where it saturates.
- Do it as a function of the detector network (all three ET combinations in one plot),
then ET-2L-par vs ET-2L-par + CE.
- Do it as a function of the frequency bin width df to see how finer or coarser grids
affect the calculation (this is done already in the notebook)
- Do it for the analytic spectral density, and for a catalog which uses IMRPhenomXAS_NRTidalv3
to see whether the merger+ringdown + tidal effects are important here

### Log-likelihood ratio

No need for the normalization subplot.

## mcmc_example_models

- We need corner plots! Just like the corner plots we setup in the paper notebooks.
We can use arviz for this, since it integrates well with numpyro
- 





