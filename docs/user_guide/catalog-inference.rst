Predicting a background and evaluating a likelihood
===================================================

Set ``JAX_ENABLE_X64=1`` before starting Python to run the examples
in double precision. Run the code blocks in order in the same session.

A reference catalog places caller-drawn intrinsic sources at one reference redshift,
face-on, and generates their intensities. The caller supplies the source model, the
proposal log density of the draws and the reference redshift:

.. code:: python

   import jax
   import numpy as np

   from astrogwb.background.importance import CatalogModel, ReferenceCatalog
   from astrogwb.populations import coba
   from astrogwb.utils.numpyro import sample_and_logprob
   from astrogwb.waveform import WaveformConfig, waveform_generator

   params = {
       "H0": 67.66,
       "Omega_m": 0.3096,
       "gamma": 2.35,
       "kappa": 4.4,
       "z_peak": 2.14,
       "local_merger_rate": 240.0,
       "minimum_mass": 1.0,
       "mass_width": 1.5,
   }
   waveform = WaveformConfig(
       approximant="AnalyticInspiral",
       minimum_frequency=2.0,
       maximum_frequency=2048.0,
       reference_frequency=2.0,
       sampling_frequency=4096.0,
       frequency_resolution=1.0,
   )
   mass_sites = ("source_frame_mass_1", "source_frame_mass_2")
   samples, proposal_log_prob = sample_and_logprob(
       coba.source_model(params, mass_model="uniform"),
       jax.random.key(42),
       num_samples=1024,
       density_sites=mass_sites,
   )
   catalog = ReferenceCatalog.from_samples(
       waveform_generator(waveform),
       samples,
       intrinsic_sites=mass_sites,
       proposal_log_prob=proposal_log_prob,
       reference_redshift=0.01,
       parameters=params,
   )

A spectrum prediction uses the reference catalog, with every source placed at a common
reference redshift and face-on, so that a source at any redshift is a rescaling of its
reference power. Intrinsic parameters are reweighted to a target population; redshift is
integrated by quadrature. The proposal and target log densities must use the same
variables and measure. Here the spin and tidal distributions are shared, so their
densities cancel and only the ordered mass pair enters the ratio.

.. code:: python

   import jax.numpy as jnp
   import numpy as np

   from astrogwb.rates import madau_dickinson_rate


   def ordered_mass_log_prob(intrinsic, target):
       primary, secondary = (intrinsic[name] for name in mass_sites)
       low, width = target["minimum_mass"], target["mass_width"]
       support = (low <= secondary) & (secondary <= primary) & (primary <= low + width)
       return jnp.where(support, jnp.log(2.0) - 2.0 * jnp.log(width), -jnp.inf)


   def merger_rate(redshift, target):
       return madau_dickinson_rate(
           redshift,
           target["gamma"],
           target["kappa"],
           target["z_peak"],
           target["local_merger_rate"],
       )


   frequencies = np.arange(10.0, 200.0, 2.0)
   model = CatalogModel.from_catalog(
       catalog,
       merger_rate=merger_rate,
       log_prob=ordered_mass_log_prob,
       frequencies=frequencies,
       num_redshift_nodes=32,
       minimum_redshift=0.01,
       maximum_redshift=20.0,
   )
   prediction = model(
       {**params, "mass_width": 1.2, "H0": 70.0, "xi_0": 1.2, "xi_n": 1.91},
       shot_noise=True,
   )

When the target intrinsic density matches the proposal throughout the analysis, pass
``target_matches_proposal=True`` in place of ``log_prob`` to cache the unit-weight
moments.

``prediction.spectral_density`` is the background’s mean, :math:`S_h = 2\mathcal
R\langle I\rangle`, and ``prediction.intensity_second_moment`` is :math:`4\mathcal
R\langle I I'\rangle`, the raw second moment that sets the shot noise of a Poisson
catalog. A detector network’s likelihood of its per-pair cross powers is the dense
Gaussian with covariance ``Sigma + intensity_second_moment * connected_overlap / T``
(elementwise), with ``Sigma`` the detector noise, ``connected_overlap`` the network’s
sky and polarization average and ``T`` the observation time. The likelihood builds the
network from the detectors; the data are keyed by the sorted detector-name pair:

.. code:: python

   from astrogwb.constants import SECONDS_PER_YEAR
   from astrogwb.detector import load_detector
   from astrogwb.inference import DenseGaussianLikelihood

   # Placeholder observations on the model frequency grid.
   c_12 = c_13 = c_23 = np.zeros_like(frequencies)

   likelihood = DenseGaussianLikelihood(
       model,
       {name: load_detector(name) for name in ("E1", "E2", "E3")},
       frequencies=frequencies,
       observation_seconds=SECONDS_PER_YEAR,
       observed={("E1", "E2"): c_12, ("E1", "E3"): c_13, ("E2", "E3"): c_23},
   )
   log_likelihood, extras = likelihood.log_likelihood(params)

``likelihood.masked_generalized_orfs`` holds the network's four weighted generalized ORFs and
``likelihood.covariance_components(params)`` splits the covariance into its
detector-noise, count, intensity and polarization terms. Its no-shot-noise limit,
the diagonal detector noise alone, is ``DiagonalGaussianLikelihood``, which takes the
same arguments without ``key``, ``num_draws`` and ``generalized_orfs``.

Any pytree implementing ``SpectralDensityModel`` works in place of a catalog model, for
example a ``jax.tree_util.Partial`` of a module-level function returning a
``SpectralDensityPrediction``.
