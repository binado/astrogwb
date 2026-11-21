Public API migration
====================

Detector networks
-----------------

Pass a mapping from detector names to ``Detector`` instances to the response,
noise, estimation, and likelihood APIs. Use ``load_network(name)`` to resolve
a registered network. ``network_pairs(network)`` replaces ``pair_names()`` and
gives the canonical order of pair rows.

If you stored a pair array from an unsorted detector sequence, use the detector
names in their original order to recover the old row labels. For a spectrum
whose last two axes are pair and frequency, reorder the rows as follows.

.. code-block:: python

   from itertools import combinations

   from astrogwb.detector import network_pairs

   old_pairs = [tuple(sorted(pair)) for pair in combinations(old_detector_names, 2)]
   indices = [old_pairs.index(pair) for pair in network_pairs(network)]
   spectra = spectra[..., indices, :]

Mappings of observed spectra keyed by pair can be passed directly. Recompute
precomputed generalized ORFs using the canonical network mapping.

``point_source_overlap_reduction_function()`` replaces
``overlap_reduction_function()`` for one sky position, polarization, and
inclination. ``pair_response()`` stacks those ORFs for a whole network. Use
``map_over_pairs()`` for pair-keyed results, including custom observables.
Call the point-source function on explicit source draws to replace
``sample_pair_response()``. Use ``map_over_pairs()`` with the isotropic ORF
to obtain the distinct baselines previously available in the symmetric ORF matrix.

Waveforms and background predictions
------------------------------------

Construct ``WaveformConfig`` with the same fields as ``WaveformMetadata`` and
pass it to ``waveform_generator()``. Use ``generator.frequencies`` for the
frequency grid and ``generator.intensity(sources)`` for the former callable and
``generate_batch()``. Intensity is now shaped ``(sources, frequencies)``.
The generator's ``metadata`` attribute is now ``config``.
For one source, pass length-one columns and take the first row. Use
``generator.polarizations(sources)`` for the former
``generate_polarizations_batch()``. Both polarizations use frequency last.

Import ``ReferenceCatalog`` and ``CatalogModel`` from
``astrogwb.background.importance``. Build the catalog with
``ReferenceCatalog.from_samples(...)`` in place of ``reference_catalog(...)``.
For standalone intensity calculations, use the generator directly. Catalog
intensity on disk still has the ``(frequencies, sources)`` layout.
Use ``inspiral_spectral_density()`` in place of ``analytic_spectral_density()``.

Simulation, noise, and likelihoods
----------------------------------

Replace ``network_spectra_ensemble(root, seed, count, names, start=start)``
with ``read_background_ensemble(paths, names)``, where ``paths`` contains
``root / f"{seed}-{i}.h5"`` for each index in ``range(start, count)``.
Index the result by network name. Existing background files and CLI options
use the same format and units.

Observation times stay in years: the likelihoods, ``effective_psd`` noise
scales and ``characteristic_noise()`` take ``observation_time`` in years
(``characteristic_noise()`` by keyword), and convert with
``years_to_seconds()``. The SNR functions and ``project_background()`` take
durations in seconds under the name ``observation_seconds``. Use
``pair_cross_power_std()`` and ``network_cross_power_std()`` in place of
``pair_bin_scale()`` and ``gaussian_bin_scale()``.

Networks are mappings keyed by detector name, and ``network_pairs()`` raises
``ValueError`` for fewer than two detectors or a key that differs from its
detector's name. Use ``network_overlap_reduction_function()`` for the isotropic
ORF of every pair, ``(pairs, frequencies)``, and ``pair_weights()`` for the
optimal cross-correlation weights.

Recover the four fields previously returned by ``network_noise()`` per network
from a saved draw and its configuration as follows.

.. code-block:: python

   import jax.numpy as jnp
   from astrogwb.detector import (
       effective_psd,
       isotropic_overlap_reduction_function,
       map_over_pairs,
       network_cross_power_std,
       network_pairs,
       pair_cross_power_std,
   )
   from astrogwb.utils import stack_pairs, years_to_seconds

   frequencies = draw.frequencies
   network = config.networks[name]
   years = config.observation_seconds / years_to_seconds(1.0)
   pair_overlap = stack_pairs(
       map_over_pairs(
           lambda first, second: isotropic_overlap_reduction_function(
               frequencies, first, second
           ),
           network,
       ),
       network_pairs(network),
   )
   pair_scale = pair_cross_power_std(frequencies, network, years)
   psd = effective_psd(frequencies, network)
   scale = network_cross_power_std(jnp.asarray(psd), years, frequencies)

Use ``likelihood.masked_generalized_orfs`` to inspect the matrices on the
counted entries. The likelihood constructor still takes full-grid matrices
through ``generalized_orfs=``.
