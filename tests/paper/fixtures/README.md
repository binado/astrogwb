These fixtures were captured before the gwmock convention migration, from
commit `657fc6182a91750084200c3a61636023d29ab937` with the locked dependencies.

`detector_networks_before_gwmock.npz` contains the pairwise analytic ORFs and
effective PSDs for all six networks in `config/detectors.toml`. Its frequency
grid includes zero and 1,000 logarithmically spaced points from 0.01 to 10,000
Hz, covering the low-frequency series, oscillatory ORFs, and PSD band edges.
Regression checks require unchanged finite masks and relative tolerance
`1e-10` (absolute tolerance `1e-10` for ORFs, zero for PSDs).

`legacy_detector_registry.json` is the complete resolved registry in the old
degree fields, including PSD references, labels, and network membership. It
exercises saved-config migration without consulting new packaged defaults.
These are historical references; do not regenerate them from canonical geometry.
