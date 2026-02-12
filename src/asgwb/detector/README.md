# Noise sensitivity curves for interferometric detectors

All noise curves are two-column files, with frequency in the first column and PSD in the second.
To include a file that contains an ASD curve, use the [`asd_to_psd.sh`](./asd_to_psd.sh) helper script.

## Further references

- [Bilby PSD repository](https://github.com/bilby-dev/bilby/tree/main/bilby/gw/detector/noise_curves)
- [GWFast PSD repository](https://github.com/CosmoStatGW/gwfast/blob/master/psds/PSDs_README.md)

## Available detectors

### Einstein Telescope

- [ET_D](./noise_curves/ET-000A-18_ETD_hlf_psd.txt): ET_D sensitivity curve with combined HF and LF interferometers
- [ET_L_COBA_10km](./noise_curves/ET_COBA_10km_psd.txt): sensitivity for 10km L-shaped detector from COBA study
- [ET_L_COBA_15km](./noise_curves/ET_COBA_15km_psd.txt): sensitivity for 15km L-shaped detector from COBA study
- [ET_L_COBA_20km](./noise_curves/ET_COBA_20km_psd.txt): sensitivity for 20km L-shaped detector from COBA study

#### References

- ET-D sensitivity curve: [https://apps.et-gw.eu/tds/?content=3&r=14065](https://apps.et-gw.eu/tds/?content=3&r=14065)
- COBA sensitivity curves in ET-TDS: [https://apps.et-gw.eu/tds/?content=3&r=18213](https://apps.et-gw.eu/tds/?content=3&r=18213)

### Cosmic Explorer

- [CE_40km](./noise_curves/cosmic_explorer_strain_psd.txt): the baseline 40 km detector
- [CE_20km_CBC](./noise_curves/cosmic_explorer_20km_pm_strain_psd.txt): the baseline 20 km detector ("compact binary tuned")
- [CE_20km_post_merger](./noise_curves/cosmic_explorer_20km_pm_strain_psd.txt): the 20 km detector tuned for post-merger signals
- [CE_40km_low_frequency](./noise_curves/cosmic_explorer_40km_lf_strain_psd.txt): the 40 km detector tuned for low-freqency signals

#### References

- Sensitivity curves: [https://dcc.cosmicexplorer.org/CE-T2000017/public](https://dcc.cosmicexplorer.org/CE-T2000017/public)
- [2109.09882](https://arxiv.org/abs/2109.09882)
- [2201.10668](https://arxiv.org/abs/2201.10668)

### LIGO, Virgo, Kagra

- [AplusDesign](./noise_curves/AplusDesign_psd.txt): LIGO A+ target design sensitivity for O5
- [avirgo_O5high_NEW](./noise_curves/avirgo_O5high_NEW_psd.txt): Virgo target sensitivity for O5 in the high noise, low range limit
- [avirgo_O5low_NEW](./noise_curves/avirgo_O5low_NEW_psd.txt): Virgo target sensitivity for O5 in the low noise, high range limit
- [kagra_128Mpc](./noise_curves/kagra_128Mpc_psd.txt): Kagra O5 sensitivity

#### References

- [https://dcc.ligo.org/LIGO-T2000012/public](https://dcc.ligo.org/LIGO-T2000012/public)
