# astrogwb

Model the stochastic gravitational-wave background from compact-binary
populations and perform Bayesian inference with detector networks.

## Installation

Requires Python 3.12 or newer.

```bash
pip install astrogwb
```

Install `astrogwb[cli]` to use the simulation command.

## Usage

```python
from astrogwb.detector import load_detector, load_sensitivity

hanford = load_detector("H1")
sensitivity = load_sensitivity("H1")
```

See the [documentation](https://astrogwb.readthedocs.io/en/latest/) for
installation extras, examples, and the API reference.

## Features

- Analytic inspiral spectra and importance-weighted population predictions.
- Detector responses, sensitivity curves, and network signal-to-noise ratios.
- Segmented background simulations and a likelihood that includes shot noise.

## License

[MIT](LICENSE)
