Installation
============

``astrogwb`` requires Python 3.12 or newer.

.. code-block:: bash

   pip install astrogwb

Use ``pip install "astrogwb[cli]"`` for the simulation command,
or ``pip install "astrogwb[io]"`` for HDF5 catalog serialization.
JAX accelerator extras are available as ``astrogwb[cuda]`` (CUDA 12)
and ``astrogwb[tpu]``.

From a source checkout, build the documentation with ``just docs build``.
The build writes HTML to ``docs/_build/html``.
