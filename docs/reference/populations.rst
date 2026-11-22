Reference catalogs and population models
========================================

.. currentmodule:: astrogwb

Reference catalogs
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.importance.ReferenceCatalog
   ~background.importance.spectrum.INCLINATION_MEAN
   ~background.importance.spectrum.INCLINATION_RAW_SECOND_MOMENT
   ~background.importance.spectrum.phinney_kernel

Catalog models and diagnostics
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~background.importance.CatalogModel
   ~background.importance.diagnostics.relative_ess

Population models
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~populations.Parameters
   ~populations.PopulationModel
   ~populations.SourceModel
   ~populations.coba.source_model
   ~populations.mass.gaussian_mass_pair.gaussian_mass_pair_model
   ~populations.mass.uniform_mass_pair.uniform_mass_pair_model

Population I/O
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~populations.io.read_population_file
