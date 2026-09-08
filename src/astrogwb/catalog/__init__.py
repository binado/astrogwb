"""Array-native catalog container with its own population record and I/O."""

from astrogwb.catalog.catalog import Catalog
from astrogwb.catalog.metadata import PopulationMetadata, ScalarProvenance

__all__ = ["Catalog", "PopulationMetadata", "ScalarProvenance"]
