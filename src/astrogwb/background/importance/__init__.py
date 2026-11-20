"""Background predictions from importance reweighting a reference catalog."""

from astrogwb.background.importance.catalog import ReferenceCatalog
from astrogwb.background.importance.model import CatalogModel

__all__ = ["CatalogModel", "ReferenceCatalog"]
