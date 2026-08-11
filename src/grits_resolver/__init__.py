from .runtime import configure_gdal_data

configure_gdal_data()

__version__ = "0.1.0"

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .aliases import COMMON_OPTICAL_ALIASES, resolve_alias
from .resolver import MultiSensorResolver
from .temporal import TemporalMatch, join_times

__all__ = ["COMMON_OPTICAL_ALIASES", "CatalogConsistencyError", "CatalogEntry", "MultiSensorResolver", "TemporalMatch", "__version__", "join_times", "load_catalog", "resolve_alias"]
