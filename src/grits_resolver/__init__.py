from .runtime import configure_gdal_data

configure_gdal_data()

__version__ = "0.2.0"

from .catalog import CatalogConsistencyError, CatalogEntry, load_catalog
from .aliases import COMMON_OPTICAL_ALIASES, resolve_alias
from .resolver import MultiSensorResolver
from .s1 import S1GRiTSResolver, S1TrackPair, STATIC_BANDS
from .temporal import TemporalMatch, join_times
from .s2_quality import SceneQualityPair

__all__ = ["COMMON_OPTICAL_ALIASES", "CatalogConsistencyError", "CatalogEntry", "MultiSensorResolver", "S1GRiTSResolver", "S1TrackPair", "STATIC_BANDS", "TemporalMatch", "SceneQualityPair", "__version__", "join_times", "load_catalog", "resolve_alias"]
