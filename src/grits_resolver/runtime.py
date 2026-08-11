from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_gdal_data() -> Path | None:
    """Make direct conda-environment Python launches as reliable as activation."""
    configured = os.environ.get("GDAL_DATA")
    if configured and (Path(configured) / "gdalvrt.xsd").exists():
        return Path(configured)
    candidates = (
        Path(sys.prefix) / "Library" / "share" / "gdal",
        Path(sys.prefix) / "share" / "gdal",
    )
    for candidate in candidates:
        if (candidate / "gdalvrt.xsd").exists():
            os.environ["GDAL_DATA"] = str(candidate)
            return candidate
    return None
