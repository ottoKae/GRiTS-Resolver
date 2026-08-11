from __future__ import annotations


COMMON_OPTICAL_ALIASES = ("blue", "green", "red", "nir_narrow", "swir_1", "swir_2")

SENSOR_ALIASES = {
    "sentinel-2": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir_narrow": "nir_narrow",
        "swir_1": "swir_1",
        "swir_2": "swir_2",
    },
    # HLS names are included as a contract for the later HLS adapter.
    "hls-s30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B8A",
        "swir_1": "B11",
        "swir_2": "B12",
    },
    "hls-l30": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
    },
}


def resolve_alias(sensor: str, alias: str) -> str:
    try:
        return SENSOR_ALIASES[sensor][alias]
    except KeyError as exc:
        raise KeyError(f"Unsupported alias {alias!r} for sensor {sensor!r}") from exc
