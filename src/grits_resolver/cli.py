from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .resolver import MultiSensorResolver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-root", required=True)
    parser.add_argument("--s2-root", required=True)
    parser.add_argument("--tile", default="17MPU")
    args = parser.parse_args()
    resolver = MultiSensorResolver(Path(args.s1_root), Path(args.s2_root))
    matches = resolver.time_join(args.tile, tolerance=np.timedelta64(20, "D"))
    print(f"S1 entries: {len(resolver.entries('sentinel-1', args.tile))}")
    print(f"S2 entries: {len(resolver.entries('sentinel-2', args.tile))}")
    print(f"Temporal matches: {sum(match.right_index is not None for match in matches)}/{len(matches)}")
