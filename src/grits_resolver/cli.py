from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .resolver import MultiSensorResolver


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-root", required=True)
    parser.add_argument("--s2-root", required=True)
    parser.add_argument("--tile", default="17MPU")
    parser.add_argument("--s1-track", help="Explicit S1 track for multitrack tiles")
    parser.add_argument("--s1-store", help="Explicit S1 monthly store")
    parser.add_argument("--s2-store", help="Explicit S2 store when several match")
    args = parser.parse_args(argv)
    resolver = MultiSensorResolver(Path(args.s1_root), Path(args.s2_root))
    matches = resolver.time_join(args.tile, tolerance=np.timedelta64(20, "D"),
                                 s1_track=args.s1_track, s1_store=args.s1_store,
                                 s2_store=args.s2_store)
    print(f"S1 entries: {len(resolver.entries('sentinel-1', args.tile))}")
    print(f"S2 entries: {len(resolver.entries('sentinel-2', args.tile))}")
    print(f"Temporal matches: {sum(match.right_index is not None for match in matches)}/{len(matches)}")
