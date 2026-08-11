from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


JoinMethod = Literal["exact", "nearest", "backward", "forward"]


@dataclass(frozen=True)
class TemporalMatch:
    left_index: int
    right_index: int | None
    left_time: np.datetime64
    right_time: np.datetime64 | None
    delta: np.timedelta64 | None


def join_times(
    left: np.ndarray,
    right: np.ndarray,
    *,
    method: JoinMethod = "nearest",
    tolerance: np.timedelta64 | None = None,
) -> list[TemporalMatch]:
    left_ns = np.asarray(left).astype("datetime64[ns]")
    right_ns = np.asarray(right).astype("datetime64[ns]")
    order = np.argsort(right_ns)
    sorted_right = right_ns[order]
    matches: list[TemporalMatch] = []
    for left_index, timestamp in enumerate(left_ns):
        pos = int(np.searchsorted(sorted_right, timestamp))
        candidates: list[int] = []
        if method in {"exact", "nearest", "backward"} and pos > 0:
            candidates.append(pos - 1)
        if method in {"exact", "nearest", "forward"} and pos < len(sorted_right):
            candidates.append(pos)
        if (
            method == "backward"
            and pos < len(sorted_right)
            and sorted_right[pos] == timestamp
        ):
            candidates.append(pos)
        if method == "exact":
            candidates = [i for i in candidates if sorted_right[i] == timestamp]
        elif method == "backward":
            candidates = [i for i in candidates if sorted_right[i] <= timestamp]
        elif method == "forward":
            candidates = [i for i in candidates if sorted_right[i] >= timestamp]
        if not candidates:
            matches.append(TemporalMatch(left_index, None, timestamp, None, None))
            continue
        chosen = min(candidates, key=lambda i: abs(sorted_right[i] - timestamp))
        delta = sorted_right[chosen] - timestamp
        if tolerance is not None and abs(delta) > tolerance:
            matches.append(TemporalMatch(left_index, None, timestamp, None, None))
            continue
        right_index = int(order[chosen])
        matches.append(TemporalMatch(left_index, right_index, timestamp, right_ns[right_index], delta))
    return matches
