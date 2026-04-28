"""
Generate virtual dense body points by interpolating along pose skeleton segments.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import mediapipe as mp
import numpy as np


_POSE_ENUM = mp.solutions.pose.PoseLandmark
_POSE_NAMES = [lm.name.lower() for lm in _POSE_ENUM]
_POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


def _pose_name_pairs() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for a, b in _POSE_CONNECTIONS:
        a_idx = int(a.value) if hasattr(a, "value") else int(a)
        b_idx = int(b.value) if hasattr(b, "value") else int(b)
        if a_idx >= len(_POSE_NAMES) or b_idx >= len(_POSE_NAMES):
            continue
        n1 = _POSE_NAMES[a_idx]
        n2 = _POSE_NAMES[b_idx]
        key = tuple(sorted((n1, n2)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((n1, n2))
    return pairs


_POSE_SEGMENTS = _pose_name_pairs()


def build_virtual_body_names(base_names: Sequence[str], subdivisions: int) -> List[str]:
    """
    Return deterministic names for virtual body points appended to base_names.
    """
    if subdivisions <= 0:
        return []
    name_set = set(base_names)
    out: List[str] = []
    for n1, n2 in _POSE_SEGMENTS:
        if n1 not in name_set or n2 not in name_set:
            continue
        for k in range(1, subdivisions + 1):
            out.append(f"body_{n1}_{n2}_{k}")
    return out


def append_virtual_body_points(
    base_names: Sequence[str],
    xyzv: np.ndarray,
    subdivisions: int,
) -> np.ndarray:
    """
    Append virtual interpolated points along pose segments.

    Args:
        base_names: names corresponding to rows in xyzv.
        xyzv: shape (N, 4)
        subdivisions: number of points inserted per segment.
    """
    if subdivisions <= 0:
        return xyzv

    name_to_idx = {n: i for i, n in enumerate(base_names)}
    extras: List[np.ndarray] = []

    for n1, n2 in _POSE_SEGMENTS:
        i1 = name_to_idx.get(n1)
        i2 = name_to_idx.get(n2)
        if i1 is None or i2 is None:
            continue
        p1 = xyzv[i1]
        p2 = xyzv[i2]
        valid = np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))
        for k in range(1, subdivisions + 1):
            t = float(k) / float(subdivisions + 1)
            if not valid:
                extras.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
                continue
            interp = ((1.0 - t) * p1[:3]) + (t * p2[:3])
            vis = min(float(p1[3]), float(p2[3]))
            extras.append(np.array([interp[0], interp[1], interp[2], vis], dtype=np.float32))

    if not extras:
        return xyzv
    extra_arr = np.vstack(extras).astype(np.float32, copy=False)
    return np.vstack([xyzv, extra_arr]).astype(np.float32, copy=False)

