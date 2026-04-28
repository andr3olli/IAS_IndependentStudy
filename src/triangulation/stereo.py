"""
Stereo calibration loading and landmark triangulation utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


@dataclass
class StereoCalibration:
    k_left: np.ndarray
    d_left: np.ndarray
    k_right: np.ndarray
    d_right: np.ndarray
    r: np.ndarray
    t: np.ndarray

    @property
    def p1_norm(self) -> np.ndarray:
        return np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])

    @property
    def p2_norm(self) -> np.ndarray:
        return np.hstack([self.r, self.t.reshape(3, 1)])


def _matrix(data: Dict[str, Any], keys: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.shape != shape:
                raise ValueError(f"Expected {key} shape {shape}, got {arr.shape}")
            return arr
    raise KeyError(f"Missing one of keys {keys}")


def _vector(data: Dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in data:
            arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
            return arr
    raise KeyError(f"Missing one of keys {keys}")


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """
    Load stereo calibration from YAML.

    Supported keys:
    - Flattened:
      K1, D1, K2, D2, R, T
    - Nested:
      left.K / left.dist
      right.K / right.dist
      and R/T at root.
    """
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Stereo calibration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("Stereo calibration YAML must be a mapping.")

    left = data.get("left", {})
    right = data.get("right", {})
    if left and not isinstance(left, dict):
        raise ValueError("stereo calibration `left` must be a mapping.")
    if right and not isinstance(right, dict):
        raise ValueError("stereo calibration `right` must be a mapping.")

    k_left = _matrix(
        {**data, **left},
        ("K1", "k1", "K_left", "k_left", "K"),
        (3, 3),
    )
    d_left = _vector(
        {**data, **left},
        ("D1", "d1", "dist_left", "dist", "D"),
    )
    k_right = _matrix(
        {**data, **right},
        ("K2", "k2", "K_right", "k_right", "K"),
        (3, 3),
    )
    d_right = _vector(
        {**data, **right},
        ("D2", "d2", "dist_right", "dist", "D"),
    )
    r = _matrix(data, ("R", "r"), (3, 3))
    t = _vector(data, ("T", "t"))
    if t.size != 3:
        raise ValueError(f"T must have 3 values, got {t.size}")

    return StereoCalibration(
        k_left=k_left,
        d_left=d_left,
        k_right=k_right,
        d_right=d_right,
        r=r,
        t=t,
    )


class StereoTriangulator:
    """
    Triangulate corresponding landmarks from two calibrated cameras.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        min_visibility: float = 0.4,
        max_reprojection_error: float = 0.02,
    ) -> None:
        self.calib = calibration
        self.min_visibility = float(min_visibility)
        self.max_reprojection_error = float(max_reprojection_error)

    def triangulate(self, xyzv_left: np.ndarray, xyzv_right: np.ndarray) -> np.ndarray:
        """
        Triangulate xyz landmarks using left/right pixel points.

        Args:
            xyzv_left: (N,4) x,y,z,visibility from left camera (z ignored)
            xyzv_right: (N,4) x,y,z,visibility from right camera (z ignored)

        Returns:
            xyzv_3d: (N,4) x,y,z,visibility in left-camera coordinates.
        """
        left = np.asarray(xyzv_left, dtype=np.float64)
        right = np.asarray(xyzv_right, dtype=np.float64)
        n = int(min(left.shape[0], right.shape[0]))
        out = np.full((n, 4), np.nan, dtype=np.float32)
        if n == 0:
            return out

        valid_indices = []
        pts_left = []
        pts_right = []
        vis_min = []
        for idx in range(n):
            x_l, y_l, _z_l, v_l = left[idx]
            x_r, y_r, _z_r, v_r = right[idx]
            if (
                not np.isfinite(x_l)
                or not np.isfinite(y_l)
                or not np.isfinite(v_l)
                or not np.isfinite(x_r)
                or not np.isfinite(y_r)
                or not np.isfinite(v_r)
            ):
                continue
            if v_l < self.min_visibility or v_r < self.min_visibility:
                continue
            valid_indices.append(idx)
            pts_left.append([x_l, y_l])
            pts_right.append([x_r, y_r])
            vis_min.append(float(min(v_l, v_r)))

        if not valid_indices:
            return out

        pts_left_arr = np.asarray(pts_left, dtype=np.float64).reshape(-1, 1, 2)
        pts_right_arr = np.asarray(pts_right, dtype=np.float64).reshape(-1, 1, 2)

        und_left = cv2.undistortPoints(
            pts_left_arr,
            self.calib.k_left,
            self.calib.d_left.reshape(-1, 1),
        ).reshape(-1, 2)
        und_right = cv2.undistortPoints(
            pts_right_arr,
            self.calib.k_right,
            self.calib.d_right.reshape(-1, 1),
        ).reshape(-1, 2)

        points_4d = cv2.triangulatePoints(
            self.calib.p1_norm,
            self.calib.p2_norm,
            und_left.T,
            und_right.T,
        )
        w = points_4d[3, :]
        points_3d = np.full((points_4d.shape[1], 3), np.nan, dtype=np.float64)
        valid_w = np.abs(w) > 1e-9
        points_3d[valid_w] = (points_4d[:3, valid_w] / w[valid_w]).T

        z1 = points_3d[:, 2]
        z1_safe = np.where(np.abs(z1) > 1e-9, z1, np.nan)
        x1_proj = points_3d[:, :2] / z1_safe[:, None]
        p2 = points_3d @ self.calib.r.T + self.calib.t.reshape(1, 3)
        z2 = p2[:, 2]
        z2_safe = np.where(np.abs(z2) > 1e-9, z2, np.nan)
        x2_proj = p2[:, :2] / z2_safe[:, None]

        err_left = np.linalg.norm(x1_proj - und_left, axis=1)
        err_right = np.linalg.norm(x2_proj - und_right, axis=1)
        err = 0.5 * (err_left + err_right)
        depth_ok = (points_3d[:, 2] > 0.0) & (p2[:, 2] > 0.0)
        reproj_ok = np.isfinite(err) & (err <= self.max_reprojection_error)

        for i, lm_idx in enumerate(valid_indices):
            if not depth_ok[i] or not reproj_ok[i]:
                continue
            out[lm_idx, :3] = points_3d[i].astype(np.float32)
            out[lm_idx, 3] = np.float32(vis_min[i])

        return out
