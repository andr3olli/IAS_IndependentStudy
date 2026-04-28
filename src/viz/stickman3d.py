"""
3D stickman renderer using Matplotlib for MediaPipe pose landmarks.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import mediapipe as mp

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


class Stickman3DRenderer:
    """
    Render MediaPipe pose landmarks as a 3D stickman frame-by-frame.
    """

    def __init__(
        self,
        names: Sequence[str],
        visibility_threshold: float = 0.2,
        fig_size: Tuple[float, float] = (4.0, 4.0),
        dpi: int = 200,
        elev: float = 15.0,
        azim: float = -60.0,
        axis_limits: Dict[str, Tuple[float, float]] | None = None,
        line_color: Tuple[float, float, float] = (0.9, 0.4, 0.1),
        joint_color: Tuple[float, float, float] = (0.1, 0.8, 0.9),
        background_color: Tuple[float, float, float] = (0.05, 0.05, 0.05),
        line_width: float = 2.0,
        joint_size: float = 20.0,
        x_scale: float = 1.0,
    ) -> None:
        self.names = list(names)
        self.name_to_index = {name: idx for idx, name in enumerate(self.names)}
        self.visibility_threshold = visibility_threshold
        self.line_color = line_color
        self.joint_color = joint_color
        self.background_color = background_color
        self.line_width = float(line_width)
        self.joint_size = float(joint_size)
        self.scale_x = float(x_scale)

        self.fig = plt.figure(figsize=fig_size, dpi=dpi)
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(0, 0, 1, 1)

        self.elev = elev
        self.azim = azim
        axis_limits = axis_limits or {}
        self.xlim = axis_limits.get("x", (-2.5, 2.5))
        self.ylim = axis_limits.get("y", (-3.0, 2.0))
        self.zlim = axis_limits.get("z", (-2.0, 2.0))

    @property
    def frame_size(self) -> Tuple[int, int]:
        """
        Return (width, height) in pixels for the rendered canvas.
        """
        width, height = self.canvas.get_width_height()
        return width, height

    def render(self, xyzv: np.ndarray) -> np.ndarray:
        coords, visibility = self._normalize_coords(xyzv)

        self.ax.cla()
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_facecolor(self.background_color)
        self.fig.patch.set_facecolor(self.background_color)
        self.ax.set_box_aspect((1, 1, 1))

        # Apply subtle pane tint; handle Matplotlib API differences across versions.
        pane_color = (*self.background_color, 0.05)
        axes = []
        for attr in ("w_xaxis", "xaxis"):
            if hasattr(self.ax, attr):
                axes.append(getattr(self.ax, attr))
        for attr in ("w_yaxis", "yaxis"):
            if hasattr(self.ax, attr):
                axes.append(getattr(self.ax, attr))
        for attr in ("w_zaxis", "zaxis"):
            if hasattr(self.ax, attr):
                axes.append(getattr(self.ax, attr))

        seen = set()
        for axis in axes:
            if id(axis) in seen:
                continue
            seen.add(id(axis))
            if hasattr(axis, "set_pane_color"):
                axis.set_pane_color(pane_color)
            if hasattr(axis, "line"):
                try:
                    axis.line.set_color((0.6, 0.6, 0.6, 0.3))
                except Exception:
                    pass

        for start_landmark, end_landmark in POSE_CONNECTIONS:
            start_idx = self._landmark_index(start_landmark)
            end_idx = self._landmark_index(end_landmark)
            if start_idx is None or end_idx is None:
                continue

            if (
                visibility[start_idx] >= self.visibility_threshold
                and visibility[end_idx] >= self.visibility_threshold
            ):
                xs = [coords[start_idx, 0], coords[end_idx, 0]]
                ys = [coords[start_idx, 1], coords[end_idx, 1]]
                zs = [coords[start_idx, 2], coords[end_idx, 2]]
                self.ax.plot(xs, ys, zs, color=self.line_color, linewidth=self.line_width)

        vis_mask = visibility >= self.visibility_threshold
        if np.any(vis_mask):
            vis_count = int(np.count_nonzero(vis_mask))
            point_size = self.joint_size
            if vis_count > 80:
                point_size = max(2.0, self.joint_size * (80.0 / vis_count))
            self.ax.scatter(
                coords[vis_mask, 0],
                coords[vis_mask, 1],
                coords[vis_mask, 2],
                color=self.joint_color,
                s=point_size,
            )

        self.canvas.draw()
        width, height = self.canvas.get_width_height()
        buffer = np.asarray(self.canvas.buffer_rgba(), dtype=np.uint8)
        buffer = buffer.reshape((height, width, 4))
        image = buffer[:, :, :3].copy()
        return image

    def _normalize_coords(self, xyzv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(xyzv[:, :3], dtype=np.float32)
        visibility = np.asarray(xyzv[:, 3], dtype=np.float32)

        left_hip = self._get_coord(coords, "left_hip")
        right_hip = self._get_coord(coords, "right_hip")
        left_shoulder = self._get_coord(coords, "left_shoulder")
        right_shoulder = self._get_coord(coords, "right_shoulder")

        centers = [p for p in (left_hip, right_hip) if p is not None]
        if centers:
            center = np.mean(centers, axis=0)
        else:
            center = np.nanmean(coords, axis=0)
        if not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=np.float32)

        coords = coords - center

        ref_pairs = [
            (left_shoulder, right_shoulder),
            (left_hip, right_hip),
        ]
        scale = None
        for p1, p2 in ref_pairs:
            if p1 is not None and p2 is not None:
                dist = np.linalg.norm(p1 - p2)
                if np.isfinite(dist) and dist > 1e-6:
                    scale = dist
                    break
        if scale is None or not np.isfinite(scale):
            scale = 100.0

        coords = coords / scale
        coords[:, 0] *= self.scale_x
        coords[:, 1] *= -1.0  # make upward positive

        return coords, visibility

    def _get_coord(self, coords: np.ndarray, name: str) -> np.ndarray | None:
        idx = self.name_to_index.get(name)
        if idx is None:
            return None
        point = coords[idx]
        if np.any(np.isnan(point)):
            return None
        return point

    def _landmark_index(self, value) -> int | None:
        if isinstance(value, int):
            return value if value < len(self.names) else None
        if hasattr(value, "value"):
            return int(value.value)
        name = getattr(value, "name", "").lower()
        return self.name_to_index.get(name)
