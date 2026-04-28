"""
Interactive stickman viewer for recorded keypoints.

Usage:
    python -m viewers.stickman_viewer --csv data/processed/test_keypoints.csv --head-scan --head-scan-mode compare

Controls:
    Left / Right arrow : previous / next frame
    Space              : play / pause
    - / +              : zoom out / zoom in
    0                  : reset zoom
    Q or Esc           : quit
Mouse:
    Drag to rotate, scroll to zoom (Matplotlib 3D).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import mediapipe as mp

from analytics.head_scan import HeadScanner, HeadScannerV2, HeadScannerV3


POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive 3D stickman viewer.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to keypoints CSV (frame, landmark, x, y, z, visibility).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback FPS for autoplay.",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=0.2,
        help="Minimum visibility to show a joint/segment.",
    )
    parser.add_argument("--head-scan", action="store_true", help="Show head scan count/yaw using HeadScanner.")
    parser.add_argument(
        "--head-scan-mode",
        type=str,
        default="v3",
        choices=["v1", "v2", "v3", "compare", "compare_v2_v3"],
        help="Scanner mode for the HUD.",
    )
    parser.add_argument("--head-left", type=float, default=-0.25, help="Head scan left threshold.")
    parser.add_argument("--head-right", type=float, default=0.25, help="Head scan right threshold.")
    parser.add_argument("--head-deadband", type=float, default=0.04, help="Head scan deadband.")
    parser.add_argument("--head-min-vis", type=float, default=0.5, help="Head scan min visibility for landmarks.")
    parser.add_argument("--head-torso-weight", type=float, default=0.6, help="V2 torso compensation weight.")
    parser.add_argument("--head-face-weight", type=float, default=0.55, help="V2 face landmark yaw weight.")
    parser.add_argument("--head-smooth-alpha", type=float, default=0.25, help="V2 EMA smoothing alpha.")
    parser.add_argument("--head-dwell-frames", type=int, default=2, help="V2 required frames before state transition.")
    parser.add_argument("--head-refractory-frames", type=int, default=10, help="V3 refractory frames after counted event.")
    parser.add_argument("--head-min-excursion", type=float, default=0.03, help="V3 additional margin beyond threshold.")
    parser.add_argument("--head-min-quality", type=float, default=0.35, help="V3 minimum quality required for side transitions.")
    parser.add_argument("--head-adaptive", action="store_true", help="Enable V3 adaptive threshold calibration.")
    parser.add_argument("--head-warmup-frames", type=int, default=45, help="V3 calibration warmup frames.")
    parser.add_argument("--head-adaptive-sigma-scale", type=float, default=2.0, help="V3 sigma multiplier for adaptive thresholds.")
    parser.add_argument("--head-adaptive-deadband-scale", type=float, default=0.5, help="V3 adaptive deadband scale.")
    parser.add_argument("--head-adaptive-min-abs-threshold", type=float, default=0.12, help="V3 minimum absolute side threshold.")
    parser.add_argument("--head-adaptive-min-sigma", type=float, default=0.02, help="V3 minimum sigma floor.")
    parser.add_argument("--head-v3-face", action="store_true", help="Enable V3 face cue fusion if face landmarks exist.")
    parser.add_argument("--head-v3-face-weight", type=float, default=0.35, help="V3 face cue blend weight.")
    parser.add_argument(
        "--base-span",
        type=float,
        default=3.0,
        help="Base half-range for x/y/z limits in normalized space (larger = zoomed out).",
    )
    return parser.parse_args()


def load_keypoints(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load CSV -> (frames, num_landmarks, 4), ordered by encounter order."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    frames: List[List[Tuple[float, float, float, float]]] = []
    names_order: List[str] = []
    name_to_idx: Dict[str, int] = {}
    current_frame_idx = None
    current: List[Tuple[float, float, float, float]] = []

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            f_idx = int(row["frame"])
            name = row["landmark"]
            x, y, z, vis = (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                float(row["visibility"]),
            )

            if current_frame_idx is None:
                current_frame_idx = f_idx
            if f_idx != current_frame_idx:
                frames.append(current)
                current = []
                current_frame_idx = f_idx

            if name not in name_to_idx:
                name_to_idx[name] = len(names_order)
                names_order.append(name)
            # ensure list long enough
            while len(current) < len(names_order):
                current.append((np.nan, np.nan, np.nan, np.nan))
            current[name_to_idx[name]] = (x, y, z, vis)

    if current:
        frames.append(current)

    arr = np.asarray(frames, dtype=np.float32)
    return arr, names_order


class StickmanViewer:
    def __init__(
        self,
        xyzv: np.ndarray,
        names: List[str],
        fps: float,
        visibility_thresh: float,
        base_span: float = 3.0,
        head_scan_mode: str = "v2",
        head_scanner_v1: Optional[HeadScanner] = None,
        head_scanner_v2: Optional[HeadScannerV2] = None,
        head_scanner_v3: Optional[HeadScannerV3] = None,
    ) -> None:
        self.xyzv = xyzv  # shape (T, 33, 4)
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.fps = fps
        self.visibility_thresh = visibility_thresh
        self.base_span = float(base_span)
        self.zoom_scale = 1.0
        self.head_scan_mode = head_scan_mode
        self.head_scanner_v1 = head_scanner_v1
        self.head_scanner_v2 = head_scanner_v2
        self.head_scanner_v3 = head_scanner_v3
        self._state_cache = {
            "v1": {"prev": None, "text": "", "color": (1, 1, 1)},
            "v2": {"prev": None, "text": "", "color": (1, 1, 1)},
            "v3": {"prev": None, "text": "", "color": (1, 1, 1)},
        }
        self.state_colors = {
            "neutral": (0.6, 0.6, 0.6),
            "left": (0.2, 0.7, 0.9),
            "right": (0.9, 0.6, 0.2),
        }

        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_facecolor((0.05, 0.05, 0.05))
        self.fig.patch.set_facecolor((0.05, 0.05, 0.05))
        self._grid_divisions = 8
        self._cube_lines: List[Line2D] = []
        self._floor_grid_lines: List[Line2D] = []
        self._axis_lines: List[Line2D] = []
        self._base_dist = float(getattr(self.ax, "dist", 10.0))
        self._style_axes()
        self._init_reference_frame_artists()

        self.lines = []
        for _ in POSE_CONNECTIONS:
            ln, = self.ax.plot(
                [],
                [],
                [],
                color=(0.9, 0.4, 0.1),
                linewidth=2,
                zorder=3,
            )
            self.lines.append(ln)
        self.sc = self.ax.scatter(
            [],
            [],
            [],
            color=(0.1, 0.8, 0.9),
            s=30,
            zorder=4,
        )
        self.text_v1 = self.fig.text(0.02, 0.96, "", color="w", fontsize=10)
        self.text_v1_state = self.fig.text(0.02, 0.93, "", color="w", fontsize=9)
        self.text_v2 = self.fig.text(0.02, 0.89, "", color="w", fontsize=10)
        self.text_v2_state = self.fig.text(0.02, 0.86, "", color="w", fontsize=9)
        self.text_v3 = self.fig.text(0.02, 0.82, "", color="w", fontsize=10)
        self.text_v3_state = self.fig.text(0.02, 0.79, "", color="w", fontsize=9)

        # Yaw bar overlay (2D axes in figure coords)
        self.bar_v1 = self._create_bar([0.05, 0.03, 0.35, 0.045], (0.0, 0.65, 1.0))
        self.bar_v2 = self._create_bar([0.05, 0.08, 0.35, 0.045], (0.0, 0.9, 0.5))
        self.bar_v3 = self._create_bar([0.05, 0.13, 0.35, 0.045], (0.0, 0.9, 0.9))
        if self.head_scan_mode == "v1":
            self.text_v2.set_visible(False)
            self.text_v2_state.set_visible(False)
            self.bar_v2["ax"].set_visible(False)
            self.text_v3.set_visible(False)
            self.text_v3_state.set_visible(False)
            self.bar_v3["ax"].set_visible(False)
        elif self.head_scan_mode == "v2":
            self.text_v1.set_visible(False)
            self.text_v1_state.set_visible(False)
            self.bar_v1["ax"].set_visible(False)
            self.text_v3.set_visible(False)
            self.text_v3_state.set_visible(False)
            self.bar_v3["ax"].set_visible(False)
        elif self.head_scan_mode == "v3":
            self.text_v1.set_visible(False)
            self.text_v1_state.set_visible(False)
            self.bar_v1["ax"].set_visible(False)
            self.text_v2.set_visible(False)
            self.text_v2_state.set_visible(False)
            self.bar_v2["ax"].set_visible(False)
        elif self.head_scan_mode == "compare":
            self.text_v3.set_visible(False)
            self.text_v3_state.set_visible(False)
            self.bar_v3["ax"].set_visible(False)
        elif self.head_scan_mode == "compare_v2_v3":
            self.text_v1.set_visible(False)
            self.text_v1_state.set_visible(False)
            self.bar_v1["ax"].set_visible(False)

        self.frame_idx = 0
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._on_timer)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._set_limits()
        self._draw_frame()

    def _set_limits(self):
        # normalize first frame to get scale
        coords, _ = self._normalize(self.xyzv[0])
        max_range = np.nanmax(np.ptp(coords, axis=0))
        if not np.isfinite(max_range) or max_range <= 0:
            max_range = 1.0
        span = max(max_range * 0.9, self.base_span)
        self.base_span = span
        self._apply_zoom_limits()

    def _apply_zoom_limits(self):
        # Keep world bounds fixed so the coordinate system stays continuous.
        # Zoom is applied with camera distance instead of shrinking limits.
        span = self.base_span
        self.ax.set_xlim(-span, span)
        self.ax.set_ylim(-span, span)
        self.ax.set_zlim(-span, span)
        self._update_reference_frame_artists(span * 1.05)
        if hasattr(self.ax, "dist"):
            # Larger dist => zoom out, smaller dist => zoom in.
            self.ax.dist = float(np.clip(self._base_dist * self.zoom_scale, 4.0, 40.0))

    def _style_axes(self):
        self.ax.grid(True)
        pane_color = (0.10, 0.10, 0.10, 0.45)
        axis_line = (1.0, 1.0, 1.0, 0.90)
        grid_line = (1.0, 1.0, 1.0, 0.14)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            if hasattr(axis, "set_pane_color"):
                try:
                    axis.set_pane_color(pane_color)
                except Exception:
                    pass
            if hasattr(axis, "line"):
                try:
                    axis.line.set_color(axis_line)
                    axis.line.set_linewidth(1.0)
                except Exception:
                    pass
            # mpl3d exposes grid style through private _axinfo.
            try:
                axis._axinfo["grid"]["color"] = grid_line
                axis._axinfo["grid"]["linewidth"] = 0.8
                axis._axinfo["grid"]["linestyle"] = "-"
            except Exception:
                pass

    def _init_reference_frame_artists(self):
        for _ in range(12):
            ln, = self.ax.plot(
                [],
                [],
                [],
                color=(1.0, 1.0, 1.0, 0.35),
                linewidth=1.0,
                zorder=1,
            )
            self._cube_lines.append(ln)

        for _ in range(2 * (self._grid_divisions + 1)):
            ln, = self.ax.plot(
                [],
                [],
                [],
                color=(1.0, 1.0, 1.0, 0.18),
                linewidth=0.8,
                zorder=0,
            )
            self._floor_grid_lines.append(ln)

        for _ in range(3):
            ln, = self.ax.plot(
                [],
                [],
                [],
                color=(1.0, 1.0, 1.0, 0.95),
                linewidth=1.6,
                zorder=2,
            )
            self._axis_lines.append(ln)

    def _update_reference_frame_artists(self, span: float):
        s = float(max(1e-6, span))
        corners = [
            (-s, -s, -s),
            (s, -s, -s),
            (s, s, -s),
            (-s, s, -s),
            (-s, -s, s),
            (s, -s, s),
            (s, s, s),
            (-s, s, s),
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for ln, (i, j) in zip(self._cube_lines, edges):
            p1 = corners[i]
            p2 = corners[j]
            ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            ln.set_3d_properties([p1[2], p2[2]])

        grid_vals = np.linspace(-s, s, self._grid_divisions + 1)
        floor_y = -s
        idx = 0
        for z in grid_vals:
            ln = self._floor_grid_lines[idx]
            idx += 1
            ln.set_data([-s, s], [floor_y, floor_y])
            ln.set_3d_properties([z, z])
        for x in grid_vals:
            ln = self._floor_grid_lines[idx]
            idx += 1
            ln.set_data([x, x], [floor_y, floor_y])
            ln.set_3d_properties([-s, s])

        axis_segments = [
            ((-s, 0.0, 0.0), (s, 0.0, 0.0)),
            ((0.0, -s, 0.0), (0.0, s, 0.0)),
            ((0.0, 0.0, -s), (0.0, 0.0, s)),
        ]
        for ln, (p1, p2) in zip(self._axis_lines, axis_segments):
            ln.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            ln.set_3d_properties([p1[2], p2[2]])

    def _normalize(self, xyzv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(xyzv[:, :3], dtype=np.float32)
        visibility = np.asarray(xyzv[:, 3], dtype=np.float32)

        def get(name: str):
            idx = self.name_to_idx.get(name)
            if idx is None:
                return None
            pt = coords[idx]
            if np.any(np.isnan(pt)):
                return None
            return pt

        left_hip = get("left_hip")
        right_hip = get("right_hip")
        left_shoulder = get("left_shoulder")
        right_shoulder = get("right_shoulder")

        centers = [p for p in (left_hip, right_hip) if p is not None]
        center = np.mean(centers, axis=0) if centers else np.nanmean(coords, axis=0)
        if not np.all(np.isfinite(center)):
            center = np.zeros(3, dtype=np.float32)
        coords = coords - center

        scale = None
        for p1, p2 in ((left_shoulder, right_shoulder), (left_hip, right_hip)):
            if p1 is not None and p2 is not None:
                dist = np.linalg.norm(p1 - p2)
                if np.isfinite(dist) and dist > 1e-6:
                    scale = dist
                    break
        if scale is None:
            scale = 100.0
        coords = coords / scale
        coords[:, 1] *= -1.0
        return coords, visibility

    def _draw_frame(self):
        frame_xyzv = self.xyzv[self.frame_idx]
        coords, visibility = self._normalize(frame_xyzv)
        vis_mask = visibility >= self.visibility_thresh

        if self.head_scanner_v1 is not None:
            info_v1 = self.head_scanner_v1.update(frame_xyzv)
            self._render_scan_hud(
                key="v1",
                info=info_v1,
                main_text=self.text_v1,
                state_text=self.text_v1_state,
                bar=self.bar_v1,
                label="V1",
            )
        else:
            self.text_v1.set_text("")
            self.text_v1_state.set_text("")
            self._update_bar(np.nan, None, self.bar_v1)

        if self.head_scanner_v2 is not None:
            info_v2 = self.head_scanner_v2.update(frame_xyzv)
            self._render_scan_hud(
                key="v2",
                info=info_v2,
                main_text=self.text_v2,
                state_text=self.text_v2_state,
                bar=self.bar_v2,
                label="V2",
            )
        else:
            self.text_v2.set_text("")
            self.text_v2_state.set_text("")
            self._update_bar(np.nan, None, self.bar_v2)

        if self.head_scanner_v3 is not None:
            info_v3 = self.head_scanner_v3.update(frame_xyzv)
            self._render_scan_hud(
                key="v3",
                info=info_v3,
                main_text=self.text_v3,
                state_text=self.text_v3_state,
                bar=self.bar_v3,
                label="V3",
            )
        else:
            self.text_v3.set_text("")
            self.text_v3_state.set_text("")
            self._update_bar(np.nan, None, self.bar_v3)

        for (conn, ln) in zip(POSE_CONNECTIONS, self.lines):
            s_idx = self._lm_index(conn[0])
            e_idx = self._lm_index(conn[1])
            if s_idx is None or e_idx is None:
                ln.set_data([], [])
                ln.set_3d_properties([])
                continue
            if visibility[s_idx] < self.visibility_thresh or visibility[e_idx] < self.visibility_thresh:
                ln.set_data([], [])
                ln.set_3d_properties([])
                continue
            xs = [coords[s_idx, 0], coords[e_idx, 0]]
            ys = [coords[s_idx, 1], coords[e_idx, 1]]
            zs = [coords[s_idx, 2], coords[e_idx, 2]]
            ln.set_data(xs, ys)
            ln.set_3d_properties(zs)

        xs = coords[vis_mask, 0]
        ys = coords[vis_mask, 1]
        zs = coords[vis_mask, 2]
        self.sc._offsets3d = (xs, ys, zs)
        n_vis = int(np.count_nonzero(vis_mask))
        point_size = 30.0
        if n_vis > 80:
            point_size = max(3.0, 30.0 * (80.0 / n_vis))
        self.sc.set_sizes(np.full(max(len(xs), 1), point_size, dtype=np.float32))
        self.fig.canvas.draw_idle()

    def _render_scan_hud(self, key, info, main_text, state_text, bar, label):
        yaw = info.get("yaw_norm", np.nan)
        count = info.get("count", 0)
        state = str(info.get("state", "neutral"))
        text = (
            f"{label} scans: {count} | {label} yaw: {yaw:.2f}"
            if np.isfinite(yaw)
            else f"{label} scans: {count} | {label} yaw: --"
        )
        if key == "v2":
            yaw_face = info.get("yaw_face", np.nan)
            yaw_pose = info.get("yaw_pose", np.nan)
            if np.isfinite(yaw_face):
                text += f" | face: {yaw_face:.2f}"
            if np.isfinite(yaw_pose):
                text += f" | pose: {yaw_pose:.2f}"
        elif key == "v3":
            yaw_face = info.get("yaw_face", np.nan)
            yaw_pose = info.get("yaw_pose", np.nan)
            quality = info.get("quality", np.nan)
            event = int(info.get("event", 0))
            event_dir = info.get("event_direction", "none")
            calibrated = int(info.get("calibrated", 0))
            if np.isfinite(yaw_face):
                text += f" | face: {yaw_face:.2f}"
            if np.isfinite(yaw_pose):
                text += f" | pose: {yaw_pose:.2f}"
            if np.isfinite(quality):
                text += f" | q:{quality:.2f}"
            text += f" | cal:{calibrated}"
            if event:
                text += f" | EVENT:{event_dir}"
        main_text.set_text(text)
        transition_text, transition_color = self._update_state(key, state)
        if key == "v2":
            cand = info.get("candidate_state", "none")
            pending = info.get("pending_frames", 0)
            dwell = info.get("dwell_frames", 0)
            face_pts = info.get("face_points", 0)
            transition_text = (
                f"{transition_text} | cand:{cand} p:{pending}/{dwell} face_pts:{face_pts}"
            )
        elif key == "v3":
            cand = info.get("candidate_state", "none")
            pending = info.get("pending_frames", 0)
            dwell = info.get("dwell_frames", 0)
            face_pts = info.get("face_points", 0)
            lt = info.get("left_threshold", np.nan)
            rt = info.get("right_threshold", np.nan)
            transition_text = (
                f"{transition_text} | cand:{cand} p:{pending}/{dwell} "
                f"face_pts:{face_pts} lt:{lt:.2f} rt:{rt:.2f}"
            )
        state_text.set_text(transition_text)
        state_text.set_color(transition_color)
        if key == "v1":
            scanner = self.head_scanner_v1
        elif key == "v2":
            scanner = self.head_scanner_v2
        else:
            scanner = self.head_scanner_v3
        self._update_bar(yaw, scanner, bar)

    def _lm_index(self, value) -> int | None:
        if isinstance(value, int):
            return value if value < len(self.names) else None
        if hasattr(value, "value"):
            return int(value.value)
        name = getattr(value, "name", "").lower()
        return self.name_to_idx.get(name)

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self.frame_idx = min(self.frame_idx + 1, len(self.xyzv) - 1)
            self._draw_frame()
        elif event.key in ("left", "p"):
            self.frame_idx = max(self.frame_idx - 1, 0)
            self._draw_frame()
        elif event.key in ("-", "_"):
            self.zoom_scale = min(5.0, self.zoom_scale * 1.15)
            self._apply_zoom_limits()
            self.fig.canvas.draw_idle()
        elif event.key in ("+", "="):
            self.zoom_scale = max(0.2, self.zoom_scale / 1.15)
            self._apply_zoom_limits()
            self.fig.canvas.draw_idle()
        elif event.key == "0":
            self.zoom_scale = 1.0
            self._apply_zoom_limits()
            self.fig.canvas.draw_idle()
        elif event.key == " ":
            self.playing = not self.playing
            if self.playing:
                self.timer.start()
            else:
                self.timer.stop()
        elif event.key in ("q", "escape"):
            plt.close(self.fig)

    def _on_timer(self):
        if not self.playing:
            return
        self.frame_idx = (self.frame_idx + 1) % len(self.xyzv)
        self._draw_frame()

    def _update_state(self, key: str, state: str):
        cache = self._state_cache[key]
        prev = cache["prev"]
        if prev is None:
            cache["prev"] = state
            cache["text"] = f"{key.upper()} state: neutral -> {state}"
            cache["color"] = self.state_colors.get(state, (1, 1, 1))
        elif state != prev:
            cache["text"] = f"{key.upper()} state: {prev} -> {state}"
            cache["color"] = self.state_colors.get(state, (1, 1, 1))
            cache["prev"] = state
        return cache["text"], cache["color"]

    def _update_bar(self, yaw: float, scanner, bar):
        # yaw in [-1,1]; map to [0,1]
        fill_center = 0.5
        if np.isfinite(yaw):
            fill_center = 0.5 + float(np.clip(yaw, -1.0, 1.0)) * 0.5
        width = abs(fill_center - 0.5)
        bar["fill"].set_x(min(fill_center, 0.5))
        bar["fill"].set_width(width if width > 0 else 0.0)

        # Update ticks: left/right thresholds and deadband (optional)
        thresholds = [
            (-scanner.deadband if scanner else None),
            (scanner.deadband if scanner else None),
            (scanner.left_threshold if scanner else None),
            (scanner.right_threshold if scanner else None),
        ]
        colors = [
            (0.4, 0.7, 0.4),
            (0.4, 0.7, 0.4),
            (0.8, 0.6, 0.6),
            (0.8, 0.6, 0.6),
        ]
        for ln, th, color in zip(bar["ticks"], thresholds, colors):
            if th is None or not np.isfinite(th):
                ln.set_data([], [])
                continue
            x = 0.5 + np.clip(th, -1.0, 1.0) * 0.5
            ln.set_data([x, x], [0.2, 0.8])
            ln.set_color(color)
        bar["ax"].figure.canvas.draw_idle()

    def _create_bar(self, rect, fill_color):
        ax = self.fig.add_axes(rect)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor((0.05, 0.05, 0.05))
        outline = patches.Rectangle(
            (0, 0.25),
            1,
            0.5,
            fill=False,
            edgecolor=(0.6, 0.6, 0.6),
            linewidth=1.5,
        )
        fill = patches.Rectangle((0.5, 0.25), 0.0, 0.5, color=fill_color, alpha=0.7)
        ax.add_patch(outline)
        ax.add_patch(fill)
        ticks: List[Line2D] = []
        for _ in range(4):
            ln = Line2D([], [], color=(0.7, 0.5, 0.5), linewidth=1)
            ax.add_line(ln)
            ticks.append(ln)
        return {"ax": ax, "fill": fill, "ticks": ticks}

    def show(self):
        plt.show()


def main():
    args = parse_args()
    xyzv, names = load_keypoints(args.csv)
    mode = args.head_scan_mode
    head_scanner_v1 = None
    head_scanner_v2 = None
    head_scanner_v3 = None
    if args.head_scan:
        if mode in ("v1", "compare"):
            head_scanner_v1 = HeadScanner(
                names=names,
                right_threshold=args.head_right,
                left_threshold=args.head_left,
                deadband=args.head_deadband,
                min_visibility=args.head_min_vis,
            )
        if mode in ("v2", "compare", "compare_v2_v3"):
            head_scanner_v2 = HeadScannerV2(
                names=names,
                right_threshold=args.head_right,
                left_threshold=args.head_left,
                deadband=args.head_deadband,
                min_visibility=args.head_min_vis,
                torso_weight=args.head_torso_weight,
                face_weight=args.head_face_weight,
                smooth_alpha=args.head_smooth_alpha,
                dwell_frames=args.head_dwell_frames,
            )
        if mode in ("v3", "compare_v2_v3"):
            head_scanner_v3 = HeadScannerV3(
                names=names,
                right_threshold=args.head_right,
                left_threshold=args.head_left,
                deadband=args.head_deadband,
                min_visibility=args.head_min_vis,
                smooth_alpha=args.head_smooth_alpha,
                dwell_frames=args.head_dwell_frames,
                refractory_frames=args.head_refractory_frames,
                min_excursion=args.head_min_excursion,
                min_quality=args.head_min_quality,
                adaptive=args.head_adaptive,
                warmup_frames=args.head_warmup_frames,
                adaptive_sigma_scale=args.head_adaptive_sigma_scale,
                adaptive_deadband_scale=args.head_adaptive_deadband_scale,
                adaptive_min_abs_threshold=args.head_adaptive_min_abs_threshold,
                adaptive_min_sigma=args.head_adaptive_min_sigma,
                use_face=args.head_v3_face,
                face_weight=args.head_v3_face_weight,
            )
    viewer = StickmanViewer(
        xyzv,
        names,
        fps=args.fps,
        visibility_thresh=args.visibility_threshold,
        base_span=args.base_span,
        head_scan_mode=mode,
        head_scanner_v1=head_scanner_v1,
        head_scanner_v2=head_scanner_v2,
        head_scanner_v3=head_scanner_v3,
    )
    viewer.show()


if __name__ == "__main__":
    main()
