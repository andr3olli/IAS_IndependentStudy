"""
Simple head scanning heuristic: detect left/right head turns and count them.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class HeadScanner:
    """
    Track head yaw normalized by shoulder width and count directional changes.
    """

    def __init__(
        self,
        names: List[str],
        right_threshold: float = 0.18,
        left_threshold: float = -0.18,
        deadband: float = 0.05,
        min_visibility: float = 0.5,
    ) -> None:
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.right_threshold = float(right_threshold)
        self.left_threshold = float(left_threshold)
        self.deadband = float(deadband)
        self.min_visibility = float(min_visibility)
        self.state = "neutral"
        self.count = 0

    def update(self, xyzv: np.ndarray) -> Dict[str, float | int | str]:
        yaw = self._compute_yaw_norm(xyzv)

        if not np.isfinite(yaw):
            return {"yaw_norm": np.nan, "state": self.state, "count": self.count}

        if yaw > self.right_threshold:
            if self.state != "right":
                self.count += 1
            self.state = "right"
        elif yaw < self.left_threshold:
            if self.state != "left":
                self.count += 1
            self.state = "left"
        elif abs(yaw) < self.deadband:
            self.state = "neutral"

        return {"yaw_norm": float(yaw), "state": self.state, "count": self.count}

    def _compute_yaw_norm(self, xyzv: np.ndarray) -> float:
        idx_nose = self.name_to_idx.get("nose")
        idx_l_sh = self.name_to_idx.get("left_shoulder")
        idx_r_sh = self.name_to_idx.get("right_shoulder")
        if None in (idx_nose, idx_l_sh, idx_r_sh):
            return np.nan

        pts = xyzv[[idx_nose, idx_l_sh, idx_r_sh], :]
        vis = pts[:, 3]
        if np.any(vis < self.min_visibility):
            return np.nan

        nose_x = pts[0, 0]
        left_x = pts[1, 0]
        right_x = pts[2, 0]
        span = abs(left_x - right_x)
        if not np.isfinite(span) or span < 1e-3:
            return np.nan

        mid = 0.5 * (left_x + right_x)
        yaw_norm = (nose_x - mid) / span
        return float(yaw_norm)


class HeadScannerV2:
    """
    Improved scanner with:
    - multi-point head center (nose/eyes/ears) instead of only nose
    - torso compensation (shoulder midpoint vs hip midpoint)
    - EMA smoothing
    - dwell frames before state transition (reduces flicker)
    """

    def __init__(
        self,
        names: List[str],
        right_threshold: float = 0.25,
        left_threshold: float = -0.25,
        deadband: float = 0.04,
        min_visibility: float = 0.5,
        torso_weight: float = 0.6,
        face_weight: float = 0.55,
        smooth_alpha: float = 0.25,
        dwell_frames: int = 2,
    ) -> None:
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self._face_indices = [
            i for i, name in enumerate(names) if name.startswith("face_")
        ]
        self._face_nose_idx = self.name_to_idx.get("face_1")
        self.right_threshold = float(right_threshold)
        self.left_threshold = float(left_threshold)
        self.deadband = float(deadband)
        self.min_visibility = float(min_visibility)
        self.torso_weight = float(torso_weight)
        self.face_weight = float(np.clip(face_weight, 0.0, 1.0))
        self.smooth_alpha = float(smooth_alpha)
        self.dwell_frames = int(max(1, dwell_frames))

        self.state = "neutral"
        self.count = 0

        self._smoothed_yaw = np.nan
        self._pending_state: str | None = None
        self._pending_frames = 0
        self._prev_state_for_transition = "neutral"
        self._last_face_points = 0

    def update(self, xyzv: np.ndarray) -> Dict[str, float | int | str]:
        yaw_pose, yaw_face, raw, face_points = self._compute_compensated_yaw(xyzv)
        self._last_face_points = int(face_points)
        if not np.isfinite(raw):
            return {
                "yaw_pose": np.nan,
                "yaw_face": np.nan,
                "yaw_raw": np.nan,
                "yaw_norm": np.nan,
                "state": self.state,
                "prev_state": self._prev_state_for_transition,
                "candidate_state": "none",
                "pending_frames": 0,
                "dwell_frames": self.dwell_frames,
                "face_points": self._last_face_points,
                "count": self.count,
            }

        if not np.isfinite(self._smoothed_yaw):
            self._smoothed_yaw = raw
        else:
            a = self.smooth_alpha
            self._smoothed_yaw = (a * raw) + ((1.0 - a) * self._smoothed_yaw)

        candidate = self._candidate_state(self._smoothed_yaw)
        if candidate is not None:
            if candidate == self._pending_state:
                self._pending_frames += 1
            else:
                self._pending_state = candidate
                self._pending_frames = 1

            if (
                candidate != self.state
                and self._pending_frames >= self.dwell_frames
            ):
                old = self.state
                self.state = candidate
                self._prev_state_for_transition = old
                if self.state in ("left", "right"):
                    self.count += 1
                self._pending_state = None
                self._pending_frames = 0
        else:
            self._pending_state = None
            self._pending_frames = 0

        return {
            "yaw_pose": float(yaw_pose) if np.isfinite(yaw_pose) else np.nan,
            "yaw_face": float(yaw_face) if np.isfinite(yaw_face) else np.nan,
            "yaw_raw": float(raw),
            "yaw_norm": float(self._smoothed_yaw),
            "state": self.state,
            "prev_state": self._prev_state_for_transition,
            "candidate_state": candidate if candidate is not None else "none",
            "pending_frames": int(self._pending_frames),
            "dwell_frames": int(self.dwell_frames),
            "face_points": self._last_face_points,
            "count": self.count,
        }

    def _candidate_state(self, yaw: float) -> str | None:
        if yaw > self.right_threshold:
            return "right"
        if yaw < self.left_threshold:
            return "left"
        if abs(yaw) < self.deadband:
            return "neutral"
        return None

    def _compute_compensated_yaw(self, xyzv: np.ndarray) -> tuple[float, float, float, int]:
        shoulder = self._pair_mid_span(xyzv, "left_shoulder", "right_shoulder")
        hip = self._pair_mid_span(xyzv, "left_hip", "right_hip")
        head_x = self._weighted_head_x(xyzv)

        if shoulder is None or not np.isfinite(head_x):
            return np.nan, np.nan, np.nan, 0

        shoulder_mid_x, shoulder_span = shoulder
        if not np.isfinite(shoulder_span) or shoulder_span < 1e-3:
            return np.nan, np.nan, np.nan, 0

        head_offset = (head_x - shoulder_mid_x) / shoulder_span
        torso_terms = []
        if hip is not None:
            hip_mid_x, hip_span = hip
            denom = max(shoulder_span, hip_span, 1e-3)
            torso_terms.append((shoulder_mid_x - hip_mid_x) / denom)

        left_term = self._side_torso_offset(
            xyzv, "left_shoulder", "left_hip", shoulder_span
        )
        right_term = self._side_torso_offset(
            xyzv, "right_shoulder", "right_hip", shoulder_span
        )
        if np.isfinite(left_term):
            torso_terms.append(left_term)
        if np.isfinite(right_term):
            torso_terms.append(right_term)

        torso_offset = float(np.mean(torso_terms)) if torso_terms else 0.0

        yaw_pose = head_offset - (self.torso_weight * torso_offset)
        yaw_face, face_points = self._face_mesh_yaw(xyzv)
        if np.isfinite(yaw_face):
            yaw = ((1.0 - self.face_weight) * yaw_pose) + (self.face_weight * yaw_face)
        else:
            yaw = yaw_pose
        return (
            float(yaw_pose),
            float(yaw_face),
            float(np.clip(yaw, -2.0, 2.0)),
            int(face_points),
        )

    def _side_torso_offset(
        self,
        xyzv: np.ndarray,
        shoulder_name: str,
        hip_name: str,
        ref_span: float,
    ) -> float:
        s = self._point(xyzv, shoulder_name)
        h = self._point(xyzv, hip_name)
        if s is None or h is None:
            return np.nan
        denom = max(ref_span, 1e-3)
        return float((s[0] - h[0]) / denom)

    def _pair_mid_span(
        self, xyzv: np.ndarray, left_name: str, right_name: str
    ) -> tuple[float, float] | None:
        left = self._point(xyzv, left_name)
        right = self._point(xyzv, right_name)
        if left is None or right is None:
            return None
        left_x = left[0]
        right_x = right[0]
        span = abs(left_x - right_x)
        if not np.isfinite(span) or span < 1e-3:
            return None
        mid = 0.5 * (left_x + right_x)
        return float(mid), float(span)

    def _weighted_head_x(self, xyzv: np.ndarray) -> float:
        names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        xs = []
        ws = []
        for name in names:
            p = self._point(xyzv, name)
            if p is None:
                continue
            x, _, _, vis = p
            xs.append(x)
            ws.append(max(vis, 1e-3))
        if not xs:
            return np.nan
        return float(np.average(np.asarray(xs), weights=np.asarray(ws)))

    def _face_mesh_yaw(self, xyzv: np.ndarray) -> tuple[float, int]:
        if not self._face_indices:
            return np.nan, 0

        xs = []
        for idx in self._face_indices:
            if idx >= xyzv.shape[0]:
                continue
            x, _y, _z, vis = xyzv[idx]
            if np.isfinite(x) and np.isfinite(vis) and vis >= self.min_visibility:
                xs.append(float(x))
        n_points = len(xs)
        if n_points < 20:
            return np.nan, n_points

        xs_arr = np.asarray(xs, dtype=np.float32)
        q_lo = float(np.quantile(xs_arr, 0.1))
        q_hi = float(np.quantile(xs_arr, 0.9))
        span = q_hi - q_lo
        if not np.isfinite(span) or span < 1e-3:
            return np.nan, n_points
        mid = 0.5 * (q_lo + q_hi)

        nose_x = np.nan
        if self._face_nose_idx is not None and self._face_nose_idx < xyzv.shape[0]:
            x, _y, _z, vis = xyzv[self._face_nose_idx]
            if np.isfinite(x) and np.isfinite(vis) and vis >= self.min_visibility:
                nose_x = float(x)
        if not np.isfinite(nose_x):
            nose_x = self._weighted_head_x(xyzv)
        if not np.isfinite(nose_x):
            return np.nan, n_points

        return float((nose_x - mid) / span), n_points

    def _point(self, xyzv: np.ndarray, name: str):
        idx = self.name_to_idx.get(name)
        if idx is None:
            return None
        x, y, z, vis = xyzv[idx]
        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or not np.isfinite(z)
            or not np.isfinite(vis)
            or vis < self.min_visibility
        ):
            return None
        return float(x), float(y), float(z), float(vis)


class HeadScannerV3:
    """
    Higher-robustness scanner with:
    - torso-frame yaw (head offset projected on torso-right axis)
    - optional face cue fusion in the same torso frame
    - adaptive threshold calibration during warmup
    - quality gating + dwell + refractory + minimum excursion
    """

    def __init__(
        self,
        names: List[str],
        right_threshold: float = 0.25,
        left_threshold: float = -0.25,
        deadband: float = 0.04,
        min_visibility: float = 0.5,
        smooth_alpha: float = 0.25,
        dwell_frames: int = 2,
        refractory_frames: int = 10,
        min_excursion: float = 0.03,
        min_quality: float = 0.35,
        adaptive: bool = True,
        warmup_frames: int = 45,
        adaptive_sigma_scale: float = 2.0,
        adaptive_deadband_scale: float = 0.5,
        adaptive_min_abs_threshold: float = 0.12,
        adaptive_min_sigma: float = 0.02,
        use_face: bool = True,
        face_weight: float = 0.35,
    ) -> None:
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self._face_indices = [
            i for i, name in enumerate(names) if name.startswith("face_")
        ]
        self._face_nose_idx = self.name_to_idx.get("face_1")

        self.base_right_threshold = float(right_threshold)
        self.base_left_threshold = float(left_threshold)
        self.base_deadband = float(deadband)
        self.right_threshold = float(right_threshold)
        self.left_threshold = float(left_threshold)
        self.deadband = float(deadband)

        self.min_visibility = float(min_visibility)
        self.smooth_alpha = float(smooth_alpha)
        self.dwell_frames = int(max(1, dwell_frames))
        self.refractory_frames = int(max(0, refractory_frames))
        self.min_excursion = float(max(0.0, min_excursion))
        self.min_quality = float(np.clip(min_quality, 0.0, 1.0))

        self.adaptive = bool(adaptive)
        self.warmup_frames = int(max(1, warmup_frames))
        self.adaptive_sigma_scale = float(max(0.0, adaptive_sigma_scale))
        self.adaptive_deadband_scale = float(max(0.0, adaptive_deadband_scale))
        self.adaptive_min_abs_threshold = float(max(0.0, adaptive_min_abs_threshold))
        self.adaptive_min_sigma = float(max(1e-6, adaptive_min_sigma))
        self.is_calibrated = False
        self._calibration_values: List[float] = []

        self.use_face = bool(use_face)
        self.face_weight = float(np.clip(face_weight, 0.0, 1.0))

        self.state = "neutral"
        self.count = 0
        self._smoothed_yaw = np.nan
        self._pending_state: str | None = None
        self._pending_frames = 0
        self._prev_state_for_transition = "neutral"
        self._last_face_points = 0
        self._frame_idx = -1
        self._last_event_frame = -10**9
        self._last_raw_yaw = np.nan

    def update(self, xyzv: np.ndarray) -> Dict[str, float | int | str]:
        self._frame_idx += 1
        (
            yaw_pose,
            yaw_face,
            raw_yaw,
            quality,
            face_points,
        ) = self._compute_torso_frame_yaw(xyzv)
        self._last_face_points = int(face_points)

        event = 0
        event_direction = "none"

        if not np.isfinite(raw_yaw):
            return {
                "yaw_pose": np.nan,
                "yaw_face": np.nan,
                "yaw_raw": np.nan,
                "yaw_norm": np.nan,
                "quality": float(quality) if np.isfinite(quality) else np.nan,
                "state": self.state,
                "prev_state": self._prev_state_for_transition,
                "candidate_state": "none",
                "pending_frames": 0,
                "dwell_frames": self.dwell_frames,
                "face_points": self._last_face_points,
                "left_threshold": float(self.left_threshold),
                "right_threshold": float(self.right_threshold),
                "deadband": float(self.deadband),
                "calibrated": int(self.is_calibrated),
                "event": event,
                "event_direction": event_direction,
                "count": self.count,
            }

        if not np.isfinite(self._smoothed_yaw):
            self._smoothed_yaw = raw_yaw
        else:
            a = self.smooth_alpha
            self._smoothed_yaw = (a * raw_yaw) + ((1.0 - a) * self._smoothed_yaw)

        self._update_adaptive_thresholds(float(self._smoothed_yaw), float(quality))

        candidate = self._candidate_state(float(self._smoothed_yaw))
        if candidate is not None:
            if candidate == self._pending_state:
                self._pending_frames += 1
            else:
                self._pending_state = candidate
                self._pending_frames = 1

            if candidate != self.state and self._pending_frames >= self.dwell_frames:
                can_transition = True
                if candidate in ("left", "right"):
                    can_transition = bool(quality >= self.min_quality)

                if can_transition:
                    old_state = self.state
                    self.state = candidate
                    self._prev_state_for_transition = old_state

                    if (
                        candidate in ("left", "right")
                        and old_state == "neutral"
                        and self._refractory_ok()
                        and self._excursion_ok(candidate, float(self._smoothed_yaw))
                    ):
                        self.count += 1
                        event = 1
                        event_direction = candidate
                        self._last_event_frame = self._frame_idx

                self._pending_state = None
                self._pending_frames = 0
        else:
            self._pending_state = None
            self._pending_frames = 0

        return {
            "yaw_pose": float(yaw_pose) if np.isfinite(yaw_pose) else np.nan,
            "yaw_face": float(yaw_face) if np.isfinite(yaw_face) else np.nan,
            "yaw_raw": float(raw_yaw),
            "yaw_norm": float(self._smoothed_yaw),
            "quality": float(quality) if np.isfinite(quality) else np.nan,
            "state": self.state,
            "prev_state": self._prev_state_for_transition,
            "candidate_state": candidate if candidate is not None else "none",
            "pending_frames": int(self._pending_frames),
            "dwell_frames": int(self.dwell_frames),
            "face_points": self._last_face_points,
            "left_threshold": float(self.left_threshold),
            "right_threshold": float(self.right_threshold),
            "deadband": float(self.deadband),
            "calibrated": int(self.is_calibrated),
            "event": int(event),
            "event_direction": event_direction,
            "count": self.count,
        }

    def _candidate_state(self, yaw: float) -> str | None:
        if yaw > self.right_threshold:
            return "right"
        if yaw < self.left_threshold:
            return "left"
        if abs(yaw) < self.deadband:
            return "neutral"
        return None

    def _refractory_ok(self) -> bool:
        return (self._frame_idx - self._last_event_frame) >= self.refractory_frames

    def _excursion_ok(self, candidate: str, yaw: float) -> bool:
        if candidate == "right":
            return yaw >= (self.right_threshold + self.min_excursion)
        if candidate == "left":
            return yaw <= (self.left_threshold - self.min_excursion)
        return True

    def _update_adaptive_thresholds(self, yaw: float, quality: float) -> None:
        if not self.adaptive:
            return

        if not self.is_calibrated and np.isfinite(yaw) and np.isfinite(quality):
            if quality >= (self.min_quality * 0.7):
                self._calibration_values.append(float(yaw))

            if len(self._calibration_values) >= self.warmup_frames:
                arr = np.asarray(self._calibration_values, dtype=np.float32)
                mu = float(np.mean(arr))
                sigma = float(np.std(arr))
                sigma_eff = max(sigma, self.adaptive_min_sigma)
                thr_mag = max(
                    self.adaptive_sigma_scale * sigma_eff,
                    self.adaptive_min_abs_threshold,
                )

                self.left_threshold = mu - thr_mag
                self.right_threshold = mu + thr_mag
                self.deadband = max(
                    self.base_deadband,
                    self.adaptive_deadband_scale * sigma_eff,
                )
                self.is_calibrated = True

    def _compute_torso_frame_yaw(
        self, xyzv: np.ndarray
    ) -> tuple[float, float, float, float, int]:
        l_sh = self._point(xyzv, "left_shoulder")
        r_sh = self._point(xyzv, "right_shoulder")
        l_hip = self._point(xyzv, "left_hip")
        r_hip = self._point(xyzv, "right_hip")
        head = self._weighted_head_point(xyzv)

        if (
            l_sh is None
            or r_sh is None
            or l_hip is None
            or r_hip is None
            or head is None
        ):
            return np.nan, np.nan, np.nan, 0.0, 0

        shoulder_vec = r_sh[:3] - l_sh[:3]
        shoulder_span = float(np.linalg.norm(shoulder_vec))
        if not np.isfinite(shoulder_span) or shoulder_span < 1e-6:
            return np.nan, np.nan, np.nan, 0.0, 0

        shoulder_mid = 0.5 * (l_sh[:3] + r_sh[:3])
        hip_mid = 0.5 * (l_hip[:3] + r_hip[:3])
        torso_up = shoulder_mid - hip_mid

        right_u = self._unit(shoulder_vec)
        if right_u is None:
            return np.nan, np.nan, np.nan, 0.0, 0
        # Remove right component from torso_up for stability.
        torso_up = torso_up - (np.dot(torso_up, right_u) * right_u)
        up_u = self._unit(torso_up)
        if up_u is None:
            return np.nan, np.nan, np.nan, 0.0, 0

        head_vec = head[:3] - shoulder_mid
        yaw_pose = float(np.dot(head_vec, right_u) / shoulder_span)

        yaw_face = np.nan
        face_points = 0
        if self.use_face:
            yaw_face, face_points = self._face_torso_yaw(xyzv, right_u)

        if np.isfinite(yaw_face):
            yaw_raw = ((1.0 - self.face_weight) * yaw_pose) + (
                self.face_weight * yaw_face
            )
        else:
            yaw_raw = yaw_pose

        vis_torso = np.mean([l_sh[3], r_sh[3], l_hip[3], r_hip[3]])
        vis_head = float(head[3])
        hip_span = float(np.linalg.norm(r_hip[:3] - l_hip[:3]))
        if np.isfinite(hip_span) and hip_span > 1e-6:
            span_sym = 1.0 - abs(shoulder_span - hip_span) / max(shoulder_span, hip_span)
            span_sym = float(np.clip(span_sym, 0.0, 1.0))
        else:
            span_sym = 0.5

        face_score = (
            float(np.clip(face_points / 80.0, 0.0, 1.0))
            if self.use_face and len(self._face_indices) > 0
            else 0.5
        )
        if np.isfinite(self._last_raw_yaw):
            stability = 1.0 - min(1.0, abs(yaw_raw - self._last_raw_yaw) / 0.4)
        else:
            stability = 1.0
        quality = (
            (0.35 * np.clip(vis_torso, 0.0, 1.0))
            + (0.35 * np.clip(vis_head, 0.0, 1.0))
            + (0.15 * span_sym)
            + (0.10 * face_score)
            + (0.05 * stability)
        )
        self._last_raw_yaw = float(yaw_raw)

        return (
            float(yaw_pose),
            float(yaw_face) if np.isfinite(yaw_face) else np.nan,
            float(np.clip(yaw_raw, -2.0, 2.0)),
            float(np.clip(quality, 0.0, 1.0)),
            int(face_points),
        )

    def _weighted_head_point(self, xyzv: np.ndarray):
        keys = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
        pts = []
        ws = []
        for name in keys:
            p = self._point(xyzv, name)
            if p is None:
                continue
            pts.append(np.asarray(p[:3], dtype=np.float32))
            ws.append(max(float(p[3]), 1e-3))
        if not pts:
            return None
        pts_arr = np.asarray(pts, dtype=np.float32)
        ws_arr = np.asarray(ws, dtype=np.float32)
        xyz = np.average(pts_arr, axis=0, weights=ws_arr)
        vis = float(np.mean(ws_arr))
        return np.asarray([xyz[0], xyz[1], xyz[2], vis], dtype=np.float32)

    def _face_torso_yaw(self, xyzv: np.ndarray, right_u: np.ndarray) -> tuple[float, int]:
        if not self._face_indices:
            return np.nan, 0

        values = []
        face_points = []
        for idx in self._face_indices:
            if idx >= xyzv.shape[0]:
                continue
            x, y, z, vis = xyzv[idx]
            if not np.isfinite(vis) or vis < self.min_visibility:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            p = np.asarray([x, y, z], dtype=np.float32)
            face_points.append(p)
            values.append(float(np.dot(p, right_u)))

        n = len(values)
        if n < 20:
            return np.nan, n

        vals = np.asarray(values, dtype=np.float32)
        lo = float(np.quantile(vals, 0.1))
        hi = float(np.quantile(vals, 0.9))
        span = hi - lo
        if not np.isfinite(span) or span < 1e-6:
            return np.nan, n
        mid = 0.5 * (lo + hi)

        nose_val = np.nan
        if self._face_nose_idx is not None and self._face_nose_idx < xyzv.shape[0]:
            x, y, z, vis = xyzv[self._face_nose_idx]
            if (
                np.isfinite(x)
                and np.isfinite(y)
                and np.isfinite(z)
                and np.isfinite(vis)
                and vis >= self.min_visibility
            ):
                nose_val = float(np.dot(np.asarray([x, y, z], dtype=np.float32), right_u))

        if not np.isfinite(nose_val):
            head = self._weighted_head_point(xyzv)
            if head is not None:
                nose_val = float(np.dot(head[:3], right_u))

        if not np.isfinite(nose_val):
            return np.nan, n
        return float((nose_val - mid) / span), n

    @staticmethod
    def _unit(vec: np.ndarray) -> np.ndarray | None:
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return (vec / norm).astype(np.float32)

    def _point(self, xyzv: np.ndarray, name: str):
        idx = self.name_to_idx.get(name)
        if idx is None or idx >= xyzv.shape[0]:
            return None
        x, y, z, vis = xyzv[idx]
        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or not np.isfinite(z)
            or not np.isfinite(vis)
            or vis < self.min_visibility
        ):
            return None
        return np.asarray([x, y, z, vis], dtype=np.float32)
