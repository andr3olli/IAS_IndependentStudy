"""
MediaPipe pose backend (single-person) for the PlaiPose pipeline.

Responsibilities:
- Initialize MediaPipe BlazePose for video streams
- Run inference on BGR frames
- Return per-frame keypoints as a dict and as a NumPy array
- Remain data-only (no drawing here)

Dependencies: mediapipe, numpy

Usage:
    from pose.mediapipe_pose import MediaPipePose
    mp_pose = MediaPipePose(model_complexity=1, smooth_landmarks=True)
    keypoints = mp_pose.infer(frame_bgr)  # dict with x,y,visibility in pixels
    mp_pose.close()

Returned structure (for a W x H frame):
    {
        'ok': True/False,                  # detection happened
        'width': W, 'height': H,           # frame size
        'names': [...33 landmark names...],
        'xyzv_dict': { name: (x_px, y_px, z_rel, visibility), ... },
        'xyzv': np.ndarray shape (33, 4),  # rows ordered like 'names'
        'world_ok': True/False,
        'world_xyzv_dict': { name: (x_w, y_w, z_w, visibility), ... },
        'world_xyzv': np.ndarray shape (33, 4),
        'timestamp_ms': float | None       # optional (not set here)
    }

This module keeps MediaPipe-specific concerns isolated from the rest of the pipeline.
"""
from typing import Dict, List, Tuple

import numpy as np
import mediapipe as mp


class MediaPipePose:
    """Thin wrapper around MediaPipe BlazePose for video inference."""

    # Stable list of landmark names in MediaPipe index order (0..32)
    _LM_ENUM = mp.solutions.pose.PoseLandmark
    NAMES: List[str] = [lm.name.lower() for lm in _LM_ENUM]

    def __init__(
        self,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        static_image_mode: bool = False,
    ) -> None:
        """
        Args:
            model_complexity: 0 (lite), 1 (full), 2 (heavy)
            smooth_landmarks: temporal smoothing inside MediaPipe
            enable_segmentation: not needed fr our use-case
            static_image_mode: set False for video streams
        """
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            smooth_landmarks=smooth_landmarks,
        )

    @classmethod
    def names(cls) -> List[str]:
        """Ordered landmark names (len == 33)."""
        return list(cls.NAMES)

    def infer(self, frame_bgr: np.ndarray) -> Dict:
        """
        Run pose inference on a single BGR frame.

        Returns a dict with keys:
            ok: bool
            width, height: ints
            names: List[str]
            xyzv_dict: Dict[name, (x_px, y_px, z_rel, visibility)]
            xyzv: np.ndarray of shape (33, 4)
            world_xyzv_dict: Dict[name, (x_w, y_w, z_w, visibility)]
            world_xyzv: np.ndarray of shape (33, 4)
        """
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("infer expects a color BGR frame (H,W,3)")

        height, width = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            # No detection: return NaNs but keep shapes predictable
            xyzv = np.full((len(self.NAMES), 4), np.nan, dtype=np.float32)
            world_xyzv = np.full((len(self.NAMES), 4), np.nan, dtype=np.float32)
            return {
                'ok': False,
                'width': width,
                'height': height,
                'names': self.names(),
                'xyzv_dict': {name: (np.nan, np.nan, np.nan, np.nan) for name in self.NAMES},
                'xyzv': xyzv,
                'world_ok': False,
                'world_xyzv_dict': {name: (np.nan, np.nan, np.nan, np.nan) for name in self.NAMES},
                'world_xyzv': world_xyzv,
                'timestamp_ms': None,
            }

        lm = results.pose_landmarks.landmark
        xyzv = np.zeros((len(self.NAMES), 4), dtype=np.float32)
        xyzv_dict: Dict[str, Tuple[float, float, float, float]] = {}

        for i, name in enumerate(self.NAMES):
            p = lm[i]
            x_px = float(p.x * width)
            y_px = float(p.y * height)
            z_rel = float(p.z)
            vis = float(getattr(p, 'visibility', np.nan))
            xyzv[i] = (x_px, y_px, z_rel, vis)
            xyzv_dict[name] = (x_px, y_px, z_rel, vis)

        world_xyzv = np.full((len(self.NAMES), 4), np.nan, dtype=np.float32)
        world_xyzv_dict: Dict[str, Tuple[float, float, float, float]] = {
            name: (np.nan, np.nan, np.nan, np.nan) for name in self.NAMES
        }
        world_ok = bool(results.pose_world_landmarks)
        if world_ok:
            lm_world = results.pose_world_landmarks.landmark
            for i, name in enumerate(self.NAMES):
                p = lm_world[i]
                x_w = float(p.x)
                y_w = float(p.y)
                z_w = float(p.z)
                vis = float(getattr(p, 'visibility', np.nan))
                world_xyzv[i] = (x_w, y_w, z_w, vis)
                world_xyzv_dict[name] = (x_w, y_w, z_w, vis)

        return {
            'ok': True,
            'width': width,
            'height': height,
            'names': self.names(),
            'xyzv_dict': xyzv_dict,
            'xyzv': xyzv,
            'world_ok': world_ok,
            'world_xyzv_dict': world_xyzv_dict,
            'world_xyzv': world_xyzv,
            'timestamp_ms': None,
        }

    def close(self) -> None:
        if hasattr(self, '_pose') and self._pose is not None:
            self._pose.close()
            self._pose = None
