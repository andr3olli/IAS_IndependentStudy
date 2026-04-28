"""
MediaPipe face mesh backend (single-face) for dense head landmarks.

Returns per-frame face points as:
    xyzv: np.ndarray shape (N, 4) where N is 468 or 478
    names: ["face_0", ..., "face_{N-1}"]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import mediapipe as mp
import numpy as np


class MediaPipeFaceMesh:
    """Thin wrapper around MediaPipe FaceMesh for video inference."""

    def __init__(
        self,
        refine_landmarks: bool = True,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_face = mp.solutions.face_mesh
        self._refine_landmarks = bool(refine_landmarks)
        self._num_landmarks = (
            self._mp_face.FACEMESH_NUM_LANDMARKS_WITH_IRISES
            if self._refine_landmarks
            else self._mp_face.FACEMESH_NUM_LANDMARKS
        )
        self._names: List[str] = [f"face_{i}" for i in range(self._num_landmarks)]
        self._face = self._mp_face.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=self._refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def names(self) -> List[str]:
        return list(self._names)

    def infer(self, frame_bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Dict:
        if frame_bgr is None or frame_bgr.ndim != 3:
            raise ValueError("infer expects a color BGR frame (H,W,3)")

        height, width = frame_bgr.shape[:2]
        x0, y0, x1, y1 = self._sanitize_roi(roi, width, height)
        patch = frame_bgr[y0:y1, x0:x1]
        patch_h, patch_w = patch.shape[:2]
        if patch_h < 2 or patch_w < 2:
            x0, y0, x1, y1 = 0, 0, width, height
            patch = frame_bgr
            patch_h, patch_w = height, width

        patch_rgb = patch[:, :, ::-1]
        results = self._face.process(patch_rgb)

        xyzv = np.full((self._num_landmarks, 4), np.nan, dtype=np.float32)
        xyzv_dict: Dict[str, Tuple[float, float, float, float]] = {
            name: (np.nan, np.nan, np.nan, np.nan) for name in self._names
        }

        if not results.multi_face_landmarks:
            return {
                "ok": False,
                "width": width,
                "height": height,
                "names": self.names(),
                "xyzv_dict": xyzv_dict,
                "xyzv": xyzv,
                "timestamp_ms": None,
            }

        # Single-face baseline: pick the first detected face.
        lm = results.multi_face_landmarks[0].landmark
        n = min(len(lm), self._num_landmarks)
        for i in range(n):
            p = lm[i]
            x_px = float((p.x * patch_w) + x0)
            y_px = float((p.y * patch_h) + y0)
            z_rel = float(p.z)
            # FaceMesh landmarks don't expose visibility; mark detected points as visible.
            vis = 1.0
            xyzv[i] = (x_px, y_px, z_rel, vis)
            xyzv_dict[self._names[i]] = (x_px, y_px, z_rel, vis)

        return {
            "ok": True,
            "width": width,
            "height": height,
            "names": self.names(),
            "xyzv_dict": xyzv_dict,
            "xyzv": xyzv,
            "timestamp_ms": None,
        }

    def _sanitize_roi(
        self,
        roi: tuple[int, int, int, int] | None,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if roi is None:
            return 0, 0, width, height
        x0, y0, x1, y1 = roi
        x0 = max(0, min(int(x0), width - 1))
        y0 = max(0, min(int(y0), height - 1))
        x1 = max(x0 + 1, min(int(x1), width))
        y1 = max(y0 + 1, min(int(y1), height))
        return x0, y0, x1, y1

    def close(self) -> None:
        if hasattr(self, "_face") and self._face is not None:
            self._face.close()
            self._face = None
