"""
Drawing helpers for overlaying MediaPipe skeletons on frames.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import cv2
import mediapipe as mp


PoseLandmark = mp.solutions.pose.PoseLandmark
POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)
NAME_BY_LANDMARK = {lm: lm.name.lower() for lm in PoseLandmark}


def _visible(point: Tuple[float, float, float, float], thresh: float) -> bool:
    return point is not None and point[3] >= thresh


def draw_skeleton(
    frame_bgr,
    xyzv_dict: Mapping[str, Tuple[float, float, float, float]],
    visibility_threshold: float = 0.2,
    color=(0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Draw MediaPipe pose connections and keypoints on the given frame in-place.
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr must be a valid image array")

    for connection in POSE_CONNECTIONS:
        start_name = NAME_BY_LANDMARK[connection[0]]
        end_name = NAME_BY_LANDMARK[connection[1]]
        start_pt = xyzv_dict.get(start_name)
        end_pt = xyzv_dict.get(end_name)
        if _visible(start_pt, visibility_threshold) and _visible(
            end_pt, visibility_threshold
        ):
            cv2.line(
                frame_bgr,
                (int(start_pt[0]), int(start_pt[1])),
                (int(end_pt[0]), int(end_pt[1])),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    radius = max(1, thickness + 1)
    for name, point in xyzv_dict.items():
        if _visible(point, visibility_threshold):
            cv2.circle(
                frame_bgr,
                (int(point[0]), int(point[1])),
                radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

