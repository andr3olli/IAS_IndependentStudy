"""
Utility helpers for configuration, video IO, and CSV export.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import yaml
import numpy as np


__all__ = [
    "load_config",
    "open_video",
    "video_writer",
    "write_keypoints_csv",
    "write_scan_metrics_csv",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Read a YAML config file and return a dict.
    """
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping. Got: {type(data)}")

    return data


def open_video(path: str | Path) -> Tuple[cv2.VideoCapture, float, int, int]:
    """
    Open a video file for reading.

    Returns:
        cap: cv2.VideoCapture
        fps: float (fallback to 30.0 if metadata missing)
        width, height: ints
    """
    video_path = Path(path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    return cap, float(fps), width, height


def video_writer(path: str | Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Initialize a cv2.VideoWriter with mp4v codec.
    """
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    return writer


def write_keypoints_csv(
    path: str | Path,
    rows: Sequence[Dict[str, Any]],
    names: Sequence[str],
) -> None:
    """
    Persist per-frame keypoints to CSV in a long format:
        frame, landmark, x, y, z, visibility

    Args:
        path: destination CSV path.
        rows: iterable of {"frame_idx": int, "xyzv": np.ndarray shape (N, 4)}
        names: landmark names matching xyzv order.
    """
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["frame", "landmark", "x", "y", "z", "visibility"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            frame_idx = row["frame_idx"]
            xyzv = row["xyzv"]
            if not isinstance(xyzv, np.ndarray):
                xyzv = np.asarray(xyzv)

            for landmark, values in zip(names, xyzv):
                x, y, z, vis = map(float, values.tolist())
                writer.writerow(
                    {
                        "frame": frame_idx,
                        "landmark": landmark,
                        "x": x,
                        "y": y,
                        "z": z,
                        "visibility": vis,
                    }
                )


def write_scan_metrics_csv(
    path: str | Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    """
    Persist per-frame scan analytics in wide format.

    Args:
        path: destination CSV path.
        rows: iterable of dict rows; keys are unioned to build CSV columns.
    """
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame"])
        return

    key_order = ["frame"]
    seen = {"frame"}
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                key_order.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=key_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
