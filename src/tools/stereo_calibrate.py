from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a stereo camera pair from checkerboard image pairs."
    )
    parser.add_argument(
        "--left-glob",
        type=str,
        required=True,
        help="Glob pattern for left camera calibration images.",
    )
    parser.add_argument(
        "--right-glob",
        type=str,
        required=True,
        help="Glob pattern for right camera calibration images.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        required=True,
        help="Checkerboard inner corners along rows.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        required=True,
        help="Checkerboard inner corners along cols.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        required=True,
        help="Checkerboard square size in meters (or your chosen unit).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/stereo_calibration.yaml",
        help="Output YAML path.",
    )
    parser.add_argument(
        "--use-sb",
        action="store_true",
        help="Use findChessboardCornersSB when available.",
    )
    return parser.parse_args()


def _glob_paths(pattern: str) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(pattern))


def _object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    obj[:, :2] = grid * float(square_size)
    return obj


def _find_corners(
    gray: np.ndarray,
    pattern_size: Tuple[int, int],
    use_sb: bool,
) -> tuple[bool, np.ndarray | None]:
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
        if found:
            return True, corners.reshape(-1, 1, 2).astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def _calibrate_single(
    obj_points: List[np.ndarray],
    img_points: List[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    rms, k, d, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
    )
    return float(rms), k, d.reshape(-1)


def main() -> None:
    args = parse_args()
    pattern_size = (int(args.cols), int(args.rows))

    left_paths = _glob_paths(args.left_glob)
    right_paths = _glob_paths(args.right_glob)
    if len(left_paths) == 0 or len(right_paths) == 0:
        raise RuntimeError("No calibration images found.")
    if len(left_paths) != len(right_paths):
        raise RuntimeError(
            f"Left/right image count mismatch: {len(left_paths)} vs {len(right_paths)}"
        )

    obj_template = _object_points(pattern_size, args.square_size)
    obj_points: List[np.ndarray] = []
    img_points_left: List[np.ndarray] = []
    img_points_right: List[np.ndarray] = []
    image_size_left = None
    image_size_right = None

    for left_path, right_path in zip(left_paths, right_paths):
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue

        if image_size_left is None:
            image_size_left = (left.shape[1], left.shape[0])
        if image_size_right is None:
            image_size_right = (right.shape[1], right.shape[0])

        found_l, corners_l = _find_corners(left, pattern_size, args.use_sb)
        found_r, corners_r = _find_corners(right, pattern_size, args.use_sb)
        if not found_l or not found_r:
            continue

        obj_points.append(obj_template.copy())
        img_points_left.append(corners_l)
        img_points_right.append(corners_r)

    if len(obj_points) < 8:
        raise RuntimeError(
            f"Only {len(obj_points)} valid stereo pairs found. Need at least 8."
        )
    if image_size_left is None or image_size_right is None:
        raise RuntimeError("Failed to read calibration image sizes.")
    if image_size_left != image_size_right:
        raise RuntimeError(
            f"Left/right image sizes differ: {image_size_left} vs {image_size_right}"
        )

    rms_left, k1, d1 = _calibrate_single(obj_points, img_points_left, image_size_left)
    rms_right, k2, d2 = _calibrate_single(
        obj_points, img_points_right, image_size_right
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    flags = cv2.CALIB_FIX_INTRINSIC
    stereo_rms, k1, d1_cv, k2, d2_cv, r, t, e, f = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        k1,
        d1.reshape(-1, 1),
        k2,
        d2.reshape(-1, 1),
        image_size_left,
        criteria=criteria,
        flags=flags,
    )

    r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
        k1,
        d1_cv,
        k2,
        d2_cv,
        image_size_left,
        r,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    output = {
        "left": {"K": k1.tolist(), "dist": d1_cv.reshape(-1).tolist()},
        "right": {"K": k2.tolist(), "dist": d2_cv.reshape(-1).tolist()},
        "R": r.tolist(),
        "T": t.reshape(-1).tolist(),
        "R1": r1.tolist(),
        "R2": r2.tolist(),
        "P1": p1.tolist(),
        "P2": p2.tolist(),
        "Q": q.tolist(),
        "metrics": {
            "pairs_used": int(len(obj_points)),
            "rms_left": float(rms_left),
            "rms_right": float(rms_right),
            "stereo_rms": float(stereo_rms),
            "image_width": int(image_size_left[0]),
            "image_height": int(image_size_left[1]),
            "roi1": [int(v) for v in roi1],
            "roi2": [int(v) for v in roi2],
            "E": np.asarray(e).tolist(),
            "F": np.asarray(f).tolist(),
        },
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(output, fh, sort_keys=False)

    print(f"Used {len(obj_points)} stereo pairs")
    print(f"Left RMS: {rms_left:.4f}")
    print(f"Right RMS: {rms_right:.4f}")
    print(f"Stereo RMS: {stereo_rms:.4f}")
    print(f"Wrote stereo calibration to {out_path}")


if __name__ == "__main__":
    main()
