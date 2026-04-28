from __future__ import annotations

import argparse
from typing import List, Tuple

import cv2
import numpy as np

from pose.mediapipe_pose import MediaPipePose
from pose.mediapipe_face import MediaPipeFaceMesh
from pose.body_points import build_virtual_body_names, append_virtual_body_points
from triangulation.stereo import StereoTriangulator, load_stereo_calibration
from utils.io import (
    load_config,
    open_video,
    video_writer,
    write_keypoints_csv,
    write_scan_metrics_csv,
)
from viz.draw_skeleton import draw_skeleton
from viz.stickman3d import Stickman3DRenderer
from analytics.head_scan import HeadScanner, HeadScannerV2, HeadScannerV3


def _tuple_from_config(value, default, length: int) -> Tuple[float, ...]:
    if isinstance(value, (list, tuple)):
        if len(value) == length:
            return tuple(value)
        if len(value) == 1:
            return tuple(value * length)
    if isinstance(value, (int, float)):
        return tuple([value] * length)
    return tuple(default)


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    scale = target_height / frame.shape[0]
    target_width = int(round(frame.shape[1] * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if right is None:
        return left
    target_height = max(left.shape[0], right.shape[0])
    left_resized = _resize_to_height(left, target_height)
    right_resized = _resize_to_height(right, target_height)
    return np.hstack([left_resized, right_resized])


def _skip_initial_frames(cap: cv2.VideoCapture, count: int) -> None:
    remaining = int(max(0, count))
    while remaining > 0:
        ret, _ = cap.read()
        if not ret:
            break
        remaining -= 1


def _append_scan_row(
    scan_rows: List[dict],
    frame_idx: int,
    v1: dict | None = None,
    v2: dict | None = None,
    v3: dict | None = None,
) -> None:
    row = {"frame": int(frame_idx)}
    for prefix, info in (("v1", v1), ("v2", v2), ("v3", v3)):
        if info is None:
            continue
        for key, value in info.items():
            row[f"{prefix}_{key}"] = value
    scan_rows.append(row)


def _landmark_xy(
    xyzv: np.ndarray,
    name_to_idx: dict,
    name: str,
    min_visibility: float = 0.2,
) -> tuple[float, float] | None:
    idx = name_to_idx.get(name)
    if idx is None:
        return None
    x, y, _z, vis = xyzv[idx]
    if (
        not np.isfinite(x)
        or not np.isfinite(y)
        or not np.isfinite(vis)
        or vis < min_visibility
    ):
        return None
    return float(x), float(y)


def _estimate_head_roi(
    xyzv_pose: np.ndarray,
    pose_names: List[str],
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.2,
    min_size_px: int = 120,
    pad_scale: float = 1.8,
) -> tuple[int, int, int, int] | None:
    name_to_idx = {name: i for i, name in enumerate(pose_names)}
    head_keys = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
    ]
    shoulder_keys = ["left_shoulder", "right_shoulder"]

    pts = []
    for key in head_keys + shoulder_keys:
        p = _landmark_xy(xyzv_pose, name_to_idx, key, min_visibility=min_visibility)
        if p is not None:
            pts.append(p)
    if len(pts) < 3:
        return None

    xs = np.asarray([p[0] for p in pts], dtype=np.float32)
    ys = np.asarray([p[1] for p in pts], dtype=np.float32)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)
    size = max(width, height) * max(1.0, float(pad_scale))
    size = max(size, float(min_size_px))

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    # bias ROI up slightly so face occupies more of the crop.
    cy -= 0.1 * size

    x0 = int(round(cx - (size * 0.5)))
    y0 = int(round(cy - (size * 0.5)))
    x1 = int(round(cx + (size * 0.5)))
    y1 = int(round(cy + (size * 0.5)))

    x0 = max(0, min(x0, frame_width - 1))
    y0 = max(0, min(y0, frame_height - 1))
    x1 = max(x0 + 1, min(x1, frame_width))
    y1 = max(y0 + 1, min(y1, frame_height))
    return x0, y0, x1, y1


def _draw_head_scan(
    frame: np.ndarray,
    yaw_info: dict,
    text_color: Tuple[float, float, float],
    bar_color: Tuple[float, float, float],
    left_threshold: float,
    right_threshold: float,
    deadband: float,
    label: str = "Head",
    text_offset: int = 0,
    bar_index: int = 0,
) -> None:
    h, w = frame.shape[:2]
    count = yaw_info.get("count", 0)
    yaw = yaw_info.get("yaw_norm", np.nan)
    state = yaw_info.get("state", "neutral")

    pad = 20
    anchor_x = pad
    anchor_y = pad + 20 + text_offset

    cv2.putText(
        frame,
        f"{label} scans: {count}",
        (anchor_x, anchor_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        lineType=cv2.LINE_AA,
    )

    if np.isnan(yaw):
        return

    yaw_clamped = float(np.clip(yaw, -1.0, 1.0))
    cv2.putText(
        frame,
        f"{label} yaw: {yaw:.2f} ({state})",
        (anchor_x, anchor_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2,
        lineType=cv2.LINE_AA,
    )
    bar_w = int(w * 0.30)
    bar_h = 12
    x0 = pad
    x1 = x0 + bar_w
    y0 = h - pad - bar_h - 10 - (bar_index * (bar_h + 12))
    y1 = y0 + bar_h
    cx = x0 + bar_w // 2

    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), thickness=2)
    cv2.line(frame, (cx, y0), (cx, y1), (120, 120, 120), 1, lineType=cv2.LINE_AA)

    # threshold markers
    def _tick_at(th: float, color):
        th_clamped = np.clip(th, -1.0, 1.0)
        tx = int(cx + th_clamped * (bar_w // 2))
        cv2.line(frame, (tx, y0), (tx, y1), color, 1, lineType=cv2.LINE_AA)

    _tick_at(left_threshold, (180, 120, 120))
    _tick_at(right_threshold, (180, 120, 120))
    if deadband > 0:
        _tick_at(deadband, (120, 180, 120))
        _tick_at(-deadband, (120, 180, 120))

    fill_x = int(cx + yaw_clamped * (bar_w // 2))
    cv2.rectangle(frame, (cx, y0), (fill_x, y1), bar_color, thickness=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MediaPipe Pose baseline pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mvp.yaml",
        help="Path to YAML config describing input/output paths.",
    )
    return parser.parse_args()


def _run_stereo_pipeline(config: dict) -> None:
    overlay_video = config["overlay_video"]
    keypoints_csv = config["keypoints_csv"]

    stereo_cfg = config.get("stereo", {})
    pose_kwargs = config.get("pose", {})
    body_cfg = config.get("body_enrichment", {})
    overlay_cfg = config.get("overlay", {})
    stickman_cfg = config.get("stickman", {})
    head_cfg = config.get("analytics", {}).get("head_scan", {})

    left_video = stereo_cfg.get("left_video")
    right_video = stereo_cfg.get("right_video")
    calibration_path = stereo_cfg.get("calibration")
    if not left_video or not right_video:
        raise ValueError(
            "Stereo mode requires stereo.left_video and stereo.right_video."
        )
    if not calibration_path:
        raise ValueError("Stereo mode requires stereo.calibration path.")

    calibration = load_stereo_calibration(calibration_path)
    triangulator = StereoTriangulator(
        calibration=calibration,
        min_visibility=float(stereo_cfg.get("min_visibility", 0.4)),
        max_reprojection_error=float(
            stereo_cfg.get("max_reprojection_error", 0.02)
        ),
    )

    mp_pose_left = MediaPipePose(
        model_complexity=pose_kwargs.get("model_complexity", 1),
        smooth_landmarks=pose_kwargs.get("smooth_landmarks", True),
        enable_segmentation=pose_kwargs.get("enable_segmentation", False),
        static_image_mode=pose_kwargs.get("static_image_mode", False),
    )
    mp_pose_right = MediaPipePose(
        model_complexity=pose_kwargs.get("model_complexity", 1),
        smooth_landmarks=pose_kwargs.get("smooth_landmarks", True),
        enable_segmentation=pose_kwargs.get("enable_segmentation", False),
        static_image_mode=pose_kwargs.get("static_image_mode", False),
    )

    cap_left, fps_left, _width_left, _height_left = open_video(left_video)
    cap_right, fps_right, _width_right, _height_right = open_video(right_video)
    _skip_initial_frames(cap_left, int(stereo_cfg.get("left_start_frame", 0)))
    _skip_initial_frames(cap_right, int(stereo_cfg.get("right_start_frame", 0)))

    fps_cfg = stereo_cfg.get("fps")
    fps = float(fps_cfg) if fps_cfg is not None else float(min(fps_left, fps_right))
    if abs(fps_left - fps_right) > 0.5:
        print(
            f"Warning: left fps ({fps_left:.2f}) != right fps ({fps_right:.2f}); "
            f"using output fps={fps:.2f}"
        )

    visibility_threshold = overlay_cfg.get("visibility_threshold", 0.2)

    rows: List[dict] = []
    scan_rows: List[dict] = []
    pose_names = mp_pose_left.names()
    base_names = pose_names
    body_dense_enabled = bool(body_cfg.get("enabled", False))
    body_subdivisions = int(body_cfg.get("subdivisions", 1))
    body_virtual_names: List[str] = []
    if body_dense_enabled and body_subdivisions > 0:
        body_virtual_names = build_virtual_body_names(base_names, body_subdivisions)
    names = base_names + body_virtual_names
    frame_idx = 0

    overlay_writer = None
    stickman_writer = None
    composite_writer = None

    stickman_renderer = None
    stickman_enabled = bool(stickman_cfg.get("enabled", False))
    if stickman_enabled:
        stickman_renderer = Stickman3DRenderer(
            names=names,
            visibility_threshold=stickman_cfg.get("visibility_threshold", 0.2),
            fig_size=_tuple_from_config(
                stickman_cfg.get("fig_size", (4.0, 4.0)),
                default=(4.0, 4.0),
                length=2,
            ),
            dpi=int(stickman_cfg.get("dpi", 200)),
            elev=float(stickman_cfg.get("elev", 15.0)),
            azim=float(stickman_cfg.get("azim", -60.0)),
            axis_limits=stickman_cfg.get("axis_limits"),
            line_color=_tuple_from_config(
                stickman_cfg.get("line_color", (0.9, 0.4, 0.1)),
                default=(0.9, 0.4, 0.1),
                length=3,
            ),
            joint_color=_tuple_from_config(
                stickman_cfg.get("joint_color", (0.1, 0.8, 0.9)),
                default=(0.1, 0.8, 0.9),
                length=3,
            ),
            background_color=_tuple_from_config(
                stickman_cfg.get("background_color", (0.05, 0.05, 0.05)),
                default=(0.05, 0.05, 0.05),
                length=3,
            ),
            line_width=float(stickman_cfg.get("line_width", 2.0)),
            joint_size=float(stickman_cfg.get("joint_size", 20.0)),
            x_scale=float(stickman_cfg.get("x_scale", 1.0)),
        )

    head_scan_enabled = bool(head_cfg.get("enabled", False))
    head_mode = str(head_cfg.get("mode", "v3")).lower()
    if head_mode not in {"v1", "v2", "v3", "compare", "compare_v2_v3"}:
        head_mode = "v3"
    head_scanner_v1 = None
    head_scanner_v2 = None
    head_scanner_v3 = None
    scan_metrics_csv = head_cfg.get("metrics_csv")
    if head_scan_enabled and head_mode in {"v1", "compare"}:
        head_scanner_v1 = HeadScanner(
            pose_names,
            right_threshold=head_cfg.get("right_threshold", 0.18),
            left_threshold=head_cfg.get("left_threshold", -0.18),
            deadband=head_cfg.get("deadband", 0.05),
            min_visibility=head_cfg.get("min_visibility", 0.5),
        )
    if head_scan_enabled and head_mode in {"v2", "compare", "compare_v2_v3"}:
        head_scanner_v2 = HeadScannerV2(
            pose_names,
            right_threshold=head_cfg.get("right_threshold", 0.25),
            left_threshold=head_cfg.get("left_threshold", -0.25),
            deadband=head_cfg.get("deadband", 0.04),
            min_visibility=head_cfg.get("min_visibility", 0.5),
            torso_weight=head_cfg.get("torso_weight", 0.6),
            face_weight=head_cfg.get("face_weight", 0.0),
            smooth_alpha=head_cfg.get("smooth_alpha", 0.25),
            dwell_frames=head_cfg.get("dwell_frames", 2),
        )
    if head_scan_enabled and head_mode in {"v3", "compare_v2_v3"}:
        head_scanner_v3 = HeadScannerV3(
            pose_names,
            right_threshold=head_cfg.get("right_threshold", 0.25),
            left_threshold=head_cfg.get("left_threshold", -0.25),
            deadband=head_cfg.get("deadband", 0.04),
            min_visibility=head_cfg.get("min_visibility", 0.5),
            smooth_alpha=head_cfg.get("smooth_alpha", 0.25),
            dwell_frames=head_cfg.get("dwell_frames", 2),
            refractory_frames=head_cfg.get("refractory_frames", 10),
            min_excursion=head_cfg.get("min_excursion", 0.03),
            min_quality=head_cfg.get("min_quality", 0.35),
            adaptive=head_cfg.get("adaptive", True),
            warmup_frames=head_cfg.get("warmup_frames", 45),
            adaptive_sigma_scale=head_cfg.get("adaptive_sigma_scale", 2.0),
            adaptive_deadband_scale=head_cfg.get("adaptive_deadband_scale", 0.5),
            adaptive_min_abs_threshold=head_cfg.get(
                "adaptive_min_abs_threshold", 0.12
            ),
            adaptive_min_sigma=head_cfg.get("adaptive_min_sigma", 0.02),
            use_face=head_cfg.get("v3_use_face", False),
            face_weight=head_cfg.get("v3_face_weight", 0.35),
        )

    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            if not ret_left or not ret_right:
                break

            result_left = mp_pose_left.infer(frame_left)
            result_right = mp_pose_right.infer(frame_right)

            xyzv_pose_3d = triangulator.triangulate(
                result_left["xyzv"], result_right["xyzv"]
            )
            xyzv_output = xyzv_pose_3d
            if body_dense_enabled and body_subdivisions > 0:
                xyzv_output = append_virtual_body_points(
                    base_names=base_names,
                    xyzv=xyzv_pose_3d,
                    subdivisions=body_subdivisions,
                )

            yaw_info_v1 = None
            yaw_info_v2 = None
            yaw_info_v3 = None
            if head_scanner_v1 is not None:
                yaw_info_v1 = head_scanner_v1.update(xyzv_pose_3d)
            if head_scanner_v2 is not None:
                yaw_info_v2 = head_scanner_v2.update(xyzv_pose_3d)
            if head_scanner_v3 is not None:
                yaw_info_v3 = head_scanner_v3.update(xyzv_pose_3d)

            left_vis = frame_left.copy()
            right_vis = frame_right.copy()
            if overlay_cfg.get("enabled", True):
                if result_left.get("xyzv_dict"):
                    draw_skeleton(
                        left_vis,
                        result_left["xyzv_dict"],
                        visibility_threshold=visibility_threshold,
                    )
                if result_right.get("xyzv_dict"):
                    draw_skeleton(
                        right_vis,
                        result_right["xyzv_dict"],
                        visibility_threshold=visibility_threshold,
                    )
            overlay_pair = _stack_side_by_side(left_vis, right_vis)

            if overlay_cfg.get("write_video", True):
                if overlay_writer is None:
                    frame_size = (overlay_pair.shape[1], overlay_pair.shape[0])
                    overlay_writer = video_writer(overlay_video, fps, frame_size)
                overlay_writer.write(overlay_pair)

            if stickman_renderer is not None:
                stickman_frame_rgb = stickman_renderer.render(xyzv_output)
                stickman_frame_bgr = cv2.cvtColor(
                    stickman_frame_rgb, cv2.COLOR_RGB2BGR
                )
                if yaw_info_v1 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v1,
                        text_color=_tuple_from_config(
                            head_cfg.get("v1_text_color", (80, 210, 255)),
                            default=(80, 210, 255),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v1_bar_color", (0, 165, 255)),
                            default=(0, 165, 255),
                            length=3,
                        ),
                        left_threshold=float(head_cfg.get("left_threshold", -0.25)),
                        right_threshold=float(head_cfg.get("right_threshold", 0.25)),
                        deadband=float(head_cfg.get("deadband", 0.04)),
                        label="V1",
                        text_offset=0 if head_mode != "compare" else 0,
                        bar_index=0 if head_mode != "compare" else 1,
                    )
                if yaw_info_v2 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v2,
                        text_color=_tuple_from_config(
                            head_cfg.get("v2_text_color", (0, 255, 170)),
                            default=(0, 255, 170),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v2_bar_color", (0, 220, 120)),
                            default=(0, 220, 120),
                            length=3,
                        ),
                        left_threshold=float(head_cfg.get("left_threshold", -0.25)),
                        right_threshold=float(head_cfg.get("right_threshold", 0.25)),
                        deadband=float(head_cfg.get("deadband", 0.04)),
                        label="V2",
                        text_offset=60 if head_mode == "compare" else 0,
                        bar_index=0 if head_mode not in {"compare", "compare_v2_v3"} else 1,
                    )
                if yaw_info_v3 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v3,
                        text_color=_tuple_from_config(
                            head_cfg.get("v3_text_color", (255, 220, 0)),
                            default=(255, 220, 0),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v3_bar_color", (0, 220, 220)),
                            default=(0, 220, 220),
                            length=3,
                        ),
                        left_threshold=float(
                            yaw_info_v3.get(
                                "left_threshold",
                                head_cfg.get("left_threshold", -0.25),
                            )
                        ),
                        right_threshold=float(
                            yaw_info_v3.get(
                                "right_threshold",
                                head_cfg.get("right_threshold", 0.25),
                            )
                        ),
                        deadband=float(
                            yaw_info_v3.get(
                                "deadband",
                                head_cfg.get("deadband", 0.04),
                            )
                        ),
                        label="V3",
                        text_offset=60 if head_mode == "compare_v2_v3" else 0,
                        bar_index=0,
                    )

                if stickman_cfg.get("write_stickman_video", True):
                    stickman_path = stickman_cfg.get("video")
                    if not stickman_path:
                        raise ValueError(
                            "stickman.video path is required when write_stickman_video is True."
                        )
                    if stickman_writer is None:
                        frame_size = (
                            stickman_frame_bgr.shape[1],
                            stickman_frame_bgr.shape[0],
                        )
                        stickman_writer = video_writer(stickman_path, fps, frame_size)
                    stickman_writer.write(stickman_frame_bgr)

                if stickman_cfg.get("write_composite_video", True):
                    composite_path = stickman_cfg.get("composite_video")
                    if not composite_path:
                        raise ValueError(
                            "stickman.composite_video path is required when write_composite_video is True."
                        )
                    composite_frame = _stack_side_by_side(
                        overlay_pair, stickman_frame_bgr
                    )
                    if composite_writer is None:
                        comp_size = (
                            composite_frame.shape[1],
                            composite_frame.shape[0],
                        )
                        composite_writer = video_writer(composite_path, fps, comp_size)
                    composite_writer.write(composite_frame)

            rows.append({"frame_idx": frame_idx, "xyzv": xyzv_output})
            if head_scan_enabled and scan_metrics_csv:
                _append_scan_row(
                    scan_rows,
                    frame_idx=frame_idx,
                    v1=yaw_info_v1,
                    v2=yaw_info_v2,
                    v3=yaw_info_v3,
                )
            frame_idx += 1

    finally:
        cap_left.release()
        cap_right.release()
        mp_pose_left.close()
        mp_pose_right.close()
        if overlay_writer is not None:
            overlay_writer.release()
        if stickman_writer is not None:
            stickman_writer.release()
        if composite_writer is not None:
            composite_writer.release()

    write_keypoints_csv(keypoints_csv, rows, names)
    if head_scan_enabled and scan_metrics_csv:
        write_scan_metrics_csv(scan_metrics_csv, scan_rows)
    if overlay_cfg.get("write_video", True):
        print(f"Wrote stereo overlay video to {overlay_video}")
    print(f"Wrote triangulated keypoints CSV to {keypoints_csv}")
    if head_scan_enabled and scan_metrics_csv:
        print(f"Wrote scan metrics CSV to {scan_metrics_csv}")
    if (
        stickman_enabled
        and stickman_cfg.get("write_stickman_video", True)
        and stickman_cfg.get("video")
    ):
        print(f"Wrote stickman video to {stickman_cfg['video']}")
    if (
        stickman_enabled
        and stickman_cfg.get("write_composite_video", True)
        and stickman_cfg.get("composite_video")
    ):
        print(f"Wrote composite video to {stickman_cfg['composite_video']}")


def run_pipeline(config_path: str) -> None:
    config = load_config(config_path)
    stereo_cfg = config.get("stereo", {})
    if bool(stereo_cfg.get("enabled", False)):
        _run_stereo_pipeline(config)
        return

    input_video = config["input_video"]
    overlay_video = config["overlay_video"]
    keypoints_csv = config["keypoints_csv"]

    pose_kwargs = config.get("pose", {})
    mono3d_cfg = config.get("monocular_3d", {})
    face_cfg = config.get("face", {})
    body_cfg = config.get("body_enrichment", {})
    overlay_cfg = config.get("overlay", {})
    stickman_cfg = config.get("stickman", {})
    head_cfg = config.get("analytics", {}).get("head_scan", {})
    mono3d_enabled = bool(mono3d_cfg.get("enabled", False))
    mono3d_source = str(mono3d_cfg.get("source", "pose_world")).lower()
    use_world_output = mono3d_enabled and mono3d_source in {
        "pose_world",
        "world",
        "world_landmarks",
    }
    mono3d_fallback_to_relative = bool(
        mono3d_cfg.get("fallback_to_relative", True)
    )

    mp_pose = MediaPipePose(
        model_complexity=pose_kwargs.get("model_complexity", 1),
        smooth_landmarks=pose_kwargs.get("smooth_landmarks", True),
        enable_segmentation=pose_kwargs.get("enable_segmentation", False),
        static_image_mode=pose_kwargs.get("static_image_mode", False),
    )

    cap, fps, width, height = open_video(input_video)
    writer = video_writer(overlay_video, fps, (width, height))

    visibility_threshold = overlay_cfg.get("visibility_threshold", 0.2)

    rows: List[dict] = []
    scan_rows: List[dict] = []
    pose_names = mp_pose.names()
    face_enabled = bool(face_cfg.get("enabled", False))
    if use_world_output and face_enabled:
        print(
            "monocular_3d is enabled: disabling face mesh fusion to keep output "
            "coordinates in one consistent 3D frame."
        )
        face_enabled = False
    mp_face = None
    face_names: List[str] = []
    if face_enabled:
        mp_face = MediaPipeFaceMesh(
            refine_landmarks=face_cfg.get("refine_landmarks", True),
            static_image_mode=face_cfg.get("static_image_mode", False),
            max_num_faces=face_cfg.get("max_num_faces", 1),
            min_detection_confidence=face_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=face_cfg.get("min_tracking_confidence", 0.5),
        )
        face_names = mp_face.names()
    face_use_pose_roi = bool(face_cfg.get("use_pose_head_roi", True))
    face_fallback_full_frame = bool(face_cfg.get("fallback_full_frame", True))
    face_roi_min_visibility = float(face_cfg.get("roi_min_visibility", 0.2))
    face_roi_min_size_px = int(face_cfg.get("roi_min_size_px", 120))
    face_roi_pad_scale = float(face_cfg.get("roi_pad_scale", 1.8))
    base_names = pose_names + face_names
    body_dense_enabled = bool(body_cfg.get("enabled", False))
    body_subdivisions = int(body_cfg.get("subdivisions", 1))
    body_virtual_names: List[str] = []
    if body_dense_enabled and body_subdivisions > 0:
        body_virtual_names = build_virtual_body_names(base_names, body_subdivisions)
    names = base_names + body_virtual_names
    frame_idx = 0

    stickman_renderer = None
    stickman_writer = None
    composite_writer = None
    stickman_enabled = stickman_cfg.get("enabled", False)
    head_scan_enabled = head_cfg.get("enabled", False)
    head_mode = str(head_cfg.get("mode", "v3")).lower()
    if head_mode not in {"v1", "v2", "v3", "compare", "compare_v2_v3"}:
        head_mode = "v3"
    scan_metrics_csv = head_cfg.get("metrics_csv")

    head_scanner_v1 = None
    head_scanner_v2 = None
    head_scanner_v3 = None
    if head_scan_enabled and head_mode in {"v1", "compare"}:
        head_scanner_v1 = HeadScanner(
            pose_names,
            right_threshold=head_cfg.get("right_threshold", 0.18),
            left_threshold=head_cfg.get("left_threshold", -0.18),
            deadband=head_cfg.get("deadband", 0.05),
            min_visibility=head_cfg.get("min_visibility", 0.5),
        )
    if head_scan_enabled and head_mode in {"v2", "compare", "compare_v2_v3"}:
        head_scanner_v2 = HeadScannerV2(
            base_names,
            right_threshold=head_cfg.get("right_threshold", 0.25),
            left_threshold=head_cfg.get("left_threshold", -0.25),
            deadband=head_cfg.get("deadband", 0.04),
            min_visibility=head_cfg.get("min_visibility", 0.5),
            torso_weight=head_cfg.get("torso_weight", 0.6),
            face_weight=head_cfg.get("face_weight", 0.55),
            smooth_alpha=head_cfg.get("smooth_alpha", 0.25),
            dwell_frames=head_cfg.get("dwell_frames", 2),
        )
    if head_scan_enabled and head_mode in {"v3", "compare_v2_v3"}:
        head_scanner_v3 = HeadScannerV3(
            base_names,
            right_threshold=head_cfg.get("right_threshold", 0.25),
            left_threshold=head_cfg.get("left_threshold", -0.25),
            deadband=head_cfg.get("deadband", 0.04),
            min_visibility=head_cfg.get("min_visibility", 0.5),
            smooth_alpha=head_cfg.get("smooth_alpha", 0.25),
            dwell_frames=head_cfg.get("dwell_frames", 2),
            refractory_frames=head_cfg.get("refractory_frames", 10),
            min_excursion=head_cfg.get("min_excursion", 0.03),
            min_quality=head_cfg.get("min_quality", 0.35),
            adaptive=head_cfg.get("adaptive", True),
            warmup_frames=head_cfg.get("warmup_frames", 45),
            adaptive_sigma_scale=head_cfg.get("adaptive_sigma_scale", 2.0),
            adaptive_deadband_scale=head_cfg.get("adaptive_deadband_scale", 0.5),
            adaptive_min_abs_threshold=head_cfg.get(
                "adaptive_min_abs_threshold", 0.12
            ),
            adaptive_min_sigma=head_cfg.get("adaptive_min_sigma", 0.02),
            use_face=head_cfg.get("v3_use_face", True),
            face_weight=head_cfg.get("v3_face_weight", 0.35),
        )

    if stickman_enabled:
        stickman_renderer = Stickman3DRenderer(
            names=names,
            visibility_threshold=stickman_cfg.get("visibility_threshold", 0.2),
            fig_size=_tuple_from_config(
                stickman_cfg.get("fig_size", (4.0, 4.0)),
                default=(4.0, 4.0),
                length=2,
            ),
            dpi=int(stickman_cfg.get("dpi", 200)),
            elev=float(stickman_cfg.get("elev", 15.0)),
            azim=float(stickman_cfg.get("azim", -60.0)),
            axis_limits=stickman_cfg.get("axis_limits"),
            line_color=_tuple_from_config(
                stickman_cfg.get("line_color", (0.9, 0.4, 0.1)),
                default=(0.9, 0.4, 0.1),
                length=3,
            ),
            joint_color=_tuple_from_config(
                stickman_cfg.get("joint_color", (0.1, 0.8, 0.9)),
                default=(0.1, 0.8, 0.9),
                length=3,
            ),
            background_color=_tuple_from_config(
                stickman_cfg.get("background_color", (0.05, 0.05, 0.05)),
                default=(0.05, 0.05, 0.05),
                length=3,
            ),
            line_width=float(stickman_cfg.get("line_width", 2.0)),
            joint_size=float(stickman_cfg.get("joint_size", 20.0)),
            x_scale=float(stickman_cfg.get("x_scale", 1.0)),
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = mp_pose.infer(frame)
            pose_xyzv_for_output = result["xyzv"]
            if use_world_output:
                if bool(result.get("world_ok", False)):
                    pose_xyzv_for_output = result["world_xyzv"]
                elif not mono3d_fallback_to_relative:
                    pose_xyzv_for_output = np.full(
                        result["xyzv"].shape,
                        np.nan,
                        dtype=np.float32,
                    )
            face_result = None
            if mp_face is not None:
                head_roi = None
                if face_use_pose_roi:
                    head_roi = _estimate_head_roi(
                        result["xyzv"],
                        pose_names,
                        frame_width=width,
                        frame_height=height,
                        min_visibility=face_roi_min_visibility,
                        min_size_px=face_roi_min_size_px,
                        pad_scale=face_roi_pad_scale,
                    )
                face_result = mp_face.infer(frame, roi=head_roi)
                if (
                    face_fallback_full_frame
                    and not face_result.get("ok", False)
                    and head_roi is not None
                ):
                    face_result = mp_face.infer(frame, roi=None)

            xyzv_combined = (
                np.vstack([pose_xyzv_for_output, face_result["xyzv"]])
                if face_result is not None
                else pose_xyzv_for_output
            )
            xyzv_output = xyzv_combined
            if body_dense_enabled and body_subdivisions > 0:
                xyzv_output = append_virtual_body_points(
                    base_names=base_names,
                    xyzv=xyzv_combined,
                    subdivisions=body_subdivisions,
                )
            yaw_info_v1 = None
            yaw_info_v2 = None
            yaw_info_v3 = None
            if head_scanner_v1 is not None:
                yaw_info_v1 = head_scanner_v1.update(pose_xyzv_for_output)
            if head_scanner_v2 is not None:
                yaw_info_v2 = head_scanner_v2.update(xyzv_combined)
            if head_scanner_v3 is not None:
                yaw_info_v3 = head_scanner_v3.update(xyzv_combined)

            if overlay_cfg.get("enabled", True) and result.get("xyzv_dict"):
                draw_skeleton(
                    frame,
                    result["xyzv_dict"],
                    visibility_threshold=visibility_threshold,
                )

            if overlay_cfg.get("write_video", True):
                writer.write(frame)

            stickman_frame_bgr = None
            if stickman_renderer is not None:
                stickman_frame_rgb = stickman_renderer.render(xyzv_output)
                stickman_frame_bgr = cv2.cvtColor(
                    stickman_frame_rgb, cv2.COLOR_RGB2BGR
                )
                if yaw_info_v1 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v1,
                        text_color=_tuple_from_config(
                            head_cfg.get("v1_text_color", (80, 210, 255)),
                            default=(80, 210, 255),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v1_bar_color", (0, 165, 255)),
                            default=(0, 165, 255),
                            length=3,
                        ),
                        left_threshold=float(head_cfg.get("left_threshold", -0.25)),
                        right_threshold=float(head_cfg.get("right_threshold", 0.25)),
                        deadband=float(head_cfg.get("deadband", 0.04)),
                        label="V1",
                        text_offset=0 if head_mode != "compare" else 0,
                        bar_index=0 if head_mode != "compare" else 1,
                    )
                if yaw_info_v2 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v2,
                        text_color=_tuple_from_config(
                            head_cfg.get("v2_text_color", (0, 255, 170)),
                            default=(0, 255, 170),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v2_bar_color", (0, 220, 120)),
                            default=(0, 220, 120),
                            length=3,
                        ),
                        left_threshold=float(head_cfg.get("left_threshold", -0.25)),
                        right_threshold=float(head_cfg.get("right_threshold", 0.25)),
                        deadband=float(head_cfg.get("deadband", 0.04)),
                        label="V2",
                        text_offset=60 if head_mode == "compare" else 0,
                        bar_index=0 if head_mode not in {"compare", "compare_v2_v3"} else 1,
                    )
                if yaw_info_v3 is not None:
                    _draw_head_scan(
                        stickman_frame_bgr,
                        yaw_info_v3,
                        text_color=_tuple_from_config(
                            head_cfg.get("v3_text_color", (255, 220, 0)),
                            default=(255, 220, 0),
                            length=3,
                        ),
                        bar_color=_tuple_from_config(
                            head_cfg.get("v3_bar_color", (0, 220, 220)),
                            default=(0, 220, 220),
                            length=3,
                        ),
                        left_threshold=float(
                            yaw_info_v3.get(
                                "left_threshold",
                                head_cfg.get("left_threshold", -0.25),
                            )
                        ),
                        right_threshold=float(
                            yaw_info_v3.get(
                                "right_threshold",
                                head_cfg.get("right_threshold", 0.25),
                            )
                        ),
                        deadband=float(
                            yaw_info_v3.get(
                                "deadband",
                                head_cfg.get("deadband", 0.04),
                            )
                        ),
                        label="V3",
                        text_offset=60 if head_mode == "compare_v2_v3" else 0,
                        bar_index=0,
                    )

                if stickman_cfg.get("write_stickman_video", True):
                    stickman_path = stickman_cfg.get("video")
                    if not stickman_path:
                        raise ValueError(
                            "stickman.video path is required when write_stickman_video is True."
                        )
                    if stickman_writer is None:
                        frame_size = (
                            stickman_frame_bgr.shape[1],
                            stickman_frame_bgr.shape[0],
                        )
                        stickman_writer = video_writer(
                            stickman_path, fps, frame_size
                        )
                    stickman_writer.write(stickman_frame_bgr)

                if stickman_cfg.get("write_composite_video", True):
                    composite_path = stickman_cfg.get("composite_video")
                    if not composite_path:
                        raise ValueError(
                            "stickman.composite_video path is required when write_composite_video is True."
                        )
                    composite_frame = _stack_side_by_side(frame, stickman_frame_bgr)
                    if composite_writer is None:
                        comp_size = (
                            composite_frame.shape[1],
                            composite_frame.shape[0],
                        )
                        composite_writer = video_writer(
                            composite_path, fps, comp_size
                        )
                    composite_writer.write(composite_frame)

            rows.append({"frame_idx": frame_idx, "xyzv": xyzv_output})
            if head_scan_enabled and scan_metrics_csv:
                _append_scan_row(
                    scan_rows,
                    frame_idx=frame_idx,
                    v1=yaw_info_v1,
                    v2=yaw_info_v2,
                    v3=yaw_info_v3,
                )
            frame_idx += 1

    finally:
        cap.release()
        writer.release()
        mp_pose.close()
        if mp_face is not None:
            mp_face.close()
        if stickman_writer is not None:
            stickman_writer.release()
        if composite_writer is not None:
            composite_writer.release()

    write_keypoints_csv(keypoints_csv, rows, names)
    if head_scan_enabled and scan_metrics_csv:
        write_scan_metrics_csv(scan_metrics_csv, scan_rows)
    print(f"Wrote overlay video to {overlay_video}")
    print(f"Wrote keypoints CSV to {keypoints_csv}")
    if head_scan_enabled and scan_metrics_csv:
        print(f"Wrote scan metrics CSV to {scan_metrics_csv}")
    if use_world_output:
        print("Monocular 3D source: MediaPipe pose_world_landmarks")
    if (
        stickman_enabled
        and stickman_cfg.get("write_stickman_video", True)
        and stickman_cfg.get("video")
    ):
        print(f"Wrote stickman video to {stickman_cfg['video']}")
    if (
        stickman_enabled
        and stickman_cfg.get("write_composite_video", True)
        and stickman_cfg.get("composite_video")
    ):
        print(f"Wrote composite video to {stickman_cfg['composite_video']}")


def main() -> None:
    args = parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
