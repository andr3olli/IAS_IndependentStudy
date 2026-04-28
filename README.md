# PitchPose

PitchPose is a computer-vision pipeline for analyzing presentation body language from video, with a focus on head scan behavior (left/right audience scanning).

It uses MediaPipe Pose (and optional Face Mesh), supports monocular and stereo workflows, renders visual overlays + a 3D stickman, and exports per-frame analytics for evaluation.

## What It Does

1. Reads a video (or stereo pair) from config.
2. Runs pose estimation per frame.
3. Optionally fuses dense face landmarks and virtual body points.
4. Computes head scan signals with three scanner variants (`v1`, `v2`, `v3`).
5. Writes:
   - Overlay video with 2D skeleton
   - Optional 3D stickman video + side-by-side composite
   - Keypoints CSV (`frame, landmark, x, y, z, visibility`)
   - Head-scan metrics CSV (wide per-frame analytics)

## Project Structure

- `src/main.py`: Main runtime entrypoint for monocular and stereo pipelines.
- `src/pose/`: Pose and face backends.
- `src/analytics/head_scan.py`: Head scanner versions (`HeadScanner`, `HeadScannerV2`, `HeadScannerV3`).
- `src/triangulation/stereo.py`: Stereo calibration loader and triangulation.
- `src/viz/`: 2D skeleton drawing and 3D stickman rendering.
- `src/viewers/stickman_viewer.py`: Interactive playback/debug viewer for exported keypoints.
- `src/tools/stereo_calibrate.py`: Stereo calibration utility.
- `src/tools/evaluate_scan_events.py`: Event-level evaluation against ground truth.
- `src/tools/run_report_suite.py`: Batch experiment runner across scanner versions.
- `configs/mvp.yaml`: Main config example.
- `configs/stereo_calibration.example.yaml`: Stereo calibration format example.
- `umls/`: PlantUML source diagrams.

## Setup

Python 3.10+ recommended.

Install dependencies:

```bash
pip install numpy opencv-python mediapipe matplotlib pyyaml
```

## Run

Monocular pipeline:

```bash
PYTHONPATH=src python src/main.py --config configs/mvp.yaml
```

If `stereo.enabled: true` in config, the stereo pipeline runs automatically.

## Key Tools

Run interactive viewer:

```bash
PYTHONPATH=src python -m viewers.stickman_viewer --csv data/processed/test_keypoints.csv --head-scan --head-scan-mode v3
```

Run stereo calibration:

```bash
PYTHONPATH=src python -m tools.stereo_calibrate --left-glob "data/calib/left/*.png" --right-glob "data/calib/right/*.png" --rows 6 --cols 9 --square-size 0.024 --output configs/stereo_calibration.yaml
```

Evaluate predicted scan events:

```bash
PYTHONPATH=src python -m tools.evaluate_scan_events --pred-csv data/processed/test_head_scan_metrics.csv --gt-csv data/processed/labels/gt_events_template.csv --prefix v3
```

Run report suite across versions:

```bash
PYTHONPATH=src python -m tools.run_report_suite --base-config configs/mvp.yaml --versions v1,v2,v3 --report-name final_report_run
```

## Scanner Modes

- `v1`: Thresholding on pose nose vs shoulder midpoint.
- `v2`: Smoothed, torso-compensated yaw with optional face cue blend and dwell logic.
- `v3`: Torso-frame yaw + quality gating + adaptive thresholds + refractory/event logic.

## UML Diagrams

The UML/architecture visuals below summarize the system from different perspectives.

### 1) Runtime Architecture

![Runtime Architecture](umls/images/Architecture_big.png)

This diagram shows the high-level runtime flow: video input, pose/face processing, analytics, and output artifacts (videos + CSVs). It is useful for quickly understanding the main modules and how data moves across them.

### 2) Pipeline Components

![Pipeline Components](umls/images/pipeline.png)

This component-level view focuses on the scan-analysis pipeline internals, including detector outputs, enrichment/fusion steps, and scanner logic. It helps explain which subsystem is responsible for each transformation in the head-scan metric path.

### 3) Per-Frame Visualization Sequence

![Per-Frame Visualization Sequence](umls/images/Visualization.png)

This sequence diagram illustrates what happens within a single frame cycle, from frame acquisition to rendering overlays/stickman outputs. It is useful for debugging frame-level timing and understanding rendering order.
