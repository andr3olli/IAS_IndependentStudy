from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
import time

import yaml

from tools.evaluate_scan_events import evaluate_events, load_gt_events, load_pred_events
from utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run versioned scan experiments (v1/v2/v3), save all artifacts, "
            "evaluate against GT, and generate viewer commands."
        )
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/mvp.yaml",
        help="Base YAML config to clone per run.",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="*",
        default=[],
        help="Optional list of input videos. If omitted, uses input_video from base config.",
    )
    parser.add_argument(
        "--versions",
        type=str,
        default="v1,v2,v3",
        help="Comma-separated head scan modes to run.",
    )
    parser.add_argument(
        "--report-root",
        type=str,
        default="data/reports",
        help="Root folder for all report runs.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="",
        help="Optional report folder name. If empty, timestamp is used.",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="",
        help="Optional GT folder. Expected file: <clip_stem><gt_suffix>.",
    )
    parser.add_argument(
        "--gt-suffix",
        type=str,
        default="_gt.csv",
        help="Suffix for GT files in gt-dir.",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=5,
        help="Tolerance (frames) for event matching.",
    )
    parser.add_argument(
        "--ignore-direction",
        action="store_true",
        help="Ignore left/right direction in evaluation.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to run pipeline and viewers.",
    )
    parser.add_argument(
        "--run-viewers",
        action="store_true",
        help=(
            "Launch viewer windows sequentially at the end (close each to continue). "
            "Otherwise only write viewer_commands.sh."
        ),
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help=(
            "Stream pipeline stdout/stderr live while also writing log files. "
            "Useful for immediate feedback."
        ),
    )
    return parser.parse_args()


def _ensure_mapping(root: dict, key: str) -> dict:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mode_set(mode: str) -> str:
    if mode not in {"v1", "v2", "v3", "compare", "compare_v2_v3"}:
        return "v3"
    return mode


def _normalize_versions(raw: str) -> List[str]:
    items = [v.strip().lower() for v in raw.split(",") if v.strip()]
    versions = []
    for item in items:
        mode = _mode_set(item)
        if mode in {"v1", "v2", "v3"} and mode not in versions:
            versions.append(mode)
    if not versions:
        versions = ["v1", "v2", "v3"]
    return versions


def _build_run_config(
    base_cfg: dict,
    input_video: Path,
    out_dir: Path,
    mode: str,
) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["input_video"] = str(input_video)
    cfg["overlay_video"] = str(out_dir / "overlay.mp4")
    cfg["keypoints_csv"] = str(out_dir / "keypoints.csv")

    # Keep this runner scoped to monocular sweeps.
    stereo_cfg = _ensure_mapping(cfg, "stereo")
    stereo_cfg["enabled"] = False

    overlay_cfg = _ensure_mapping(cfg, "overlay")
    overlay_cfg["enabled"] = True
    overlay_cfg["write_video"] = True

    stickman_cfg = _ensure_mapping(cfg, "stickman")
    stickman_cfg["enabled"] = True
    stickman_cfg["write_stickman_video"] = True
    stickman_cfg["write_composite_video"] = True
    stickman_cfg["video"] = str(out_dir / "stickman.mp4")
    stickman_cfg["composite_video"] = str(out_dir / "side_by_side.mp4")

    analytics_cfg = _ensure_mapping(cfg, "analytics")
    head_cfg = _ensure_mapping(analytics_cfg, "head_scan")
    head_cfg["enabled"] = True
    head_cfg["mode"] = mode
    head_cfg["metrics_csv"] = str(out_dir / "scan_metrics.csv")
    return cfg


def _pipeline_cmd(python_bin: str, config_path: Path) -> List[str]:
    return [python_bin, "src/main.py", "--config", str(config_path)]


def _viewer_cmd(python_bin: str, keypoints_csv: Path, mode: str) -> List[str]:
    cmd = [
        python_bin,
        "-m",
        "viewers.stickman_viewer",
        "--csv",
        str(keypoints_csv),
        "--head-scan",
        "--head-scan-mode",
        mode,
    ]
    if mode == "v3":
        cmd.append("--head-adaptive")
    return cmd


def _run_command(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    live_output: bool = False,
    extra_env: Dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        if not live_output:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return int(proc.returncode)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_fh.write(line)
        proc.wait()
        return int(proc.returncode)


def _tail_text(path: Path, lines: int = 12) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        return ""
    return "\n".join(text[-lines:])


def _evaluate_if_possible(
    metrics_csv: Path,
    gt_csv: Path | None,
    mode: str,
    tolerance_frames: int,
    ignore_direction: bool,
) -> dict | None:
    if gt_csv is None or not gt_csv.exists():
        return None
    if not metrics_csv.exists():
        return None

    pred = load_pred_events(metrics_csv, prefix=mode)
    gt = load_gt_events(gt_csv)
    summary, _matches = evaluate_events(
        pred_events=pred,
        gt_events=gt,
        tolerance=tolerance_frames,
        ignore_direction=ignore_direction,
    )
    return summary


def _write_viewer_script(path: Path, commands: List[List[str]], repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'cd "{repo_root}"',
        'export PYTHONPATH="src:${PYTHONPATH:-}"',
        "",
    ]
    for cmd in commands:
        q = " ".join(f'"{x}"' if " " in x else x for x in cmd)
        lines.append(f"echo 'Running: {q}'")
        lines.append(q)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    base_cfg_path = Path(args.base_config).expanduser()
    base_cfg = load_config(base_cfg_path)

    versions = _normalize_versions(args.versions)
    if args.inputs:
        inputs = [Path(p).expanduser() for p in args.inputs]
    else:
        default_input = base_cfg.get("input_video")
        if not default_input:
            raise ValueError("No --inputs provided and base config has no input_video.")
        inputs = [Path(str(default_input)).expanduser()]

    for inp in inputs:
        if not inp.exists():
            raise FileNotFoundError(f"Input video not found: {inp}")

    report_name = args.report_name.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_root = Path(args.report_root).expanduser()
    report_dir = report_root / report_name
    configs_dir = report_dir / "configs"
    logs_dir = report_dir / "logs"
    summary_rows: List[Dict[str, object]] = []
    viewer_commands: List[List[str]] = []

    gt_dir = Path(args.gt_dir).expanduser() if args.gt_dir else None
    tolerance_frames = max(0, int(args.tolerance_frames))

    total_runs = len(inputs) * len(versions)
    print(f"Starting report suite with {total_runs} runs")
    run_idx = 0

    for input_video in inputs:
        clip = input_video.stem
        gt_csv = (gt_dir / f"{clip}{args.gt_suffix}") if gt_dir else None

        for mode in versions:
            run_idx += 1
            run_out_dir = report_dir / clip / mode
            run_out_dir.mkdir(parents=True, exist_ok=True)

            cfg = _build_run_config(
                base_cfg=base_cfg,
                input_video=input_video,
                out_dir=run_out_dir,
                mode=mode,
            )
            cfg_path = configs_dir / f"{clip}_{mode}.yaml"
            _write_yaml(cfg_path, cfg)

            log_path = logs_dir / f"{clip}_{mode}.log"
            cmd = _pipeline_cmd(args.python_bin, cfg_path)
            print(
                f"[{run_idx}/{total_runs}] clip={clip} mode={mode} "
                f"-> running pipeline..."
            )
            t0 = time.perf_counter()
            code = _run_command(
                cmd,
                cwd=repo_root,
                log_path=log_path,
                live_output=bool(args.live_output),
            )
            dt = time.perf_counter() - t0

            metrics_csv = run_out_dir / "scan_metrics.csv"
            keypoints_csv = run_out_dir / "keypoints.csv"
            eval_summary = _evaluate_if_possible(
                metrics_csv=metrics_csv,
                gt_csv=gt_csv,
                mode=mode,
                tolerance_frames=tolerance_frames,
                ignore_direction=bool(args.ignore_direction),
            )

            row: Dict[str, object] = {
                "clip": clip,
                "mode": mode,
                "status": "ok" if code == 0 else "failed",
                "return_code": code,
                "config": str(cfg_path),
                "log": str(log_path),
                "overlay_video": str(run_out_dir / "overlay.mp4"),
                "stickman_video": str(run_out_dir / "stickman.mp4"),
                "composite_video": str(run_out_dir / "side_by_side.mp4"),
                "keypoints_csv": str(keypoints_csv),
                "metrics_csv": str(metrics_csv),
                "gt_csv": str(gt_csv) if gt_csv else "",
            }

            if eval_summary is not None:
                row.update(
                    {
                        "tp": eval_summary["tp"],
                        "fp": eval_summary["fp"],
                        "fn": eval_summary["fn"],
                        "precision": eval_summary["precision"],
                        "recall": eval_summary["recall"],
                        "f1": eval_summary["f1"],
                        "mae_frames": eval_summary["mae_frames"],
                        "pred_events": eval_summary["pred_events"],
                        "gt_events": eval_summary["gt_events"],
                    }
                )

            summary_rows.append(row)
            status_txt = "OK" if code == 0 else "FAILED"
            print(
                f"[{run_idx}/{total_runs}] clip={clip} mode={mode} "
                f"-> {status_txt} ({dt:.1f}s)"
            )
            if code != 0:
                print(f"  Log: {log_path}")
                tail = _tail_text(log_path, lines=14)
                if tail:
                    print("  Last log lines:")
                    for ln in tail.splitlines():
                        print(f"    {ln}")
            elif eval_summary is not None:
                print(
                    "  Eval: "
                    f"P={eval_summary['precision']:.3f} "
                    f"R={eval_summary['recall']:.3f} "
                    f"F1={eval_summary['f1']:.3f} "
                    f"TP={eval_summary['tp']} FP={eval_summary['fp']} FN={eval_summary['fn']}"
                )
            if code == 0 and keypoints_csv.exists():
                viewer_commands.append(_viewer_cmd(args.python_bin, keypoints_csv, mode))

    summary_path = report_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        for row in summary_rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        _write_csv(summary_path, summary_rows, fieldnames)

    viewer_script = report_dir / "viewer_commands.sh"
    _write_viewer_script(viewer_script, viewer_commands, repo_root=repo_root)

    print(f"Report directory: {report_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Viewer commands script: {viewer_script}")

    if args.run_viewers and viewer_commands:
        print("Launching viewer sessions sequentially...")
        for cmd in viewer_commands:
            print(" ".join(cmd))
            subprocess.run(
                cmd,
                cwd=str(repo_root),
                env={**os.environ, "PYTHONPATH": "src"},
                check=False,
            )


if __name__ == "__main__":
    main()
