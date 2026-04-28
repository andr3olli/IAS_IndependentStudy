from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted head-scan events against ground truth."
    )
    parser.add_argument(
        "--pred-csv",
        type=str,
        required=True,
        help="Predicted scan metrics CSV (from analytics.head_scan.metrics_csv).",
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        required=True,
        help="Ground-truth events CSV.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="v3",
        help="Prediction prefix in CSV (v1/v2/v3).",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=5,
        help="Max absolute frame error to consider a match.",
    )
    parser.add_argument(
        "--ignore-direction",
        action="store_true",
        help="Ignore left/right direction during matching.",
    )
    parser.add_argument(
        "--matches-csv",
        type=str,
        default="",
        help="Optional path to write detailed match table.",
    )
    return parser.parse_args()


def _to_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_pred_events(path: str | Path, prefix: str) -> List[Tuple[int, str]]:
    pred_path = Path(path).expanduser()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")

    event_col = f"{prefix}_event"
    dir_col = f"{prefix}_event_direction"
    count_col = f"{prefix}_count"
    state_col = f"{prefix}_state"

    events: List[Tuple[int, str]] = []
    prev_count = None

    with pred_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return events
        has_event_col = event_col in reader.fieldnames

        for row in reader:
            frame = int(float(row.get("frame", "0")))

            if has_event_col:
                is_event = _to_bool(row.get(event_col))
                if not is_event:
                    continue
                direction = str(row.get(dir_col, "none")).strip().lower()
                if direction not in {"left", "right"}:
                    direction = str(row.get(state_col, "none")).strip().lower()
                events.append((frame, direction))
                continue

            if count_col not in row:
                continue
            try:
                count = int(float(row.get(count_col, "0")))
            except (TypeError, ValueError):
                continue

            if prev_count is None:
                prev_count = count
                continue

            if count > prev_count:
                direction = str(row.get(state_col, "none")).strip().lower()
                events.append((frame, direction))
            prev_count = count

    return events


def load_gt_events(path: str | Path) -> List[Tuple[int, str]]:
    gt_path = Path(path).expanduser()
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {gt_path}")

    events: List[Tuple[int, str]] = []
    with gt_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return events

        has_frame = "frame" in reader.fieldnames
        has_span = "start_frame" in reader.fieldnames and "end_frame" in reader.fieldnames
        if not has_frame and not has_span:
            raise ValueError(
                "Ground-truth CSV must contain either `frame` or "
                "`start_frame` + `end_frame` columns."
            )

        for row in reader:
            if has_frame:
                frame = int(float(row.get("frame", "0")))
            else:
                start = int(float(row.get("start_frame", "0")))
                end = int(float(row.get("end_frame", str(start))))
                frame = int(round(0.5 * (start + end)))

            direction = str(row.get("direction", "none")).strip().lower()
            events.append((frame, direction))

    return events


def evaluate_events(
    pred_events: List[Tuple[int, str]],
    gt_events: List[Tuple[int, str]],
    tolerance: int,
    ignore_direction: bool = False,
) -> tuple[dict, List[dict]]:
    used_gt = set()
    matches: List[dict] = []
    tp = 0

    for pred_idx, (p_frame, p_dir) in enumerate(pred_events):
        best_idx = None
        best_err = None
        for gt_idx, (g_frame, g_dir) in enumerate(gt_events):
            if gt_idx in used_gt:
                continue
            if not ignore_direction and p_dir in {"left", "right"} and g_dir in {"left", "right"}:
                if p_dir != g_dir:
                    continue
            err = abs(p_frame - g_frame)
            if err > tolerance:
                continue
            if best_err is None or err < best_err:
                best_err = err
                best_idx = gt_idx

        if best_idx is not None:
            used_gt.add(best_idx)
            g_frame, g_dir = gt_events[best_idx]
            matches.append(
                {
                    "pred_index": pred_idx,
                    "pred_frame": p_frame,
                    "pred_direction": p_dir,
                    "gt_index": best_idx,
                    "gt_frame": g_frame,
                    "gt_direction": g_dir,
                    "abs_error": abs(p_frame - g_frame),
                    "matched": 1,
                }
            )
            tp += 1
        else:
            matches.append(
                {
                    "pred_index": pred_idx,
                    "pred_frame": p_frame,
                    "pred_direction": p_dir,
                    "gt_index": -1,
                    "gt_frame": "",
                    "gt_direction": "",
                    "abs_error": "",
                    "matched": 0,
                }
            )

    fp = max(0, len(pred_events) - tp)
    fn = max(0, len(gt_events) - tp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    mae = (
        sum(m["abs_error"] for m in matches if m["matched"] == 1) / tp
        if tp > 0
        else float("nan")
    )

    summary = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae_frames": float(mae),
        "pred_events": int(len(pred_events)),
        "gt_events": int(len(gt_events)),
    }
    return summary, matches


def write_matches(path: str | Path, rows: List[dict]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pred_index",
        "pred_frame",
        "pred_direction",
        "gt_index",
        "gt_frame",
        "gt_direction",
        "abs_error",
        "matched",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    pred = load_pred_events(args.pred_csv, prefix=args.prefix)
    gt = load_gt_events(args.gt_csv)
    summary, matches = evaluate_events(
        pred_events=pred,
        gt_events=gt,
        tolerance=max(0, int(args.tolerance_frames)),
        ignore_direction=bool(args.ignore_direction),
    )

    print(f"Pred events: {summary['pred_events']}")
    print(f"GT events: {summary['gt_events']}")
    print(f"TP: {summary['tp']} FP: {summary['fp']} FN: {summary['fn']}")
    print(
        f"Precision: {summary['precision']:.3f} "
        f"Recall: {summary['recall']:.3f} F1: {summary['f1']:.3f}"
    )
    if summary["tp"] > 0:
        print(f"MAE (frames): {summary['mae_frames']:.2f}")
    else:
        print("MAE (frames): n/a")

    if args.matches_csv:
        write_matches(args.matches_csv, matches)
        print(f"Wrote matches to {Path(args.matches_csv).expanduser()}")


if __name__ == "__main__":
    main()
