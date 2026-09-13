"""
logger.py

Writes the automatically-collected metrics to disk:
  - <output_dir>/<run_name>/frame_log.csv   -- per-frame centroiding error etc.
  - <output_dir>/<run_name>/summary.json    -- aggregate performance report

This is the "auto performance log generation" requirement -- nothing here
is entered by hand, it is all pulled from MetricsCollector.
"""

import csv
import json
import time
from pathlib import Path


def save_run_logs(metrics_collector, output_dir: str, scenario_name: str) -> Path:
    """
    Save frame-level CSV + summary JSON for one run.

    Returns
    -------
    Path to the run's output directory.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"{scenario_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-frame CSV: this is the centroiding-error log ---
    csv_path = run_dir / "frame_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "t_s", "error_x_px", "error_y_px",
            "error_magnitude_px", "tracker_status", "detection_found",
            "processing_time_ms",
        ])
        for r in metrics_collector.records:
            writer.writerow([
                r.frame_index, round(r.t, 4),
                round(r.error_x, 3) if r.error_x is not None else "",
                round(r.error_y, 3) if r.error_y is not None else "",
                round(r.error_magnitude_px, 3) if r.error_magnitude_px is not None else "",
                r.tracker_status, r.detection_found,
                round(r.processing_time_s * 1000, 4),
            ])

    # --- Summary JSON: the auto-generated performance report ---
    summary = metrics_collector.summary()
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return run_dir
