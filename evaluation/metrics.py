"""
metrics.py

Collects real, measured data during a simulation/benchmark run and
computes the performance statistics the SIH problem statement asks for.

NOTHING in this file invents numbers -- every value here is derived from
actual per-frame records appended during the run. If a run hasn't happened,
there is no MetricsCollector instance with data in it, and summary() cannot
be called meaningfully.
"""

import time
from dataclasses import dataclass, field


@dataclass
class FrameRecord:
    frame_index: int
    t: float
    error_x: float | None
    error_y: float | None
    error_magnitude_px: float | None
    tracker_status: str
    detection_found: bool
    processing_time_s: float


@dataclass
class MetricsCollector:
    performance_targets: dict = field(default_factory=dict)
    records: list = field(default_factory=list)
    _run_start_wall: float = field(default_factory=time.perf_counter)

    def record_frame(
        self,
        frame_index: int,
        t: float,
        control_result,
        tracker_status: str,
        detection_found: bool,
        processing_time_s: float,
    ):
        """Append one frame's worth of real measurements."""

        if control_result is not None:
            error_x = control_result.error_x
            error_y = control_result.error_y
            error_mag = control_result.error_magnitude_px
        else:
            error_x = None
            error_y = None
            error_mag = None

        self.records.append(
            FrameRecord(
                frame_index=frame_index,
                t=t,
                error_x=error_x,
                error_y=error_y,
                error_magnitude_px=error_mag,
                tracker_status=tracker_status,
                detection_found=detection_found,
                processing_time_s=processing_time_s,
            )
        )

    def summary(self) -> dict:
        """
        Compute summary statistics from everything recorded so far.
        Returns None or an explanatory status where a metric genuinely
        cannot be computed.
        """

        if not self.records:
            return {
                "status": "Not measured yet -- no frames recorded."
            }

        n = len(self.records)

        total_wall_time = (
            time.perf_counter() - self._run_start_wall
        )

        # ------------------------------------------------------------
        # Tracking errors
        # ------------------------------------------------------------

        tracked_errors = [
            r.error_magnitude_px
            for r in self.records
            if r.tracker_status == "TRACKING"
            and r.error_magnitude_px is not None
        ]

        # ------------------------------------------------------------
        # Acquisition
        # ------------------------------------------------------------

        acquisition_time_s = None

        for r in self.records:
            if r.tracker_status == "TRACKING":
                acquisition_time_s = r.t
                break

        target_acquired = acquisition_time_s is not None

        # ------------------------------------------------------------
        # Target loss
        # ------------------------------------------------------------

        lost_frames = sum(
            1
            for r in self.records
            if r.tracker_status == "LOST"
        )

        tracking_frames = sum(
            1
            for r in self.records
            if r.tracker_status == "TRACKING"
        )

        if target_acquired:
            # Once the target has been acquired, LOST frames represent
            # actual tracking loss.
            target_loss_pct = 100 * lost_frames / n

        else:
            # The target was never acquired, so "loss percentage" is not
            # a meaningful metric. We report this explicitly instead of
            # incorrectly showing 0%.
            target_loss_pct = None

        # ------------------------------------------------------------
        # Re-acquisition events
        # ------------------------------------------------------------
        #
        # Re-acquisition time measures how long it takes the tracker to
        # return to TRACKING after the target becomes detectable again.
        #
        # We therefore:
        #   1. Detect when the tracker enters LOST.
        #   2. Wait for detection_found=True.
        #   3. Record that detection time.
        #   4. Measure until TRACKING is restored.
        #
        # This avoids incorrectly counting the intentional occlusion
        # period itself as reacquisition time.
        # ------------------------------------------------------------

        reacquisition_times = []

        was_lost = False
        detection_return_time = None

        for r in self.records:

            if r.tracker_status == "LOST":
                was_lost = True

                # The target is still lost, so don't overwrite an
                # already-recorded detection return time.
                continue

            if was_lost and r.detection_found:
                if detection_return_time is None:
                    detection_return_time = r.t

            if (
                was_lost
                and detection_return_time is not None
                and r.tracker_status == "TRACKING"
            ):
                reacquisition_times.append(
                    r.t - detection_return_time
                )

                was_lost = False
                detection_return_time = None

        # ------------------------------------------------------------
        # Processing performance
        # ------------------------------------------------------------

        avg_processing_time = (
            sum(r.processing_time_s for r in self.records)
            / n
        )

        fps_measured = (
            n / total_wall_time
            if total_wall_time > 0
            else None
        )

        # ------------------------------------------------------------
        # RMSE
        # ------------------------------------------------------------

        rmse = None

        if tracked_errors:
            rmse = (
                sum(e ** 2 for e in tracked_errors)
                / len(tracked_errors)
            ) ** 0.5

        # ------------------------------------------------------------
        # Build summary
        # ------------------------------------------------------------

        summary = {
            "frames_processed": n,

            "simulation_duration_s": self.records[-1].t,

            "wall_clock_duration_s": round(
                total_wall_time,
                4,
            ),

            "measured_fps": (
                round(fps_measured, 2)
                if fps_measured
                else None
            ),

            "avg_processing_time_ms": round(
                avg_processing_time * 1000,
                3,
            ),

            "acquisition_time_s": acquisition_time_s,

            "avg_tracking_error_px": (
                round(
                    sum(tracked_errors)
                    / len(tracked_errors),
                    3,
                )
                if tracked_errors
                else None
            ),

            "max_tracking_error_px": (
                round(
                    max(tracked_errors),
                    3,
                )
                if tracked_errors
                else None
            ),

            "rmse_tracking_error_px": (
                round(rmse, 3)
                if rmse is not None
                else None
            ),

            "target_loss_pct": (
                round(target_loss_pct, 2)
                if target_loss_pct is not None
                else None
            ),

            "lock_retention_pct": round(
                100 * tracking_frames / n,
                2,
            ),

            "num_reacquisitions": len(
                reacquisition_times
            ),

            "avg_reacquisition_time_s": (
                round(
                    sum(reacquisition_times)
                    / len(reacquisition_times),
                    3,
                )
                if reacquisition_times
                else None
            ),

            "target_acquired": target_acquired,
        }

        # Compare against configured performance targets.
        if self.performance_targets:
            summary["targets_met"] = (
                self._check_targets(summary)
            )

        return summary

    def _check_targets(self, summary: dict) -> dict:
        """Honest pass/fail check against configured targets."""

        pt = self.performance_targets

        checks = {}

        # ------------------------------------------------------------
        # Acquisition target
        # ------------------------------------------------------------

        if summary["acquisition_time_s"] is not None:

            checks["acquisition_time"] = (
                summary["acquisition_time_s"]
                <= pt.get(
                    "acquisition_time_s_max",
                    float("inf"),
                )
            )

        else:

            checks["acquisition_time"] = (
                "Not measured yet "
                "(target never acquired)"
            )

        # ------------------------------------------------------------
        # Tracking error target
        # ------------------------------------------------------------

        if summary["avg_tracking_error_px"] is not None:

            checks["tracking_error"] = (
                summary["avg_tracking_error_px"]
                <= pt.get(
                    "tracking_error_px_max",
                    float("inf"),
                )
            )

        else:

            checks["tracking_error"] = (
                "Not measured yet"
            )

        # ------------------------------------------------------------
        # Target loss
        # ------------------------------------------------------------

        if summary["target_loss_pct"] is not None:

            checks["target_loss"] = (
                summary["target_loss_pct"]
                <= pt.get(
                    "target_loss_pct_max",
                    100,
                )
            )

        else:

            checks["target_loss"] = (
                "Not measured yet "
                "(target never acquired)"
            )

        # ------------------------------------------------------------
        # Processing FPS
        # ------------------------------------------------------------

        if summary["measured_fps"] is not None:

            checks["processing_fps"] = (
                summary["measured_fps"]
                >= pt.get(
                    "processing_fps_min",
                    0,
                )
            )

        else:

            checks["processing_fps"] = (
                "Not measured yet"
            )

        return checks