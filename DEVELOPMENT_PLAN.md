# AI-VISTA — Development Plan

**SIH26169:** Development of an AI-Based Virtual Camera Tracking System for
Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals.

This document tracks every phase of development. We do **not** skip phases,
and we do **not** move to the next phase until the current one has been
tested and explicitly approved.

Status legend: ✅ Done &nbsp; 🔄 In progress &nbsp; ⬜ Not started

---

| Phase | Name | What it delivers | Status |
|-------|------|-------------------|--------|
| 0 | Understand requirements & architecture | Shared understanding of the problem, data flow, and design principles | ✅ |
| 1 | Project structure & configuration system | Folder layout, `configs/*.yaml`, `config_loader.py`, tests | ✅ |
| 2 | Virtual world | A 2000×2000 coordinate space the beacon lives in (`src/simulation/environment.py`) | ✅ |
| 3 | Beacon trajectories | Straight line / circular / figure-8 / random motion generators (`src/simulation/motion.py`, `target.py`) | ✅ |
| 4 | Virtual camera & viewport | A movable 640×480 window with FOV into the world (`src/camera/virtual_camera.py`) | ✅ |
| 5 | Frame generation | Render the beacon inside the camera's current viewport as an actual image (`src/simulation/renderer.py`) | ✅ |
| 6 | Basic beacon detection | Classical CV detector: threshold + connected components (`src/detection/classical_detector.py`) | ✅ |
| 7 | Target centre extraction | Compute beacon centroid from detected blob | ✅ (part of Phase 6's detector output) |
| 8 | Tracking | Kalman filter (constant-velocity model), survives brief detection loss (`src/tracking/kalman_tracker.py`) | ✅ |
| 9 | Camera-centre / error calculation | `error_x, error_y = target_center - camera_center` (`src/camera/controller.py`) | ✅ |
| 10 | Virtual camera control | Proportional controller: error → LEFT/RIGHT/UP/DOWN/CENTER + camera movement | ✅ |
| 11 | Camera geometry / angular error | Pixel error → angular error using FOV + resolution (`src/camera/geometry.py`) | ⬜ (pixel-based control confirmed working first, per plan) |
| 12 | Disturbances | Gaussian/salt-pepper/Poisson noise, low light, haze, fog, camera jitter (`src/disturbances/`) | ✅ (partial: noise + atmosphere + jitter implemented; rain/platform-motion pending) |
| 13 | Robustness benchmarking | Run detector/tracker against each disturbance, measure real numbers | 🔄 (noisy.yaml tested once — see Known Results below; full sweep pending) |
| 14 | AI/YOLO evaluation | Compare classical CV vs. a learned detector **with evidence**, decide which to use where | ⬜ |
| 15 | .mp4 benchmark input mode | Reuse the same detection/tracking pipeline on recorded video (`app/` video mode) | ⬜ |
| 16 | Real-time dashboard | Live view of frame, tracking box, error, metrics | ⬜ |
| 17 | Automated evaluation & reporting | CSV/JSON logs + plots generated automatically per experiment (`src/evaluation/`) | ⬜ |
| 18 | Packaging | Turn this into a runnable standalone app | ⬜ |
| 19 | Documentation & demo prep | Final README, technical notes, judge Q&A prep | ⬜ |

---

## Working agreement (how we build each phase)

For every phase, before writing code, you will get:
1. **What** we're about to build.
2. **Why** it's needed (how it fits the data flow).
3. **Which files** will be created/changed.
4. **Data flow** for that phase specifically.

Then: implementation → actual test/run → real results reported → how you can
verify it yourself → **stop and wait for approval** before the next phase.

## Known results (measured, from actual runs — see `data/output/`)

These are real numbers from runs already executed in this repo. Re-run
yourself with `python main.py --scenario <name>` to reproduce.

| Scenario | Acquisition (s) | Avg error (px) | Target loss (%) | FPS | Targets met |
|---|---|---|---|---|---|
| `default` (straight line, clean) | 4.77 | 6.1 | 0.0 | 416 | 3/4 (acquisition time fails — see below) |
| `circular` (radius 200px) | 0.0 | 7.0 | 0.0 | 411 | 4/4 |
| `noisy` (Gaussian + salt-pepper) | 0.13 | 17.2 | 1.67 | 42 | 3/4 (tracking error fails) |

## Known limitations (honest, current)

1. **No active search behaviour.** The camera only reacts once the beacon
   is already inside its 640×480 viewport. If a scenario starts the beacon
   far outside the initial viewport, acquisition time is however long it
   takes the beacon to drift into view (this is why `default` fails the
   acquisition-time target — the beacon starts intentionally out of frame
   to test this). A real system would need an active search/scan pattern
   when uninitialized. Flagged as a strong candidate for Phase 21
   (Innovation) if time allows.
2. **Geometric FOV limit.** The camera's centre can only range over
   `[res_width/2, world_width - res_width/2]` (and similarly for y) while
   still showing a full viewport. A target closer than that to the world
   edge leaves a permanent residual error even with perfect tracking. This
   was discovered by testing (see the `default.yaml` comment on
   `end_xy`) — not a controller bug.
3. **Classical detector under noise.** The `noisy` scenario's tracking
   error (17.2px) exceeds the 10px target. Likely cause: Gaussian noise
   occasionally pushes background pixels above the detection threshold,
   creating spurious blobs that shift the measured centroid. Candidate
   fixes (not yet implemented/tested): a light blur/median filter before
   thresholding, or an adaptive threshold. This is exactly the kind of
   evidence Phase 13/14 is meant to produce before deciding whether a
   learned detector is justified.
4. Angular/geometric conversion (Phase 11), full disturbance set (rain,
   platform motion), GUI/dashboard, and packaging are not yet built.

## Ground rules we're holding ourselves to

- No hard-coded parameters — everything tunable lives in `configs/*.yaml`.
- No performance numbers are ever claimed unless they were actually measured
  by the evaluation subsystem. If something hasn't been benchmarked yet, the
  answer is "Not measured yet."
- Detection method (classical CV vs. AI) is chosen using evidence collected
  in Phase 13/14, not assumed in advance.
- Every phase's core logic (motion math, geometry conversion, control logic)
  gets at least one automated test.
