 
# AI-VISTA

## AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals

**Smart India Hackathon 2026 \| Problem Statement: SIH26169**

AI-VISTA is a software-first simulation and tracking framework for the
**coarse alignment of mobile Free Space Optical Communication (FSOC)
terminals**. It simulates a moving optical beacon, generates a virtual
camera view, detects and tracks the beacon, measures alignment error,
and automatically repositions the virtual camera to keep the target near
the optical axis.

The project is designed so that the core alignment pipeline can be
developed, tested, benchmarked, and demonstrated without requiring
expensive physical FSOC hardware.

------------------------------------------------------------------------

## Problem Statement

FSOC systems transmit information using highly directional optical
beams. For communication to be established, the transmitting and
receiving terminals must first be aligned.

Before fine-pointing hardware can perform precise alignment, a **coarse
alignment system** must:

1.  Locate the optical beacon.
2.  Determine its position in the camera frame.
3.  Track the beacon as the terminal or target moves.
4.  Calculate the displacement between the beacon and camera centre.
5.  Reposition the camera toward the target.
6.  Continue correcting the alignment in real time.

AI-VISTA implements this complete feedback loop in a configurable
virtual environment.

------------------------------------------------------------------------

## Core Pipeline

``` text
Beacon Motion
     ↓
Virtual Environment
     ↓
Virtual Camera
     ↓
Rendered Camera Frame
     ↓
Environmental / Sensor Disturbances
     ↓
Beacon Detection
     ↓
Target Centroid Extraction
     ↓
Kalman Tracking
     ↓
Alignment Error Calculation
     ↓
Proportional Camera Controller
     ↓
Virtual Camera Repositioning
     ↓
Performance Measurement & Logging
```

The loop runs continuously for every simulated frame.

------------------------------------------------------------------------

## Key Features

-   **Configurable virtual FSOC environment** with a 2D world and
    movable camera.
-   **Multiple beacon trajectories**, including straight-line, circular,
    figure-8, and random motion.
-   **Classical computer-vision beacon detector** using denoising,
    thresholding, and connected-component analysis.
-   **Kalman-filter tracking** using a constant-velocity model.
-   **Target-loss handling** with short-term prediction during missed
    detections.
-   **Closed-loop camera control** using the target-to-camera-centre
    error.
-   **Configurable disturbances** including Gaussian noise,
    salt-and-pepper noise, Poisson noise, low light, haze, fog, and
    camera jitter.
-   **Scenario-based YAML configuration** instead of hard-coded
    experiment parameters.
-   **Automatic performance evaluation** from actual frame-level
    measurements.
-   **CSV and JSON experiment logs** for reproducible evaluation.
-   **Optional annotated MP4 output** for demonstrations.
-   **Automated tests** for configuration, motion, detection, and
    controller logic.

------------------------------------------------------------------------

## Why Classical Computer Vision First?

The current target is a small, bright optical beacon against a
relatively dark background. For this type of target, a lightweight
detector can be faster and easier to validate than immediately
introducing a large object-detection network.

The implemented detector uses:

``` text
BGR Frame
   ↓
Grayscale Conversion
   ↓
Gaussian Denoising
   ↓
Brightness Thresholding
   ↓
Connected Components
   ↓
Largest Valid Bright Blob
   ↓
Beacon Centroid
```

This approach requires no trained model or GPU.

A learned detector such as YOLO is intentionally reserved for later
comparison. The final detector should be selected from measured evidence
such as robustness, accuracy, latency, and FPS rather than from model
complexity alone.

------------------------------------------------------------------------

## Tracking and Camera Control

After detection, the beacon centroid is passed to a **Kalman filter**.
The tracker estimates target position and velocity and can continue
predicting the target for a limited number of missed frames.

The controller calculates:

``` text
error_x = target_x - image_center_x
error_y = target_y - image_center_y
```

A proportional controller then moves the virtual camera by a fraction of
this error while respecting the configured camera speed limit.

For demonstration and logging, the error is also translated into:

-   `LEFT`
-   `RIGHT`
-   `UP`
-   `DOWN`
-   `CENTER`

A target is treated as centred when the error magnitude falls below the
controller's lock threshold.

------------------------------------------------------------------------

## Project Structure

``` text
AI-VISTA/
│
├── main.py
│   Main simulation entry point and end-to-end pipeline.
│
├── configs/
│   ├── default.yaml
│   ├── straight_line.yaml
│   ├── circular.yaml
│   ├── figure8.yaml
│   ├── random_motion.yaml
│   ├── noisy.yaml
│   ├── low_light.yaml
│   ├── noise_test.yaml
│   ├── target_loss.yaml
│   └── stress_test.yaml
│
├── src/
│   ├── simulation/
│   │   ├── environment.py
│   │   ├── motion.py
│   │   ├── renderer.py
│   │   └── target.py
│   │
│   ├── camera/
│   │   ├── virtual_camera.py
│   │   └── controller.py
│   │
│   ├── detection/
│   │   └── classical_detector.py
│   │
│   ├── tracking/
│   │   └── kalman_tracker.py
│   │
│   ├── disturbances/
│   │   ├── noise.py
│   │   ├── atmosphere.py
│   │   └── jitter.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── logger.py
│   │
│   └── utils/
│       └── config_loader.py
│
├── tests/
│   ├── test_config_loader.py
│   ├── test_motion.py
│   ├── test_classical_detector.py
│   └── test_controller.py
│
├── data/
│   ├── input/
│   └── output/
│
├── models/
├── notebooks/
├── logs/
├── app/
│
├── configs_backup/
├── DEVELOPMENT_PLAN.md
├── requirements.txt
└── README.md
```

`configs_backup/`, `Untitled-1.py`, and `Untitled-1_backup.py` are
present in the supplied development archive but are not part of the main
runtime pipeline.

------------------------------------------------------------------------

## Main Modules

### `src/simulation`

Creates the virtual environment in which the optical beacon moves.

-   `environment.py` defines the 2D world.
-   `motion.py` implements target trajectories.
-   `target.py` represents the beacon and connects it to the selected
    motion model.
-   `renderer.py` converts the beacon's world position into an actual
    camera image.

### `src/camera`

Models the camera and the feedback controller.

-   `virtual_camera.py` stores camera resolution, FOV, optical-axis
    position, and movement limits.
-   `controller.py` calculates alignment error and performs proportional
    camera correction.

### `src/detection`

`classical_detector.py` detects the bright beacon using OpenCV and
returns information including whether a target was found, its centroid,
blob area, confidence, and threshold used.

### `src/tracking`

`kalman_tracker.py` provides frame-to-frame target tracking using an
OpenCV Kalman filter with a constant-velocity state model.

### `src/disturbances`

Used to test robustness under degraded conditions.

Implemented disturbance functions include:

-   Gaussian noise
-   Salt-and-pepper noise
-   Poisson noise
-   Low-light simulation
-   Haze
-   Fog
-   Camera/platform jitter

The configuration also contains planned disturbance options such as rain
and platform motion, but these are not yet integrated into the main
disturbance pipeline.

### `src/evaluation`

Collects measured performance data and automatically writes experiment
results.

Each run can produce:

``` text
data/output/<scenario>_<timestamp>/
├── frame_log.csv
└── summary.json
```

`frame_log.csv` stores per-frame measurements such as alignment error,
tracker state, detection status, and processing time.

`summary.json` stores aggregate performance statistics generated from
those measurements.

------------------------------------------------------------------------

## Configuration

The baseline configuration is:

``` text
configs/default.yaml
```

Scenario files are loaded as **partial overrides** on top of the default
configuration.

The default configuration includes parameters for:

-   World dimensions
-   Camera resolution
-   Horizontal and vertical FOV
-   Camera update rate
-   Maximum camera angular speed
-   Beacon size and colour
-   Motion model
-   Noise and atmospheric disturbances
-   Simulation duration
-   Performance targets
-   Output paths

This makes experiments reproducible and avoids scattering tuning values
throughout the source code.

------------------------------------------------------------------------

## Available Scenarios

The supplied repository contains configuration files for:

  Scenario          Purpose
  ----------------- --------------------------------------
  `default`         Baseline clean simulation
  `straight_line`   Straight target motion
  `circular`        Circular target trajectory
  `figure8`         Figure-8 target trajectory
  `random_motion`   Random target movement
  `noisy`           Detection/tracking under image noise
  `low_light`       Reduced-brightness environment
  `noise_test`      Additional detector-noise testing
  `target_loss`     Target-loss/reacquisition testing
  `stress_test`     More demanding robustness testing

------------------------------------------------------------------------

## Requirements

The project currently uses a deliberately small dependency set:

-   Python
-   NumPy
-   OpenCV
-   PyYAML
-   Matplotlib

Pinned package versions are available in `requirements.txt`.

PyTorch and Ultralytics are **not currently required** by the main
pipeline. They are intended to be added only if later experiments
justify a learned detector.

------------------------------------------------------------------------

## Installation

### 1. Clone or download the repository

``` bash
git clone <your-repository-url>
cd AI-VISTA
```

If you are using the project ZIP, extract it and open a terminal inside
the `AI-VISTA` directory.

### 2. Create a virtual environment

Windows:

``` bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running AI-VISTA

### Baseline simulation

``` bash
python main.py --scenario default
```

### Circular-motion scenario

``` bash
python main.py --scenario circular
```

### Noisy environment

``` bash
python main.py --scenario noisy
```

### Save an annotated demonstration video

``` bash
python main.py --scenario circular --save-video
```

### Override simulation duration

``` bash
python main.py --scenario noisy --duration 15
```

### Change proportional controller gain

``` bash
python main.py --scenario default --gain 0.5
```

### Change the detector brightness threshold

``` bash
python main.py --scenario default --detector-threshold 100
```

Command-line options can be combined, for example:

``` bash
python main.py --scenario stress_test --duration 20 --gain 0.4 --detector-threshold 110 --save-video
```

------------------------------------------------------------------------

## Running Tests

The repository uses Python's built-in `unittest` framework.

``` bash
python -m unittest discover tests -v
```

The supplied tests cover major components including:

-   Configuration loading and validation
-   Beacon motion
-   Classical beacon detection
-   Camera control logic

------------------------------------------------------------------------

## Output

At the end of a run, AI-VISTA prints a measured performance summary and
saves the experiment logs.

Typical output location:

``` text
data/output/default_YYYYMMDD_HHMMSS/
```

The metrics subsystem is designed around SIH-relevant targets such as:

-   Acquisition time
-   Tracking/alignment error
-   Target-loss percentage
-   Reacquisition behaviour
-   Processing FPS

Performance values are calculated from the actual run rather than
inserted manually.

When `--save-video` is enabled, an annotated MP4 is also written to the
configured output directory.

------------------------------------------------------------------------

## Current Development Status

The current repository already implements the central
simulation-to-control loop:

``` text
Motion → Rendering → Disturbances → Detection
       → Tracking → Error → Camera Control → Metrics
```

Completed or substantially implemented components include the virtual
world, beacon trajectories, virtual camera, frame rendering, classical
detection, centroid extraction, Kalman tracking, pixel-error
calculation, proportional camera control, several disturbances, and
automatic CSV/JSON logging.

The development plan identifies the following major areas as pending or
still being expanded:

-   Pixel-error to angular-error geometry
-   Complete disturbance coverage
-   Full robustness benchmark sweep
-   Evidence-based classical CV vs AI/YOLO comparison
-   Recorded `.mp4` benchmark input
-   Real-time GUI/dashboard
-   Standalone application packaging
-   Final hardware-integration pathway

See `DEVELOPMENT_PLAN.md` for the phase-by-phase roadmap.

------------------------------------------------------------------------

## Current Limitations

1.  **No active search pattern yet.**\
    The camera responds after the beacon enters its field of view. A
    future acquisition mode can scan the scene when the target is
    initially outside the viewport.

2.  **Finite virtual-camera field of view.**\
    World-edge geometry can prevent perfect centring when the target
    approaches regions where a full camera viewport cannot be
    maintained.

3.  **Disturbance robustness is still under evaluation.**\
    Classical threshold-based detection can become less accurate under
    severe noise or environmental degradation.

4.  **Not all configured disturbances are active.**\
    Rain and platform-motion options exist in configuration but are not
    yet applied by the current main disturbance function.

5.  **AI detection is not yet integrated into the main pipeline.**\
    This is deliberate. The project plans to compare a learned detector
    against the lightweight classical approach before selecting the
    final method.

6.  **Recorded-video and GUI modes are not yet implemented.**

------------------------------------------------------------------------

## Future Scope

AI-VISTA can be extended with:

-   Active target search/scanning before acquisition
-   Angular error estimation from camera FOV and image geometry
-   PID or adaptive camera control
-   AI/YOLO-based beacon detection when experimentally justified
-   Automatic switching between classical and learned detectors
-   Real recorded-video benchmarking
-   Rain and richer atmospheric models
-   Platform-motion simulation
-   Live tracking dashboard
-   Experiment plots and comparison reports
-   Hardware-in-the-loop testing
-   Integration with pan-tilt units, gimbals, or real optical terminals
-   Fine-pointing handoff after successful coarse alignment

------------------------------------------------------------------------

## SIH Demonstration Value

The project separates the system into independent modules so that each
stage can be tested and explained clearly:

``` text
Environment → Perception → Tracking → Control → Evaluation
```

This gives the team a reproducible way to demonstrate not only that the
beacon can be tracked, but also **how accurately, how quickly, and under
what conditions the coarse-alignment loop succeeds or fails**.

That measured, modular approach also creates a practical path from
software simulation to recorded-video testing and eventually to physical
FSOC terminal control.

------------------------------------------------------------------------

## Project

**AI-VISTA**\
**AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile
Free Space Optical Communication Terminals**

**Smart India Hackathon 2026**\
**Problem Statement ID: SIH26169**
