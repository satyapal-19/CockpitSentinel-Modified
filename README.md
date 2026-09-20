<h1 align="center">
  🚗 CockpitSentinel
</h1>

<p align="center">
  <strong>Intelligent AI Driver Monitoring System with Automotive-Grade Algorithmic Robustness</strong>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#algorithmic-innovations">Algorithmic Innovations</a> •
  <a href="#audio-escalation">Audio Escalation</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#team">Team</a>
</p>

---

## Key Features

| Module | What It Detects / Solves | How It Works |
| :--- | :--- | :--- |
| **P80 PERCLOS & Micro-Sleep** | Progressive fatigue & acute micro-sleeps | Rolling 30s/60s temporal sliding window; $>1.5\text{s}$ continuous eye closure triggers immediate `CRITICAL` |
| **Speech vs. Yawn Discrimination** | Distinguishes conversational speech from yawns | Temporal derivative $\frac{d(\text{MAR})}{dt}$ and oscillation frequency analysis over a rolling buffer |
| **Sunglasses Occlusion Fallback** | Detects dark/polarized sunglasses and glare | Eye ROI luminance contrast variance ($\sigma < 12.0$); transitions to secondary indicators (head pitch droop & yawns) |
| **3D Head Pose & Nodding** | Downward head droop & cyclic nod-offs | Full 3D Euler angles (Pitch, Yaw, Roll) via PnP decomposition; pitch $\ge 15^\circ$ forward droop detection |
| **Distraction Detection** | Handheld mobile phone use & smoking | YOLOv8 + YOLOWorld open-vocabulary zero-shot object detection |
| **3-Tier Contextual Audio** | Acoustic alerts matched to severity | Non-blocking background worker thread with cooldown throttling (Caution, Warning, Critical siren) |
| **Device Agnostic** | GPU/NPU acceleration across platforms | Auto-detects NVIDIA CUDA, Apple Silicon MPS, and CPU fallback |

---

## Algorithmic Innovations (Solving Real-World CV Edge Cases)

### 1. Speech vs. Yawn Discrimination (Eliminating False Positives)
* **The Problem:** Standard Mouth Aspect Ratio (MAR) static thresholding triggers constant false alarms when the driver talks to passengers, sings along to music, or chews gum.
* **The Solution (`YawnSpeechDiscriminator`):**
  * Speech manifests as rapid, oscillating mouth aperture fluctuations ($\ge 1.5\text{ Hz}$) with short durations ($< 0.5\text{s}$).
  * A true fatigue yawn has a distinct temporal profile: slow expansion, sustained wide aperture ($\text{MAR} \ge 0.60$ for $\ge 1.5\text{s}$), low oscillation rate, and involuntary eye squinting.
  * When speech oscillations are detected, false yawn alarms are automatically suppressed.

### 2. Automotive Industry-Standard PERCLOS Metric
* **The Problem:** Simple consecutive frame counting fails to assess cumulative, progressive drowsiness and slow eyelid droops.
* **The Solution (`PERCLOSTracker`):**
  * Implements the **P80 PERCLOS** standard recognized by NHTSA and Euro NCAP:
    $$\text{PERCLOS} = \frac{\text{Time eyes are closed}}{\text{Observation Window (e.g. 30s – 60s)}} \times 100$$
  * Evaluates progressive fatigue ($\text{PERCLOS} \ge 20\%$).
  * Tracks continuous closure duration and triggers `microsleep_detected` if eyes remain closed for $\ge 1.5\text{s}$.

### 3. Sunglasses / Occlusion Fallback Mode
* **The Problem:** Drivers frequently wear sunglasses or glare-resistant glasses, causing eye landmarks to collapse or become completely unreliable.
* **The Solution (`EyeOcclusionDetector`):**
  * Evaluates grayscale standard deviation and mean luminance in the eye bounding box:
    $$\sigma = \text{std}(\text{ROI}_{\text{eye}})$$
  * Sunglasses exhibit flat, uniform dark tint ($\sigma < 12.0$ or $\mu < 28.0$).
  * When occlusion is detected:
    * System displays `[SUNGLASSES MODE]` on the HUD.
    * Bypasses eye landmark metrics to eliminate dark-frame false alarms.
    * Automatically elevates **Secondary Fatigue Indicators**:
      1. **3D Head Nodding / Micro-droops (`head_nodding`)**: Downward head tilt ($\text{pitch} \ge 15^\circ$) or cyclic nods trigger immediate alert escalation.
      2. **Mouth Yawning (`yawning`)**: Monitored via visible mouth landmarks with speech filtering.

---

## 3-Tier Contextual Audio Escalation Subsystem

Audio alerts are generated asynchronously on a dedicated daemon worker thread consuming from a bounded queue, ensuring **zero frame drops** and steady FPS in video processing.

| Tier | Trigger Severity | Acoustic Cue | Cooldown |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Caution)** | Score $\ge 2$ (e.g., yawning, PERCLOS fatigue) | Soft chime (800 Hz, 150ms) | 5.0 seconds |
| **Tier 2 (Warning)** | Score $\ge 4$ (e.g., phone, head droop) | Urgent double-tone (1200 Hz, 250ms $\times$ 2) | 6.0 seconds |
| **Tier 3 (Critical)**| Score $\ge 6$ (e.g., micro-sleep, combined hazards) | Emergency pulsing siren (2000 Hz, 120ms $\times$ 4) | 2.0 seconds |

> Use the `--no-audio` flag if you wish to run the monitor silently with visual overlay alerts only.

---

## Architecture

```
                             ┌──────────────────────────┐
                             │   Camera / Video Frame   │
                             └─────────────┬────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
              DrowsinessDetector                     DistractionDetector
          (MediaPipe Face Landmarker)                (YOLOv8 + YOLOWorld)
                        │                                     │
         ┌──────────────┼──────────────┐                      │
         ▼              ▼              ▼                      │
    Eye Contrast    Yawn/Speech     3D Pose                   │
    (Occlusion)     (Derivative)  (Pitch/Yaw)                 │
         │              │              │                      │
         │              ▼              ▼                      │
         │       [yawn vs talk]  [head nod]                   │
         ▼              │              │                      │
   PERCLOS (P80)        │              │                      │
         │              └───────┬──────┘                      │
         └──────────────┐       │                             │
                        ▼       ▼                             ▼
                      DriverSignals ◄─────────────────────────┘
                   (eyes_closed, yawning, talking,
                    looking_away, phone, smoking,
                    perclos_fatigue, microsleep,
                    head_nodding, eye_occluded)
                                │
                                ▼
                           AlertPolicy
             (Risk fusion & occlusion fallback mode)
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
      OpenCV Live Overlay                    AudioManager
    - Status & Risk Level               - Tier 1 Caution Chime
    - [SUNGLASSES MODE] Badge           - Tier 2 Warning Tone
    - Telemetry (EAR, MAR, Pitch, Yaw)  - Tier 3 Critical Siren
    - Emergency Flashing Banners        - Zero-lag Daemon Thread
```

---

## Quick Start

> **Prerequisites:** Python 3.11 or 3.12 · Git · Webcam

### 1. Windows Setup

```powershell
# Clone the repository
git clone https://github.com/satyapal-19/CockpitSentinel.git CockpitSentinelAntigravity
cd CockpitSentinelAntigravity

# Bootstrap virtual environment & models
.\scripts\setup_environment.ps1

# Activate environment and verify
.\.venv\Scripts\Activate.ps1
python scripts\verify_environment.py

# Run test suite (65 passing unit tests)
pytest --cov=cockpit_sentinel

# Launch the live monitor
python -m cockpit_sentinel.pipeline.live_monitor
```

### 2. macOS / Linux Setup

```bash
git clone https://github.com/satyapal-19/CockpitSentinel.git CockpitSentinelAntigravity
cd CockpitSentinelAntigravity

bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_environment.py
pytest
python -m cockpit_sentinel.pipeline.live_monitor
```

---

## Usage & CLI Options

```bash
# Default webcam live monitoring (with 3-tier audio alerts)
python -m cockpit_sentinel.pipeline.live_monitor

# Run with custom video file
python -m cockpit_sentinel.pipeline.live_monitor --source path/to/driving_video.mp4

# Run with GPU acceleration (NVIDIA CUDA)
python -m cockpit_sentinel.pipeline.live_monitor --device cuda

# Run silently (audio alerts muted, HUD visual alerts remain active)
python -m cockpit_sentinel.pipeline.live_monitor --no-audio

# Inspect all available options
python -m cockpit_sentinel.pipeline.live_monitor --help
```

Press **Q** or **Esc** inside the video display window to stop cleanly.

---

## Configuration

All runtime thresholds are managed via YAML in [`configs/`](configs/):

| Configuration File | Controlled Parameters |
| :--- | :--- |
| [`drowsiness.yaml`](configs/drowsiness.yaml) | `eye_aspect_ratio_threshold` (0.22), `perclos_window_seconds` (30.0), `perclos_fatigue_threshold` (0.20), `microsleep_threshold_seconds` (1.5), `yawn_min_duration_seconds` (1.5), `speech_mar_threshold` (0.38), `head_pitch_threshold_degrees` (15.0), `eye_contrast_threshold` (12.0) |
| [`distraction.yaml`](configs/distraction.yaml) | YOLO confidence thresholds for mobile phone and smoking detection |
| [`alerts.yaml`](configs/alerts.yaml) | Signal weights (`microsleep`: 6, `head_nodding`: 4, `phone`: 4, `eyes_closed`: 3, `smoking`: 3, `perclos`: 3, `yawning`: 2, `looking_away`: 2, `talking`: 0, `eye_occluded`: 0) and risk thresholds (Caution $\ge 2$, Warning $\ge 4$, Critical $\ge 6$) |
| [`models.yaml`](configs/models.yaml) | Model filenames and resolution registry |

---

## Project Structure

```
CockpitSentinelAntigravity/
├── configs/                      # YAML configuration files
│   ├── alerts.yaml               #   Weights, score thresholds & fallback
│   ├── distraction.yaml          #   YOLO detection confidences
│   ├── drowsiness.yaml           #   EAR, MAR, PERCLOS, speech & pitch limits
│   └── models.yaml               #   Model asset registry
├── scripts/                      # Bootstrap and validation utilities
│   ├── run_drowsiness_monitor.py #   Convenience launcher script
│   ├── setup_environment.ps1     #   Windows setup bootstrap
│   ├── setup.sh                  #   Unix setup bootstrap
│   └── verify_environment.py     #   Health and hardware check
├── src/cockpit_sentinel/         # Core package
│   ├── alerts/                   #   Policy scoring and audio escalation
│   │   ├── audio.py              #     Non-blocking 3-tier AudioManager
│   │   └── policy.py             #     Weighted assessment engine
│   ├── detection/                #   Object detection (YOLOv8 + YOLOWorld)
│   ├── drowsiness/               #   MediaPipe geometry & fatigue trackers
│   │   ├── detector.py           #     FaceLandmarker, EAR, MAR & Head pose
│   │   ├── occlusion.py          #     Sunglasses & eye contrast detector
│   │   └── yawn_speech.py        #     Speech vs. Yawn discriminator
│   ├── pipeline/                 #   Live video capture and HUD rendering
│   │   └── live_monitor.py       #     LiveMonitor frame loop & overlay
│   ├── utils/                    #   Cross-device, GPU resolution & paths
│   └── domain.py                 #   Shared dataclasses, signals & contracts
├── tests/unit/                   # Comprehensive unit test suite (65 tests)
│   ├── test_alert_policy.py
│   ├── test_audio.py
│   ├── test_device.py
│   ├── test_distraction_detector.py
│   ├── test_drowsiness_detector.py
│   ├── test_environment.py
│   ├── test_head_pose.py
│   ├── test_live_monitor.py
│   ├── test_occlusion.py
│   ├── test_paths.py
│   ├── test_perclos.py
│   ├── test_risk_scoring.py
│   └── test_yawn_speech.py
├── pyproject.toml                # Build configuration & tool settings
└── README.md
```

---

## Verification & Testing

The test suite contains **65 unit tests** verifying all mathematical calculations, temporal buffers, edge-case classifications, and policy transitions:

```powershell
# Run the test suite with coverage report
pytest --cov=cockpit_sentinel --cov-report=term-missing
```

**Quality Standards:**
- Fully linted with `ruff check .` (0 warnings).
- Code formatting enforced with `ruff format .`.
- Full typing safety and validation across all configurations.

---

## Team

**Walchand College of Engineering, Sangli** — Department of Computer Science & Engineering  
*Mini Project 2026–27*

- **Satyapal Gaikwad**
- **Darshan Patil**
- **Yashodhan Jadhav**

---

<p align="center">
  <sub>CockpitSentinel • Safe Driving Through Intelligent Computer Vision</sub>
</p>
