<h1 align="center">
  🚗 CockpitSentinel
</h1>

<p align="center">
  <strong>Intelligent AI Driver Monitoring System with Automotive-Grade Algorithmic Robustness</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Tests-78%20Passing-brightgreen?logo=pytest" alt="78 Tests Passing">
  <img src="https://img.shields.io/badge/Coverage-72%25-green" alt="72% Test Coverage">
  <img src="https://img.shields.io/badge/Code%20Style-Ruff-black" alt="Ruff Formatted">
  <img src="https://img.shields.io/badge/Compute-CUDA%20%7C%20MPS%20%7C%20CPU-orange" alt="Hardware Agnostic">
  <img src="https://img.shields.io/badge/Framework-FastAPI%20%2B%20OpenCV-teal" alt="FastAPI + OpenCV">
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#algorithmic-innovations">Algorithmic Innovations</a> •
  <a href="#auto-driver-recognition--multi-driver-profiles">Driver Recognition</a> •
  <a href="#telematics-web-dashboard">Web Dashboard</a> •
  <a href="#3-tier-contextual-audio-escalation-subsystem">Audio Escalation</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage--cli-options">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#project-structure">Structure</a> •
  <a href="#team">Team</a>
</p>

---

## Key Features

| Module | What It Detects / Solves | How It Works |
| :--- | :--- | :--- |
| **Auto Driver Recognition** | Personalized baseline & multi-driver identity | 20-D normalized bone distance signature matching ($\ge 92\%$); auto-binds personalized thresholds upon startup |
| **3s Auto-Calibration** | Custom EAR & MAR thresholds per facial anatomy | 3-second baseline capture computes optimal thresholds ($\text{EAR}_{th} = \mu \times 0.70$, $\text{MAR}_{th} = \mu \times 2.80$) |
| **Telematics Web Dashboard** | Real-time monitoring & profile management | FastAPI + WebSockets HUD, live video feed, real-time gauges, profile manager, and interactive boundary sliders |
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

### 4. Auto Driver Recognition & Multi-Driver Profiles
* **The Problem:** Generic thresholds fail across diverse drivers. A driver with naturally narrow eyes will suffer false drowsiness alarms under static $0.22$ EAR, while a driver with wide eyes might be drowsy at $0.24$.
* **The Solution (`DriverRecognizer` & `ProfileManager`):**
  * **Privacy-First Bone Distance Signatures:** Instead of saving unencrypted face images, CockpitSentinel extracts a 20-dimensional scale-invariant geometric ratio vector:
    $$\vec{S} = \left[ \frac{\|\mathbf{p}_{i} - \mathbf{p}_{j}\|_2}{D_{\text{interocular}}} \right] \quad (i, j \in \text{key facial bone landmarks})$$
  * **Seamless Startup Recognition:** During the first 60 frames after ignition/start, the recognizer calculates cosine similarity against enrolled profiles ($\ge 0.92$ match confidence):
    $$\text{Cosine Similarity} = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
  * **Zero-Intervention Threshold Binding:** Once recognized, the monitor dynamically swaps active thresholds (`ear_threshold`, `mar_threshold`, `head_pitch`) to match the identified driver's personal anatomy.

### 5. One-Click 3-Second Auto-Calibration
* **The Problem:** Drivers shouldn't need to manually guess or tweak numerical ratios in configuration files.
* **The Solution:**
  * Driver sits comfortably in normal driving posture for 3 seconds (90 frames).
  * CockpitSentinel calculates resting geometry baselines ($\mu_{\text{EAR}}$, $\mu_{\text{MAR}}$) and derives custom bounds:
    $$\text{EAR}_{\text{threshold}} = \max\left(0.14, \min(0.28, \mu_{\text{EAR}} \times 0.70)\right)$$
    $$\text{MAR}_{\text{threshold}} = \max\left(0.52, \min(0.75, \mu_{\text{MAR}} \times 2.80)\right)$$
  * Updates disk persistence (`data/profiles.json`) and binds immediately into the live pipeline.

---

## Telematics Web Dashboard

CockpitSentinel includes a full-stack, responsive web dashboard built with **FastAPI**, **HTML5**, **Tailwind CSS**, and **WebSockets**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🚗 CockpitSentinel Telematics HUD            [ACTIVE DRIVER: Satyapal 97%]  │
├──────────────────────────────────────┬──────────────────────────────────────┤
│                                      │  🎛️ DRIVER PROFILES                  │
│       LIVE MJPEG VIDEO STREAM        │  Active: [ Satyapal Gaikwad     ▼ ]  │
│        (Sub-10ms latency)            │  [+ New Driver]  [⏱️ Auto-Calibrate] │
│                                      ├──────────────────────────────────────┤
│   - Bounding Boxes & Facial Mesh     │  🎚️ BOUNDARY ADJUSTMENTS (Live)      │
│   - Real-time HUD Badges             │  EAR Threshold: [──●────] 0.22       │
│   - Dynamic Hazard Banners           │  MAR Threshold: [────●──] 0.60       │
│                                      │  Pitch Limit:   [──●────] 15.0°      │
├──────────────────────────────────────┴──────────────────────────────────────┤
│  📊 REAL-TIME TELEMETRY GAUGES                                              │
│  [ EAR: 0.31 ]    [ MAR: 0.18 ]    [ PERCLOS: 4.2% ]    [ Pitch: -2.1° ]    │
│  [ Yaw: 3.5° ]    [ Risk: 0 SAFE ] [ FPS: 30.0 ]        [ Safety: 98/100 ]  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Capabilities:
- **Live Video Streaming**: Real-time MJPEG feed streamed at 30 FPS directly to the browser.
- **25 Hz WebSocket Telemetry**: Instantaneous pushing of facial angles, closure ratios, fatigue metrics, and active alert state.
- **Driver Profile Hub**: Create, switch, or inspect drivers on-the-fly without restarting the video feed.
- **Dynamic Boundary Sliders**: Adjust EAR, MAR, and Pitch limits with instantaneous REST synchronization to disk.
- **1-Click 3-Second Auto-Calibration**: Captures live resting geometry and recalibrates driver baseline on demand.
- **Trip Safety Score**: Dynamic 100-point safety index penalized by microsleep, distraction, nodding, and yawning events.

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

## API Reference

The dashboard provides a complete RESTful and WebSocket API:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Responsive HTML5 telematics dashboard web interface |
| `GET` | `/api/profiles` | List all registered driver profiles |
| `POST` | `/api/profiles` | Create a new driver profile (`name`, resting & threshold EAR/MAR) |
| `GET` | `/api/profiles/{id}` | Retrieve details for a specific driver profile |
| `PUT` | `/api/profiles/{id}` | Update thresholds (EAR, MAR, Pitch) for a profile |
| `POST` | `/api/profiles/{id}/calibrate` | Run 3-second auto-calibration on the driver profile |
| `POST` | `/api/profiles/{id}/activate` | Set active driver profile for the live pipeline |
| `GET` | `/api/telemetry` | Get latest instantaneous telemetry snapshot (JSON) |
| `GET` | `/api/video_feed` | Multipart MJPEG real-time video stream |
| `WS` | `/ws/telemetry` | Bi-directional WebSocket streaming telemetry at 25 Hz |

---

## Architecture

```
                             ┌──────────────────────────┐
                             │   Camera / Video Frame   │
                             └─────────────┬────────────┘
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
               DrowsinessDetector                   DistractionDetector
           (MediaPipe Face Landmarker)              (YOLOv8 + YOLOWorld)
                         │                                   │
          ┌──────────────┼──────────────┐                    │
          ▼              ▼              ▼                    │
     Eye Contrast   Yawn/Speech      3D Pose                 │
     (Occlusion)    (Derivative)   (Pitch/Yaw)               │
          │              │              │                    │
          │              ▼              ▼                    │
          │       [yawn vs talk]   [head nod]                │
          ▼              │              │                    │
    PERCLOS (P80)        │              │                    │
          │              └───────┬──────┘                    │
          └──────────────┐       │                           │
                         ▼       ▼                           ▼
                       DriverSignals ◄───────────────────────┘
                    (eyes_closed, yawning, talking,
                     looking_away, phone, smoking,
                     perclos_fatigue, microsleep,
                     head_nodding, eye_occluded)
                                 │
                                 ▼
                            AlertPolicy
              (Risk fusion & occlusion fallback mode)
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
    Live OpenCV HUD      AudioManager (Daemon)   Web Dashboard (FastAPI)
  - Biometric Badge    - Tier 1: Soft Chime    - Live MJPEG Video Feed
  - 3D Euler Vectors   - Tier 2: Double Tone   - Real-Time WebSockets
  - Warning Banners    - Tier 3: Alarm Siren   - Profile Calibration
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

# Run test suite (78 passing unit tests)
pytest --cov=cockpit_sentinel

# Launch the live desktop monitor
python -m cockpit_sentinel.pipeline.live_monitor

# Or launch the Telematics Web Dashboard
python scripts/run_dashboard.py
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

### 1. Telematics Web Dashboard (Recommended)

```bash
# Launch dashboard at http://127.0.0.1:8000
python scripts/run_dashboard.py

# Custom host, port, or video source
python scripts/run_dashboard.py --host 0.0.0.0 --port 8000 --source 0 --device cuda
```

### 2. Desktop HUD Monitor

```bash
# Default webcam live monitoring (with 3-tier audio alerts & auto driver recognition)
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
├── data/                         # Persistent runtime storage
│   └── profiles.json             #   Driver profiles & biometric signatures
├── scripts/                      # Bootstrap and validation utilities
│   ├── run_dashboard.py          #   Telematics Web Dashboard launcher
│   ├── run_drowsiness_monitor.py #   Convenience launcher script
│   ├── setup_environment.ps1     #   Windows setup bootstrap
│   ├── setup.sh                  #   Unix setup bootstrap
│   └── verify_environment.py     #   Health and hardware check
├── src/cockpit_sentinel/         # Core package
│   ├── alerts/                   #   Policy scoring and audio escalation
│   │   ├── audio.py              #     Non-blocking 3-tier AudioManager
│   │   └── policy.py             #     Weighted assessment engine
│   ├── dashboard/                #   Telematics Web Dashboard (FastAPI)
│   │   ├── templates/index.html  #     Real-time HTML5/Tailwind dashboard UI
│   │   └── app.py                #     REST & WebSocket telemetry server
│   ├── detection/                #   Object detection (YOLOv8 + YOLOWorld)
│   │   └── distraction.py        #     Phone use and smoking inference
│   ├── drowsiness/               #   MediaPipe geometry & fatigue trackers
│   │   ├── detector.py           #     FaceLandmarker, EAR, MAR & Head pose
│   │   ├── occlusion.py          #     Sunglasses & eye contrast detector
│   │   ├── recognition.py        #     Biometric face recognition & profiles
│   │   └── yawn_speech.py        #     Speech vs. Yawn discriminator
│   ├── pipeline/                 #   Live video capture and HUD rendering
│   │   └── live_monitor.py       #     LiveMonitor frame loop & overlay
│   ├── utils/                    #   Cross-device, GPU resolution & paths
│   │   ├── device.py             #     CUDA, MPS, and CPU resolution
│   │   └── paths.py              #     Cross-platform path resolution
│   └── domain.py                 #   Shared dataclasses, signals & contracts
├── tests/unit/                   # Comprehensive unit test suite (78 tests)
│   ├── test_alert_policy.py      #   Alert policy scoring & weights
│   ├── test_audio.py             #   Audio manager queue & cooldowns
│   ├── test_dashboard_api.py     #   FastAPI REST endpoints & telemetry
│   ├── test_device.py            #   Hardware device detection & fallback
│   ├── test_distraction_detector.py # YOLO distraction detection
│   ├── test_drowsiness_detector.py # MediaPipe geometry & thresholds
│   ├── test_environment.py       #   Package import & version check
│   ├── test_head_pose.py         #   3D Euler angles & head droop
│   ├── test_live_monitor.py      #   Pipeline fusion & overlay loop
│   ├── test_occlusion.py         #   Sunglasses contrast & fallback mode
│   ├── test_paths.py             #   Path resolution & project markers
│   ├── test_perclos.py           #   P80 PERCLOS & micro-sleep detection
│   ├── test_profiles.py          #   Profile CRUD & auto-calibration
│   ├── test_recognition.py       #   Scale invariance & face recognition
│   ├── test_risk_scoring.py      #   Signal risk synthesis
│   └── test_yawn_speech.py       #   Oscillation-based speech filtering
├── pyproject.toml                # Build configuration & tool settings
└── README.md
```

---

## Verification & Testing

The test suite contains **78 unit tests** verifying all mathematical calculations, temporal buffers, edge-case classifications, biometric face signatures, and policy transitions:

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
