# Road Safety Audit (RSA) Telemetry System

An automated road safety audit pipeline that turns a forward-facing dashcam video into measurable road-safety telemetry. It combines YOLO segmentation, Depth Anything V2 metric depth, SAM 3 and classical computer vision (undistortion, bird's-eye-view transform, polynomial lane fitting) to check roads against IRC-style safety criteria.

## Demo

https://youtu.be/4rYpicGCoUE

## What it measures

| Module | Output |
|---|---|
| **Lane integrity** | Counts detected centre-line and shoulder markings vs. the number you expect; flags missing centre/shoulder lines, crash barriers and kerbs |
| **Lane width** | Per-lane width in metres (BEV pixel distance x calibrated metres-per-pixel), compared to a minimum-width threshold |
| **Road curvature** | Radius of curvature in metres from polynomial fits of lane markings in BEV space, smoothed over 30 frames, compared to a minimum safe radius |
| **Sign clearance** | Tracks traffic signs (BoT-SORT), estimates distance with metric depth, computes the height of the sign's bottom edge above the road and checks it against the 2.0 m minimum |
| **Dynamic horizon / pitch** | Estimates the horizon from the vanishing point of lane lines to compensate for camera pitch |

### Detected classes (custom YOLO model)

`0` Centre Lane Marking, `1` Chevron, `2` Kerb, `3` Rigid Crash Barrier, `4` Shoulder Marking, `5` Traffic Sign Damaged, `6` Traffic Sign Not Visible, `7` W-Beam Crash Barrier, `8` Mandatory Sign, `9` Informative Sign, `10` Warning Sign

## Pipeline overview

```
Video frame
  -> Undistort (camera intrinsics K + distortion coefficients)
  -> YOLO segmentation   : lanes, kerbs, crash barriers
  -> YOLO + BoT-SORT     : traffic signs with persistent IDs
  -> Depth Anything V2   : metric depth map
  -> Perspective warp (BEV) -> quadratic lane fit -> radius & lane widths (metres)
  -> Sign distance + height-above-road
  -> HUD overlay + annotated video + BEV video + CSV log
```

Before the main run, an interactive **calibration phase** is performed:

1. **BEV matrix** - click 4 road points on a frame (or paste a known 3x3 matrix).
2. **MPP_X** (metres per pixel, lateral) - manual value, auto-calculated from the known road width, or measured from two clicked BEV points.
3. **MPP_Y** (metres per pixel, longitudinal) - manual value, SAM 3 on a dashed lane strip of known length, or two clicked BEV points.

## Requirements

- Python 3.10+
- An **NVIDIA GPU with CUDA** (the script uses CUDA mixed precision and SAM 3 on `device=0`)
- A desktop environment (the calibration steps use OpenCV GUI windows)

## Installation

```bash
git clone https://github.com/<your-username>/road-safety-audit.git
cd road-safety-audit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

### Depth Anything V2 (required)

The script imports `metric_depth.depth_anything_v2` from the official repository:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

Download the **metric depth, VKITTI, ViT-B** checkpoint (`depth_anything_v2_metric_vkitti_vitb.pth`) from the [Depth-Anything-V2 repo](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) and place it anywhere you like.

### SAM 3 (optional)

Only needed if you choose "Calculate Dynamically using SAM 3" for MPP_Y.

1. Request access to `sam3.pt` on the SAM 3 page on Hugging Face (it is not downloaded automatically).
2. Install Ultralytics' CLIP fork:
   ```bash
   pip uninstall clip -y
   pip install git+https://github.com/ultralytics/CLIP.git
   ```

### YOLO weights

You need your trained road-audit segmentation model (`best.pt`) with the 11 classes listed above. **Weights are not included in this repository**; they are git-ignored. Host them on Hugging Face / Google Drive / a GitHub Release and link them here:

> **Weights:** `<add download link here>`

## Usage

```bash
python Road_Safety_audit.py \
  --video path/to/road_video.mp4 \
  --yolo-weights path/to/best.pt \
  --depth-checkpoint path/to/depth_anything_v2_metric_vkitti_vitb.pth \
  --depth-anything-dir ./Depth-Anything-V2 \
  --sam3-weights path/to/sam3.pt        # optional
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--video` | yes | - | Input road video |
| `--yolo-weights` | yes | - | Trained YOLO segmentation weights |
| `--depth-checkpoint` | yes | - | Depth Anything V2 metric checkpoint |
| `--depth-anything-dir` | no | `Depth-Anything-V2` | Cloned Depth-Anything-V2 repo |
| `--sam3-weights` | no | `sam3.pt` | SAM 3 weights (MPP_Y option 2 only) |
| `--output-dir` | no | `outputs` | Where results are written |

The script then asks for:

- expected number of lanes, camera height (m, default 1.25)
- minimum safe curve radius (m) and minimum lane width (m)
- whether a crash barrier / kerb is present
- whether to generate the BEV video
- a starting timestamp (s) for calibration

and walks you through the BEV / MPP_X / MPP_Y calibration. Press **q** in the video window to stop early.

### Camera calibration

The script contains the intrinsic matrix `K` and distortion coefficients `dist_coefs` of the camera used during development. **Replace them with your own camera's values** (e.g. from an OpenCV checkerboard calibration), otherwise distances and widths will be wrong.

The `H_c` / `pitch_deg` block at the top and the cone-based `X_ai_distances` table are used for the distance-calibration regression (cones placed at 10 m, 20 m, 30 m); update them if your set-up differs.

## Outputs

Written to `--output-dir`:

| File | Description |
|---|---|
| `<video>_audit.mp4` | Annotated video with lane, curvature, sign-clearance and integrity HUDs |
| `<video>_bev.mp4` | Bird's-eye-view video (only if BEV generation is enabled) |
| `<video>_log.csv` | Telemetry sampled every 0.1 s |

CSV columns: `Timestamp (sec)`, `Expected Total Lanes`, `Centre Lanes Detected`, `Shoulder Lanes Detected`, `Lane Integrity Status`, `Kerb Available`, `Crash Barrier Available`, `Radius of Curvature (m)`, `Curvature Severity`, `Tracked Lane Widths (m)`, `Traffic Signs [ID|Class|Height|Dist]`.

A `distance_calibration_model.pkl` and a `frames/` folder are also created in the working directory.

## Repository structure

```
road-safety-audit/
├── Road_Safety_audit.py   # main pipeline
├── requirements.txt
├── README.md
├── .gitignore
└── assets/                # screenshots / demo GIFs
```

## Limitations

- Accuracy depends on correct camera intrinsics, camera height and the BEV / MPP calibration.
- Intended for a forward-facing, roughly level camera on a vehicle; results on heavily occluded or poorly marked roads will degrade.
- Thresholds (2.0 m sign height, 80 km/h design speed) are set as constants in the script.

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO, BoT-SORT, SAM 3 integration)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Meta SAM 3](https://huggingface.co/facebook/sam3)

## License

Add a license of your choice (e.g. MIT) as `LICENSE`. Note that Ultralytics is AGPL-3.0, and the Depth Anything V2 and SAM 3 weights have their own licence terms, which may affect how you can distribute this project.
