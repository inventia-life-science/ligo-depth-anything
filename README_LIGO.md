# Depth Anything 3 - LIGO Setup Guide

This document describes the setup steps needed to get Depth Anything 3 working with `uv` package manager, including CUDA-enabled PyTorch installation and model downloads.

## 1. Installing Dependencies with `uv`

### Prerequisites

- Python 3.10-3.13 (required for Gradio support)
- `uv` package manager installed
- CUDA-capable GPU (optional, but recommended)

### Step-by-Step Installation

#### 1.1 Create and Activate Virtual Environment

```bash
cd $YOUR_REPO_DIRECTORY

# Create a virtual environment (not ".venv" parameter optional as uv will default the venv name to .venv)
uv venv .venv

# Activate it
source .venv/bin/activate
```

#### 1.2 Install CUDA-Enabled PyTorch

**Important**: Install PyTorch with CUDA support **first**, before other dependencies. This prevents dependency resolution conflicts.

```bash
# Install PyTorch 2.9.1 with CUDA 12.8 support
uv pip install 'torch==2.9.1+cu128' --index-url https://download.pytorch.org/whl/cu128
```

Verify the installation:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Should output: 2.9.1+cu128 True
```

#### 1.3 Install Project Dependencies

After PyTorch is installed, install the rest of the project:

```bash
# Install build dependencies first
uv pip install hatchling

# Install the project with all extras (no build isolation needed since torch is already installed)
uv pip install --no-build-isolation -e '.[all]'
```

**Note**: The `--no-build-isolation` flag is needed because:
- `xformers` requires `torch` during build but doesn't declare it properly
- The project itself needs `hatchling` in the environment when building

**Alternative**: If you encounter issues, you can install dependencies from `requirements.txt`:

```bash
uv pip install -r requirements.txt
uv pip install -e '.[all]'
```

### Troubleshooting

- **Out of Memory (OOM) errors**: Use a smaller model (`DA3-SMALL` instead of `DA3-LARGE`) or lower `process_res` in your scripts
- **xformers build failures**: The codebase handles missing `xformers` gracefully - it's optional and will fall back to pure PyTorch implementations
- **Gradio/Python version conflicts**: Ensure Python >= 3.10 (see `pyproject.toml`)

## 2. Downloading Models Locally

Models are automatically downloaded from Hugging Face on first use, but you can pre-download them to a local directory for offline use or faster loading.

### 2.1 Download Model Using Python API

```bash
# Make sure you're in the project root with venv activated
cd /home/tom/repos/Depth-Anything-3
source .venv/bin/activate

# Download model to local directory
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('depth-anything/DA3-BASE', local_dir='models/DA3-BASE', local_dir_use_symlinks=False)"
```

Note that with 4GB VRAM, the DA3-LARGE model was working up to ~800 res, but took about 1 second for inference. On the contrary, the DA3-BASE model took 0.6 seconds for 800 res and 0.5s for 384 res

Replace `DA3-BASE` with your desired model:
- `depth-anything/DA3-SMALL` - Smallest, best for 4GB VRAM
- `depth-anything/DA3-BASE` - Base model
- `depth-anything/DA3-LARGE` - Large model (requires more VRAM)
- `depth-anything/DA3-GIANT` - Largest model
- `depth-anything/DA3METRIC-LARGE` - **Metric depth model** (see below)

#### DA3METRIC-LARGE - Metric Depth Estimation

**Use Case**: The `DA3METRIC-LARGE` model is specialized for **metric depth estimation** in monocular settings. Unlike the standard models which predict relative depth (depth values are normalized/scaled), this model predicts depth in **real-world units** (meters, centimeters, etc.).

**Key Features:**
- ✅ Metric depth (real-world scale)
- ✅ Relative depth estimation
- ✅ Sky segmentation
- ❌ **No pose estimation** (monocular only)
- ❌ No multi-view capabilities
- ❌ No Gaussian splatting

**When to Use:**
- Applications requiring actual distance measurements (e.g., robotics, AR/VR, surveying)
- When you need depth values in meters/centimeters rather than normalized values
- Single-image depth estimation where scale matters
- **Not suitable** if you need camera pose estimation or multi-view depth fusion

**Example Usage:**
```python
model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
prediction = model.inference([image])
# prediction.depth will contain metric depth values (in meters)
# prediction.is_metric will be True
```

### 2.2 Using Local Models in Code

Once downloaded, point your code to the local directory:

```python
# Instead of:
model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")

# Use:
model = DepthAnything3.from_pretrained("/home/tom/repos/Depth-Anything-3/models/DA3-BASE")
```

### Model Size Recommendations

- **4GB VRAM**: Use `DA3-SMALL` with `process_res=320-384`
- **8GB VRAM**: Use `DA3-BASE` or `DA3METRIC-LARGE` with `process_res=384-504`
- **16GB+ VRAM**: Can use `DA3-LARGE` or `DA3-GIANT` with higher resolutions

**Note**: `DA3METRIC-LARGE` has the same parameter count (0.35B) as `DA3-LARGE`, so VRAM requirements are similar. Use it when you need metric depth values rather than relative depth.

## 3. Run Files

### `run.py` - Batch Image Processing

Processes multiple images from a directory and exports depth visualizations.

**What it does:**
1. Loads the `DA3-BASE` model
2. Processes all PNG images from `assets/examples/SOH/`
3. Runs depth estimation inference
4. Exports side-by-side RGB + depth visualizations to `outputs/depth_vis/`

**Usage:**
```bash
uv run python run.py
```

**Output:**
- `outputs/depth_vis/0000.jpg`, `0001.jpg`, etc. - Side-by-side RGB and depth visualizations

### `run_with_capture.py` - Webcam Capture with Pose Estimation

Captures a single frame from your webcam, runs depth estimation, and estimates camera poses.

**What it does:**
1. Loads the `DA3-BASE` model (configured to use local model path)
2. Captures a single frame from the default webcam (index 0)
3. Runs depth estimation and **camera pose estimation** (extrinsics + intrinsics)
4. Prints estimated camera parameters:
   - Extrinsics matrix (4x4 world-to-camera transformation)
   - Camera position in world space
   - Intrinsics matrix (3x3 camera calibration)
   - Focal length and principal point
5. Exports depth visualization to `outputs_webcam/depth_vis/`
6. Saves camera poses to `outputs_webcam/camera_poses.npz`

**Usage:**
```bash
uv run python run_with_capture.py
```

**Output:**
- `outputs_webcam/depth_vis/0000.jpg` - Side-by-side RGB and depth visualization
- `outputs_webcam/camera_poses.npz` - NumPy archive containing:
  - `extrinsics`: (1, 4, 4) array - Camera extrinsics matrices
  - `intrinsics`: (1, 3, 3) array - Camera intrinsics matrices

**Loading saved poses:**
```python
import numpy as np
data = np.load("outputs_webcam/camera_poses.npz")
extrinsics = data["extrinsics"]  # (1, 4, 4)
intrinsics = data["intrinsics"]  # (1, 3, 3)
```

### Key Differences

| Feature | `run.py` | `run_with_capture.py` |
|---------|----------|----------------------|
| Input | Directory of images | Webcam capture |
| Pose Estimation | No | Yes (automatic) |
| Output Location | `outputs/` | `outputs_webcam/` |
| Model Path | Hugging Face repo ID | Local directory path |
| Process Resolution | 384 (lower for VRAM) | 800 (adjustable) |

## Quick Reference

### Common Commands

```bash
# Activate environment
source .venv/bin/activate

# Run batch processing
uv run python run.py

# Run webcam capture
uv run python run_with_capture.py

# Check GPU memory usage
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Paths

- **Hugging Face (online)**: `"depth-anything/DA3-BASE"` (or `DA3-SMALL`, `DA3-LARGE`, `DA3METRIC-LARGE`, etc.)
- **Local (offline)**: `"/home/tom/repos/Depth-Anything-3/models/DA3-BASE"` (adjust path as needed)

### Output Directories

- `outputs/depth_vis/` - Batch processing results
- `outputs_webcam/depth_vis/` - Webcam capture results
- `outputs_webcam/camera_poses.npz` - Saved camera poses

