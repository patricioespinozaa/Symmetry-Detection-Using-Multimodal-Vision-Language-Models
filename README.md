# Symmetry Detection Using Multimodal Vision-Language Models

This repository contains the tooling to render 3D objects, generate multi-view images under different lighting conditions and resolutions, and build the dataset used for symmetry detection experiments.

---

## Installation

### GPU server setup

```bash
# Check CUDA version
nvidia-smi

# Create a clean environment
conda create -n tesis_env python=3.10 -y
conda activate tesis_env

# Recomendado: limpiar versiones previas de torch
pip uninstall -y torch torchvision torchaudio pytorch3d

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install pytorch3d==0.7.9+pt2.6.0cu124 \
  --extra-index-url https://miropsota.github.io/torch_packages_builder

pip install transformers==4.57.1 accelerate==1.10.1 \
  pillow==11.3.0 einops==0.8.1 tqdm==4.67.1 molmo_utils
---

## Dataset generation

### 1. Download objects and symmetry data

> *(download instructions here)*

### 2. Generate rendered images

Use `data_render.py` to batch-render all `.obj` files in a folder. It calls `export_fibonacci_views.py` internally for each object and configuration.

Before rendering starts, the script prints an **execution plan** and asks for confirmation (`OK`) so you can verify the total image count.

#### Single GPU

```bash
python3 -m utils.data_render \
  --input-folder ../data/objects/curated_axis_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --sizes 224 448 1024 \
  --lightings darker flat brighter
```

#### Multiple GPUs (one symmetry type per GPU)

```bash
# GPU 0 — axis symmetry objects
CUDA_VISIBLE_DEVICES=0 python3 -m utils.data_render \
  --input-folder ../data/objects/curated_axis_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type axis_sym \
  --repo-views 114 \
  --sizes 224 448 1134 \
  --lightings darker flat brighter

# GPU 1 — plane symmetry objects
CUDA_VISIBLE_DEVICES=1 python3 -m utils.data_render \
  --input-folder ../data/objects/curated_plane_sym_obj \
  --output-folder ../data/renders \
  --symmetry-type plane_sym \
  --repo-views 114 \
  --sizes 224 448 1134 \
  --lightings darker flat brighter
```

#### `data_render.py` arguments

| Argument | Default | Description |
|---|---|---|
| `--input-folder` | *(required)* | Folder containing `.obj` files |
| `--output-folder` | *(required)* | Output base folder |
| `--symmetry-type` | *(required)* | `axis_sym` or `plane_sym` |
| `--repo-views` | `114` | Number of Fibonacci viewpoints |
| `--sizes` | `224` | One or more image sizes (e.g. `224 448 1134`) |
| `--lightings` | `flat` | One or more lighting modes: `flat`, `darker`, `brighter` |

#### Output structure

```
<output-folder>/<symmetry_type>/<object_id>/<image_size>/<illumination>/
```

Each configuration produces `<repo-views> × 4 rotations` images. For example, 114 views × 4 rotations × 3 sizes × 3 lightings = **5,472 images per object**.

---

### Running long jobs with tmux

When using multiple GPUs, it is recommended to run each job in a separate `tmux` session so the process survives terminal disconnections.

```bash
# 1. Create a new session
tmux new -s render_axis_sym
tmux new -s render_plane_sym

# 2. Launch the render command inside the session
CUDA_VISIBLE_DEVICES=0 python3 -m utils.data_render ...

# 3. Detach without killing the process
#    Press: Ctrl+B, then D

# 4. Reattach to a session
tmux attach -t render_axis_sym
tmux attach -t render_plane_sym

# 5. List active sessions
tmux ls
```