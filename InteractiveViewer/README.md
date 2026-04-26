# Interactive 3D Viewer

This folder contains an interactive visualization tool with support for 3D meshes and symmetry planes, built using Open3D.

---

## Features

1. **Interactive 3D rendering** with mouse controls

- Rotation, zoom, and panning (translation)
- Coordinate axis visualization

2. **Symmetry visualization**

- Load symmetry planes from `.txt` files
- Display each plane with a distinct color
- Visualize plane normals and reference points

3. **Mesh support**

- Load `.obj` files
- Automatic normal computation
- Support for complex meshes

---

## Usage

### Basic usage

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\InteractiveViewer\example\example.obj"
```

### Using a specific symmetry file

If the `.txt` file has a different name than the `.obj`:

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\InteractiveViewer\example\example.obj" `
  --symmetries ".\InteractiveViewer\example\example.txt"
```

### Adjusting the size of the visualized planes

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\InteractiveViewer\example\example.obj" `
  --size 0.3
```

---

## Interactive Controls

| Control | Action |
|---|---|
| **Left click + drag** | Rotate object |
| **Mouse scroll** | Zoom in/out |
| **Right click + drag** | Pan (translation) |
| **R** | Reset view |
| **Q** | Close viewer |

---

## Visual Interpretation

| Element | Color | Meaning |
|---|---|---|
| Mesh | Gray | Main 3D object |
| Plane | Red / Green / Blue / etc. | Symmetry plane |
| Line | Yellow | Plane normal |
| Sphere | Green | Reference point |

---

## Input Format

The `.txt` file must follow this structure:

```
N
plane nx ny nz px py pz [confidence]
plane ...
```

**Example** — single XY plane with normal along −Z:

```
1
plane 0.0007492030056283876 -0.0004077757667688796 -0.9999996405271714 0.0 0.0 0.0
```

---

## Project Structure

```
InteractiveViewer/
├── view_symmetries.py     # Main script
├── /example/example.obj   # Object script
├── /example/example.txt   # Symmetries
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## Core API / Functions

### `load_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh`

Loads a `.obj` mesh and computes vertex normals.

---

### `load_symmetries(sym_path: str) -> list`

Reads a `.txt` file containing symmetry planes and returns a list of dictionaries with:

- `normal` — normal vector `(nx, ny, nz)`
- `point` — point on the plane `(px, py, pz)`
- `confidence` — confidence score in the range `[0, 1]`

---

### `create_plane_geometry(normal, point, size, color) -> list`

Generates Open3D geometry objects to visualize a plane:

- Square mesh representing the plane surface
- Line segment representing the normal vector
- Sphere marking the reference point
