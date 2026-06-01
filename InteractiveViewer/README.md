# Interactive 3D Viewer

Interactive visualization tool for 3D meshes and symmetries (planar and axial), built with Open3D.

> **Platform note:** This tool requires a graphical display.

---

## Features

- **Interactive 3D rendering** — rotation, zoom, pan, coordinate frame
- **Planar symmetry visualization** — colored square + normal vector + reference point
- **Axial symmetry visualization** — colored line segment + directional cones + reference point
- **Auto-detection of symmetry file** — if `--symmetries` is omitted, looks for a `.txt` with the same name as the mesh

---

## Usage

### Basic usage (symmetry file auto-detected)

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\objects\my_object.obj"
```

### Specifying a symmetry file explicitly

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\objects\my_object.obj" `
  --symmetries ".\objects\my_object.txt"
```

### Adjusting the size of the visualized elements

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\objects\my_object.obj" `
  --size 0.3
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--mesh` | *(required)* | Path to the `.obj` mesh file |
| `--symmetries` | same name as mesh, `.txt` | Path to the `.txt` symmetry file |
| `--size` | `0.2` | Base size for all visualized symmetry elements |

---

## Examples — axis symmetry object

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\Examples\objects\axis_sym_obj.obj" `
  --symmetries ".\Examples\objects\axis_sym_obj.txt"
```

Expected output:

```
Loading mesh: Examples\objects\axis_sym_obj.obj
  1234 vertices, 2468 triangles
Loading symmetries: Examples\objects\axis_sym_obj.txt
  1 symmetry (0 plane, 1 axis)
  [ axis] vector=[0.0, 1.0, 0.0]  point=[0.0, 0.0, 0.0]  confidence=1.0000
```

## Examples — plane symmetry object

```powershell
.\venv\Scripts\python.exe .\InteractiveViewer\view_symmetries.py `
  --mesh ".\Examples\objects\plane_sym_obj.obj" `
  --symmetries ".\Examples\objects\plane_sym_obj.txt"
```

Expected output:

```
Loading mesh: Examples\objects\plane_sym_obj.obj
  987 vertices, 1974 triangles
Loading symmetries: Examples\objects\plane_sym_obj.txt
  1 symmetry (1 plane, 0 axis)
  [plane] vector=[0.0007, -0.0004, -1.0]  point=[0.0, 0.0, 0.0]  confidence=1.0000
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

| Element | Color | Type | Meaning |
|---|---|---|---|
| Mesh | Gray | Both | 3D object |
| Square | Red / Blue / Green / … | Plane | Symmetry plane surface |
| Line (short) | Yellow | Plane | Plane normal vector |
| Sphere (small) | Green | Plane | Reference point on plane |
| Line (long) | Red / Blue / Green / … | Axis | Symmetry axis segment |
| Cones | Same as line | Axis | Axis directionality |
| Sphere | Same as line | Axis | Reference point on axis |

Multiple symmetries are drawn in different colors (Red, Blue, Green, Orange, Magenta, Cyan).

---

## Input Format

The `.txt` file supports both symmetry types:

```
N
plane <nx> <ny> <nz> <px> <py> <pz> [<confidence>]
axis  <nx> <ny> <nz> <px> <py> <pz> [<confidence>]
```

- `N` — number of symmetry entries
- `plane` — planar symmetry: `(nx, ny, nz)` is the plane normal, `(px, py, pz)` is a point on the plane
- `axis` — axial symmetry: `(nx, ny, nz)` is the rotation axis direction, `(px, py, pz)` is a point on the axis
- `confidence` — optional score in `[0, 1]` (default: 1.0)

**Example — mixed file:**

```
2
axis  0.0 1.0 0.0  0.0 0.0 0.0
plane 1.0 0.0 0.0  0.0 0.0 0.0
```

---

## Core API

### `load_mesh(mesh_path) -> o3d.geometry.TriangleMesh`

Loads a `.obj` mesh, computes vertex normals, and applies uniform gray color.

### `load_symmetries(sym_path) -> list[dict]`

Parses a `.txt` file and returns a list of dicts with keys:
- `type` — `"plane"` or `"axis"`
- `vector` — normalized normal (plane) or direction (axis)
- `point` — reference point on the element
- `confidence` — confidence score

### `create_plane_geometry(normal, point, size, color) -> list`

Returns Open3D geometries: plane square mesh + normal line + reference sphere.

### `create_axis_geometry(direction, point, size, color) -> list`

Returns Open3D geometries: axis line segment + directional cones + reference sphere.
