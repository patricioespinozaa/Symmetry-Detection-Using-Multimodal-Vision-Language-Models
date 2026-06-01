import argparse
import numpy as np
import open3d as o3d
from pathlib import Path


def load_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a .obj mesh using Open3D.
    Args:
        mesh_path: Path to the .obj file
    Returns:
        An Open3D TriangleMesh object with vertex normals computed and colored gray.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    return mesh


def load_symmetries(sym_path: str) -> list:
    """
    Load symmetry entries from a .txt file.

    Supported formats:
        N
        plane <nx> <ny> <nz> <px> <py> <pz> [<confidence>]
        axis  <nx> <ny> <nz> <px> <py> <pz> [<confidence>]
    
    Args:
        sym_path: Path to the symmetry .txt file
    Returns:
        A list of symmetry dictionaries with keys:
            'type': 'plane' or 'axis'
            'vector': np.array of shape (3,) - normal for plane, direction for axis
            'point': np.array of shape (3,) - reference point on the plane or axis
            'confidence': float (optional, default=1.0)
    """
    symmetries = []
    try:
        with open(sym_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        n = int(lines[0])
        for line in lines[1:n + 1]:
            parts = line.split()
            kind = parts[0].lower()
            if kind not in ('plane', 'axis'):
                print(f"  [warn] Unknown symmetry type '{parts[0]}', skipping: {line}")
                continue
            try:
                vec   = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                point = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                confidence = float(parts[7]) if len(parts) > 7 else 1.0
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                symmetries.append({
                    'type':       kind,
                    'vector':     vec,    # normal for plane, direction for axis
                    'point':      point,
                    'confidence': confidence,
                })
            except (ValueError, IndexError) as e:
                print(f"  [warn] Error parsing line: {line} ({e})")
    except FileNotFoundError:
        print(f"Symmetry file not found: {sym_path}")
    return symmetries


def create_plane_geometry(
    normal: np.ndarray,
    point: np.ndarray,
    size: float = 0.2,
    color: list = None,
) -> list:
    """
    Create Open3D geometries for a symmetry plane:
      - Square mesh (the plane surface)
      - Line segment (the normal vector)
      - Sphere (the reference point)

    Args:
        normal: np.array of shape (3,) - normal vector of the plane
        point: np.array of shape (3,) - reference point on the plane
        size: float - base size for the plane and normal visualization
        color: list of 3 floats - RGB color for the plane and normal (default: [1.0, 0.0, 0.0])
    Returns:
        A list of Open3D geometries representing the plane, normal, and reference point.
    """
    if color is None:
        color = [1.0, 0.0, 0.0]

    # Two vectors orthogonal to the normal
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v1 = ref - np.dot(ref, normal) * normal
    v1 /= np.linalg.norm(v1) + 1e-8
    v2 = np.cross(normal, v1)
    v2 /= np.linalg.norm(v2) + 1e-8

    corners = np.array([
        point - v1 * size - v2 * size,
        point + v1 * size - v2 * size,
        point + v1 * size + v2 * size,
        point - v1 * size + v2 * size,
    ])
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices  = o3d.utility.Vector3dVector(corners)
    plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane_mesh.paint_uniform_color(color)
    plane_mesh.compute_vertex_normals()

    normal_end = point + normal * size * 0.5
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([point, normal_end])
    normal_line.lines  = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.1)
    sphere.translate(point)
    sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    return [plane_mesh, normal_line, sphere]


def create_axis_geometry(
    direction: np.ndarray,
    point: np.ndarray,
    size: float = 0.2,
    color: list = None,
) -> list:
    """
    Create Open3D geometries for a symmetry axis:
      - Line segment centered at point, extending ±size along direction
      - Sphere at the reference point
      - Small arrows (cones) at both ends to indicate direction

    Args:
        direction: np.array of shape (3,) - direction vector of the axis
        point: np.array of shape (3,) - reference point on the axis
        size: float - base size for the axis visualization
        color: list of 3 floats - RGB color for the axis (default: [1.0, 0.0, 0.0])
    Returns:
        A list of Open3D geometries representing the axis line, reference point, and direction
    """
    if color is None:
        color = [1.0, 0.0, 0.0]

    half = size * 3.0  # make the axis segment longer than the plane
    p0 = point - direction * half
    p1 = point + direction * half

    axis_line = o3d.geometry.LineSet()
    axis_line.points = o3d.utility.Vector3dVector([p0, p1])
    axis_line.lines  = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.paint_uniform_color(color)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.12)
    sphere.translate(point)
    sphere.paint_uniform_color(color)

    # Cone at each end to show directionality
    cone_h = size * 0.4
    cone_r = size * 0.12
    geoms   = [axis_line, sphere]
    for tip, sign in [(p1, 1), (p0, -1)]:
        cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_r, height=cone_h)
        # Default cone points along +Z; rotate to align with direction * sign
        d = direction * sign
        z = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z, d)
        cross_norm = np.linalg.norm(cross)
        if cross_norm > 1e-6:
            angle = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                cross / cross_norm * angle
            )
            cone.rotate(R, center=(0, 0, 0))
        cone.translate(tip)
        cone.paint_uniform_color(color)
        geoms.append(cone)

    return geoms


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D visualizer for objects and symmetries (plane + axis)."
    )
    parser.add_argument("--mesh",       type=str, required=True,
                        help="Path to the .obj file")
    parser.add_argument("--symmetries", type=str, default=None,
                        help="Path to the .txt symmetry file (default: same name as mesh)")
    parser.add_argument("--size",       type=float, default=0.2,
                        help="Base size for visualized symmetry elements")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    sym_path  = Path(args.symmetries) if args.symmetries else mesh_path.with_suffix('.txt')

    print(f"Loading mesh: {mesh_path}")
    if not mesh_path.exists():
        print(f"Error: mesh file not found: {mesh_path}")
        return
    mesh = load_mesh(str(mesh_path))
    print(f"  {len(np.asarray(mesh.vertices))} vertices, "
          f"{len(np.asarray(mesh.triangles))} triangles")

    COLORS = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.5, 1.0],  # Blue
        [0.0, 0.85, 0.0], # Green
        [1.0, 0.5, 0.0],  # Orange
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ]

    geometries = [mesh]
    if sym_path.exists():
        print(f"Loading symmetries: {sym_path}")
        symmetries = load_symmetries(str(sym_path))
        n_plane = sum(1 for s in symmetries if s['type'] == 'plane')
        n_axis  = sum(1 for s in symmetries if s['type'] == 'axis')
        print(f"  {len(symmetries)} symmetr{'y' if len(symmetries)==1 else 'ies'} "
              f"({n_plane} plane, {n_axis} axis)")
        for i, sym in enumerate(symmetries):
            color = COLORS[i % len(COLORS)]
            if sym['type'] == 'plane':
                geoms = create_plane_geometry(sym['vector'], sym['point'], args.size, color)
            else:
                geoms = create_axis_geometry(sym['vector'], sym['point'], args.size, color)
            geometries.extend(geoms)
            print(f"  [{sym['type']:>5}] vector={np.round(sym['vector'], 4).tolist()}  "
                  f"point={np.round(sym['point'], 4).tolist()}  "
                  f"confidence={sym['confidence']:.4f}")
    else:
        print(f"No symmetry file found at: {sym_path}  (mesh only)")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Viewer: {mesh_path.name}", width=1200, height=800)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.get_render_option().point_size          = 1.0
    vis.get_render_option().show_coordinate_frame = True
    vis.get_view_control().set_zoom(0.8)
    vis.run()
    vis.destroy_window()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
