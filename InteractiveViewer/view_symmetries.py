import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path


def load_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh:
    """Load a .obj mesh using Open3D"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    return mesh


def load_symmetries(sym_path: str) -> list:
    """
    Load symmetry planes from a .txt file
    
    Expected format:
    N
    plane <nx> <ny> <nz> <px> <py> <pz> [<confidence>]
    plane ...
    """
    symmetries = []
    try:
        with open(sym_path, 'r') as f:
            num_planes = int(f.readline().strip())
            for _ in range(num_planes):
                line = f.readline().strip()
                if not line:
                    continue
                parts = line.split()
                if parts[0].lower() == 'plane':
                    try:
                        normal = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        point = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
                        confidence = float(parts[7]) if len(parts) > 7 else 1.0
                        # Normalize normal
                        normal = normal / (np.linalg.norm(normal) + 1e-8)
                        symmetries.append({
                            'normal': normal,
                            'point': point,
                            'confidence': confidence
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line} ({e})")
    except FileNotFoundError:
        print(f"Symmetry file not found: {sym_path}")
    
    return symmetries


def create_plane_geometry(normal: np.ndarray, point: np.ndarray, size: float = 0.2, color: list = None) -> list:
    """
    Create geometry to visualize a symmetry plane
    
    Returns a list of geometries: [plane, normal_line]
    """
    if color is None:
        color = [1.0, 0.0, 0.0]  # Red
    
    geometries = []
    # Create two vectors orthogonal to the normal
    if abs(normal[0]) < 0.9:
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([0.0, 1.0, 0.0])
    v1 = v1 - np.dot(v1, normal) * normal
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = np.cross(normal, v1)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    # Create a square in the plane
    corners = np.array([
        point - v1 * size - v2 * size,
        point + v1 * size - v2 * size,
        point + v1 * size + v2 * size,
        point - v1 * size + v2 * size,
    ])
    # Create the plane mesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
    plane_mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [0, 2, 3]
    ])
    plane_mesh.paint_uniform_color(color)
    plane_mesh.compute_vertex_normals()
    # Create a line for the normal
    normal_end = point + normal * size * 0.5
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([point, normal_end])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
    # Create a sphere at the point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.1)
    sphere.translate(point)
    sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    geometries.append(plane_mesh)
    geometries.append(normal_line)
    geometries.append(sphere)
    return geometries


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D visualizer for objects and symmetry planes")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the .obj file")
    parser.add_argument("--symmetries", type=str, default=None, help="Path to the .txt file with symmetries (optional)")
    parser.add_argument("--size", type=float, default=0.2, help="Size of the visualized planes")
    args = parser.parse_args()
    mesh_path = Path(args.mesh)
    # If no symmetries file is specified, try to use the same name as the mesh
    if args.symmetries is None:
        sym_path = mesh_path.with_suffix('.txt')
    else:
        sym_path = Path(args.symmetries)
    print(f"Loading mesh: {mesh_path}")
    if not mesh_path.exists():
        print(f"Error: mesh file not found: {mesh_path}")
        return
    # Load mesh
    mesh = load_mesh(str(mesh_path))
    print(f"Mesh loaded: {len(np.asarray(mesh.vertices))} vertices, {len(np.asarray(mesh.triangles))} triangles")
    # Load symmetries
    geometries = [mesh]
    if sym_path.exists():
        print(f"Loading symmetries: {sym_path}")
        symmetries = load_symmetries(str(sym_path))
        print(f"Symmetries loaded: {len(symmetries)} plane(s)")
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]
        for i, sym in enumerate(symmetries):
            color = colors[i % len(colors)]
            plane_geoms = create_plane_geometry(sym['normal'], sym['point'], args.size, color)
            geometries.extend(plane_geoms)
            print(f"  Plane {i+1}: normal={sym['normal']}, point={sym['point']}, confidence={sym['confidence']:.4f}")
    else:
        print(f"Symmetries file not found: {sym_path}")
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Viewer: {mesh_path.name}", width=1200, height=800)
    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    # Configure view
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True
    vis.get_view_control().set_zoom(0.8)
    # Show
    vis.run()
    vis.destroy_window()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
