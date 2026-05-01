import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRendererWithFragments,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.renderer import TexturesVertex

def get_min_camera_distance(r: float, f: float, aspect_ratio: float = 1) -> float:
    """
    Calculate the minimum camera distance to ensure the mesh fits in the field of view.
    Args:
        r: Mesh radius
        f: Field of view (degrees)
        aspect_ratio: Aspect ratio (default 1)
    Returns:
        Minimum camera distance
    """
    f_rad = np.deg2rad(f)
    fov_h = 2 * np.arctan(np.tan(f_rad/2) * np.sqrt(aspect_ratio**2 / (1 + aspect_ratio**2)))
    fov_v = 2 * np.arctan(np.tan(f_rad/2) * np.sqrt(1 / (1 + aspect_ratio**2)))
    fov_min = min(fov_h, fov_v)
    d = r / np.tan(fov_min / 2)
    return d

def rotate_points(points, angle, axis=None):
    """
    Rotate a set of 3D points around a given axis by a given angle.
    Args:
        points: Array of 3D points
        angle: Rotation angle (radians)
        axis: Rotation axis (default Y)
    Returns:
        Rotated points
    """
    if axis is None:
        axis = np.array([0, 1, 0], dtype=np.float64)
    else:
        axis = np.array(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    c_inv = 1 - c
    rotation_matrix = np.array([
        [c + x * x * c_inv, x * y * c_inv - z * s, x * z * c_inv + y * s],
        [y * x * c_inv + z * s, c + y * y * c_inv, y * z * c_inv - x * s],
        [z * x * c_inv - y * s, z * y * c_inv + x * s, c + z * z * c_inv],
    ])
    return np.array([rotation_matrix @ p for p in points], dtype=np.float64)

def cartesian_to_spherical(points):
    """
    Convert cartesian coordinates to spherical (azimuth, elevation, radius).
    Args:
        points: Array of 3D points
    Returns:
        List of dicts with azimuth, elevation, radius
    """
    angles_info = []
    for point in points:
        x, y, z = point
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.degrees(np.arctan2(z, x))
        elevation = np.degrees(np.arcsin(np.clip(y / (radius + 1e-8), -1.0, 1.0)))
        angles_info.append({
            'azimuth': int(azimuth) % 360,
            'elevation': int(elevation),
            'radius': radius
        })
    return angles_info

def sample_fibonacci_viewpoints(radius, num_points):
    """
    Generate camera positions on a sphere using the Fibonacci method.
    Args:
        radius: Sphere radius
        num_points: Number of viewpoints
    Returns:
        (positions, angle_info)
    """
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append([x * radius, y * radius, z * radius])
    # Small rotation to avoid pole singularity
    rotated_points = rotate_points(points, 0.001, axis=[1, 0, 0])
    angle_info = cartesian_to_spherical(np.array(rotated_points))
    return np.array(rotated_points), angle_info

def stabilize_eye_positions(eye_positions: np.ndarray) -> np.ndarray:
    """
    Avoid degenerate camera positions (e.g., exactly on the pole).
    """
    stabilized = np.array(eye_positions, dtype=np.float64, copy=True)
    norms = np.linalg.norm(stabilized, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    unit = stabilized / norms
    degenerate = np.abs(unit[:, 1]) > 0.999
    if np.any(degenerate):
        stabilized[degenerate, 0] += 1e-4
        stabilized[degenerate, 2] += 1e-4
    return stabilized

def load_mesh(mesh_path: str, device: torch.device):
    """
    Load a mesh as a PyTorch3D object, set uniform gray color (or other illumination value).
    """
    mesh = load_objs_as_meshes([mesh_path], load_textures=False, device=device)
    # The gray value will be set outside this function for flexibility
    return mesh

def mesh_radius(mesh):
    """
    Estimate mesh radius using sampled points (as in EnhancedBackProjection).
    """
    return sample_points_from_meshes(mesh, 1000).norm(dim=-1).max().item()

def save_image_tensor(image_tensor: torch.Tensor, output_path: Path):
    """
    Save a single image tensor as a PNG file.
    """
    image_rgb = image_tensor[..., :3].detach().cpu().numpy()
    image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(image_rgb).save(output_path)

def save_image_tensor_with_rotations(image_tensor: torch.Tensor, output_path_base: Path, meta_base: dict, records: list, idx: int, az: int, el: int):
    """
    Save the image tensor with 4 rotations (0, 90, 180, 270 degrees) and append metadata for each.
    Args:
        image_tensor: Rendered image tensor
        output_path_base: Output directory
        meta_base: Base metadata dict
        records: List to append metadata
        idx: Viewpoint index
        az: Azimuth
        el: Elevation
    """
    image_rgb = image_tensor[..., :3].detach().cpu().numpy()
    image_rgb = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image_rgb)
    for rot_idx, rot_deg in enumerate([0, 90, 180, 270]):
        img_rot = pil_img.rotate(-rot_deg, expand=True)  # PIL rotates counterclockwise
        filename = f"IND_{idx:02d}_AZ_{az:03d}_EL_{el:+03d}_ROT_{rot_deg:03d}.png"
        img_path = output_path_base / filename
        img_rot.save(img_path)
        meta = meta_base.copy()
        meta["filename"] = filename
        meta["rotation_deg"] = rot_deg
        meta["rotation_index"] = rot_idx
        records.append(meta)

def render_fibonacci(mesh, eye_positions, output_dir: Path, fov: float, image_size: int, device: torch.device, angle_info=None):
    """
    Render the mesh from all Fibonacci viewpoints and save 4 rotated images per view.
    Args:
        mesh: PyTorch3D mesh
        eye_positions: Camera positions
        output_dir: Output directory
        fov: Field of view
        image_size: Output image size
        device: Device
        angle_info: List of angle dicts (optional)
    Returns:
        List of metadata records for all images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eye_positions = stabilize_eye_positions(eye_positions)
    eye_tensor = torch.tensor(eye_positions, dtype=torch.float32, device=device)
    R, T = look_at_view_transform(eye=eye_tensor, device=device)
    raster_settings = RasterizationSettings(
        image_size=(image_size, image_size),
        faces_per_pixel=5,
        bin_size=0,
        cull_backfaces=True,
    )
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardFlatShader(device=device),
    )
    cameras = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=device)
    eye_positions_normalized = eye_positions / np.linalg.norm(eye_positions, axis=1, keepdims=True)
    lights = DirectionalLights(
        direction=torch.tensor(eye_positions_normalized, dtype=torch.float32, device=device), 
        device=device
    )
    with torch.no_grad():
        images_list = []
        for i in range(0, len(eye_positions), 4):
            end_idx = min(i + 4, len(eye_positions))
            batch_size = end_idx - i
            R_batch = R[i:end_idx]
            T_batch = T[i:end_idx]
            cameras_batch = FoVPerspectiveCameras(R=R_batch, T=T_batch, fov=fov, device=device)
            lights_batch = DirectionalLights(
                direction=torch.tensor(eye_positions_normalized[i:end_idx], dtype=torch.float32, device=device),
                device=device
            )
            mesh_batch_ext = mesh.extend(batch_size)
            images_batch, _ = renderer(mesh_batch_ext, cameras=cameras_batch, lights=lights_batch)
            images_batch = images_batch[..., :3]
            images_list.append(images_batch)
        images = torch.cat(images_list, dim=0)
    if angle_info is None:
        angle_info = cartesian_to_spherical(eye_positions)
    records = []
    for i in range(len(eye_positions)):
        az = angle_info[i]['azimuth']
        el = angle_info[i]['elevation']
        meta_base = {
            "index": i,
            "azimuth": az,
            "elevation": el,
            "angle_info": angle_info[i],
            "eye": [float(x) for x in eye_positions[i].tolist()],
            "R": R[i].detach().cpu().tolist(),
            "T": T[i].detach().cpu().tolist(),
        }
        # Save 4 rotated images and metadata per viewpoint
        save_image_tensor_with_rotations(images[i], output_dir, meta_base, records, i, az, el)
    return records

def write_metadata(object_output_dir: Path, all_records):
    """
    Write all metadata records to a JSON file.
    """
    json_path = object_output_dir / "metadata_all.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

def main():
    """
    Main entry point: parse arguments, render views, save images and metadata.
    """
    parser = argparse.ArgumentParser(description="Export mesh views using Fibonacci method with 4 2D rotations.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to the .obj file")
    parser.add_argument("--output", type=str, default="output", help="Output base folder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device: cuda:0 or cpu")
    parser.add_argument("--fov", type=float, default=60.0, help="Field of view in degrees")
    parser.add_argument("--image-size", type=int, default=512, help="Output image size (square)")
    parser.add_argument("--repo-views", type=int, default=26, help="Number of Fibonacci views")
    parser.add_argument("--camera-distance-factor", type=float, default=1.2, help="Camera distance multiplier (default 1.2)")
    parser.add_argument("--illumination", type=str, default="flat", choices=["flat", "darker", "brighter"], help="Illumination mode: flat (default), darker, brighter")
    parser.add_argument("--symmetry-type", type=str, required=True, choices=["axis_sym", "plane_sym"], help="Type of symmetry to detect")
    args = parser.parse_args()
    start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    mesh_path = Path(args.mesh)
    object_name = mesh_path.stem
    output_root = Path(args.output)
    # Estructura: <output>/<symmetry_type>/<object_id>/<size>/<lighting>/
    object_output_dir = (
        output_root
        / args.symmetry_type
        / object_name
        / str(args.image_size)
        / args.illumination
    )
    object_output_dir.mkdir(parents=True, exist_ok=True)

    # Illumination value map
    illumination_map = {
        "flat": 0.7,
        "darker": 0.3,
        "brighter": 0.95,
    }
    illumination_value = illumination_map[args.illumination]

    # Load mesh
    mesh = load_mesh(str(mesh_path), device)
    # Set the mesh to the selected gray value
    mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_packed()[None]) * illumination_value)

    # Estimate camera distance
    radius = mesh_radius(mesh)
    camera_distance = get_min_camera_distance(radius, args.fov) * args.camera_distance_factor
    # Generate Fibonacci viewpoints
    fib_views, fib_angles = sample_fibonacci_viewpoints(camera_distance, args.repo_views)
    # Render and save images with 4 rotations per view
    all_records = render_fibonacci(
        mesh=mesh,
        eye_positions=fib_views,
        output_dir=object_output_dir,
        fov=args.fov,
        image_size=args.image_size,
        device=device,
        angle_info=fib_angles,
    )
    # Save metadata
    write_metadata(object_output_dir, all_records)
    elapsed_time = time.time() - start_time
    manifest = {
        "mesh": str(mesh_path),
        "device": str(device),
        "fov": args.fov,
        "image_size": args.image_size,
        "camera_distance": camera_distance,
        "total_images": len(all_records),
        "processing_time_seconds": round(elapsed_time, 2),
        "processing_time_human": f"{int(elapsed_time // 60)}m {elapsed_time % 60:.2f}s",
        "illumination": args.illumination,
        "illumination_value": illumination_value,
    }
    with open(object_output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    # print("Export completed")
    # print(f"Object: {object_name}")
    # print(f"Output: {object_output_dir}")
    # print(f"Total images: {len(all_records)}")
    # print(f"Illumination: {args.illumination} (gray value {illumination_value})")
    # print(f"\nProcessing time: {int(elapsed_time // 60)}m {elapsed_time % 60:.2f}s ({elapsed_time:.2f}s total)")

if __name__ == "__main__":
    main()
