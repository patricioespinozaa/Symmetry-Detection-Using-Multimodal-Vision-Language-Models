import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Batch render .obj files with export_fibonacci_views.py")
    parser.add_argument('--input-folder', type=str, default='data/objects/curated_axis_sym_obj')
    parser.add_argument('--output-folder', type=str, default='renders')
    parser.add_argument('--illumination', type=str, default='flat', choices=['flat', 'darker', 'brighter'])
    parser.add_argument('--repo-views', type=int, default=114)
    parser.add_argument('--image-size', type=int, default=224)
    args = parser.parse_args()

    obj_folder = Path(args.input_folder)
    obj_files = list(obj_folder.glob("*.obj"))

    print("\nBatch rendering configuration:")
    print(f"  Input folder:   {args.input_folder}")
    print(f"  Output folder:  {args.output_folder}")
    print(f"  Illumination:   {args.illumination}")
    print(f"  Repo views:     {args.repo_views}")
    print(f"  Image size:     {args.image_size}")
    print(f"  Total objects:  {len(obj_files)}\n")

    with tqdm(
        total=len(obj_files),
        desc="Waiting...",
        unit="obj",
        dynamic_ncols=True,         
        bar_format="{desc:<40} {bar:40} {n_fmt}/{total_fmt}  [{elapsed}<{remaining}]"
    ) as pbar:
        for obj_path in obj_files:
            pbar.set_description(f"Rendering {obj_path.stem:<30}")
            cmd = [
                sys.executable, "ImagesGenerator/export_fibonacci_views.py",
                "--mesh", str(obj_path),
                "--output", args.output_folder,
                "--repo-views", str(args.repo_views),
                "--image-size", str(args.image_size),
                "--illumination", args.illumination
            ]
            subprocess.run(cmd, check=True)
            pbar.update(1)

        pbar.set_description("Done!")

if __name__ == "__main__":
    main()