import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import sys


def preview_execution(obj_files, sizes, lightings, repo_views, output_folder, symmetry_type):
    """
    Show execution summary and ask for confirmation.
    """

    total_objects = len(obj_files)
    total_configs = len(sizes) * len(lightings)
    images_per_object = repo_views * total_configs  
    total_images = total_objects * images_per_object

    print("\n========== EXECUTION PLAN ==========")
    print(f"Input folder:     {obj_files[0].parent if obj_files else 'N/A'}")
    print(f"Output folder:    {output_folder}")
    print(f"Symmetry type:    {symmetry_type}")
    print(f"Objects:          {total_objects}")
    print(f"Views per object: {repo_views}")
    print(f"Sizes:            {sizes}")
    print(f"Lightings:        {lightings}")
    print(f"Images/object:    {images_per_object}")
    print(f"TOTAL IMAGES:     {total_images}")
    print("====================================\n")

    confirm = input("Type 'OK' to start: ")
    if confirm.strip() != "OK":
        print("Execution cancelled.")
        exit()


def main():
    parser = argparse.ArgumentParser(description="Batch render .obj files with export_fibonacci_views.py")

    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--symmetry-type', type=str, required=True, choices=['axis_sym', 'plane_sym'])

    parser.add_argument('--repo-views', type=int, default=114)
    parser.add_argument('--sizes', type=int, nargs='+', default=[224])
    parser.add_argument('--lightings', nargs='+', default=['flat'], choices=['flat', 'darker', 'brighter'])

    args = parser.parse_args()

    obj_folder = Path(args.input_folder)
    obj_files = sorted(obj_folder.glob("*.obj"))

    if len(obj_files) == 0:
        print("No .obj files found.")
        return

    # 🔍 PREVIEW + CONFIRMATION
    preview_execution(
        obj_files,
        args.sizes,
        args.lightings,
        args.repo_views,
        args.output_folder,
        args.symmetry_type
    )

    # =========================
    # PROCESS LOOP
    # =========================
    with tqdm(
        total=len(obj_files),
        desc="Starting...",
        unit="obj",
        dynamic_ncols=True,
        bar_format="{desc:<45} {bar:40} {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ) as pbar:

        for obj_path in obj_files:
            object_id = obj_path.stem
            for size in args.sizes:
                for lighting in args.lightings:
                    pbar.set_description(f"{object_id} | {size}px | {lighting}")
                    # Pasar solo la raíz, la estructura la arma export_fibonacci_views.py
                    cmd = [
                        sys.executable, "ImagesGenerator/export_fibonacci_views.py",
                        "--mesh", str(obj_path),
                        "--output", str(args.output_folder),
                        "--repo-views", str(args.repo_views),
                        "--image-size", str(size),
                        "--illumination", lighting,
                        "--symmetry-type", args.symmetry_type
                    ]
                    subprocess.run(cmd, check=True)
            pbar.update(1)
        pbar.set_description("Done!")


if __name__ == "__main__":
    main()