# Variante "_subset" de data_render.py para renders parciales (--subset-n-views).
# Copia deliberada: el original lo usan pipelines en curso y no se modifica.
# Diferencia: las vistas se eligen sobre la secuencia Fibonacci completa
# ("fibonacci_total" en metadata_all.json), no sobre len(metadata).
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

import numpy as np

EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_fibonacci_views_subset.py"


def subset_indices(total, subset_n_views):
    """Same union of indices as export_fibonacci_views.subset_indices (kept here to avoid importing torch)."""
    chosen = set()
    for n in subset_n_views:
        if n >= total:
            chosen.update(range(total))
        else:
            chosen.update(int(round(i)) for i in np.linspace(0, total - 1, n))
    return sorted(chosen)


def preview_execution(obj_files, sizes, lightings, repo_views, output_folder, symmetry_type,
                      views_per_object=None, assume_yes=False):
    """
    Show execution summary and ask for confirmation (skipped with assume_yes).
    """
    if views_per_object is None:
        views_per_object = repo_views

    total_objects = len(obj_files)
    total_configs = len(sizes) * len(lightings)
    images_per_object = views_per_object * total_configs
    total_images = total_objects * images_per_object

    print("\n========== EXECUTION PLAN ==========")
    print(f"Input folder:     {obj_files[0].parent if obj_files else 'N/A'}")
    print(f"Output folder:    {output_folder}")
    print(f"Symmetry type:    {symmetry_type}")
    print(f"Objects:          {total_objects}")
    print(f"Views per object: {views_per_object} (of {repo_views} Fibonacci)")
    print(f"Sizes:            {sizes}")
    print(f"Lightings:        {lightings}")
    print(f"Images/object:    {images_per_object}")
    print(f"TOTAL IMAGES:     {total_images}")
    print("====================================\n")

    if assume_yes:
        return
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
    parser.add_argument('--subset-n-views', type=int, nargs='+', default=None,
                        help='Render only the views the pointing runners pick for these view counts '
                             '(e.g. 6 14 26 -> 36 of 114); passed through to export_fibonacci_views.py')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip object/size/lighting combos whose manifest.json already exists')
    parser.add_argument('--yes', action='store_true', help='Do not ask for interactive confirmation')

    args = parser.parse_args()

    obj_folder = Path(args.input_folder)
    obj_files = sorted(obj_folder.glob("*.obj"))

    if len(obj_files) == 0:
        print("No .obj files found.")
        return

    views_per_object = args.repo_views
    if args.subset_n_views:
        views_per_object = len(subset_indices(args.repo_views, args.subset_n_views))

    # 🔍 PREVIEW + CONFIRMATION
    preview_execution(
        obj_files,
        args.sizes,
        args.lightings,
        args.repo_views,
        args.output_folder,
        args.symmetry_type,
        views_per_object=views_per_object,
        assume_yes=args.yes,
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
                    out_dir = Path(args.output_folder) / args.symmetry_type / object_id / str(size) / lighting
                    if args.skip_existing and (out_dir / "manifest.json").exists():
                        continue
                    # Pasar solo la raíz, la estructura la arma export_fibonacci_views.py
                    cmd = [
                        sys.executable, str(EXPORT_SCRIPT),
                        "--mesh", str(obj_path),
                        "--output", str(args.output_folder),
                        "--repo-views", str(args.repo_views),
                        "--image-size", str(size),
                        "--illumination", lighting,
                        "--symmetry-type", args.symmetry_type,
                        "--device", args.device,
                    ]
                    if args.subset_n_views:
                        cmd += ["--subset-n-views", *map(str, args.subset_n_views)]
                    subprocess.run(cmd, check=True)
            pbar.update(1)
        pbar.set_description("Done!")


if __name__ == "__main__":
    main()