# estimate_symmetry_variants.py
#
# Sibling of Mapping/estimate_symmetry_no_mesh.py that runs one of the
# triangulation ablations (EXP-A..F, triangulation_variants/) instead of the
# baseline estimator, over a SINGLE already-run --experiment-id (a prior
# Molmo2 prompt execution). Reads the same molmo_multiview_<EXP>.json files,
# writes predicted_symmetry_<EXP>_<variant>.json in the exact schema
# Mapping/evaluate.py already understands ("triangulation" /
# "triangulation_multiplane" method keys) -- no changes needed downstream.
#
# For running this over MANY prompt executions at once, see run_batch.py,
# which drives this module's process_object() directly from a YAML config
# instead of shelling out once per experiment_id.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common.datasets import OBJECTS_SUBDIR, load_mesh
from pipeline_common.naming import exp_filename

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Mapping"))
from evaluate import N_SAMPLES_DEFAULT, SDE_REF_SEED_DEFAULT, sample_surface_points  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VariantConfig, build_output_experiment_id  # noqa: E402
from triangulation_variants import get_variant  # noqa: E402
from view_filtering import filter_points_on_object  # noqa: E402

MOLMO_JSON = "molmo_multiview.json"
MANIFEST_FILE = "manifest.json"
OUTPUT_FILE = "predicted_symmetry.json"
METHOD_KEY = "triangulation"

DEFAULT_FOV = 60.0


def _build_mesh_geometry(objects_root: Path, symmetry_type: str, object_id: str) -> dict | None:
    """ Load a mesh and its area-weighted surface sample, for variants that need real geometry (EXP-F).

    Does NOT depend on any variant hyperparameter (sde_gate) -- safe to
    cache by object_id alone and reuse across multiple VariantConfig runs.

    Args:
        * objects_root: root folder holding curated_{axis,plane}_sym_obj/
        * symmetry_type: "axis_sym" or "plane_sym"
        * object_id: object identifier

    Returns:
        * dict | None: {"vertices", "faces", "surface_sample"}, or None if
          the .obj file is missing

    """
    obj_path = Path(objects_root) / OBJECTS_SUBDIR[symmetry_type] / f"{object_id}.obj"
    if not obj_path.exists():
        return None
    mesh = load_mesh(obj_path)
    vertices, faces, sample = sample_surface_points(mesh, N_SAMPLES_DEFAULT, SDE_REF_SEED_DEFAULT)
    return {"vertices": vertices, "faces": faces, "surface_sample": sample}


def process_object(
    object_dir: Path,
    symmetry_type: str,
    sizes: list[int],
    lightings: list[str],
    source_experiment_id: str,
    variant_config: VariantConfig,
    objects_root: Path | None = None,
    overwrite: bool = False,
    mesh_ctx_cache: dict | None = None,
) -> None:
    """ Run one triangulation variant, for one object, across every (size, lighting, n_views) config.

    Args:
        * object_dir: <renders_root>/<symmetry_type>/<object_id>
        * symmetry_type: "axis_sym" or "plane_sym"
        * sizes: render sizes to pool across (same convention as
          Mapping/estimate_symmetry_no_mesh.py)
        * lightings: illuminations to pool across
        * source_experiment_id: the prior --experiment-id prompt run to read
          points from (its molmo_multiview_<ID>.json must already exist)
        * variant_config: which EXP-A..F variant + hyperparameters to run
        * objects_root: needed only when variant_config.variant == "expF"
          (real-mesh SDE gate)
        * overwrite: overwrite an existing output file instead of skipping
        * mesh_ctx_cache: optional dict shared by the caller across objects,
          so each object's mesh/surface-sample is only built once even if
          this function runs for several (size, lighting) configs of it

    """
    variant = get_variant(variant_config.variant)
    if symmetry_type not in variant.SYMMETRY_TYPES:
        raise ValueError(f"variant {variant_config.variant!r} does not support {symmetry_type!r} "
                          f"(supports {variant.SYMMETRY_TYPES})")

    output_experiment_id = build_output_experiment_id(source_experiment_id, variant_config)
    input_file = exp_filename(MOLMO_JSON, source_experiment_id)
    output_file = exp_filename(OUTPUT_FILE, output_experiment_id)
    output_path = object_dir / output_file
    if output_path.exists() and not overwrite:
        return

    mesh_ctx = None
    if variant.NEEDS_MESH:
        if objects_root is None:
            raise ValueError(f"variant {variant_config.variant!r} needs --objects-root (real-mesh gate)")
        object_id = object_dir.name
        if mesh_ctx_cache is not None and object_id in mesh_ctx_cache:
            geometry = mesh_ctx_cache[object_id]
        else:
            geometry = _build_mesh_geometry(objects_root, symmetry_type, object_id)
            if mesh_ctx_cache is not None:
                mesh_ctx_cache[object_id] = geometry
        if geometry is None:
            return   # no .obj for this object -- can't run a mesh-gated variant
        mesh_ctx = {**geometry, "sde_gate": variant_config.sde_gate}

    configs: dict[str, list[tuple]] = defaultdict(list)
    image_cache: dict = {}

    for size in sizes:
        for lighting in lightings:
            render_dir = object_dir / str(size) / lighting
            molmo_path = render_dir / input_file
            if not molmo_path.exists():
                continue

            manifest_path = render_dir / MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                image_size = manifest.get("image_size", size)
                fov_deg = manifest.get("fov", DEFAULT_FOV)
            else:
                image_size, fov_deg = size, DEFAULT_FOV

            with open(molmo_path, encoding="utf-8") as f:
                molmo_data = json.load(f)

            for n_views_key, group in molmo_data.items():
                images_sent = group.get("images_sent", [])
                points_by_image = group.get("points_by_image", {})
                if variant_config.filter_policy != "none" and points_by_image:
                    points_by_image = filter_points_on_object(
                        points_by_image, images_sent, render_dir,
                        policy=variant_config.filter_policy,
                        min_points=variant_config.min_points_on_object,
                        image_cache=image_cache,
                    )
                if not points_by_image:
                    continue
                configs[n_views_key].append((points_by_image, images_sent, fov_deg, image_size))

    if not configs:
        return

    n_views_predictions: dict = {}

    for n_views_key, cfg_list in configs.items():
        try:
            if symmetry_type == "axis_sym":
                # Best-of-cfg_list, not cross-config pooling: unlike
                # Mapping/estimate_symmetry_no_mesh.py (which pools raw
                # centers/normals across every matching size/lighting config
                # for axis specifically), every variant here consumes one
                # self-contained points_by_image/images_sent pair, so pooling
                # would require reindexing across configs with overlapping
                # view indices. In practice sizes/lightings sweeps of a
                # single prompt run are rare (typically one config), and this
                # matches the baseline's OWN plane-side behavior already
                # ("known simplification", see estimate_plane_no_mesh caller).
                best_pred, best_n = None, -1
                for points_by_image, images_sent, fov_deg, image_size in cfg_list:
                    try:
                        pred = variant.estimate_axis(points_by_image, images_sent, fov_deg, image_size)
                    except ValueError:
                        continue
                    if pred["n_views_used"] > best_n:
                        best_pred, best_n = pred, pred["n_views_used"]
                if best_pred is None:
                    continue

                n_views_predictions[n_views_key] = {
                    "n_points_raw": best_n, "n_points_fit": best_n,
                    METHOD_KEY: {
                        "direction": best_pred["direction"], "origin": best_pred["origin"],
                        "n_points": best_n, "n_views_used": best_n,
                        "n_inliers": None, "sde": None, "accepted": None,
                    },
                }

            else:
                best_planes: list[dict] | None = None
                for points_by_image, images_sent, fov_deg, image_size in cfg_list:
                    try:
                        planes = variant.detect_planes(
                            points_by_image, images_sent, fov_deg, image_size,
                            edge_on_thresh=variant_config.edge_on_thresh,
                            max_planes=variant_config.max_planes,
                            dup_angle_thresh_deg=variant_config.dup_angle_thresh,
                            mesh_ctx=mesh_ctx,
                        )
                    except ValueError:
                        continue
                    if not planes:
                        continue
                    if best_planes is None or planes[0]["n_views_used"] > best_planes[0]["n_views_used"]:
                        best_planes = planes

                if not best_planes:
                    continue

                n_points_total = sum(p["n_views_used"] for p in best_planes)
                if variant_config.max_planes == 1:
                    p = best_planes[0]
                    n_views_predictions[n_views_key] = {
                        "n_points_raw": n_points_total, "n_points_fit": n_points_total,
                        METHOD_KEY: {
                            "normal": p["normal"], "origin": p["origin"],
                            "n_points": p["n_views_used"], "n_views_used": p["n_views_used"],
                            "n_inliers": None, "sde": None, "accepted": None,
                        },
                    }
                else:
                    n_views_predictions[n_views_key] = {
                        "n_points_raw": n_points_total, "n_points_fit": n_points_total,
                        f"{METHOD_KEY}_multiplane": {
                            "planes": [
                                {"normal": p["normal"], "origin": p["origin"],
                                 "n_views_used": p["n_views_used"], "n_candidates": p.get("n_candidates"),
                                 "good_views": p.get("good_views")}
                                for p in best_planes
                            ],
                        },
                    }
        except Exception:
            continue

    if not n_views_predictions:
        return

    output = {
        "object_id": object_dir.name,
        "symmetry_type": symmetry_type,
        "point_mode": METHOD_KEY,
        "source_experiment_id": source_experiment_id,
        "variant": variant_config.variant,
        "n_views_predictions": n_views_predictions,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def parse_args() -> argparse.Namespace:
    """ Parse CLI arguments for a single-variant, single-experiment_id run. """
    p = argparse.ArgumentParser(
        description="Run one triangulation ablation (EXP-A..F) over one already-run prompt "
                    "experiment_id, at full-dataset scale.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--objects-root", default=None,
                   help="Required only for --variant expF (needs the real mesh for its SDE gate).")
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--experiment-id", required=True,
                   help="Source prompt run to read points from (its molmo_multiview_<ID>.json "
                        "must already exist for every object).")
    p.add_argument("--variant", required=True, choices=["expA", "expB", "expC", "expD", "expE", "expF"])
    p.add_argument("--sizes", type=int, nargs="+", default=[224, 448, 1136])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat", "brighter", "darker"],
                   choices=["flat", "darker", "brighter"])
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--num-gpus", type=int, default=1)

    p.add_argument("--filter-policy", default="none", choices=["none", "per_point", "any", "both"],
                   help="View-discard policy for off-object points (view_filtering.py).")
    p.add_argument("--min-points-on-object", type=int, default=2)
    p.add_argument("--max-planes", type=int, default=1, help="plane_sym only.")
    p.add_argument("--edge-on-thresh", type=float, default=0.5, help="plane_sym only.")
    p.add_argument("--dup-angle-thresh", type=float, default=15.0, help="plane_sym only.")
    p.add_argument("--sde-gate", type=float, default=0.02, help="plane_sym + --variant expF only.")
    return p.parse_args()


def main() -> None:
    """ Standalone CLI entrypoint -- for batch use across many experiment_ids/variants, see run_batch.py. """
    args = parse_args()

    variant_config = VariantConfig(
        variant=args.variant, filter_policy=args.filter_policy,
        min_points_on_object=args.min_points_on_object, max_planes=args.max_planes,
        edge_on_thresh=args.edge_on_thresh, dup_angle_thresh=args.dup_angle_thresh,
        sde_gate=args.sde_gate,
    )

    symmetry_dir = Path(args.renders_root) / args.symmetry_type
    if not symmetry_dir.exists():
        print(f"[error] Not found: {symmetry_dir}")
        sys.exit(1)

    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if args.max_objects:
        all_objects = all_objects[:args.max_objects]
    objects = all_objects[args.gpu_id::args.num_gpus]

    print(f"\nRunning variant '{args.variant}' on top of experiment '{args.experiment_id}' "
          f"for {len(objects)} {args.symmetry_type} objects...")
    print(f"Output experiment_id : {build_output_experiment_id(args.experiment_id, variant_config)}")
    print(f"Filter policy        : {args.filter_policy}")
    if args.symmetry_type == "plane_sym":
        print(f"max_planes/edge_on/dup_angle/sde_gate : "
              f"{args.max_planes}/{args.edge_on_thresh}/{args.dup_angle_thresh}/{args.sde_gate}")

    mesh_ctx_cache: dict = {}
    for obj_dir in tqdm(objects, unit="obj", dynamic_ncols=True):
        process_object(
            object_dir=obj_dir, symmetry_type=args.symmetry_type, sizes=args.sizes,
            lightings=args.lightings, source_experiment_id=args.experiment_id,
            variant_config=variant_config, objects_root=args.objects_root,
            overwrite=args.overwrite, mesh_ctx_cache=mesh_ctx_cache,
        )

    print(f"\n[GPU {args.gpu_id}] Done.")


if __name__ == "__main__":
    main()
