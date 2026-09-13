# Batch, config-driven port of the "X~=500 collapse" diagnostic from
# Experiments/sandbox_pipeline_server_updated_6_pts_v2.ipynb: measures how
# often Molmo2 places a point at the horizontal image center regardless of
# camera angle (a prior/shortcut, not real geometric reasoning), separating
# legitimate cenital views (elevation near +-90 deg, where X~=500 IS the
# correct answer) from every other view (where it would indicate the bug).
# Needs no GPU and no new Molmo2 calls -- it re-reads molmo_multiview_<EXP>.json
# files that already exist, exactly like Mapping/diagnose_point_localization.py.
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline_common.naming import exp_filename  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import discover_experiment_ids  # noqa: E402

MOLMO_JSON = "molmo_multiview.json"

CENTERED_THRESH_DEFAULT = 30.0     # Molmo 0-1000 px units; |x - 500| below this counts as "centered"
CENITAL_ELEV_THRESH_DEFAULT = 70.0  # degrees; |elevation| above this is a legitimately cenital view


def _row_for_object(
    molmo_data: dict, centered_thresh: float, cenital_elev_thresh: float,
) -> dict[str, dict]:
    """ Compute per-n_views X-centering stats for one object's molmo_multiview data.

    Args:
        * molmo_data: parsed molmo_multiview_<EXP>.json contents for one object
        * centered_thresh: |x - 500| below this counts a point as "centered"
        * cenital_elev_thresh: |elevation| above this is a legitimately cenital view

    Returns:
        * dict[str, dict]: n_views_key -> {"n_points", "n_centered",
          "n_noncenital", "n_noncenital_centered"}

    """
    out: dict[str, dict] = {}
    for n_views_key, group in molmo_data.items():
        images_sent = group.get("images_sent", [])
        points_by_image = group.get("points_by_image", {})
        if not images_sent:
            continue

        n_points = n_centered = n_noncenital = n_noncenital_centered = 0
        for img_idx_str, pts in points_by_image.items():
            if not pts:
                continue
            cam = images_sent[int(img_idx_str)]
            elevation = abs(float(cam.get("elevation", 0.0)))
            is_cenital = elevation > cenital_elev_thresh

            for p in pts:
                n_points += 1
                centered = abs(p["x"] - 500.0) < centered_thresh
                if centered:
                    n_centered += 1
                if not is_cenital:
                    n_noncenital += 1
                    if centered:
                        n_noncenital_centered += 1

        out[n_views_key] = {
            "n_points": n_points, "n_centered": n_centered,
            "n_noncenital": n_noncenital, "n_noncenital_centered": n_noncenital_centered,
        }
    return out


def run_diagnostic(
    renders_root: Path, symmetry_type: str, experiment_id: str,
    sizes: list[int], lightings: list[str], max_objects: int | None = None,
    centered_thresh: float = CENTERED_THRESH_DEFAULT,
    cenital_elev_thresh: float = CENITAL_ELEV_THRESH_DEFAULT,
) -> tuple[list[dict], dict]:
    """ Run the center-bias diagnostic for one experiment_id over the whole dataset.

    Args:
        * renders_root: root folder of renders
        * symmetry_type: "axis_sym" or "plane_sym"
        * experiment_id: prompt run to diagnose (its molmo_multiview_<ID>.json
          must exist)
        * sizes: render sizes to scan
        * lightings: illuminations to scan
        * max_objects: limit to the first N objects (sorted order), or None
          for every object
        * centered_thresh: see _row_for_object
        * cenital_elev_thresh: see _row_for_object

    Returns:
        * tuple[list[dict], dict]: (detail rows, summary rows keyed by n_views)

    """
    symmetry_dir = Path(renders_root) / symmetry_type
    all_objects = sorted(d for d in symmetry_dir.iterdir() if d.is_dir())
    if max_objects:
        all_objects = all_objects[:max_objects]

    input_file = exp_filename(MOLMO_JSON, experiment_id)
    detail_rows: list[dict] = []
    totals: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    n_objects_seen: dict[str, int] = defaultdict(int)

    for obj_dir in tqdm(all_objects, unit="obj", dynamic_ncols=True,
                        desc=f"{symmetry_type}/{experiment_id}"):
        for size in sizes:
            for lighting in lightings:
                render_dir = obj_dir / str(size) / lighting
                molmo_path = render_dir / input_file
                if not molmo_path.exists():
                    continue
                with open(molmo_path, encoding="utf-8") as f:
                    molmo_data = json.load(f)

                per_nv = _row_for_object(molmo_data, centered_thresh, cenital_elev_thresh)
                for n_views_key, stats in per_nv.items():
                    detail_rows.append({
                        "experiment_id": experiment_id, "object_id": obj_dir.name,
                        "size": size, "lighting": lighting, "n_views": n_views_key,
                        **stats,
                    })
                    for key, value in stats.items():
                        totals[n_views_key][key] += value
                    n_objects_seen[n_views_key] += 1

    summary_rows = []
    for n_views_key, agg in sorted(totals.items(), key=lambda kv: int(kv[0])):
        pct_centered = agg["n_centered"] / agg["n_points"] if agg["n_points"] else 0.0
        pct_noncenital_centered = (
            agg["n_noncenital_centered"] / agg["n_noncenital"] if agg["n_noncenital"] else 0.0
        )
        if pct_noncenital_centered > 0.5:
            verdict = "PROBLEMA CONFIRMADO"
        elif pct_noncenital_centered > 0.2:
            verdict = "problema parcial"
        else:
            verdict = "sin problema generalizado"

        summary_rows.append({
            "experiment_id": experiment_id, "n_views": n_views_key,
            "n_objects": n_objects_seen[n_views_key],
            "n_points": agg["n_points"], "pct_centered": round(pct_centered, 4),
            "n_noncenital_points": agg["n_noncenital"],
            "pct_noncenital_centered": round(pct_noncenital_centered, 4),
            "verdict": verdict,
        })

    return detail_rows, summary_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    """ Write rows to path as CSV, creating parent directories as needed. """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """ CLI entrypoint: run the center-bias diagnostic over one or more experiment_ids. """
    p = argparse.ArgumentParser(
        description="Diagnose whether Molmo2 collapses points to X~=500 regardless of camera angle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--renders-root", required=True)
    p.add_argument("--symmetry-type", required=True, choices=["axis_sym", "plane_sym"])
    p.add_argument("--experiment-id", nargs="+", default=None,
                   help="One or more prompt runs to diagnose. Omit and pass --all instead to "
                        "auto-discover every molmo_multiview_<ID>.json variant already run.")
    p.add_argument("--all", action="store_true",
                   help="Diagnose every experiment_id discovered under --renders-root/--symmetry-type.")
    p.add_argument("--sizes", type=int, nargs="+", default=[224])
    p.add_argument("--lightings", type=str, nargs="+", default=["flat"])
    p.add_argument("--max-objects", type=int, default=None)
    p.add_argument("--centered-thresh", type=float, default=CENTERED_THRESH_DEFAULT)
    p.add_argument("--cenital-elev-thresh", type=float, default=CENITAL_ELEV_THRESH_DEFAULT)
    p.add_argument("--out-detail", required=True, help="Per (experiment_id, object, n_views) CSV.")
    p.add_argument("--out-summary", required=True, help="Per (experiment_id, n_views) CSV with verdict.")
    args = p.parse_args()

    if not args.experiment_id and not args.all:
        print("[error] Pass --experiment-id <id> [<id> ...] or --all.")
        sys.exit(1)

    experiment_ids = (
        discover_experiment_ids(Path(args.renders_root), args.symmetry_type)
        if args.all else args.experiment_id
    )
    if not experiment_ids:
        print(f"[error] No experiment_ids found under {args.renders_root}/{args.symmetry_type}.")
        sys.exit(1)
    print(f"Diagnosing {len(experiment_ids)} experiment_id(s): {experiment_ids}")

    all_detail, all_summary = [], []
    for experiment_id in experiment_ids:
        detail, summary = run_diagnostic(
            Path(args.renders_root), args.symmetry_type, experiment_id,
            args.sizes, args.lightings, args.max_objects,
            args.centered_thresh, args.cenital_elev_thresh,
        )
        all_detail += detail
        all_summary += summary
        for row in summary:
            print(f"  n_views={row['n_views']:>4}  pct_centered={row['pct_centered']:.2%}  "
                  f"pct_noncenital_centered={row['pct_noncenital_centered']:.2%}  -> {row['verdict']}")

    _write_csv(Path(args.out_detail), all_detail)
    _write_csv(Path(args.out_summary), all_summary)
    print(f"\nWrote {len(all_detail)} detail rows -> {args.out_detail}")
    print(f"Wrote {len(all_summary)} summary rows -> {args.out_summary}")


if __name__ == "__main__":
    main()
