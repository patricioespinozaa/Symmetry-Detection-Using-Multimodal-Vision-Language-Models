# Experiment selection helpers shared by estimate_symmetry_variants.py and
# diagnostics/: discover every molmo_multiview_<EXP>.json prompt run already
# present under a symmetry type, describe one triangulation variant
# (VariantConfig) and build its output experiment id. The YAML-driven batch
# runner (run_batch.py, RunConfig, configs/) was removed from the repo; it
# remains in the git history.
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_MOLMO_JSON_EXP_RE = re.compile(r"^molmo_multiview_(.+)\.json$")
NOMESH_SUFFIX = "_nomesh"


def discover_experiment_ids(renders_root: Path, symmetry_type: str) -> list[str]:
    """ Scan every object/size/lighting folder for molmo_multiview_<ID>.json files.

    Args:
        * renders_root: root folder of renders (same tree every Mapping/
          script uses)
        * symmetry_type: "axis_sym" or "plane_sym"

    Returns:
        * list[str]: sorted, deduplicated experiment_id values found across
          the whole dataset. The production file (molmo_multiview.json, no
          suffix -- belongs to the with-mesh pipeline) is intentionally not
          included. A "<X>_nomesh" id is dropped whenever a bare "<X>" id is
          also present: some earlier sweeps saved the exact same Molmo2
          points twice, under two --experiment-id labels, purely so
          evaluate.py/estimate_symmetry.py's with-mesh output for the bare
          prompt_id wouldn't be overwritten -- identical content, not a
          second prompt to re-run every variant against.

    """
    symmetry_dir = Path(renders_root) / symmetry_type
    if not symmetry_dir.exists():
        return []

    found: set[str] = set()
    for path in symmetry_dir.glob("*/*/*/molmo_multiview*.json"):
        match = _MOLMO_JSON_EXP_RE.match(path.name)
        if match:
            found.add(match.group(1))

    deduped = {
        exp_id for exp_id in found
        if not (exp_id.endswith(NOMESH_SUFFIX) and exp_id[: -len(NOMESH_SUFFIX)] in found)
    }
    return sorted(deduped)


@dataclass
class VariantConfig:
    """ One triangulation-ablation configuration to run (see estimate_symmetry_variants.py). """

    variant: str                              # "expE" (triangulation_variants.VARIANTS)
    filter_policy: str = "none"               # view_filtering.FILTER_POLICIES
    min_points_on_object: int = 2
    max_planes: int = 1                       # plane_sym only
    edge_on_thresh: float = 0.5               # plane_sym only
    dup_angle_thresh: float = 15.0            # plane_sym only
    sde_gate: float = 0.02                    # plane_sym + variant == "expF" only

    def output_suffix(self) -> str:
        """ The --experiment-id suffix this configuration writes its output under. """
        parts = [self.variant]
        if self.filter_policy != "none":
            parts.append(f"filt-{self.filter_policy}")
        if self.max_planes != 1:
            parts.append(f"mp{self.max_planes}")
        return "_".join(parts)


def build_output_experiment_id(source_experiment_id: str, variant_config: VariantConfig) -> str:
    """ Build the output --experiment-id for one (source prompt run, variant) combination.

    Mapping/compare_results_no_mesh.py only discovers experiment_ids ending
    in "_nomesh" (see its load_csvs docstring). Older no-mesh sweeps already
    named their source --experiment-id that way (e.g. "axis_v06_nomesh"), but
    a source prompt run's --experiment-id doesn't have to (e.g. plain
    "axis_v06", the prompt_id itself, is a common convention too) -- every
    Pipeline_Experiments output IS a no-mesh estimate by construction, so the
    trailing "_nomesh" is always appended to the output regardless of
    whether the source had it (and never doubled if it already did).

    Args:
        * source_experiment_id: the prior prompt run being re-estimated
        * variant_config: which variant + hyperparameters are being applied

    Returns:
        * str: e.g. "axis_v06_nomesh" + expA -> "axis_v06_expA_nomesh";
          "axis_v06" + expA -> "axis_v06_expA_nomesh"

    """
    base = source_experiment_id.removesuffix(NOMESH_SUFFIX)
    tag = variant_config.output_suffix()
    return f"{base}_{tag}{NOMESH_SUFFIX}"
