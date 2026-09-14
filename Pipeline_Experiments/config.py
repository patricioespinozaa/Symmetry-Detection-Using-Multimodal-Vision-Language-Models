# Config-driven experiment selection: lets every script in this module run
# over an explicit list of --experiment-id values, OR auto-discover every
# molmo_multiview_<EXP>.json prompt run already present under a symmetry
# type, instead of hardcoding one prompt the way the Experiments/ sandbox
# notebooks did. See Pipeline_Experiments/README.md for the full config
# schema and Pipeline_Experiments/configs/experiments.example.yaml for a
# worked example.
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

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

    variant: str                              # "expA".."expF"
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


@dataclass
class RunConfig:
    """ Full Pipeline_Experiments run configuration, loaded from a YAML file. """

    renders_root: Path
    objects_root: Path
    symmetry_types: list[str]
    experiment_ids: dict[str, list[str] | None]     # symmetry_type -> ids, or None = auto-discover
    sizes: list[int]
    lightings: list[str]
    max_objects: int | None
    overwrite: bool
    variants: list[VariantConfig]
    diagnostics: list[str]
    run_evaluate: bool
    with_reference_metrics: bool
    run_compare: bool
    results_dir: Path = field(default_factory=lambda: Path("results"))
    extra_evaluate_args: list[str] = field(default_factory=list)

    def resolved_experiment_ids(self, symmetry_type: str) -> list[str]:
        """ Experiment ids to process for one symmetry type -- explicit config wins over discovery.

        Args:
            * symmetry_type: "axis_sym" or "plane_sym"

        Returns:
            * list[str]: experiment_id values to iterate over

        """
        configured = self.experiment_ids.get(symmetry_type)
        if configured is not None:
            return configured
        return discover_experiment_ids(self.renders_root, symmetry_type)


def load_config(path: str | Path) -> RunConfig:
    """ Load and validate a Pipeline_Experiments YAML config file.

    Args:
        * path: path to the YAML config file

    Returns:
        * RunConfig: parsed configuration, ready for run_batch.py

    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset = raw["dataset"]
    experiment_ids_raw = raw.get("experiment_ids", {}) or {}
    experiment_ids = {
        sym: (None if experiment_ids_raw.get(sym) in (None, "auto") else list(experiment_ids_raw[sym]))
        for sym in dataset["symmetry_types"]
    }

    variants = [
        VariantConfig(
            variant=v["variant"],
            filter_policy=v.get("filter_policy", "none"),
            min_points_on_object=v.get("min_points_on_object", 2),
            max_planes=v.get("max_planes", 1),
            edge_on_thresh=v.get("edge_on_thresh", 0.5),
            dup_angle_thresh=v.get("dup_angle_thresh", 15.0),
            sde_gate=v.get("sde_gate", 0.02),
        )
        for v in raw.get("variants", [])
    ]

    evaluation = raw.get("evaluation", {}) or {}

    return RunConfig(
        renders_root=Path(dataset["renders_root"]),
        objects_root=Path(dataset["objects_root"]),
        symmetry_types=list(dataset["symmetry_types"]),
        experiment_ids=experiment_ids,
        sizes=list(dataset.get("sizes", [224])),
        lightings=list(dataset.get("lightings", ["flat"])),
        max_objects=dataset.get("max_objects"),
        overwrite=bool(dataset.get("overwrite", False)),
        variants=variants,
        diagnostics=list(raw.get("diagnostics", [])),
        run_evaluate=bool(evaluation.get("run_evaluate", False)),
        with_reference_metrics=bool(evaluation.get("with_reference_metrics", False)),
        run_compare=bool(evaluation.get("run_compare", False)),
        results_dir=Path(raw.get("results_dir", "results")),
        extra_evaluate_args=list(evaluation.get("extra_args", [])),
    )
