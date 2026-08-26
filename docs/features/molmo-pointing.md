# MolmoPointing — VLM Pointing Inference

## Description

`MolmoPointing/` is the second stage of the thesis pipeline: it drives
Molmo2-8B, a vision-language model with "pointing" capability, over the
multi-view renders produced by `ImagesGenerator/` and asks it to locate the
2D projection of a 3D object's symmetry axis (or plane) in each image.
Its output — raw text plus parsed 2D coordinates per image, tagged with the
camera metadata used to generate that view — is the only place in the
pipeline where the VLM actually "looks" at pixels; everything downstream
(`Mapping/map_to_3d.py`, `Mapping/estimate_symmetry.py`,
`Mapping/evaluate.py`) works purely with the coordinates this module
produces, never touching the model or the images again.

Beyond a single default prompt, the module is built as an experiment
harness: `prompts_registry.py` auto-discovers versioned prompt variants
(`axis_v00`..`axis_v05_1`, `plane_v00`..`plane_v05_1`) from the `prompts/`
directory so new wording can be tested without touching code, and
`molmo_multiview_runner.py` supports three orthogonal "flows" (a/b/c) that
control how much semantic context (a free-text description, and optionally
a qualitative location hint) is injected ahead of the actual pointing call.
This lets the thesis compare "blind" pointing against pointing that first
lets the model reason about what the object is and where its symmetry
element plausibly sits.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `MolmoPointing/molmo_multiview_runner.py` | `get_model()` | Lazily loads `allenai/Molmo2-8B` (`AutoProcessor`/`AutoModelForImageTextToText`) as a module-level singleton, `bfloat16`, `device_map="auto"`. |
| | `load_metadata()` / `get_n_views_entries()` | Reads `metadata_all.json`; selects `n_views` camera entries evenly spaced (via `linspace` over index) across the full Fibonacci-sphere sequence. |
| | `_call_model()` | Single low-level Molmo2 call: builds the chat-template message with 1..N images + prompt text, runs `model.generate` (`max_new_tokens=2048`), decodes, frees GPU tensors. |
| | `parse_single_coords()` | Parses the single-image `<points coords="RADIO ID X Y ...">` format into `{"0": [{obj_id, x, y}, ...]}`. |
| | `parse_multi_coords()` | Parses the multi-image `<points coords="img_idx obj_id X Y; ...">` format (1-based image index) into `{"<img_idx>": [...]}` per image. |
| | `parse_describe_and_location()` | Splits Flow C's pre-pass response on an `Axis location:`/`Plane location:` marker line into `{description, location_hint}`; falls back to treating the whole text as description if no marker is found. |
| | `augment_prompt_with_description()` | Flow B: prepends a free-text object description ahead of the base pointing prompt. |
| | `build_flow_c_prompts()` | Flow C: builds the main N-view prompt from scratch (not an augmentation) using the pre-pass description + location hint; enforces exactly `FLOW_C_POINTS_PER_IMAGE` points per image. |
| | `prepare_flow_context()` | Runs the one extra single-view pre-pass call (Flow B or C) on a seeded description view. |
| | `run_global()` / `run_single_mode()` / `run_multi()` / `run_inference()` | Dispatch to the four `--prompt-mode` strategies (`global`, `single`, `multi`, `auto`). |
| | `process_object()` | Per-object driver: iterates (size × lighting × n_views), skips already-computed keys (flow-aware), runs the pre-pass + main call, writes cumulative JSON after every `n_views`. |
| | `load_results()` / `save_results()` | JSON read/write helpers for the cumulative per-object output file. |
| | `main()` / `parse_args()` / `preview()` | CLI entry point: argument parsing, interactive confirmation, multi-GPU round-robin object splitting, token-count logging. |
| `MolmoPointing/prompts_registry.py` | `_load_prompts()` / `PROMPTS` | Scans `prompts/axis/` and `prompts/plane/` for `<variant>_single.txt` + `<variant>_multi.txt` pairs and builds the `prompt_id → {single, multi, description}` registry at import time. |
| | `get_prompt()` / `list_prompts()` | Public lookup / CLI listing of registered prompt variants. |
| | `load_flow_prompt()` | Loads the fixed Flow B/C plumbing prompts from `prompts/description/*.txt` (`describe`, `describe_and_point_axis`, `describe_and_point_plane`). |
| `pipeline_common/view_selection.py` | `select_description_view()` | Deterministic per-object seeded (`MD5(object_id) mod 2^31`) selection of one metadata entry with elevation strictly inside `(-60°, 60°)`, used by Flow B/C to pick the description view; falls back to the full entry list if none qualify. |

## Inputs & Outputs

**Inputs:**
- Rendered images and `metadata_all.json` under
  `<renders_root>/<symmetry_type>/<object_id>/<image_size>/<illumination>/`,
  produced by `ImagesGenerator/` (Fibonacci-sphere multi-view renders, one
  `ROT_000` image per viewpoint, up to 114 views per object).
- Prompt text: either the hardcoded defaults (`PROMPT_SINGLE`,
  `PROMPT_MULTI`, `PROMPT_GLOBAL`), a registered variant loaded via
  `--prompt-id` from `prompts/{axis,plane}/*.txt`, or (for Flow B/C) the
  fixed plumbing prompts in `prompts/description/*.txt`.
- CLI flags: `--renders-root`, `--symmetry-type` (`axis_sym`/`plane_sym`),
  `--prompt-mode` (`global`/`single`/`multi`/`auto`), `--flow` (`a`/`b`/`c`),
  `--view-groups`, `--sizes`, `--lightings`, `--experiment-id`,
  `--prompt-id`, `--max-objects`, `--gpu-id`/`--num-gpus` for multi-GPU
  round-robin object splitting.

**Outputs:**
- Cumulative JSON per (object, size, lighting):
  `molmo_multiview.json` (production) or `molmo_multiview_<EXPERIMENT_ID>.json`
  (experiment mode), one key per `n_views` group, containing
  `prompt_used`, `raw_output`, `points_by_image` (0–1000 scale, top-left
  origin), `images_sent` (filename + camera azimuth/elevation/`R`/`T` per
  image), `n_points`, `n_images_with_points`, and — for `--flow b`/`c` —
  `flow`, `description_used`, `description_view_idx`,
  `description_view_filename`, and (`c` only) `location_hint`.
- `token_counts[_<EXPERIMENT_ID>].txt` under `<renders_root>/<symmetry_type>/`:
  per-`n_views` mean/min/max input-token counts, written once at the end of
  a run.
- Nothing is ever overwritten in place: existing `(size, lighting, n_views)`
  keys are skipped unless the stored entry's `flow` differs from the
  current `--flow`.

## Results & Observations

- `PROMPT_IMPROVEMENTS_v1.md` documents concrete AUC deltas from the v0→v1
  prompt rewrite (SVD, best n_views, dataset `experiments_20_06_2026`), e.g.
  axis `v02`: AUC 0.070→0.354 (n_obj 27→80) after switching from silhouette
  extrema to centerline points and `midpoint`→`independent` point mode;
  plane `v05`: AUC 0.085→0.344 after reframing "most distant points" as
  "vertical, not diagonal" separation.
- `MolmoPointing/README.md` notes `PROMPT_GLOBAL` "empirically performs
  worse than `auto`" on both single- and multi-image cases (no numbers
  given in the file itself).
- An earlier Flow C iteration that asked Molmo to track named landmarks by
  identity across views "degenerated into mechanically evenly-spaced
  grid/line patterns...instead of genuine per-point reasoning (angular
  error ~90°, chance level)" — documented in code comments
  (`build_flow_c_prompts` docstring) and `README.md`, citing the same
  failure mode as the ZeroKey paper.
- The already-applied Flow C point-count fix: `FLOW_C_POINTS_PER_IMAGE = 2`
  (constant in `molmo_multiview_runner.py`, line ~529) replaced an earlier
  "at most 3, fewer OK" rule. The code comment states this was **confirmed
  empirically**: `axis_v05_1_flowC` and `plane_v04_1_flowC` "both collapsed
  to 1 point/image at n_views>1" under the flexible cap. This is a real,
  already-shipped change in the current code — not a pending TODO.
  `MolmoPointing/README.md` and `Experiments.md` were updated (2026-08-25 doc
  pass) to match: both now describe Flow C as returning exactly 2
  points/image, and still recommend `--point-mode all` in
  `estimate_symmetry.py` — not because the count is variable anymore, but
  because `all` tolerates images where only one of the two points hits the
  mesh (`independent` would discard the whole image in that case).

## Key Decisions

- **`n_views` selection uses `linspace` over the full 114-entry index
  range, not the first `n` entries** — the code comment explains that
  taking the first N of a Fibonacci-sphere sequence would concentrate
  cameras near the north pole (e.g. elevation +66° to +90° for
  `n_views=6`), so `get_n_views_entries()` guarantees uniform angular
  coverage instead.
- **Coordinate parsing drops the leading "radius" token** in
  `parse_single_coords()` (`raw = raw[1:]`) — Molmo2's single-image
  `<points coords="RADIO ID X Y ...">` format includes a radius value
  before the point list that this pipeline doesn't use.
- **`--flow a` is guaranteed byte-for-byte identical to the original
  single-flow behavior** (explicit in the module docstring and
  `process_object()`'s docstring) — Flow B/C were added without touching
  the code path used when `flow == "a"`, so existing production results
  and their resumability semantics are preserved.
- **Resumability is flow-aware**: a stored entry with no `"flow"` key
  defaults to `"a"`, so re-running with `--flow a` never reprocesses
  anything, but re-running the same `--experiment-id` under a different
  `--flow` does reprocess (rather than silently reusing another flow's
  result) — by design, per the module docstring.
- **Description view for Flow B/C is chosen with a per-object deterministic
  seed** (`MD5(object_id) mod 2^31`), not a single global seed — this keeps
  the choice reproducible per object while still varying the viewpoint used
  across different objects (`view_selection.py` docstring).
- **Description view is filtered to elevation strictly inside `(-60°,
  60°)`** — avoids near-pole views, which project an axis to a point or a
  plane to a line, making the description/location pre-pass useless for
  those objects.
- **Flow C's main pointing call drops persistent cross-view point
  identity** (`obj_id` is just a per-image enumerator, not a tracked
  landmark) — chosen after empirically observing that asking Molmo to
  re-find K named landmarks across views degenerated into grid/line
  artifacts (see Results & Observations above); documented directly in the
  `build_flow_c_prompts` docstring.
- **`--flow b`/`c` have no effect under `--prompt-mode global`** — global
  mode's prompt (`PROMPT_GLOBAL`) doesn't support prompt overrides at all,
  not even via `--prompt-id`; the runner prints an explicit warning in
  `preview()` when this combination is used.
- **Results are written to disk after every `n_views` iteration**, not
  batched at the end of an object or run — explicit in `process_object()`'s
  comment ("Write after every n_views to survive interruptions"), trading
  I/O overhead for crash resilience during long unattended runs.
- **`prompts_registry.py` requires no code changes to add a prompt
  variant** — any `<variant>_single.txt` + `<variant>_multi.txt` pair
  dropped into `prompts/axis/` or `prompts/plane/` is auto-discovered by
  `_load_prompts()`; a missing multi-image counterpart for a given variant
  is skipped with a printed warning rather than raising.

## Known Limitations

- `max_new_tokens = 2048` in `_call_model()` is a hardcoded constant with no
  CLI override and no comment explaining the choice.
  > Decision found but reason not documented.
- The elevation range `(-60.0, 60.0)` for Flow B/C description-view
  selection (`DESCRIPTION_ELEVATION_RANGE`) is a hardcoded module constant,
  not exposed as a CLI flag.
- `select_description_view()`'s fallback to the full entry list when no
  candidates fall inside the elevation range is noted in its own docstring
  as "should not happen with the standard 114-view Fibonacci sampling" —
  an untested edge case for non-standard metadata.
- `parse_multi_coords()` silently drops any group with `img_idx` outside
  `[0, n_images)` or fewer than 4 numeric tokens, and `parse_single_coords()`
  returns `{}` on fewer than 4 tokens — malformed model output degrades
  silently to zero points rather than raising or logging a warning.
- The interactive confirmation prompt in `preview()` (`Type 'OK' to start`)
  blocks unattended runs unless `--yes`/`-y` is passed; there is no
  environment-variable or config-file override.
- Multi-GPU support is limited to a fixed round-robin split
  (`objects[gpu_id::num_gpus]`) computed once at startup — no dynamic
  work-stealing or load balancing if one GPU is slower or objects vary
  greatly in per-object cost.
- `get_model()` loads the model as a process-wide singleton with no
  explicit unload/reload path — running multiple `--symmetry-type` or
  `--flow` values in the same process is not supported by the module (each
  CLI invocation is a fresh process in the documented usage).
