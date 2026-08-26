# Exploratory Data Analysis (EDA)

## Description

This component performs the initial exploratory data analysis of the raw
ShapeNet-derived dataset used throughout the thesis: the curated `.obj` mesh
files and their paired `.txt` ground-truth symmetry annotations. It runs
before any rendering, Molmo pointing, or mapping/evaluation stage and answers
basic sanity and descriptive-statistics questions about the dataset: how many
objects exist, whether every mesh has a matching ground-truth file, how
complex the meshes are (vertex count), and how many symmetry planes/axes each
object has.

The notebook validates the two curated dataset folders referenced in
`docs/Contexto.md` (`curated_axis_sym_obj` and `curated_plane_sym_obj`, 850
objects each) and produces descriptive statistics and plots for one of them.
This groundwork supports later claims in the thesis (e.g. the 692/152/6
breakdown of objects with 1/2/3 ground-truth symmetry planes) and gives an
early view of mesh complexity (vertex counts), which is relevant context for
downstream rendering and ray-casting cost.

## Key Files & Functions

| File | Function / Class | Responsibility |
|---|---|---|
| `ExploratoryDataAnalysis/EDA.ipynb` | `count_symmetries_from_txt(txt_path)` | Reads a ground-truth `.txt` file; returns the integer on the first line as the symmetry count, with a fallback that counts lines starting with `plane`/`axis` if the first line isn't a valid integer. |
| `ExploratoryDataAnalysis/EDA.ipynb` | `count_vertices_from_obj(obj_path)` | Counts lines starting with `v ` in an `.obj` file to get the mesh's vertex count. |
| `ExploratoryDataAnalysis/EDA.ipynb` | (notebook top-level cells) | Discovers `.obj`/`.txt` files per dataset folder, builds a pandas DataFrame keyed by object id (file stem), checks OBJ/TXT pairing consistency, and produces summary stats, a histogram, a boxplot, and a scatter plot. |

## Inputs & Outputs

**Inputs:**
- `../data/tesis/curated_axis_sym_obj/curated_axis_sym_obj/*.obj` and
  `*.txt` — axis-symmetry curated dataset (referenced/counted but not used
  for the detailed EDA past cell 3).
- `../data/tesis/curated_plane_sym_obj/curated_plane_sym_obj_ori/*.obj` and
  `*.txt` — plane-symmetry curated dataset. This is the dataset actually
  analyzed in detail (cells 4 onward reassign `obj_files`/`txt_files` to the
  PLANAR set).
- Each `.obj` file: a ShapeNet mesh (vertex lines `v x y z ...`).
- Each `.txt` file: ground-truth symmetry annotation, whose first line is
  expected to be an integer count of symmetries (planes, in the PLANAR
  dataset).

**Outputs:**
- No files are written to disk — all outputs are in-notebook: printed counts,
  a `pandas.DataFrame` (`df_clean`) with columns `object_id`, `vertex_count`,
  `symmetry_count`, `has_obj`, `has_txt`; a `.describe()` summary table; a
  `value_counts()` table of `symmetry_count`; a top-10-by-vertex-count table;
  and three matplotlib figures (histogram, boxplot, scatter of vertex count
  vs. symmetry count), all rendered inline and not saved to disk.

## Results & Observations

From the notebook's cached cell outputs (plane-symmetry dataset,
`curated_plane_sym_obj_ori`):

- File counts: 850 `.obj` and 850 `.txt` files found for both the AXIAL and
  PLANAR folders (cell 3).
- Pairing check: 0 objects with a `.txt` but no `.obj`, and 0 with an `.obj`
  but no `.txt` — the PLANAR dataset is fully paired (cell 5).
- `df_clean` (850 valid objects) descriptive stats (cell 6):
  - `vertex_count`: mean 5449.1, std 7784.4, min 191, 25% 882, median 2143,
    75% 6687.75, max 75563.
  - `symmetry_count`: mean 1.19, std 0.41, min 1, median 1, max 3.
- `symmetry_count` distribution (cell 7): 692 objects with 1 symmetry plane,
  152 with 2, 6 with 3 — matching the breakdown documented in
  `docs/Contexto.md` (§3, "Alcance").
- Top-10 most complex objects by vertex count (cell 8) range from 35,385 to
  75,563 vertices; the single most complex object (`8843d862a7545d0d96db382b382d7132`)
  has 75,563 vertices and only 1 GT symmetry plane.
- A histogram (cell 9), boxplot (cell 10), and scatter plot of vertex count
  vs. symmetry count (cell 11) are rendered as `image/png` outputs in the
  notebook; no numeric interpretation of these plots is written in the
  notebook text (e.g. no explicit comment on skew, outliers, or correlation
  visible in the source).
- Cell 12 (the final cell) is empty with `execution_count: None` — no content
  or output.

## Key Decisions

- **First line of the `.txt` file is treated as the ground-truth symmetry
  count**, with a fallback that counts lines prefixed with `plane`/`axis` if
  parsing the first line as `int` fails.
  > Decision found but reason not documented.
- **Vertex count is computed via a simple text-line scan** (`line.startswith('v ')`)
  rather than loading the mesh with a 3D library (e.g. `trimesh`, used
  elsewhere in the repo's `Mapping/` code).
  > Decision found but reason not documented.
- **Objects are matched between `.obj` and `.txt` by file stem** (filename
  without extension) rather than by an explicit id column or directory
  structure.
  > Decision found but reason not documented.
- **Detailed EDA (cells 4-11) is run only on the PLANAR dataset**
  (`obj_files_PLANAR`/`txt_files_PLANAR`), not the AXIAL one, even though both
  are loaded and counted in cell 3.
  > Decision found but reason not documented.

## Known Limitations

- The AXIAL dataset (`curated_axis_sym_obj`) is only used for a file-count
  sanity check (cell 3); it is never assigned to `obj_files`/`txt_files` for
  the detailed analysis, so no vertex-count/symmetry-count statistics,
  pairing checks, or plots exist for it in this notebook.
- Vertex counting via raw `v ` line-prefix matching will silently miscount if
  an `.obj` file uses different whitespace/formatting conventions, or will
  count parameter-space vertices (`vp`) incorrectly if such lines were
  present (not an issue observed here, but not guarded against).
- `count_symmetries_from_txt`'s fallback path (counting `plane`/`axis` lines)
  is present but its correctness on this dataset's actual `.txt` format is
  not directly verified in the notebook output — the first-line-integer path
  appears to always succeed here (no fallback usage is shown in any printed
  output).
- The final cell (cell 12) is empty and unexecuted (`execution_count: None`),
  suggesting the notebook was left mid-edit or a planned analysis step
  (unspecified) was never added.
- No outputs are persisted to disk (no CSV/PNG export), so downstream
  documents or scripts cannot reuse these exact statistics without re-running
  the notebook against the same data paths (`../data/tesis/...`, relative to
  the notebook's location), which are not present in this repository checkout
  and were not independently verified by this analysis.
- Per the project's own audit history (noted in the task context), cell
  `execution_count` values in this notebook are out of order relative to
  cell position (e.g. counts jump 2, 5, 6, 10, 21, 22, 23...), indicating
  cells were re-run out of sequence at some point; the cached outputs
  reported above reflect whatever state was last saved and may not reflect a
  single top-to-bottom run.
