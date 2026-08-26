# Prompt: Generate CLAUDE.md for ML/DS Thesis Project

Use this prompt at the start of a Claude Code session, in the project root.

---

Analyze this project thoroughly — folder structure, source files, notebooks,
scripts, configs, and any existing documentation — then generate a CLAUDE.md file.

This is an academic Machine Learning / Data Science thesis project in its final
stage (polishing and documenting), with the goal of keeping it well-structured
for continued development.

The CLAUDE.md must include:

## 1. Project purpose
One concise paragraph: what problem it solves, what ML approach is used,
and what the expected output or contribution is.

## 2. Tech stack
Inferred from the codebase and config files (requirements.txt, pyproject.toml,
environment.yml, etc.). Include:
- Python version
- Core ML/DS libraries (PyTorch, TensorFlow, scikit-learn, HuggingFace, etc.)
- Data handling (pandas, numpy, polars, etc.)
- Experiment tracking (MLflow, W&B, etc.) if present
- Visualization tools
- Any pipeline or orchestration tools

## 3. Project structure
Auto-detect and describe each main directory. Common patterns to look for:
- data/ (raw, processed, external)
- notebooks/ (exploration, experiments, evaluation)
- src/ or project_name/ (core source code)
- models/ (serialized models, checkpoints)
- configs/ (hyperparameters, experiment configs)
- scripts/ (training, evaluation, inference entrypoints)
- tests/ (if any)
- outputs/ or results/ (metrics, plots, predictions)

## 4. Key commands
How to:
- Set up the environment
- Run training
- Run evaluation / inference
- Run tests (if any)
- Launch notebooks

## 5. Data notes
- Where data lives and what format it is in
- Whether raw data must be downloaded separately (and how)
- Any preprocessing steps that must run before training

## 6. Experiment tracking
- How experiments are tracked (if at all)
- Where results and metrics are stored
- How to reproduce a specific run

## 7. Code conventions
Add a single line referencing the norms file:
> See @docs/code-norms.md for Python formatting and docstring standards.

## 8. What Claude should never do in this project
Infer from the codebase — for example:
- Do not overwrite files in data/raw/
- Do not modify serialized model files directly
- Do not delete outputs/ without confirmation
- Do not change random seeds without updating the experiment config

Be concise throughout. The file should be readable in under 2 minutes.
Only include what you can confirm from the codebase — do not invent or assume.
