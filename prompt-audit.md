# Prompt: Architecture Audit — ML/DS Thesis Project

Use this prompt in a Claude Code session. Read-only — do not modify any files.

---

Perform a read-only architecture audit of this ML/Data Science thesis project.
Do not modify, move, or delete any files during this session.

Produce the file docs/audits/architecture-audit.md with the following sections:

## 1. Project structure (auto-detected)
Map the actual folder structure and describe the role of each directory.
Flag any directories or files whose purpose is unclear or ambiguous.

## 2. Code quality — violations of @docs/code-norms.md
List files that violate the project's Python norms. For each violation:
- File path
- Issue (e.g. missing type hints, module-level docstring, wrong docstring format)
- Severity: High / Medium / Low

## 3. Notebook discipline
Review all .ipynb files and flag:
- Notebooks with cells run out of order (non-sequential execution counts)
- Notebooks that contain reusable logic that should be in src/ instead
- Notebooks that import from unclear or hardcoded paths
- Notebooks with no markdown explanation of what they do

## 4. Reproducibility issues
Flag anything that would prevent another person from running the project:
- Hardcoded absolute paths (e.g. /Users/yourname/...)
- Missing or incomplete environment definition (requirements.txt, environment.yml)
- Data files assumed to exist without documentation
- Random seeds not fixed or not consistent across scripts
- Model checkpoints or outputs not reproducible from the codebase alone

## 5. Structural problems
- Scripts or modules with mixed responsibilities (e.g. a training script that also does EDA)
- Duplicated logic across notebooks and src/
- Circular imports or unclear dependency direction
- Dead code: unused functions, commented-out blocks, abandoned experiments

## 6. Experiment and results organization
- Are experiment results clearly separated from source code?
- Are model outputs and metrics stored consistently?
- Is it clear which script/notebook produced which result?

## 7. Documentation gaps
- Functions or classes with missing or incomplete docstrings
- Missing README or unclear setup instructions
- Undocumented data schema or feature definitions

## 8. Recommended refactor plan
Ordered list of actions, grouped by theme, with severity rating.
Focus on changes that improve structure and maintainability without
altering the research logic or results.

Do not suggest new features or changes to the ML approach itself.
Only report what you find — do not invent issues.
