# Prompt: Refactor — ML/DS Thesis Project

Run this AFTER reviewing docs/audits/architecture-audit.md yourself.
Execute one section at a time and confirm before continuing.

---

Execute the refactor plan from @docs/audits/architecture-audit.md for this
ML/Data Science thesis project.

The goal is a clean, well-structured codebase that is easy to understand,
reproduce, and continue developing — not a rewrite.

## Rules

- Work section by section from the audit. Stop after each section and summarize
  what was done before proceeding.
- Do not alter any research logic, model architecture, or training parameters.
  If a change would affect results, flag it and wait for instruction.
- Do not overwrite files in data/raw/ under any circumstance.
- Do not delete any file without explicit confirmation — move to an
  archive/ folder instead if removal seems appropriate.
- After moving or renaming any file, update all imports and references
  that depended on it. Verify nothing is broken before continuing.
- Log every action in docs/audits/refactor-log.md using this format:

  [file or directory] | [action: moved / renamed / refactored / deleted] | [reason]

## Order of execution (High severity first)

### Phase 1 — Reproducibility fixes
- Replace all hardcoded absolute paths with relative paths or config variables
- Ensure random seeds are fixed and consistent across all scripts and notebooks
- Verify environment definition (requirements.txt or environment.yml) is complete
  and matches what the code actually imports

### Phase 2 — Structure cleanup
- Move reusable logic from notebooks into the appropriate src/ module
- Split scripts or modules with mixed responsibilities
- Remove or archive dead code (unused functions, abandoned experiment files)
- Resolve duplicate logic between notebooks and src/

### Phase 3 — Code norms (per @docs/code-norms.md)
- Add or fix docstrings to all functions and classes (use the project format)
- Add missing type hints to function signatures
- Remove module-level docstrings from the top of .py files
- Fix import order (stdlib → third-party → internal)

### Phase 4 — Notebooks
- Add a markdown cell at the top of each notebook describing its purpose,
  inputs, and outputs
- Clear and re-run notebooks that have out-of-order execution counts
  (only if safe to do so — flag otherwise)

### Phase 5 — Documentation update
- Update CLAUDE.md if the structure changed
- Ensure docs/features/ reflects the current state of the project
- Confirm docs/audits/refactor-log.md is complete

## When to stop and ask

Stop and wait for instruction if:
- A change would modify training logic, loss functions, or model configs
- A file's purpose is genuinely ambiguous
- Removing something might affect a result referenced in the thesis
- Two modules seem duplicated but with subtle differences

At the end of each phase, output a short summary:
what was changed, what was skipped, and any open questions.
