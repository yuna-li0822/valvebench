## Summary

Brief description of the change and why it is needed.

## Changes
- Adds VALVEBENCH wrapper and canonical dataset generator
- Adds CI smoke test workflow
- Adds PR and commit instructions

## Checklist
- [ ] Tests pass locally (`python -m pytest -q tests/smoke_test.py`)
- [ ] Commit message follows the project's guidelines
- [ ] Archive reviewed legacy files (if any)

## Notes for reviewers
- The canonical generator and datasets are under `aipv/heart_valve_project/`.
- Smoke tests are intentionally small for CI (<30s).
