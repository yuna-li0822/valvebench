# Commit & PR instructions

Run these commands locally to commit the current changes and open a PR.

Replace `feat/valvebench-dataset-cleanup` with your preferred branch name.

```bash
git checkout -b feat/valvebench-dataset-cleanup
git add -A
git commit -F COMMIT_MESSAGE.txt
git push -u origin feat/valvebench-dataset-cleanup

# If you have the GitHub CLI installed, create a PR:
gh pr create --fill --base main --title "ValveBench: RNG, schema, dataset v0.1, splits, and docs" --body-file COMMIT_MESSAGE.txt
```

Notes:
- `COMMIT_MESSAGE.txt` already exists and contains a prepared commit message. Review it before committing.
- The CI workflow `/.github/workflows/ci-smoke.yml` will run the smoke test on push/PR.
