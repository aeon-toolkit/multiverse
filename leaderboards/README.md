# Leaderboards

This folder contains the scripts and generated artefacts used to maintain leaderboards.

- `build_leaderboard.py` aggregates results in `results/submitted/**/metrics.csv`
- `leaderboard.md` is generated, do not edit directly

## Workflow

1. Contributors submit results via a pull request.
2. CI validates the submission structure and schema.
3. The leaderboard is regenerated and committed (or published via Pages, if configured).
