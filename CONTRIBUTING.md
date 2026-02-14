# Contributing

Thanks for contributing to the Multiverse Archive.

## What this repository accepts

- Dataset registry updates (metadata, licences, checksums, Zenodo identifiers).
- Loader and tooling improvements.
- Reproducible result submissions following `results/schema.md`.
- Documentation improvements and examples.

## Submitting leaderboard results

1. Create a folder: `results/submitted/<algorithm>/<multiverse-version>/`
2. Add:
   - `metrics.csv` (see `results/schema.md`)
   - `run-metadata.json` (software versions, hardware notes, command line, commit)
3. Open a pull request. CI will validate your submission.

## Dataset additions

- Prefer immutable, versioned Zenodo artefacts.
- Include a licence and provenance note in the registry entry.
- Provide a checksum for every artefact.
