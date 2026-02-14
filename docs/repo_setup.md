# Repository setup

1. Replace placeholders in:
   - `CITATION.cff` (`repository-code`)
   - `README.md` if needed
   - `multiverse_registry/mtsc_registry.csv` (Zenodo ids, checksums, provenance)
   - `src/multiverse/datasets/mtsc_registry.csv` (snapshot for releases)

2. Decide packaging on Zenodo:
   - one record containing all dataset zips, or
   - one record per dataset

3. Tag releases:
   - Git tag `multiverse-vX.Y.Z`
   - Zenodo versioned record for the dataset artefacts

4. Optional: publish documentation with GitHub Pages.
