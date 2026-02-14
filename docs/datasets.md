# Datasets

## Packaging convention (recommended)

For each dataset, store one zip file on Zenodo containing:

- `<NAME>_TRAIN.ts`
- `<NAME>_TEST.ts`

Optional:
- `<NAME>.ts` (full dataset)
- additional metadata files (readme, licence note)

## Registry

The registry lives in `multiverse_registry/mtsc_registry.csv` and is snapped into the package for releases.
