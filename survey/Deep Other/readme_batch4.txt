Batch 4 results

Files included
- Lin2025RFACL.csv
- Liu2024DSDCLNet.csv
- Medico2021Learning.csv
- Mu2025MPTSNet.csv
- Li2021ShapeNet.csv
- Li2026WaveState.csv
- Liang2025TCFNet.csv

Notes
- Lin2025RFACL.csv uses Table II accuracy rows only, MF1 rows omitted.
- Liu2024DSDCLNet.csv merges Table 4, Table 5, and Table 6 into a single dataset-by-method table.
  TS2Vec appears in both supervised and self-supervised settings, so the columns are named
  TS2Vec_supervised and TS2Vec_selfsupervised.
- Medico2021Learning.csv uses Table 2 mean accuracies only, standard deviations removed.
- Mu2025MPTSNet.csv uses Setting 2 / Table 2, the 25-dataset MTSC-dedicated comparison.
- Li2021ShapeNet.csv uses Table 1.
- Li2026WaveState.csv is based on explicit Table 6, the long-sequence ConvTran vs WaveState comparison.
  I did not include the paper's broader 30-dataset benchmark because that table was not reliably extractable
  into a clean dataset-by-method CSV from the PDF formatting.
- Liang2025TCFNet.csv uses Table 1 extracted from the page text; some dataset names were expanded from
  the abbreviations used in the paper.
