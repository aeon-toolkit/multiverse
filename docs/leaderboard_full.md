# Full-archive leaderboard

This ranks estimators on the 100 datasets used for the full-archive comparison in the
Multiverse archive paper. Those are the Multiverse datasets on which all 17 of the
paper's classifiers have a resample-0 result. The list is fixed, in
[`results/multiverse/paper_datasets.txt`](../results/multiverse/paper_datasets.txt), so
this table keeps describing the paper's comparison as new results arrive.

An estimator is ranked here only if it has a result on all 100 datasets, on every
metric. That differs from the [Multiverse-core leaderboard](../README.md#multiverse-core-leaderboard)
in two ways:

- **CIF-500** and **DrCIF-500** are the configurations the paper reports as CIF and
  DrCIF. The core table reports the default CIF and DrCIF instead.
- **FreshPRINCE** and **1NN-DTW** are ranked here. They are held out of the core table
  because each fails on core datasets outside these 100.

A sortable version, with an average rank beside every metric, is in
[`results/multiverse/leaderboard_full.html`](../results/multiverse/leaderboard_full.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard_full.html)).

<!-- FULL_LEADERBOARD:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **6.31** | **0.7609** | 0.7038 | **0.8440** | 0.6421 | **0.5816** | 0.6785 | **0.7736** |
| 2 | MRHydra | 6.86 | 0.7591 | **0.7145** | 0.7601 | 0.6568 | 8.6836 | 0.6925 | 0.7695 |
| 3 | RDST | 7.51 | 0.7500 | 0.6974 | 0.7474 | 0.6314 | 9.0100 | 0.6560 | 0.7718 |
| 4 | FreshPRINCE | 7.66 | 0.7473 | 0.7102 | 0.8435 | **0.6571** | 0.6086 | **0.6951** | 0.7570 |
| 5 | CIF-500 | 7.88 | 0.7465 | 0.7000 | 0.8413 | 0.6379 | 0.6084 | 0.6746 | 0.7612 |
| 6 | Arsenal | 8.00 | 0.7524 | 0.6966 | 0.7886 | 0.6316 | 4.4938 | 0.6679 | 0.7684 |
| 7 | ROCKET | 8.07 | 0.7530 | 0.6961 | 0.7456 | 0.6285 | 8.9011 | 0.6605 | 0.7714 |
| 8 | DrCIF-500 | 8.20 | 0.7314 | 0.6907 | 0.8279 | 0.6243 | 0.6299 | 0.6680 | 0.7471 |
| 9 | RIST | 8.31 | 0.7351 | 0.6935 | 0.8183 | 0.6246 | 0.6271 | 0.6660 | 0.7443 |
| 10 | QUANT | 8.73 | 0.7374 | 0.6977 | 0.8344 | 0.6384 | 0.6224 | 0.6831 | 0.7421 |
| 11 | LITETime-MV | 9.01 | 0.7161 | 0.6862 | 0.8141 | 0.6109 | 1.6554 | 0.6501 | 0.7515 |
| 12 | STC | 9.61 | 0.7334 | 0.6796 | 0.8175 | 0.6230 | 0.7178 | 0.6542 | 0.7479 |
| 13 | Catch22 | 9.67 | 0.7201 | 0.6818 | 0.8313 | 0.6188 | 0.6781 | 0.6728 | 0.7261 |
| 14 | H-InceptionTime | 10.05 | 0.6970 | 0.6672 | 0.7988 | 0.5925 | 1.7675 | 0.6494 | 0.7165 |
| 15 | TDE | 10.16 | 0.7161 | 0.6561 | 0.7811 | 0.5810 | 1.4773 | 0.6286 | 0.7306 |
| 16 | 1NN-DTW | 12.64 | 0.6467 | 0.6191 | 0.6802 | 0.5610 | 12.7343 | 0.6287 | 0.6432 |
| 17 | Dummy | 14.32 | 0.4489 | 0.3720 | 0.5000 | 0.2219 | 1.1562 | 0.3831 | 0.4231 |

Average over the 100 paper datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- FULL_LEADERBOARD:END -->

## Not yet complete

Estimators in the Multiverse-core table that do not yet have a result on all 100
datasets, so are not ranked above. The same gaps are in
[`results/multiverse/pending_full.csv`](../results/multiverse/pending_full.csv), one
`estimator,dataset` row per missing run, to queue runs from.

<!-- FULL_PENDING:START -->
| Estimator | Completed | Missing datasets |
|---|---|---|
| PatchMTSC | 60 of 100 | AsphaltPavementTypeCoordinates, ButtonPress, FeedbackButton, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| TimesNet | 60 of 100 | AsphaltPavementTypeCoordinates, ButtonPress, FeedbackButton, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| TimesURL | 60 of 100 | AsphaltPavementTypeCoordinates, ButtonPress, FeedbackButton, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| CIF | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| ConvTran | 57 of 100 | Alzheimers, AsphaltPavementTypeCoordinates, ButtonPress, EigenWorms, FeedbackButton, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PhotoStimulation, PronouncedSpeech, SPHERE-WUS, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| DisjointCNN | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| DrCIF | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| STSF | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| Summary | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| TSF | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| XCM | 57 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
| TS2Vec | 56 of 100 | AsphaltPavementTypeCoordinates, BasicMotions, ButtonPress, FeedbackButton, FingerMovements, IRDS-EFL, IRDS-EFR, IRDS-SAL, IRDS-SAR, IRDS-SFE, IRDS-SFR, IRDS-STL, IRDS-STR, ImaginedFeetHands, InnerSpeech, KERAAL-CTK, KERAAL-CTK-MC, KERAAL-ELK, KERAAL-ELK-MC, KERAAL-RTK-MC, KIMORE-LA-C, KIMORE-Sq-C, KIMORE-TR-C, KINECAL-3WFV, KINECAL-GGFV, KINECAL-QSEC, LiveFuelMoistureContent_disc, Locust2022, PronouncedSpeech, SPHERE-WUS, SelfRegulationSCP2, UCDHE-MP, UCDHE-MP-MC, UCDHE-Rowing, UIPRMD-HS-C, UIPRMD-IL-C, UIPRMD-SASLR-C, UIPRMD-SL-C, UIPRMD-SSA-C, UIPRMD-SSE-C, UIPRMD-SSIER-C, UIPRMD-SSS-C, UIPRMD-STS-C, VisualSpeech |
<!-- FULL_PENDING:END -->

All three blocks are rebuilt with `python -m multiverse.experiments.tables`.
