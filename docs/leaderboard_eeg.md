# EEG leaderboard

The EEG classification archive, `eeg2026` in aeon, has 28 datasets. Twenty-six are
multivariate and part of the Multiverse; the other two, EpilepticSeizures and Sleep, are
univariate. This ranks estimators with a resample-0 result on all 26 multivariate EEG
datasets, from the same runs as the other leaderboards.

A sortable version, with an average rank beside every metric, is in
[`results/multiverse/leaderboard_eeg.html`](../results/multiverse/leaderboard_eeg.html)
([preview](https://raw.githack.com/aeon-toolkit/multiverse/main/results/multiverse/leaderboard_eeg.html)).

<!-- EEG_LEADERBOARD:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy | AUROC | F1 | Log loss &darr; | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|---|---|
| 1 | HC2 | **4.02** | **0.5956** | 0.5812 | **0.6952** | 0.5632 | 0.7669 | 0.5803 | **0.5935** |
| 2 | DrCIF-500 | 5.15 | 0.5949 | **0.5826** | 0.6912 | 0.5681 | 0.7670 | 0.5869 | 0.5858 |
| 3 | CIF-500 | 6.10 | 0.5866 | 0.5722 | 0.6917 | 0.5554 | **0.7655** | 0.5771 | 0.5761 |
| 4 | MRHydra | 6.44 | 0.5856 | 0.5789 | 0.6388 | **0.5747** | 14.9380 | **0.5958** | 0.5661 |
| 5 | STC | 6.50 | 0.5805 | 0.5692 | 0.6776 | 0.5598 | 0.7908 | 0.5783 | 0.5677 |
| 6 | RIST | 6.60 | 0.5823 | 0.5671 | 0.6868 | 0.5461 | 0.7753 | 0.5658 | 0.5785 |
| 7 | Arsenal | 6.77 | 0.5864 | 0.5762 | 0.6760 | 0.5604 | 4.6306 | 0.5671 | 0.5920 |
| 8 | QUANT | 6.92 | 0.5784 | 0.5699 | 0.6884 | 0.5611 | 0.7936 | 0.5885 | 0.5532 |
| 9 | ROCKET | 6.94 | 0.5837 | 0.5745 | 0.6354 | 0.5619 | 15.0033 | 0.5699 | 0.5846 |
| 10 | TDE | 8.73 | 0.5513 | 0.5381 | 0.6429 | 0.5213 | 0.8573 | 0.5457 | 0.5416 |
| 11 | LITETime-MV | 9.25 | 0.5438 | 0.5435 | 0.6443 | 0.4863 | 3.1224 | 0.5064 | 0.5776 |
| 12 | RDST | 9.46 | 0.5429 | 0.5293 | 0.5970 | 0.4969 | 16.4763 | 0.5005 | 0.5683 |
| 13 | Catch22 | 10.13 | 0.5325 | 0.5172 | 0.6224 | 0.4976 | 0.8102 | 0.5324 | 0.5124 |
| 14 | Dummy | 11.98 | 0.4409 | 0.4276 | 0.5000 | 0.3172 | 0.8893 | 0.5094 | 0.3555 |

Average over the 26 EEG datasets with results for every estimator on every metric, ordered by average accuracy rank. Best in each column in bold.
<!-- EEG_LEADERBOARD:END -->

## Not yet complete

Estimators in the Multiverse-core table that do not yet have a result on all 26 EEG
datasets, so are not ranked above. The same gaps are in
[`results/multiverse/pending_eeg.csv`](../results/multiverse/pending_eeg.csv), one
`estimator,dataset` row per missing run.

<!-- EEG_PENDING:START -->
| Estimator | Completed | Missing datasets |
|---|---|---|
| H-InceptionTime | 24 of 26 | ShortIntervalTask, SitStand |
| PatchMTSC | 12 of 26 | ButtonPress, FeedbackButton, FeetHands, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| TimesNet | 12 of 26 | ButtonPress, FeedbackButton, FeetHands, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| TimesURL | 12 of 26 | ButtonPress, FeedbackButton, FeetHands, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| CIF | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| ConvTran | 10 of 26 | Alzheimers, ButtonPress, FeedbackButton, FeetHands, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PhotoStimulation, PronouncedSpeech, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| DisjointCNN | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| DrCIF | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| STSF | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| Summary | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| TS2Vec | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| TSF | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
| XCM | 10 of 26 | ButtonPress, FeedbackButton, FeetHands, FingerMovements, ImaginedFeetHands, ImaginedOpenCloseFist, InnerSpeech, LongIntervalTask, MatchingPennies, OpenCloseFist, PronouncedSpeech, SelfRegulationSCP2, ShortIntervalTask, SitStand, SongFamiliarity, VisualSpeech |
<!-- EEG_PENDING:END -->

## EEG archive study results

These are the averaged results held in [`results/eeg`](../results/eeg), for twelve
estimators including the EEG-specific CSP-SVM, R-KNN and R-MDM. They come from a separate
study, not the resample-0 runs above, so the two tables are not directly comparable.
Their 26 datasets also differ: they include the univariate Sleep and leave out
FeedbackButton.

<!-- EEG_ARCHIVE:START -->
| # | Estimator | Accuracy rank | Accuracy | Balanced accuracy |
|---|---|---|---|---|
| 1 | HC2 | **3.21** | **0.5929** | **0.5721** |
| 2 | DrCIF | 4.52 | 0.5856 | 0.5661 |
| 3 | Arsenal | 4.71 | 0.5799 | 0.5675 |
| 4 | STC | 4.87 | 0.5771 | 0.5617 |
| 5 | MRHydra | 5.62 | 0.5685 | 0.5556 |
| 6 | SVM | 6.48 | 0.5443 | 0.5180 |
| 7 | TDE | 6.96 | 0.5481 | 0.5273 |
| 8 | IT | 7.33 | 0.5344 | 0.5283 |
| 9 | CNN | 8.27 | 0.5119 | 0.4917 |
| 10 | CSP-SVM | 8.46 | 0.5076 | 0.4802 |
| 11 | R-KNN | 8.62 | 0.4779 | 0.4684 |
| 12 | R-MDM | 8.96 | 0.4690 | 0.4679 |

Average over the 26 datasets in `results/eeg`, ordered by average accuracy rank. Best in each column in bold.
<!-- EEG_ARCHIVE:END -->

All four blocks are rebuilt with `python -m multiverse.experiments.tables`.
