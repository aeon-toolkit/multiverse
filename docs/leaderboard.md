# Leaderboards

The leaderboards can be interactively generated on the WEBSITE. These are some 
illustrative static leaderboards. 

## Multiverse-core


## EEG archive

The EEG archive is a collection of EEG classification problems, described in [1]. On 
release, it contains 30 datasets. Two of these are univariate and two are not 
available on zenodo. The resulting list is contained in the multiverse

eeg = [
    "Alzheimers",
    "Blink",
    "ButtonPress",
    "EpilepticSeizures",
    "EyesOpenShut",
    "FaceDetection",
    "FeedbackButton",
    "FeetHands",
    "FingerMovements",
    "HandMovementDirection",
    "ImaginedFeetHands",
    "ImaginedOpenCloseFist",
    "InnerSpeech",
    "LongIntervalTask",
    "LowCost",
    "MatchingPennies",
    "MindReading",
    "MotorImagery",
    "OpenCloseFist",
    "PhotoStimulation",
    "PronouncedSpeech",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "ShortIntervalTask",
    "SitStand",
    "Sleep",
    "SongFamiliarity",
    "VisualSpeech",
]



We currently have results for the train/test splits with the following classifiers.

all_classifiers  = [
    "Arsenal",
    "CNN",
    "CSP-SVM",
    "DrCIF",
    "HC2",
    "IT",
    "MRHydra",
    "R-KNN",
    "R-MDM",
    "STC",
    "SVM",
    "TDE",
]

See the paper and aeon-neuro for details of these classifier. The overall accuracy 
picture is



## UEA archive

People will still use the UEA archive, so it is worth maintaining a list for sanity 
checks. The archive contains 30 datasets, but 