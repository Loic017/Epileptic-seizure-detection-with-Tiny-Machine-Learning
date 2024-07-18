# Epileptic seizure-detection with Tiny Machine Learning: A dissertation study

Epileptic seizure detection with Tiny Machine Learning: a novel embedded algorithm that can detect three types of common seizures (absence, tonic-clonic, generalised non-specific) directly on a wearable device by employing a Tiny Machine Learning framework.

## Links

### View further information about this project or other projects on my website or on my hashnode:
- https://blog.lorentel.com/tinyml-seizure-detection
- https://lorentel.com

### View the final report:
- https://drive.google.com/file/d/1W1lrBvBA_GrSyujQJyJSLgKLstS-cuaG/view?usp=sharing

## Acknowledgements

This project was supervised by Nhat Pham.

## Publications

1. The preliminary results of this study was accepted into the 2024 MobiUK Wearable and Ubiquitous Systems Research Symposium and was presented:

```
Epileptic seizure detection with Tiny Machine Learning
Loic Lorente Lemoine, Nhat Pham, MobiUK 2024 - Sixth UK Mobile, Wearable and Ubiquitous Systems Research Symposium.
```

## Awards

1. A short poster featuring the results of the study was submitted to a Cardiff University and Vietnam National University Student Poster Competition on “AI, Smart Healthcare, and IoT” and won first prize:

```
First Prize, Student Poster Competition on “AI, Smart Healthcare, and IoT”
Cardiff University and Vietnam National University- Ho Chi Minh University of Technology
```

2. I was presented with the "Best Final Year Project" award at my graduation:

```
Best Final Year Project, BSc Computer Science and Variants”
Cardiff University
```

# Overview

This repository contains all files for the project, excluding raw EEG signals and datasets. Three models for each seizure including one combined model are trained, validated and tested. The models are converted to TFLite and deployed onto an Arduino. Quantisation are applied to reduce the model size.

The original raw data is not included within this repository. The handling of EEG data should be compatible with other EEG datasets. Some tweaking might have to be made to the code, such as adapating the annotation system, in order to use other datasets.

# Report and Results

A report was written to document this project as part of the dissertation. This report highlights the complete development process of this study, including the motivation, aims and objectives, study background, methodology, implementation and results.
https://drive.google.com/file/d/1W1lrBvBA_GrSyujQJyJSLgKLstS-cuaG/view?usp=sharing

Model results and demonstration video(s) _will be_ available soon.

# Code Highlights

-   Model classes are stored in the "SZModel.py" file. This file contains alternate versions of the model.
-   The arduino folder stores the arduino files, including the deployed models and samples as header files.

# Notice

The readme and presentation of the code within this repository will be improved for better understanding, readibility and adaptability. This is to ensure that this study and its code can be understood and adopted by others. **The report or any dissertation element will not reflect any further updates as this repository will/has receive(d) updates after the submission of the dissertation. It is not part of the assessment.**
