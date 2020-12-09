# Building a preprocessing tool for physiological confounds in fMRI

This project aims at building a preprocessing tool for extracting confounders from physiological (ECG, Respiratory and Pulse) signals. 

These cleaned and filtered confounders are then to be modeled as covariate vectors and included in a design matrix for GLM analyses of fMRI data.


Goals to achieve in this project during brainhack:

- Read physiological signal from the input data files.
- Temporally filter the signals to remove artifacts (e.g., movement, gradient noise).
- Provide a summary output (including visualizations) of physiological signals for inspection.
- Provide quality metrics of signal to noise.
- Output filtered signals into [BIDS compliant format](https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/06-physiological-and-other-continuous-recordings.html).


**This tool will be part of a bigger toolbox that is under development (pyPhlem). Therefore, all contributors will be accordingly acknolewged.**

References:

- Verstynen TD. Physiological Log Extraction for Modeling (PhLEM) Toolbox. https://sites.google.com/site/phlemtoolbox/
- Verstynen TD, Deshpande V. Using pulse oximetry to account for high and low frequency physiological artifacts in the BOLD signal. Neuroimage. 2011 Apr 15;55(4):1633-44.
- Chang C, Cunningham JP, Glover GH. Influence of heart rate on the BOLD signal: the cardiac response function. Neuroimage. 2009 Feb 1;44(3):857-69.
- Birn RM, Smith MA, Jones TB, Bandettini PA. The respiration response function: the temporal dynamics of fMRI signal fluctuations related to changes in respiration. Neuroimage. 2008;40(2):644-654.

