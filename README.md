# niphlem

niphlem stands for Physiological Log Extraction for Modeling in Neuroimaging and is the cool (i.e. python) brother of PhLEM (https://sites.google.com/site/phlemtoolbox/) which was originary written in Matlab. 

This toolbox extracts physiological recordings during MRI scanning and estimates the signal phases so that they can be used as a covariate in your general linear model (GLM) with fMRI data.

niphlem can generate multiple models of physiological noise to include as regressors in your GLM model from either ECG, pneumatic breathing belt or pulse-oximetry data.  These are described in Verstynen and Deshpande (2011).

Briefly, niphlem implements three types of models:

- *RETROICOR*:  A phasic decomposition method that isolates the fourier series that best describes the spectral properties of the input signal.  This was first described by Glover and colleagues. 
- *Variation Models*:  For low frequency signals (like the pneumatic belt and low-pass filtered pulse-oximetry) this does the combined respiration variance and response function described by Birn and colleagues (2008).  For high frequency signals (i.e., ECG or high-pass filtered pulse-oximetry), this generates the heart-rate variance and cardiac response function described by Chang and colleagues (2009).
- *Downsampled Model*: Performs a simple filtering and downsampling of a raw signal as was done for the pulse-oximetry signal in Verstynen and Deshpande (2011).

## Dependencies

## Install

## References:

- Verstynen TD. Physiological Log Extraction for Modeling (PhLEM) Toolbox. https://sites.google.com/site/phlemtoolbox/
- Verstynen TD, Deshpande V. Using pulse oximetry to account for high and low frequency physiological artifacts in the BOLD signal. Neuroimage. 2011 Apr 15;55(4):1633-44.
- Chang C, Cunningham JP, Glover GH. Influence of heart rate on the BOLD signal: the cardiac response function. Neuroimage. 2009 Feb 1;44(3):857-69.
- Birn RM, Smith MA, Jones TB, Bandettini PA. The respiration response function: the temporal dynamics of fMRI signal fluctuations related to changes in respiration. Neuroimage. 2008;40(2):644-654.

