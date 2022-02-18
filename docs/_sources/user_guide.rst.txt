================
 User guide
================


1-Intro to Niphlem
====================

1.1- What is Niphlem?
------------------------

*Niphlem* is a toolbox originated to preprocess phsyiological signal collected during MRI scannings. It is able to estimate from these recording covariates that can be later used in  general linear model (GLM) analyses of fMRI data.

It is written in Python, with the aim of being simple and flexible. Such flexibility thus assumes a very basic knowledge of Python. If you do not feel confortable or are not familiar with this programming language, we recommed that you learn at least the most basic notions. For example, a good source for this may be the `scipy lecture notes <http://scipy-lectures.org/>`_.

Alternatively, Matlab users may be more interested in using PhLEM, which is what *niphlem* is based on. Check `PhLEM Toolbox <https://sites.google.com/site/phlemtoolbox/>`_ for more details on this toolbox.

1.2- Installing niphlem
------------------------

*niphlem* can be easily installed through pypi as follows::

    pip install -U niphlem

The dependencies, beyond those libraries that come with python distributions, are numpy, matplotlib, pandas,scipy, scikit-learn and outlier_utils. All of these are checked and installed if missing when installing *niphlem*.

Alternatively, if you are interested in installing the latest version under development, you may clone the github repository and install it from there directly::

    git clone https://github.com/CoAxLab/niphlem.git
    cd niphlem
    pip install -U .


2-Physiological Signal
======================

*Niphlem* is particularly designed to exctract physiological signal (cardiac, respiration) from usually electrocardiogram (ECG), pneumatic belt or pulse oximeter (pulse-ox) data recordings.

Right now, *niphlem* is able to work with physiological data in the form of log files acquired through a Multi-Band accelerated EPI Pulse sequence (visit `<https://www.cmrr.umn.edu/multiband>`_ for more details), and data in BIDS compliance. In the future, more formats of input physiological data will be incorporated.

2.1- Loading physiological data from CMRR MB sequences
------------------------------------------------------------

Under construction...

2.2- Loading physiological data in BIDS format
------------------------------------------------------------

Under construction...

3-RETROICOR Models
====================

RETROICOR stands for Retrospective Image Correction and it calculates the fourier series expansion using the phases from the cardiac and respiratory cycles.

The algorithm in these models works as follows:

1. Extract the peaks in the physiological data. In the case of cardiac, signals these would correspond to the R components of the QRS complex. In the pneumatic belt signal, they would correspnd to the peak expansion of the diaphragm. And for pulse-ox signals, the maxima in local blood oxygenation.

2. Estimation of a phase time to be between 0 and 2π between consecutive time peaks :math:`t_1` and :math:`t_2`, i.e., as having full phase cycles:

.. math::

   \phi(t) = \frac{2\pi (t - t_1)}{(t_2-t_1)}

3. Computation of sines and cosines of these phases up to a specified order, i.e. :math:`[sin(k\cdot\phi(t)), cos(k\cdot\phi(t))]`, with :math:`k=1,2,\dots`
4. Downsampling each of these terms to the scanner time resolution.

After these four steps, we obtain a series of phase regressors that we can use in our fMRI models to account for the physiological signal.



4-Variation Models
====================

Under construction...

5-Reports
====================

Under construction...