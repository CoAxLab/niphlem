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

    pip install niphlem

The dependencies, beyond those libraries that come with python distributions, are numpy, matplotlib, pandas,scipy, scikit-learn and outlier_utils. All of these are checked and installed if missing when installing *niphlem*.


2-Physiological Signal
======================

*Niphlem* handles three types of physiological signals: electrogradiogram (ECG), respiration and pulse oximeter (pulse-ox).

Right now *niphlem* is only able to use physiological data in the form of Log files acquired through a Multi-Band accelerated EPI Pulse sequence (visit `<https://www.cmrr.umn.edu/multiband>`_ for more details). In the future, more formats of input physiological data will be incorporated (e.g. BIDS format).

2.1- Loading physiological data from CMRR MB sequences
------------------------------------------------------------

Under construction...

2.2- Loading physiological data in BIDS format
------------------------------------------------------------

Soon!

3-Retroicor Models
====================

Under construction...

4-Variation Models
====================

Under construction...

5-Reports
====================

Under construction...