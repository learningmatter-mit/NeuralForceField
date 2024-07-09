"""
aRMSD
==================================
aRMSD is a Python module/program that allows for a fast and comprehensive
analysis of molecular structures parsed from different files. Unique features
are a support of crystallographic data, error propagation and specific types
of graphical representations.

Key features:
* Parses data from various file formats
* Establishes consistency and matches coordinate sequences of two molecules
* Aligns two molecular structures based on the Kabsch algorithm
* Supports different weighting functions for the superposition
* Supports error propagation for experimental structures
* Generates different visualization types of the superposition results
* Writes outfiles that can be passed to other programs

Dependencies:
* numpy
* matplotlib
* vtk
* uncertainties
* [Cython]
* [openbabel/pybel]

*** IMPORTANT NOTICE ***
It is highly recommended to install all dependencies directly from
python wheels prior to the installation of aRMSD (especially numpy and vtk)

* Recent changes are listed in CHANGELOG.md

* This project is hosted on GitHub: https://github.com/armsd/aRMSD

* Documentation: http://armsd.rtfd.io

(c) 2016 by Arne Wagner <arne.wagner@aci.uni-heidelberg.de>.
Please send feature requests, bugs and feedback to this address.

This software is released under a dual license. (1) The MIT license.
(2) Any other license, as long as it is obtained from the original
author.
"""

# Authors: Arne Wagner
# License: MIT

from __future__ import absolute_import

from aRMSD import run, __aRMSD_version__, __author__

__version__ = __aRMSD_version__
