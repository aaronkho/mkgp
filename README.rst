mkgp
====

These classes and routines were developed by Aaron Ho, and this
project repository was started in 2017. The underlying
mathematics was founded on the book, "Gaussian Process for Machine
Learning", C.E. Rasmussen, C.K.I. Williams (2006).

When using this package in any research work, please cite:
A. Ho et al 2019 Nucl. Fusion 59 056007, `DOI: 10.1088/1741-4326/ab065a
<https://doi.org/10.1088/1741-4326/ab065a>`_

Note that the package has been renamed from :code:`GPR1D` to
:code:`mkgp` in v3.0.0.


Installing the mkgp program
---------------------------

Installation is **mandatory** for this package!

For first time users, it is strongly recommended to use the GUI
developed for this Python package. To obtain the Python package
dependencies needed to use this capability, install this package
by using the following on the command line::

    pip install [--user] mkgp[gui]

Use the :code:`--user` flag if you do not have root access on the
system that you are working on. If you have already cloned the
repository, enter the top level of the repository directory and
use the following instead::

    pip install [--user] -e .[gui]

Removal of the :code:`[gui]` portion will no longer check for the
:code:`pyqt5` and :code:`matplotlib` packages needed for this
functionality. However, these packages are not crucial for the
base classes and algorithms.

To test the installation, execute the command line script::

    mkgp_1d_demo

This demonstration benefits from having :code:`matplotlib`
installed, but is not required.


Documentation
=============

Documentation of the equations used in the algorithm, along with
the available kernels and optimizers, can be found in docs/.
Documentation of the :code:`mkgp` module can be found on
`GitLab pages <https://aaronkho.gitlab.io/mkgp>`_


Using the gpr1d program
-----------------------

For those who wish to include the functionality of this package
into their own Python scripts, a sample script is provided in
:code:`src/mkgp/scripts/demo.py`. The basic syntax used to create
kernels, select optimizers, and perform the GP regression fits are
outlined there.

For any questions or to report bugs, please do so through the
proper channels in the GitLab repository.


*Important note for users!*

The following runtime warnings are common within this routine::

    RuntimeWarning: overflow encountered in double_scalars
    RuntimeWarning: invalid value encountered in true_divide
    RuntimeWarning: invalid value encountered in sqrt


They are filtered out by default but may reappear if verbosity
settings are modified. They normally occur when using the kernel
restarts option (as in the demo) and do not necessarily mean that
the final returned fit is poor.

Plotting the returned fit and errors is the recommended way to
check its quality. The log-marginal-likelihood metric can also
be used, but is only valuable when comparing different fits of
the same data, i.e. its absolute value is meaningless.

From v1.1.1, the adjusted R\ :sup:`2` and pseudo R\ :sup:`2`
metrics are now available. The adjusted R\ :sup:`2` metric provides
a measure of how close the fit is to the input data points. The
pseudo R\ :sup:`2` provides a measure of this closeness accounting
for the input data uncertainties.
