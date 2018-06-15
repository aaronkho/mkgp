GPR1D
=====

Installing the GPR1D program
----------------------------

*Author: Aaron Ho (01/06/2018)*

Installation is **mandatory** for this package!

For first time users, it is strongly recommended to use the GUI
developed for this Python package. To obtain the Python package
dependencies needed to use this capability, install this package
by using the following on the command line:

.. code-block:: console

    pip install [--user] -e GPR1D[guis]

Use the :code:`--user` flag if you do not have root access on the system
that you are working on. If you have already cloned the
repository, enter the top level of the repository directory and
use the following instead:

.. code-block:: console

    pip install [--user] -e .[guis]


Documentation
=============

Documentation of the equations used in the algorithm, along with
the available kernels and optimizers, can be found in docs/.
Documentation of the GPR1D module can be found on
`GitLab pages <https://aaronkho.gitlab.io/GPR1D>`_


Using the GPR1D program
-----------------------

For those who wish to include the functionality of this package
into their own Python scripts, a demo script is provided in
scripts/. The basic syntax used to create kernels, select
settings, and perform GPR fits are outlined there.

For any questions or to report bugs, please do so through the
proper channels in the GitLab repository.


** IMPORTANT note for users! **

The following runtime warnings are common within this routine,
but they are filtered out by default:

.. code-block:: console

    RuntimeWarning: overflow encountered in double_scalars
    RuntimeWarning: invalid value encountered in true_divide
    RuntimeWarning: invalid value encountered in sqrt


They normally occur when using the kernel restarts option (as
in the demo) and do not necessarily mean that the resulting
fit is poor.

Plotting the resulting fit and errors is the recommended way to
check its quality. The log-marginal-likelihood metric can also
be used, but is only valuable when comparing different fits of
the same data, ie. its absolute value is meaningless.
