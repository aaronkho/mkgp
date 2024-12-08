#!/usr/bin/env python

import os
import sys
import inspect
import numpy as np
import cProfile as profile
import pstats

# Assumes this is already installed
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic, Product

# Assumes this is already installed via 'pip install'
from gpr1d.core.kernels import RQ_Kernel
from gpr1d.core.routines import GaussianProcessRegression1D

def generate_sample_raw_data(n_points, x_spread, y_spread, intercept, slopeA, slopeB, boundaryA, boundaryB):

    x_values = np.linspace(0.0, 1.0, n_points)
    y_values = slopeA * x_values + intercept
    boundaryA_filter = (x_values >= boundaryA)
    y_values[boundaryA_filter] = y_values[boundaryA_filter] - slopeB * (x_values[boundaryA_filter] - boundaryA)
    boundaryB_filter = (x_values >= boundaryB)
    y_values[boundaryB_filter] = y_values[boundaryB_filter] + (slopeB + slopeA) * (x_values[boundaryB_filter] - boundaryB)

    # Add random error to generated data points
    raw_x_values = x_values + x_spread * np.random.randn(x_values.size)
    raw_y_values = y_values + y_spread * np.random.randn(y_values.size)
    raw_x_errors = np.full(raw_x_values.shape, x_spread)
    raw_y_errors = np.full(raw_y_values.shape, y_spread)
    raw_intercept = raw_y_values[0]

    return raw_x_values, raw_y_values, raw_x_errors, raw_y_errors, raw_intercept


def perform_gpr1d_with_profiler(n_fit_points, raw_x_values, raw_y_values, raw_y_errors):

    x_values = raw_x_values.copy()
    y_values = raw_y_values.copy()
    y_errors = raw_y_errors.copy()

    fit_x_values = np.linspace(0.0, 1.0, n_fit_points)
    normalized_y_errors = np.full(y_values.shape, np.mean(y_errors))

    # Define a kernel to fit the data itself
    #     Rational quadratic kernel is usually robust enough for general fitting
    kernel = RQ_Kernel(1.0e0, 1.0e0, 5.0e0)

    # This is only necessary if using kernel restart option on the data fitting
    kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, 1.0e-1, 1.0e0], [1.0e1, 1.0e0, 2.0e1]])

    # Print source location for reference, in case of multiple installations
    location = inspect.getsourcefile(type(kernel))
    print("Using GPR1D definition from: %s" % (os.path.dirname(location) + '/'))

    # GPR fit using y-errors only as weights
    #     Create class object to store raw data, kernels, and settings
    gpr_object = GaussianProcessRegression1D()

    #     Define the kernel and regularization parameter to be used in the data fitting routine
    gpr_object.set_kernel(kernel=kernel, kbounds=kernel_hyppar_bounds, regpar=1.0)

    #     Define the raw data and associated errors to be fitted
    gpr_object.set_raw_data(xdata=x_values, ydata=y_values, yerr=normalized_y_errors)

    #     Define the search criteria for data fitting routine and error fitting routine
    gpr_object.set_search_parameters(epsilon=1.0e-2, method='adam', spars=[1.0e-1, 0.4, 0.8])

    #     Perform the fit with kernel restarts and profile the operation
    prof = profile.Profile()
    prof.enable()
    gpr_object.GPRFit(fit_x_values, hsgp_flag=False, nrestarts=5)
    prof.disable()

    #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
    gp_kernel_name, gp_kernel_hyppars, gp_fit_regpar = gpr_object.get_gp_kernel_details()

    #     Grab fit results
    fit_y_values, fit_y_errors, fit_dydx_values, fit_dydx_errors = gpr_object.get_gp_results()

    #     Grab the log-marginal-likelihood of fit
    fit_lml = gpr_object.get_gp_lml()

    #    Grab the coefficient of determination of fit
    fit_r2 = gpr_object.get_gp_r2()

    # Print results
    gp_str = "\n--- GPR1D fit details ---\n\n"
    gp_str = gp_str + "Kernel name: %30s\n" % (gp_kernel_name)
    gp_str = gp_str + "Regularization parameter: %17.4f\n" % (gp_fit_regpar)
    gp_str = gp_str + "Optimized kernel hyperparameters:\n"
    for hh in range(gp_kernel_hyppars.size):
        gp_str = gp_str + "%15.6e" % (gp_kernel_hyppars[hh])
    gp_str = gp_str + "\n\n"
    gp_str = gp_str + "Log-marginal-likelihood: %18.6f\n" % (fit_lml)
    gp_str = gp_str + "R-squared: %32.6f\n" % (fit_r2)
    print(gp_str)

    # Print profile
    print("\n--- Profiler results ---\n\n")
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10) # top 10 rows


def perform_sklearn_with_profiler(n_fit_points, raw_x_values, raw_y_values, raw_y_errors):

    x_values = raw_x_values.reshape(-1, 1)
    y_values = raw_y_values.copy()
    y_errors = raw_y_errors.copy()

    fit_x_values = np.linspace(0.0, 1.0, n_fit_points).reshape(-1, 1)
    normalized_y_errors = np.full(y_values.shape, np.mean(y_errors)**2.0)

    # Define a kernel to fit the data itself
    #     Rational quadratic kernel is usually robust enough for general fitting
    kernel = Product(ConstantKernel(1.0e0, (1.0e-2, 1.0e2)), RationalQuadratic(1.0e0, 5.0e0, (1.0e-2, 1.0e2), (1.0e-1, 1.0e3)))
    gpr_setup = GaussianProcessRegressor(kernel=kernel, alpha=normalized_y_errors, n_restarts_optimizer=5, normalize_y=True, copy_X_train=True)

    prof = profile.Profile()
    prof.enable()
    gpr_object = gpr_setup.fit(x_values, y_values)
    prof.disable()

    #    Grab fit results
    fit_y_values, fit_y_errors = gpr_object.predict(fit_x_values, return_std=True)

    #     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
    parameters = gpr_object.kernel_.get_params(deep=True)
    gp_kernel_name = type(gpr_object.kernel_).__name__
    gp_kernel_hyppars = np.array([parameters["k1__constant_value"], parameters["k2__length_scale"], parameters["k2__alpha"]])
    gp_fit_regpar = 1.0

    #    Grab the log-marginal-likelihood of fit
    fit_lml = gpr_object.log_marginal_likelihood()

    #    Grab the coefficient of determination of fit
    fit_r2 = gpr_object.score(x_values, y_values)

    # Print results
    gp_str = "\n--- sklearn fit ---\n\n"
    gp_str = gp_str + "Kernel name: %30s\n" % (gp_kernel_name)
    gp_str = gp_str + "Regularization parameter: %17.4f\n" % (gp_fit_regpar)
    gp_str = gp_str + "Optimized kernel hyperparameters:\n"
    for hh in range(gp_kernel_hyppars.size):
        gp_str = gp_str + "%15.6e" % (gp_kernel_hyppars[hh])
    gp_str = gp_str + "\n\n"
    gp_str = gp_str + "Log-marginal-likelihood: %18.6f\n" % (fit_lml)
    gp_str = gp_str + "R-squared: %32.6f\n" % (fit_r2)
    print(gp_str)

    # Print profile
    print("\n--- Profiler results ---\n\n")
    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10) # top 10 rows


def run_profiler():
    """
    A script for profiling the GPR1D module against sklearn.

    Due to the iterative nature of the optimization method and the
    random nature of the kernel restart function, the results may
    not be exactly the same for consecutive runs of this demo
    script.
    """

    print("\n----- Starting profiling test -----\n")

    # Make 21-point test data vector
    x_values_21, y_values_21, x_errors_21, y_errors_21, intercept_21 = generate_sample_raw_data(21, 0.01, 0.25, 3.0, 1.0, 16.0, 0.3, 0.7)
    perform_gpr1d_with_profiler(301, x_values_21, y_values_21, y_errors_21)
    perform_sklearn_with_profiler(301, x_values_21, y_values_21, y_errors_21)

    # Make 51-point test data vector
    x_values_51, y_values_51, x_errors_51, y_errors_51, intercept_51 = generate_sample_raw_data(51, 0.01, 0.1, -2.0, 3.0, 10.0, 0.4, 0.75)
    perform_gpr1d_with_profiler(301, x_values_51, y_values_51, y_errors_51)
    perform_sklearn_with_profiler(301, x_values_51, y_values_51, y_errors_51)

    # Make 101-point test data vector
    x_values_101, y_values_101, x_errors_101, y_errors_101, intercept_101 = generate_sample_raw_data(101, 0.005, 0.2, 1.3, -2.0, 8.0, 0.25, 0.6)
    perform_gpr1d_with_profiler(301, x_values_101, y_values_101, y_errors_101)
    perform_sklearn_with_profiler(301, x_values_101, y_values_101, y_errors_101)

    print("\n----- Finished profiling test -----\n")


def main():

    run_profiler()


if __name__ == "__main__":

    main()
