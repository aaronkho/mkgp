#!/usr/bin/env python

import os
import sys
import re
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

import GPR1D   # Required import, only works after using 'pip install'

# NOTE: Due to the iterative nature of the optimization method and the
#       random nature of the kernel restart function, the results may
#       not be exactly the same for consecutive runs of this demo
#       script. However, all fits should fall within the fit error
#       ranges of each other, unless the optimization algorithm has
#       not converged.


plot_save_directory = './GPPlots/'
if not plot_save_directory.endswith('/'):
    plot_save_directory = plot_save_directory + '/'
if not os.path.isdir(plot_save_directory):
    os.makedirs(plot_save_directory)


### Generating sample data

# Make basic function data
x_spread = 0.01
y_spread = 0.25
slope1 = 1.0
x_values = np.linspace(0.0,1.0,21)
y_values = slope1 * x_values + 3.0
boundary1 = 0.3
slope2 = 16.0
boundary1_filter = (x_values >= boundary1)
y_values[boundary1_filter] = y_values[boundary1_filter] - slope2 * (x_values[boundary1_filter] - boundary1)
boundary2 = 0.7
boundary2_filter = (x_values >= boundary2)
y_values[boundary2_filter] = y_values[boundary2_filter] + (slope2 + slope1) * (x_values[boundary2_filter] - boundary2)

# Add random error to generated data points
raw_x_values = x_values + x_spread * np.random.randn(x_values.size)
raw_y_values = y_values + y_spread * np.random.randn(y_values.size)
raw_x_errors = np.full(raw_x_values.shape,x_spread)
raw_y_errors = np.full(raw_y_values.shape,y_spread)



### Fitting

fit_x_values = np.linspace(0.0,1.0,100)

# Define a kernel to fit the data itself
#     Rational quadratic kernel is usually robust enough for general fitting
kernel = GPR1D.RQ_Kernel(1.0e0,1.0e0,1.0e1)

# This is only necessary if using kernel restart option on the data fitting
kernel_hyppar_bounds = np.atleast_2d([[1.0e-1,1.0e-1,5.0e0],[1.0e1,1.0e0,2.0e1]])

# Define a kernel to fit the given y-errors, needed for rigourous estimation of fit error including data error
#     Typically a simple rational quadratic kernel is sufficient given a high regularization parameter (specified later)
#     Here, the RQ kernel is summed with a noise kernel for extra robustness and to demonstrate how to use operator kernels
error_kernel = GPR1D.Sum_Kernel(GPR1D.RQ_Kernel(1.0e0,1.0e0,1.0e0),GPR1D.Noise_Kernel(1.0e-2))

# Again, this is only necessary if using kernel restart option on the error fitting
error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1,1.0e-1,1.0e-1,1.0e-3],[1.0e1,1.0e0,1.0e1,1.0e-1]])


# GPR fit accounting only for y-errors
#     Create class object to store raw data, kernels, and settings
gpr_object = GPR1D.GaussianProcessRegression1D()

#     Define the kernel and regularization parameter to be used in the data fitting routine
gpr_object.set_kernel(kernel=kernel,kbounds=kernel_hyppar_bounds,regpar=1.0)

#     Define the kernel and regularization parameter to be used in the error fitting routine
gpr_object.set_error_kernel(kernel=error_kernel,kbounds=error_kernel_hyppar_bounds,regpar=10.0)

#     Define the raw data and associated errors to be fitted
gpr_object.set_raw_data(xdata=raw_x_values,ydata=raw_y_values,yerr=raw_y_errors,xerr=raw_x_errors, \
                        dxdata=[0.0],dydata=[0.0],dyerr=[0.0])     # Example of applying derivative constraints

#     Define the search criteria for data fitting routine and error fitting routine
gpr_object.set_search_parameters(epsilon=1.0e-2)
gpr_object.set_error_search_parameters(epsilon=1.0e-1)

#     Default optimizer is gradient ascent / descent - extremely robust but slow
#     Uncomment any of the following lines to test the recommended optimizers
#gpr_object.set_search_parameters(epsilon=1.0e-3,method='adadelta',spars=[1.0e-3,0.5])
#gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])

#     Perform the fit with kernel restarts
gpr_object.GPRFit(fit_x_values,nrestarts=5)

#     Grab optimized kernel settings - easy way to minimize data storage requirements for fit reproduction
(gp_kernel_name,gp_kernel_hyppars,gp_fit_regpar) = gpr_object.get_gp_kernel_details()
(gp_error_kernel_name,gp_error_kernel_hyppars,gp_error_fit_regpar) = gpr_object.get_gp_error_kernel_details()

#     Grab fit results
(fit_y_values,fit_y_errors,fit_dydx_values,fit_dydx_errors) = gpr_object.get_gp_results()


# GPR fit accounting for y-errors AND x-errors
#     Procedure is nearly identical to above, except for the addition of an extra option
nigpr_object = GPR1D.GaussianProcessRegression1D()
nigpr_object.set_kernel(kernel=kernel,kbounds=kernel_hyppar_bounds,regpar=1.0)
nigpr_object.set_error_kernel(kernel=error_kernel,kbounds=error_kernel_hyppar_bounds,regpar=10.0)
nigpr_object.set_raw_data(xdata=raw_x_values,ydata=raw_y_values,yerr=raw_y_errors,xerr=raw_x_errors, \
                          dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
nigpr_object.set_search_parameters(epsilon=1.0e-2)
nigpr_object.set_error_search_parameters(epsilon=1.0e-1)

#     Uncomment any of the following lines to test the recommended optimizers
#nigpr_object.set_search_parameters(epsilon=1.0e-3,method='adadelta',spars=[1.0e-3,0.5])
#nigpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])

#     Perform the fit with kernel restarts, here is the extra option to account for x-errors in fit
nigpr_object.GPRFit(fit_x_values,nigp_flag=True,nrestarts=5)

# Grab outputs
(nigp_kernel_name,nigp_kernel_hyppars,nigp_fit_regpar) = nigpr_object.get_gp_kernel_details()
(nigp_error_kernel_name,nigp_error_kernel_hyppars,nigp_error_fit_regpar) = nigpr_object.get_gp_error_kernel_details()
(ni_fit_y_values,ni_fit_y_errors,ni_fit_dydx_values,ni_fit_dydx_errors) = nigpr_object.get_gp_results()



### Plotting

plot_sigma = 2.0

# Raw data with GPR fit and error, only accounting for y-errors
plot_raw_y_errors = plot_sigma * raw_y_errors
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(raw_x_values,raw_y_values,yerr=plot_raw_y_errors,ls='',marker='.',color='k')
ax.plot(fit_x_values,fit_y_values,color='r')
plot_fit_y_lower = fit_y_values - plot_sigma * fit_y_errors
plot_fit_y_upper = fit_y_values + plot_sigma * fit_y_errors
ax.fill_between(fit_x_values,plot_fit_y_lower,plot_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(plot_save_directory+'gp_test.png')
plt.close(fig)

# Derivative of GPR fit and error, only accounting for y-errors
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fit_x_values,fit_dydx_values,color='r')
plot_fit_dydx_lower = fit_dydx_values - plot_sigma * fit_dydx_errors
plot_fit_dydx_upper = fit_dydx_values + plot_sigma * fit_dydx_errors
ax.fill_between(fit_x_values,plot_fit_dydx_lower,plot_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(plot_save_directory+'gp_dtest.png')
plt.close(fig)

# Raw data with GPR fit and error, comparison of only y-errors against y-errors AND x-errors
plot_raw_x_errors = plot_sigma * raw_x_errors
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(raw_x_values,raw_y_values,xerr=plot_raw_x_errors,yerr=plot_raw_y_errors,ls='',marker='.',color='k')
ax.plot(fit_x_values,fit_y_values,color='r')
ax.fill_between(fit_x_values,plot_fit_y_lower,plot_fit_y_upper,facecolor='r',edgecolor='None',alpha=0.2)
ax.plot(fit_x_values,ni_fit_y_values,color='b')
plot_ni_fit_y_lower = ni_fit_y_values - plot_sigma * ni_fit_y_errors
plot_ni_fit_y_upper = ni_fit_y_values + plot_sigma * ni_fit_y_errors
ax.fill_between(fit_x_values,plot_ni_fit_y_lower,plot_ni_fit_y_upper,facecolor='b',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(plot_save_directory+'nigp_test.png')
plt.close(fig)

# Derivative of GPR fit and error, comparison of only y-errors against y-errors AND x-errors
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fit_x_values,fit_dydx_values,color='r')
ax.fill_between(fit_x_values,plot_fit_dydx_lower,plot_fit_dydx_upper,facecolor='r',edgecolor='None',alpha=0.2)
ax.plot(fit_x_values,ni_fit_dydx_values,color='b')
plot_ni_fit_dydx_lower = ni_fit_dydx_values - plot_sigma * ni_fit_dydx_errors
plot_ni_fit_dydx_upper = ni_fit_dydx_values + plot_sigma * ni_fit_dydx_errors
ax.fill_between(fit_x_values,plot_ni_fit_dydx_lower,plot_ni_fit_dydx_upper,facecolor='b',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(plot_save_directory+'nigp_dtest.png')
plt.close(fig)
