#!/usr/bin/env python

import pytest
import numpy as np
import GPR1D


@pytest.fixture(scope="module")
def empty_warping_function():
    return GPR1D._WarpingFunction()

@pytest.fixture(scope="module")
def constant_warping_function():
    return GPR1D.Constant_WarpingFunction(1.0)

@pytest.fixture(scope="module")
def inverse_gaussian_warping_function():
    return GPR1D.IG_WarpingFunction(0.5,0.4,0.05,0.9,0.8)

@pytest.fixture(scope="module")
def empty_kernel():
    return GPR1D._Kernel()

@pytest.fixture(scope="module")
def constant_kernel():
    return GPR1D.Constant_Kernel(2.0)

@pytest.fixture(scope="module")
def noise_kernel():
    return GPR1D.Noise_Kernel(1.0)

@pytest.fixture(scope="module")
def linear_kernel():
    return GPR1D.Linear_Kernel(2.0)

@pytest.fixture(scope="module")
def poly_order_kernel():
    return GPR1D.Poly_Order_Kernel(2.0,1.0)

@pytest.fixture(scope="module")
def se_kernel():
    return GPR1D.SE_Kernel(1.0,0.5)

@pytest.fixture(scope="module")
def rq_kernel():
    return GPR1D.RQ_Kernel(1.0,0.5,5.0)

@pytest.fixture(scope="module")
def matern_hi_kernel():
    return GPR1D.Matern_HI_Kernel(1.0,0.5,2.5)

@pytest.fixture(scope="module")
def gibbs_constant_kernel(constant_warping_function):
    return GPR1D.Gibbs_Kernel(1.0,constant_warping_function)

@pytest.fixture(scope="module")
def gibbs_inverse_gaussian_kernel(inverse_gaussian_warping_function):
    return GPR1D.Gibbs_Kernel(1.0,inverse_gaussian_warping_function)

@pytest.fixture(scope="module")
def empty_operator_kernel():
    return GPR1D._OperatorKernel()

@pytest.fixture(scope="module")
def sum_kernel(se_kernel,noise_kernel):
    return GPR1D.Sum_Kernel(se_kernel,noise_kernel)

@pytest.fixture(scope="module")
def product_kernel(linear_kernel):
    return GPR1D.Product_Kernel(linear_kernel,linear_kernel)

@pytest.fixture(scope="module")
def empty_gpr_object():
    return GPR1D.GaussianProcessRegression1D()

@pytest.fixture(scope="module")
def basic_gpr_object(linear_kernel):
    # Made to follow y = 1.9 * x with x_error = N(0,0.02) and y_error = N(0,0.2)
    xvalues = np.array([-0.96740013, -0.87351828, -0.82642837, -0.70047194, -0.59080932,
                        -0.52655153, -0.42021946, -0.30265796, -0.18695111, -0.06505908,
                         0.01570590,  0.10126372,  0.19727240,  0.28985247,  0.41903549,
                         0.54407216,  0.61490052,  0.67747007,  0.81998975,  0.90411010,
                         0.95085251])
    xerrors = np.full(xvalues.shape,0.02)
    yvalues = np.array([-1.99033519, -1.77994779, -1.81395185, -1.34285693, -1.18439277,
                        -0.76244458, -0.69355271, -0.45994556, -0.22130082, -0.17259663,
                         0.18375909,  0.20151468,  0.20361261,  0.42478122,  0.97264555,
                         0.97186543,  1.15666269,  0.96138111,  1.69555386,  1.75694618,
                         1.77774039])
    yerrors = np.full(yvalues.shape,0.2)
    gpr_object = GPR1D.GaussianProcessRegression1D()
    gpr_object.set_kernel(kernel=linear_kernel)
    gpr_object.set_raw_data(xdata=xvalues,ydata=yvalues,yerr=yerrors,xerr=xerrors)
    return gpr_object

#@pytest.fixture(scope="module")
#def simplified_gp(rq_kernel):
