#!/usr/bin/env python

import pytest
import numpy as np
import GPR1D

@pytest.fixture(scope="module")
def constant_warping_function():
    return GPR1D.Constant_WarpingFunction(1.0)

@pytest.fixture(scope="module")
def inverse_gaussian_warping_function():
    return GPR1D.IG_WarpingFunction(0.5,0.4,0.05,0.9,0.8)

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
def sum_kernel(se_kernel,noise_kernel):
    return GPR1D.Sum_Kernel(se_kernel,noise_kernel)

@pytest.fixture(scope="module")
def product_kernel(linear_kernel):
    return GPR1D.Product_Kernel(linear_kernel,linear_kernel)

@pytest.fixture(scope="module")
def generic_gpr_object():
    return GPR1D.GaussianProcessRegression1D()

#@pytest.fixture(scope="module")
#def simplified_gp(rq_kernel):
