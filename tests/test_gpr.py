#!/usr/bin/env python

import pytest
import numpy as np
import GPR1D


@pytest.mark.usefixtures("empty_gpr_object")
class TestGPRUninitialized(object):

    def test_empty_raw_data(self,empty_gpr_object):
        assert empty_gpr_object.get_raw_data() == (None,None,None,None,None,None,None)

    def test_empty_processed_data(self,empty_gpr_object):
        assert empty_gpr_object.get_processed_data() == (None,None,None,None,None,None)

    def test_empty_x(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_x() is None

    def test_empty_mean(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_mean() is None

    def test_empty_variance(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_variance() is None

    def test_empty_std(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_std() is None

    def test_empty_derivative_mean(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_drv_mean() is None

    def test_empty_derivative_variance(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_drv_variance() is None

    def test_empty_derivative_std(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_drv_std() is None

    def test_empty_results(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_results() == (None,None,None,None)

    def test_empty_error_mean(self,empty_gpr_object):
        assert empty_gpr_object.get_error_gp_mean() is None

    def test_empty_error_variance(self,empty_gpr_object):
        assert empty_gpr_object.get_error_gp_variance() is None

    def test_empty_error_std(self,empty_gpr_object):
        assert empty_gpr_object.get_error_gp_std() is None

    def test_empty_log_marginal_likelihood(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_lml() is None

    def test_empty_null_log_marginal_likelihood(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_null_lml() is None

    def test_empty_adjusted_r_squared(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_adjusted_r2() is None

    def test_empty_generalized_r_squared(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_generalized_r2() is None

    def test_empty_input_kernel(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_input_kernel() is None

    def test_empty_kernel(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_kernel() is None

    def test_empty_kernel_details(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_kernel_details() == (None,None,None)

    def test_empty_error_kernel(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_error_kernel() is None

    def test_empty_error_kernel_details(self,empty_gpr_object):
        assert empty_gpr_object.get_gp_error_kernel_details() == (None,None,None)

    def test_empty_error_function(self,empty_gpr_object):
        assert empty_gpr_object.eval_error_function([0.0]) is None


#@pytest.mark.usefixtures("basic_gpr_object")
#class TestGPREvaluation(object):

#@pytest.mark.usefixtures("basic_gpr_object")
#class TestGPRSampling(object):

#@pytest.mark.usefixtures("basic_gpr_object","simplified_gpr_object")
#class TestGPRSimplifiedVersion(object):

#@pytest.mark.usefixtures("advanced_gpr_object")
#class TestGPROptimization(object):

#@pytest.mark.usefixtures("advanced_gpr_object")
#class TestGPRWithHeteroscedasticError(object):

#@pytest.mark.usefixtures("advanced_gpr_object")
#class TestGPRWithNoisyInput(object):
