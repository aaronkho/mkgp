#!/usr/bin/env python

import pytest
import numpy as np
from operator import itemgetter


def check_gp_results(results,cmean=None,cstd=None,cdmean=None,cdstd=None,rtol=1.0e-5,atol=1.0e-8):
    checks = [np.all(np.isclose(cmean,results[0],rtol=rtol,atol=atol)) if cmean is not None else True,
              np.all(np.isclose(cstd,results[1],rtol=rtol,atol=atol)) if cstd is not None else True,
              np.all(np.isclose(cdmean,results[2],rtol=rtol,atol=atol)) if cdmean is not None else True,
              np.all(np.isclose(cdstd,results[3],rtol=rtol,atol=atol)) if cdstd is not None else True]
    #print(np.all(np.isclose(cmean,results[0],rtol=rtol,atol=atol)))
    #print(np.all(np.isclose(cstd,results[1],rtol=rtol,atol=atol)))
    #print(np.all(np.isclose(cdmean,results[2],rtol=rtol,atol=atol)))
    #print(np.all(np.isclose(cdstd,results[3],rtol=rtol,atol=atol)))
    #print(np.isclose(cmean,results[0],rtol=rtol,atol=atol))
    #print(np.isclose(cstd,results[1],rtol=rtol,atol=atol))
    #print(np.isclose(cdmean,results[2],rtol=rtol,atol=atol))
    #print(np.isclose(cdstd,results[3],rtol=rtol,atol=atol))
    return np.all(checks)


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

    def test_empty_gp_sampling(self,empty_gpr_object):
        pytest.raises(ValueError,empty_gpr_object.sample_GP,1)

    def test_empty_gp_sampling(self,empty_gpr_object):
        pytest.raises(ValueError,empty_gpr_object.sample_GP_derivative,1)


@pytest.mark.usefixtures("unoptimized_gpr_object","linear_kernel","linear_test_data","rq_kernel")
class TestGPREvaluation(object):

    n_samples = 5
    n_stat_samples = 10000

    def test_evaluation_call(self,unoptimized_gpr_object,linear_test_data):
        xpredict = itemgetter(1)(linear_test_data)
        assert unoptimized_gpr_object.GPRFit(xpredict,hsgp_flag=False,nigp_flag=False) is None

    def test_input_kernel_equivalence(self,unoptimized_gpr_object,linear_kernel):
        assert unoptimized_gpr_object.get_gp_input_kernel() == linear_kernel

    def test_unchanged_kernel_without_optimization(self,unoptimized_gpr_object,linear_kernel):
        assert unoptimized_gpr_object.get_gp_kernel() == linear_kernel

    def test_eval_without_optimization(self,unoptimized_gpr_object,linear_test_data):
        ref_targets = itemgetter(2)(linear_test_data)
        #print(unoptimized_gpr_object.get_gp_results())
        assert check_gp_results(unoptimized_gpr_object.get_gp_results(),ref_targets[0,:],ref_targets[1,:],ref_targets[2,:],ref_targets[3,:])

    def test_cost_function_without_optimization(self,unoptimized_gpr_object,linear_test_data):
        (ref_lml,ref_null_lml) = itemgetter(3,4)(linear_test_data)
        #print(unoptimized_gpr_object.get_gp_lml(),unoptimized_gpr_object.get_gp_null_lml())
        assert np.isclose(unoptimized_gpr_object.get_gp_lml(),ref_lml) and np.isclose(unoptimized_gpr_object.get_gp_null_lml(),ref_null_lml)

    def test_non_existant_error_function_without_optimization(self,unoptimized_gpr_object):
        assert unoptimized_gpr_object.eval_error_function([0.0]) is None

    def test_sampling(self,unoptimized_gpr_object):
        assert unoptimized_gpr_object.sample_GP(self.n_samples).shape == (self.n_samples,31)

#   Test not yet operational, something strange in the neighbourhood
#    def test_sampling_statistics(self,unoptimized_gpr_object):
#        stats = unoptimized_gpr_object.sample_GP(self.n_stat_samples,simple_out=True)
#        assert check_gp_results(unoptimized_gpr_object,stats[0,:],stats[1,:],rtol=1.0e-2,atol=1.0e-2)

    def test_eval_heteroscedastic_without_optimization(self,unoptimized_gpr_object,linear_test_data,rq_kernel):
        (xpredict,hs_targets,hs_lml,hs_null_lml) = itemgetter(1,5,6,7)(linear_test_data)
        unoptimized_gpr_object.set_error_kernel(kernel=rq_kernel,regpar=1.0)
        unoptimized_gpr_object.GPRFit(xpredict,hsgp_flag=True,nigp_flag=False)
        #print(unoptimized_gpr_object.get_gp_results())
        #print(unoptimized_gpr_object.get_gp_lml(),unoptimized_gpr_object.get_gp_null_lml())
        assert check_gp_results(unoptimized_gpr_object.get_gp_results(),hs_targets[0,:],hs_targets[1,:],hs_targets[2,:],hs_targets[3,:])
        assert np.isclose(unoptimized_gpr_object.get_gp_lml(),hs_lml) and np.isclose(unoptimized_gpr_object.get_gp_null_lml(),hs_null_lml)

    def test_eval_noisy_input_without_optimization(self,unoptimized_gpr_object,linear_test_data,rq_kernel):
        (xpredict,ni_targets,ni_lml,ni_null_lml) = itemgetter(1,8,9,10)(linear_test_data)
        unoptimized_gpr_object.set_error_kernel(kernel=rq_kernel,regpar=1.0)
        unoptimized_gpr_object.GPRFit(xpredict,hsgp_flag=False,nigp_flag=True)
        #print(unoptimized_gpr_object.get_gp_results())
        #print(unoptimized_gpr_object.get_gp_lml(),unoptimized_gpr_object.get_gp_null_lml())
        assert check_gp_results(unoptimized_gpr_object.get_gp_results(),ni_targets[0,:],ni_targets[1,:],ni_targets[2,:],ni_targets[3,:])
        assert np.isclose(unoptimized_gpr_object.get_gp_lml(),ni_lml) and np.isclose(unoptimized_gpr_object.get_gp_null_lml(),ni_null_lml)

    def test_eval_double_error_without_optimization(self,unoptimized_gpr_object,linear_test_data,rq_kernel):
        (xpredict,xy_targets,xy_lml,xy_null_lml) = itemgetter(1,11,12,13)(linear_test_data)
        unoptimized_gpr_object.set_error_kernel(kernel=rq_kernel,regpar=1.0)
        unoptimized_gpr_object.GPRFit(xpredict,hsgp_flag=True,nigp_flag=True)
        #print(unoptimized_gpr_object.get_gp_results())
        #print(unoptimized_gpr_object.get_gp_lml(),unoptimized_gpr_object.get_gp_null_lml())
        assert check_gp_results(unoptimized_gpr_object.get_gp_results(),xy_targets[0,:],xy_targets[1,:],xy_targets[2,:],xy_targets[3,:])
        assert np.isclose(unoptimized_gpr_object.get_gp_lml(),xy_lml) and np.isclose(unoptimized_gpr_object.get_gp_null_lml(),xy_null_lml)


@pytest.mark.usefixtures("simplified_unoptimized_gpr_object","unoptimized_gpr_object","linear_test_data")
class TestGPRSimplifiedVersion(object):

    def test_equivalence_to_standard(self,simplified_unoptimized_gpr_object,unoptimized_gpr_object):
        error_kernel = simplified_unoptimized_gpr_object.get_gp_error_kernel()
        unoptimized_gpr_object.set_search_parameters(epsilon='None',method='adam',spars=[1.0e-2,0.4,0.8])
        unoptimized_gpr_object.set_error_kernel(kernel=error_kernel,kbounds=error_kernel.bounds,regpar=5.0,nrestarts=0)
        unoptimized_gpr_object.set_error_search_parameters(epsilon='None',method='adam',spars=[1.0e-2,0.4,0.8])
        assert simplified_unoptimized_gpr_object == unoptimized_gpr_object

    def test_eval_equivalence(self,simplified_unoptimized_gpr_object,linear_test_data):
        simplified_unoptimized_gpr_object._perform_heterogp = False
        simplified_unoptimized_gpr_object._perform_nigp = False
        (xpredict,ref_targets) = itemgetter(1,2)(linear_test_data)
        assert check_gp_results(simplified_unoptimized_gpr_object(xpredict),ref_targets[0,:],ref_targets[1,:],ref_targets[2,:],ref_targets[3,:])

    def test_sampling(self,simplified_unoptimized_gpr_object):
        assert simplified_unoptimized_gpr_object.sample(simplified_unoptimized_gpr_object.get_gp_x()).shape == (31,)


@pytest.mark.usefixtures("preoptimization_gpr_object","rq_kernel")
class TestGPROptimization(object):

    xtest = np.linspace(-1.0,1.0,5)
    rq_kbounds = np.atleast_2d([[1.0e-1,1.0e-1,5.0e0],[1.0e0,5.0e-1,2.0e1]])

    def test_proper_kernel(self,preoptimization_gpr_object,rq_kernel):
        assert preoptimization_gpr_object.get_gp_input_kernel() == rq_kernel

    def test_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='grad',spars=[1.0e-5])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_gradient_descent_with_momentum_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='mom',spars=[1.0e-4,0.9])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_nesterov_accelerated_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='nag',spars=[1.0e-4,0.9])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_adaptive_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adagrad',spars=[1.0e-2])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_decayed_accumulation_adaptive_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adadelta',spars=[1.0e-2,0.9])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_adaptive_moment_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-2,0.9,0.999])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_adaptive_moment_gradient_descent_with_l_infinity_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adamax',spars=[1.0e-3,0.9,0.999])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_nesterov_accelerated_adaptive_moment_gradient_descent_optimizer(self,preoptimization_gpr_object):
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='nadam',spars=[1.0e-3,0.9,0.999])
        assert preoptimization_gpr_object.get_gp_x() is None
        assert preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=False) is None
        assert preoptimization_gpr_object.get_gp_kernel() != preoptimization_gpr_object.get_gp_input_kernel()
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_heteroscedastic_with_optimization_and_restarts(self,preoptimization_gpr_object,rq_kernel):
        preoptimization_gpr_object.set_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=1.0)
        preoptimization_gpr_object.set_error_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=2.0,nrestarts=5)
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=True,nigp_flag=False,nrestarts=5)
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_noisy_input_with_optimization_and_restarts(self,preoptimization_gpr_object,rq_kernel):
        preoptimization_gpr_object.set_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=1.0)
        preoptimization_gpr_object.set_error_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=2.0,nrestarts=5)
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=False,nigp_flag=True,nrestarts=5)
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

    def test_double_error_with_optimization_and_restarts(self,preoptimization_gpr_object,rq_kernel):
        preoptimization_gpr_object.set_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=1.0)
        preoptimization_gpr_object.set_error_kernel(kernel=rq_kernel,kbounds=self.rq_kbounds,regpar=2.0,nrestarts=5)
        preoptimization_gpr_object.set_search_parameters(epsilon=1.0e-2,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.set_error_search_parameters(epsilon=1.0e-1,method='adam',spars=[1.0e-2,0.4,0.8])
        preoptimization_gpr_object.GPRFit(self.xtest,hsgp_flag=True,nigp_flag=True,nrestarts=5)
        (out_mean,out_std,out_derivative_mean,out_derivative_std) = preoptimization_gpr_object.get_gp_results()
        assert isinstance(out_mean,np.ndarray) and np.all(np.isfinite(out_mean))
        assert isinstance(out_std,np.ndarray) and np.all(np.isfinite(out_std))
        assert isinstance(out_derivative_mean,np.ndarray) and np.all(np.isfinite(out_derivative_mean))
        assert isinstance(out_derivative_std,np.ndarray) and np.all(np.isfinite(out_derivative_std))

