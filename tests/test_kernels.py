#!/usr/bin/env python

import pytest
import numpy as np
from gpr1d.core.baseclasses import _Kernel
from gpr1d.core.utils import KernelConstructor, KernelReconstructor


def check_kernel_evaluation(kernel, x1, x2, der, comparison):
    return np.all(np.isclose(kernel(x1, x2, der=der), comparison))

def check_kernel_transposition(kernel, x1, x2):
    return np.all(np.isclose(kernel(x1, x2, der=1), np.transpose(kernel(x1, x2, der=-1))))

def check_kernel_hyperparameter_derivatives(kernel, x1, x2, comparison):
    hderiv_tests = [False] * len(kernel.hyperparameters)
    if len(comparison) == len(hderiv_tests):
        for ihyp in np.arange(0, len(hderiv_tests)):
            hderiv_tests[ihyp] = np.all(np.isclose(kernel(x1, x2, hder=ihyp), comparison[ihyp]))
    return np.all(hderiv_tests)


@pytest.mark.usefixtures('empty_warping_function', 'empty_kernel', 'empty_operator_kernel')
class TestBaseClasses():

    def test_empty_warping_function_defaults(self, empty_warping_function):
        assert (empty_warping_function.name,
                empty_warping_function.hyperparameters.size,
                empty_warping_function.constants.size,
                empty_warping_function.bounds,
                empty_warping_function.is_hderiv_implemented()) == ('Warp', 0, 0, None, False)

    def test_eval_empty_warping_function(self, empty_warping_function):
        pytest.raises(NotImplementedError, empty_warping_function, [0.0])

    def test_empty_kernel_defaults(self, empty_kernel):
        assert (empty_kernel.name,
                empty_kernel.hyperparameters.size,
                empty_kernel.constants.size,
                empty_kernel.bounds,
                empty_kernel.is_hderiv_implemented()) == ('None', 0, 0, None, False)

    def test_eval_empty_kernel(self, empty_kernel):
        pytest.raises(NotImplementedError, empty_kernel, [0.0], [0.0])

    def test_empty_operator_kernel_defaults(self, empty_operator_kernel):
        assert (empty_operator_kernel.name,
                empty_operator_kernel.hyperparameters.size,
                empty_operator_kernel.constants.size,
                empty_operator_kernel.bounds,
                empty_operator_kernel.is_hderiv_implemented()) == ('Op()', 0, 0, None, False)

    def test_eval_empty_operator_kernel(self, empty_operator_kernel):
        pytest.raises(NotImplementedError, empty_operator_kernel, [0.0], [0.0])


@pytest.mark.usefixtures('se_kernel', 'gibbs_inverse_gaussian_kernel', 'sum_kernel')
class TestAutoKernelCreation():

    def test_constructor_kernels(self, se_kernel):
        gpobj = KernelConstructor(se_kernel.name)
        assert isinstance(gpobj, _Kernel), f'{gpobj:!r} is not Kernel object!'

    def test_constructor_warping_functions(self, gibbs_constant_kernel):
        gpobj = KernelConstructor(gibbs_constant_kernel.name)
        assert isinstance(gpobj, _Kernel), f'{gpobj:!r} is not Kernel object!'

    def test_constructor_operators(self, sum_kernel):
        gpobj = KernelConstructor(sum_kernel.name)
        assert isinstance(gpobj, _Kernel), f'{gpobj:!r} is not Kernel object!'

    def test_reconstructor_kernels(self, se_kernel):
        kernel_pars = np.hstack((se_kernel.hyperparameters, se_kernel.constants)).flatten()
        gpobj = KernelReconstructor(se_kernel.name, kernel_pars)
        assert gpobj == se_kernel

    def test_reconstructor_warping_functions(self, gibbs_inverse_gaussian_kernel):
        kernel_pars = np.hstack((gibbs_inverse_gaussian_kernel.hyperparameters, gibbs_inverse_gaussian_kernel.constants)).flatten()
        gpobj = KernelReconstructor(gibbs_inverse_gaussian_kernel.name, kernel_pars)
        assert gpobj == gibbs_inverse_gaussian_kernel

    def test_reconstructor_operators(self, sum_kernel):
        kernel_pars = np.hstack((sum_kernel.hyperparameters, sum_kernel.constants)).flatten()
        gpobj = KernelReconstructor(sum_kernel.name, kernel_pars)
        assert gpobj == sum_kernel


@pytest.mark.usefixtures('constant_kernel')
class TestConstantKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.full(x1.shape, 2.0)
    ref_dcov = np.zeros(x1.shape)
    ref_ddcov = np.zeros(x1.shape)
    ref_hdcov = [
        np.ones(x1.shape),
    ]

    def test_eval(self, constant_kernel):
        assert check_kernel_evaluation(constant_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, constant_kernel):
        assert check_kernel_evaluation(constant_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, constant_kernel):
        assert check_kernel_transposition(constant_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, constant_kernel):
        assert check_kernel_evaluation(constant_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self,constant_kernel):
        if constant_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(constant_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, constant_kernel, **kwargs)


@pytest.mark.usefixtures('noise_kernel')
class TestNoiseKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.diag(np.power(np.ones(test_x_vector.shape), 2.0))
    ref_dcov = np.zeros(x1.shape)
    ref_ddcov = np.zeros(x1.shape)
    ref_hdcov = [
        np.diag(2.0 * np.ones(test_x_vector.shape)),
    ]

    def test_eval(self, noise_kernel):
        assert check_kernel_evaluation(noise_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self,noise_kernel):
        assert check_kernel_evaluation(noise_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, noise_kernel):
        assert check_kernel_transposition(noise_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, noise_kernel):
        assert check_kernel_evaluation(noise_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, noise_kernel):
        if noise_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(noise_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, noise_kernel, **kwargs)


@pytest.mark.usefixtures('linear_kernel')
class TestLinearKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[0.0, 0.0], [0.0, 4.0]])
    ref_dcov = np.atleast_2d([[0.0, 4.0], [0.0, 4.0]])
    ref_ddcov = np.atleast_2d([[4.0, 4.0], [4.0, 4.0]])
    ref_hdcov = [
        np.atleast_2d([[0.0, 0.0], [0.0, 4.0]]),
    ]

    def test_eval(self, linear_kernel):
        assert check_kernel_evaluation(linear_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, linear_kernel):
        assert check_kernel_evaluation(linear_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, linear_kernel):
        assert check_kernel_transposition(linear_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, linear_kernel):
        assert check_kernel_evaluation(linear_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, linear_kernel):
        if linear_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(linear_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, linear_kernel, **kwargs)


@pytest.mark.usefixtures('poly_order_kernel')
class TestPolyOrderKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 1.0], [1.0, 5.0]])
    ref_dcov = np.atleast_2d([[0.0, 4.0], [0.0, 4.0]])
    ref_ddcov = np.atleast_2d([[4.0, 4.0], [4.0, 4.0]])
    ref_hdcov = [
        np.atleast_2d([[0.0, 0.0], [0.0, 4.0]]),
        np.atleast_2d([[1.0, 1.0], [1.0, 1.0]]),
    ]

    def test_eval(self, poly_order_kernel):
        assert check_kernel_evaluation(poly_order_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, poly_order_kernel):
        assert check_kernel_evaluation(poly_order_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, poly_order_kernel):
        assert check_kernel_transposition(poly_order_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, poly_order_kernel):
        assert check_kernel_evaluation(poly_order_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, poly_order_kernel):
        if poly_order_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(poly_order_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, poly_order_kernel, **kwargs)


@pytest.mark.usefixtures('se_kernel')
class TestSquareExponentialKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 0.13533528323], [0.13533528323, 1.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.54134113295], [-0.54134113295, 0.0]])
    ref_ddcov = np.atleast_2d([[4.0, -1.62402339884], [-1.62402339884, 4.0]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 0.27067056647], [0.27067056647, 2.0]]),
        np.atleast_2d([[0.0, 1.08268226589], [1.08268226589, 0.0]]),
    ]

    def test_eval(self, se_kernel):
        assert check_kernel_evaluation(se_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, se_kernel):
        assert check_kernel_evaluation(se_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, se_kernel):
        assert check_kernel_transposition(se_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, se_kernel):
        assert check_kernel_evaluation(se_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, se_kernel):
        if se_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(se_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, se_kernel, **kwargs)


@pytest.mark.usefixtures('rq_kernel')
class TestRationalQuadraticKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 0.18593443208], [0.18593443208, 1.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.53124123452], [-0.53124123452, 0.0]])
    ref_ddcov = np.atleast_2d([[4.0, -1.29015728383], [-1.29015728383, 4.0]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 0.37186886416], [0.37186886416, 2.0]]),
        np.atleast_2d([[0.0, 0.74373772833], [0.74373772833, 0.0]]),
        np.atleast_2d([[0.0, -0.00943765078], [-0.00943765078, 0.0]]),
    ]

    def test_eval(self, rq_kernel):
        assert check_kernel_evaluation(rq_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, rq_kernel):
        assert check_kernel_evaluation(rq_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, rq_kernel):
        assert check_kernel_transposition(rq_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, rq_kernel):
        assert check_kernel_evaluation(rq_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, rq_kernel):
        if rq_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(rq_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, rq_kernel, **kwargs)


@pytest.mark.usefixtures('matern_hi_kernel')
class TestMaternHalfIntegerKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 0.13866021914], [0.13866021914, 1.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.41671741677], [-0.41671741677, 0.0]])
    ref_ddcov = np.atleast_2d([[6.66666666667, -1.10633471569], [-1.10633471569, 6.66666666667]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 0.27732043828], [0.27732043828, 2.0]]),
        np.atleast_2d([[0.0, 0.18636169426], [0.18636169426, 0.0]]),
    ]

    def test_eval(self, matern_hi_kernel):
        assert check_kernel_evaluation(matern_hi_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, matern_hi_kernel):
        assert check_kernel_evaluation(matern_hi_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, matern_hi_kernel):
        assert check_kernel_transposition(matern_hi_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, matern_hi_kernel):
        assert check_kernel_evaluation(matern_hi_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, matern_hi_kernel):
        if matern_hi_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(matern_hi_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, matern_hi_kernel, **kwargs)


@pytest.mark.usefixtures('gibbs_constant_kernel')
class TestGibbsKernelWithConstant():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 0.60653065971], [0.60653065971, 1.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.60653065971], [-0.60653065971, 0.0]])
    ref_ddcov = np.atleast_2d([[1.0, 0.0], [0.0, 1.0]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 1.21306131943], [1.21306131943, 2.0]]),
        np.atleast_2d([[0.0, 0.60653065971], [0.60653065971, 0.0]]),
    ]

    def test_eval(self, gibbs_constant_kernel):
        assert check_kernel_evaluation(gibbs_constant_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, gibbs_constant_kernel):
        assert check_kernel_evaluation(gibbs_constant_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, gibbs_constant_kernel):
        assert check_kernel_transposition(gibbs_constant_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, gibbs_constant_kernel):
        assert check_kernel_evaluation(gibbs_constant_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, gibbs_constant_kernel):
        if gibbs_constant_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(gibbs_constant_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, gibbs_constant_kernel, **kwargs)


@pytest.mark.usefixtures('gibbs_inverse_gaussian_kernel')
class TestGibbsKernelWithInverseGaussian():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[1.0, 0.10737182452], [0.10737182452, 1.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.47848791644], [0.58059409716, 0.0]])
    ref_ddcov = np.atleast_2d([[4.0, 1.00715576945], [1.00715576945, 16.8232521516]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 0.21474364904], [0.21474364904, 2.0]]),
        np.atleast_2d([[-2.0, 1.88160836531], [1.88160836531, -2.78744556822]]),
        np.atleast_2d([[0.0, -0.12820614183], [-0.12820614183, 0.37723973548]]),
        np.atleast_2d([[0.0, -4.10259653856], [-4.10259653856, 12.0716715354]]),
    ]

    def test_eval(self, gibbs_inverse_gaussian_kernel):
        print(gibbs_inverse_gaussian_kernel(self.x1, self.x2))
        assert check_kernel_evaluation(gibbs_inverse_gaussian_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, gibbs_inverse_gaussian_kernel):
        assert check_kernel_evaluation(gibbs_inverse_gaussian_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, gibbs_inverse_gaussian_kernel):
        assert check_kernel_transposition(gibbs_inverse_gaussian_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, gibbs_inverse_gaussian_kernel):
        assert check_kernel_evaluation(gibbs_inverse_gaussian_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, gibbs_inverse_gaussian_kernel):
        if gibbs_inverse_gaussian_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(gibbs_inverse_gaussian_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, gibbs_inverse_gaussian_kernel, **kwargs)


@pytest.mark.usefixtures('sum_kernel')
class TestSumOperationKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[2.0, 0.13533528323], [0.13533528323, 2.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.54134113295], [-0.54134113295, 0.0]])
    ref_ddcov = np.atleast_2d([[4.0, -1.62402339884], [-1.62402339884, 4.0]])
    ref_hdcov = [
        np.atleast_2d([[2.0, 0.27067056647], [0.27067056647, 2.0]]),
        np.atleast_2d([[0.0, 1.08268226589], [1.08268226589, 0.0]]),
        np.atleast_2d([[2.0, 0.0], [0.0, 2.0]]),
    ]

    def test_eval(self, sum_kernel):
        assert check_kernel_evaluation(sum_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, sum_kernel):
        assert check_kernel_evaluation(sum_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, sum_kernel):
        assert check_kernel_transposition(sum_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, sum_kernel):
        assert check_kernel_evaluation(sum_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, sum_kernel):
        if sum_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(sum_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, sum_kernel, **kwargs)


@pytest.mark.usefixtures('product_kernel')
class TestProductOperationKernel():

    test_x_vector = np.array([0.0, 1.0])
    x1, x2 = np.meshgrid(test_x_vector, test_x_vector)
    ref_cov = np.atleast_2d([[0.0, 0.0], [0.0, 16.0]])
    ref_dcov = np.atleast_2d([[0.0, 0.0], [0.0, 32.0]])
    ref_ddcov = np.atleast_2d([[0.0, 0.0], [0.0, 64.0]])
    ref_hdcov = [
        np.atleast_2d([[0.0, 0.0], [0.0, 0.0]]), 
        np.atleast_2d([[0.0, 0.0], [0.0, 0.0]]),
    ]

    def test_eval(self, product_kernel):
        assert check_kernel_evaluation(product_kernel, self.x1, self.x2, 0, self.ref_cov)

    def test_eval_first_derivative(self, product_kernel):
        assert check_kernel_evaluation(product_kernel, self.x1, self.x2, 1, self.ref_dcov)

    def test_eval_first_derivative_transpose(self, product_kernel):
        assert check_kernel_transposition(product_kernel, self.x1, self.x2)

    def test_eval_second_derivative(self, product_kernel):
        assert check_kernel_evaluation(product_kernel, self.x1, self.x2, 2, self.ref_ddcov)

    def test_eval_hyperparameter_derivatives(self, product_kernel):
        if product_kernel.is_hderiv_implemented():
            assert check_kernel_hyperparameter_derivatives(product_kernel, self.x1, self.x2, self.ref_hdcov)
        else:
            kwargs = {'x1': self.x1, 'x2': self.x2, 'hder': 0}
            pytest.raises(NotImplementedError, product_kernel, **kwargs)
