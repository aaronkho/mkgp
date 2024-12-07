r'''
Kernel classes for Gaussian Process Regression fitting of 1D data with errorbars. Built in Python 3.x, adapted to be Python 2.x compatible.
  06/12/2024: No longer compatible with Python 2.x.

These classes were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET Contributors
    IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis, Boston, MA (2017)
    `<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

'''
#    Kernel theory: "Gaussian Process for Machine Learning", C.E. Rasmussen, C.K.I. Williams (2006)

# Required imports
import warnings
import copy
import math
import numpy as np
import scipy.special as spsp

from .baseclasses import _Kernel, _OperatorKernel, _WarpingFunction
from .definitions import number_types, array_types

__all__ = [
    'Sum_Kernel', 'Product_Kernel', 'Symmetric_Kernel',  # Kernel operator classes
    'Constant_Kernel', 'Noise_Kernel', 'Linear_Kernel', 'Poly_Order_Kernel', 'SE_Kernel', 'RQ_Kernel',
    'Matern_HI_Kernel', 'NN_Kernel', 'Gibbs_Kernel',  # Kernel classes
    'Constant_WarpingFunction', 'IG_WarpingFunction',  # Warping function classes for Gibbs Kernel
]


class Sum_Kernel(_OperatorKernel):
    r'''
    Sum Kernel: Implements the sum of two (or more) separate kernels.

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be added together. Must provide a minimum of 2.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be added together. Must contain a minimum of 2.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2 values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        covm = np.full(x1.shape, np.nan) if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1, x2, der, ihyp)
            if ihyp is not None:
                nhyps = kk.hyperparameters.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self, *args, **kwargs):
        r'''
        Initializes the :code:`Sum_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be added together. Must provide a minimum of 2.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be added together. Must contain a minimum of 2.

        :returns: none.
        '''

        klist = kwargs.get('klist')
        uklist = []
        if len(args) >= 2 and isinstance(args[0], _Kernel) and isinstance(args[1], _Kernel):
            for kk in args:
                if isinstance(kk, _Kernel):
                    uklist.append(kk)
        elif isinstance(klist, list) and len(klist) >= 2 and isinstance(klist[0], _Kernel) and isinstance(klist[1], _Kernel):
            for kk in klist:
                if isinstance(kk, _Kernel):
                    uklist.append(kk)
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel instances.')
        super().__init__('Sum', self.__calc_covm, True, uklist)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Sum_Kernel(klist=kcopy_list)
        return kcopy



class Product_Kernel(_OperatorKernel):
    r'''
    Product Kernel: Implements the product of two (or more) separate kernels.

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be multiplied together. Must provide a minimum of 2.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be multiplied together. Must contain a minimum of 2.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        covm = np.full(x1.shape, np.nan) if self._kernel_list is None else np.zeros(x1.shape)
        nks = len(self._kernel_list) if self._kernel_list is not None else 0
        dermat = np.atleast_2d([0] * nks)
        sd = int(np.sign(der))
        for ii in np.arange(0, int(sd * der)):
            for jj in np.arange(1, nks):
                deradd = dermat.copy()
                dermat = np.vstack((dermat, deradd))
            for row in np.arange(0, dermat.shape[0]):
                rem = row % nks
                fac = (row - rem) / (nks ** int(sd * der))
                idx = int((rem + fac) % nks)
                dermat[row, idx] = dermat[row, idx] + 1
        oddfilt = (np.mod(dermat, 2) != 0)
        dermat[oddfilt] = sd * dermat[oddfilt]
        for row in np.arange(0, dermat.shape[0]):
            ihyp = hder
            covterm = np.ones(x1.shape)
            for col in np.arange(0, dermat.shape[1]):
                kk = self._kernel_list[col]
                covterm = covterm * kk(x1, x2, dermat[row, col], ihyp)
                if ihyp is not None:
                    nhyps = kk.hyperparameters.size
                    ihyp = ihyp - nhyps
            covm = covm + covterm
        return covm


    def __init__(self, *args, **kwargs):
        r'''
        Initializes the :code:`Product_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be multiplied together. Must provide a minimum of 2.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be multiplied together. Must contain a minimum of 2.

        :returns: none.
        '''

        klist = kwargs.get('klist')
        uklist = []
        if len(args) >= 2 and isinstance(args[0], _Kernel) and isinstance(args[1], _Kernel):
            for kk in args:
                if isinstance(kk, _Kernel):
                    uklist.append(kk)
        elif isinstance(klist, list) and len(klist) >= 2 and isinstance(klist[0], _Kernel) and isinstance(klist[1], _Kernel):
            for kk in klist:
                if isinstance(kk, _Kernel):
                    uklist.append(kk)
        else:
            raise TypeError('Arguments to Product_Kernel must be Kernel objects.')
        super().__init__('Prod', self.__calc_covm, True, uklist)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Product_Kernel(klist=kcopy_list)
        return kcopy



class Symmetric_Kernel(_OperatorKernel):
    r'''
    1D Symmetric Kernel: Enforces even symmetry about zero for any given kernel. Although
    this class accepts multiple arguments, it only uses first :code:`_Kernel` argument.

    This is really only useful if you wish to rigourously infer data on other side of axis
    of symmetry without assuming the data can just be flipped or if data exists on other
    side but a symmetric solution is desired. **This capability is NOT fully tested!**

    :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be given flip symmetry. Must provide a minimum of 1.

    :kwarg klist: list. Python native list of :code:`_Kernel` instances to be given flip symmetry. Must contain a minimum of 1.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the covariance function at, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        covm = np.full(x1.shape, np.nan) if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1, x2, der, ihyp) + kk(-x1, x2, der, ihyp)      # Not sure if division by 2 is necessary to conserve covm
            if ihyp is not None:
                nhyps = kk.hyperparameters.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self,*args,**kwargs):
        r'''
        Initializes the :code:`Symmetric_Kernel` instance.

        :arg \*args: object. Any number of :code:`_Kernel` instance arguments, which are to be given flip symmetry. Must provide a minimum of 1.

        :kwarg klist: list. Python native list of :code:`_Kernel` instances to be given flip symmetry. Must contain a minimum of 1.

        :returns: none.
        '''

        klist = kwargs.get('klist')
        uklist = []
        if len(args) >= 1 and isinstance(args[0], _Kernel):
            if len(args) >= 2:
                print('Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.')
            kk = args[0]
            uklist.append(kk)
        elif isinstance(klist, list) and len(klist) >= 1 and isinstance(klist[0], _Kernel):
            if len(klist) >= 2:
                print('Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.')
            kk = klist[0]
            uklist.append(kk)
        else:
            raise TypeError('Arguments to Symmetric_Kernel must be Kernel objects.')
        super().__init__('Sym', self.__calc_covm, True, uklist)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current object, which can be modified without affecting the original.
        '''

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Symmetric_Kernel(klist=kcopy_list)
        return kcopy



class Constant_Kernel(_Kernel):
    r'''
    Constant Kernel: always evaluates to a constant value, regardless of input pair.

    .. warning::

        This is **NOT inherently a valid covariance function**, as it yields
        singular covariance matrices! However, it provides an alternate way
        to enforce or relax the fit smoothness. **This capability is NOT fully
        tested!**

    :kwarg cv: float. Constant value which kernel always evaluates to.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        .. warning::

            This is **not** inherently a valid covariance function, as it results
            in singular matrices!

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        c_hyp = csts[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = c_hyp * np.ones(rr.shape)
            elif hder == 0:
                covm = np.ones(rr.shape)
        return covm


    def __init__(self, cv=1.0):
        r'''
        Initializes the :code:`Constant_Kernel` instance.

        :kwarg cv: float. Constant value which kernel always evaluates to.

        :returns: none.
        '''

        csts = np.zeros((1, ))
        if isinstance(cv, number_types):
            csts[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super().__init__('C', self.__calc_covm, True, None, csts)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(csts[0])
        kcopy = Constant_Kernel(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Noise_Kernel(_Kernel):
    r'''
    Noise Kernel: adds a user-defined degree of expected noise in the GPR regression, emulates a
    constant assumed fit noise level.

    .. note::

        The noise implemented by this kernel is **conceptually not the same** as measurement error,
        which should be applied externally in GP regression implementation!!!

    :kwarg nv: float. Hyperparameter representing the noise level.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1 values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2 values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        n_hyp = hyps[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm[rr == 0.0] = n_hyp ** 2.0
            elif hder == 0:
                covm[rr == 0.0] = 2.0 * n_hyp
#       Applied second derivative of Kronecker delta, assuming it is actually a Gaussian centred on rr = 0 with small width, ss
#       Surprisingly provides good variance estimate but issues with enforcing derivative constraints (needs more work!)
#        Commented out for stability reasons.
#        elif der == 2 or der == -2:
#            drdx1 = np.sign(x1 - x2)
#            drdx1[drdx1==0] = 1.0
#            drdx2 = np.sign(x2 - x1)
#            drdx2[drdx2==0] = -1.0
#            trr = rr[rr > 0.0]
#            ss = 0.0 if trr.size == 0 else np.nanmin(trr)
#            if hder is None:
#                covm[rr == 0.0] = -drdx1[rr == 0.0] * drdx2[rr == 0.0] * 2.0 * n_hyp**2.0 / ss**2.0
#            elif hder == 0:
#                covm[rr == 0.0] = -drdx1[rr == 0.0] * drdx2[rr == 0.0] * 4.0 * n_hyp / ss**2.0
        return covm


    def __init__(self, nv=1.0):
        r'''
        Initializes the :code:`Noise_Kernel` instance.

        :kwarg nv: float. Hyperparameter representing the noise level.

        :returns: none.
        '''

        hyps = np.zeros((1, ))
        if isinstance(nv, number_types):
            hyps[0] = float(nv)
        else:
            raise ValueError('Noise hyperparameter must be a real number.')
        super().__init__('n', self.__calc_covm, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        nhp = float(hyps[0])
        kcopy = Noise_Kernel(nhp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Linear_Kernel(_Kernel):
    r'''
    Linear Kernel: Applies linear regression :code:`ax`, can be multiplied with itself
    for higher order pure polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = (v_hyp ** 2.0) * pp
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = (v_hyp ** 2.0) * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = (v_hyp ** 2.0) * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = (v_hyp ** 2.0) * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self, var=1.0):
        r'''
        Initializes the :code:`Linear_Kernel` instance.

        :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

        :returns: none.
        '''

        hyps = np.zeros((1, ))
        if isinstance(var, number_types) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        super().__init__('L', self.__calc_covm, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        kcopy = Linear_Kernel(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Poly_Order_Kernel(_Kernel):
    r'''
    Polynomial Order Kernel: Applies linear regression :code:`ax + b`, where :code:`b != 0`,
    can be multiplied with itself for higher order polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

    :kwarg cst: float. Hyperparameter added to linear component of model, ie. :code:`b`.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        b_hyp = hyps[1]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = (v_hyp ** 2.0) * pp + (b_hyp ** 2.0)
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
            elif hder == 1:
                covm = b_hyp * np.ones(pp.shape)
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = (v_hyp ** 2.0) * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = (v_hyp ** 2.0) * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = (v_hyp ** 2.0) * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self, var=1.0, cst=1.0):
        r'''
        Initializes the :code:`Poly_Order_Kernel` instance.

        :kwarg var: float. Hyperparameter multiplying linear component of model, ie. :code:`a`.

        :kwarg cst: float. Hyperparameter added to linear component of model, ie. :code:`b`.

        :returns: none.
        '''

        hyps = np.zeros((2, ))
        if isinstance(var, number_types) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Multiplicative hyperparameter must be greater than 0.')
        if isinstance(cst, number_types) and float(cst) > 0.0:
            hyps[1] = float(cst)
        else:
            raise ValueError('Additive hyperparameter must be greater than 0.')
        super().__init__('P', self.__calc_covm, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        cst = float(hyps[1])
        kcopy = Poly_Order_Kernel(chp, cst)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class SE_Kernel(_Kernel):
    r'''
    Square Exponential Kernel: Infinitely differentiable (ie. extremely smooth) covariance function.

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.
    '''

    def __calc_covm(self,x1,x2,der=0,hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        l_hyp = hyps[1]
        rr = np.abs(x1 - x2)
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (v_hyp ** 2.0) / np.power(l_hyp, nn)
            efac = np.exp(-np.power(rr, 2.0) / (2.0 * (l_hyp ** 2.0)))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp, nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * 2.0 * v_hyp / np.power(l_hyp, nn)
            efac = np.exp(-np.power(rr, 2.0) / (2.0 * (l_hyp ** 2.0)))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - jj) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp, nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (v_hyp ** 2.0) / np.power(l_hyp, nn + 1)
            efac = np.exp(-np.power(rr, 2.0) / (2.0 * (l_hyp ** 2.0)))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 3, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0, nn - ii + 2) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power(rr / l_hyp, nn - jj + 2)
            covm = afac * efac * sfac
        return covm


    def __init__(self, var=1.0, ls=1.0):
        r'''
        Initializes the :code:`SE_Kernel` instance.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter represeting variability of model in x, ie. length scale.

        :returns: none.
        '''

        hyps = np.zeros((2, ))
        if isinstance(var, number_types) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        if isinstance(ls, number_types) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Length scale hyperparameter must be greater than 0.')
        super().__init__('SE', self.__calc_covm, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        shp = float(hyps[1])
        kcopy = SE_Kernel(chp, shp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class RQ_Kernel(_Kernel):
    r'''
    Rational Quadratic Kernel: Infinitely differentiable covariance function
    but provides higher tolerance for steep slopes than the squared exponential
    kernel. Mathematically equivalent to an infinite sum of squared exponential
    kernels with harmonic length scales for :code:`alpha < 20`, but becomes
    effectively identical to the squared exponential kernel as :code:`alpha`
    approaches infinity.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

    :kwarg alpha: float. Hyperparameter representing degree of length scale mixing in model.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        rq_amp = hyps[0]
        l_hyp = hyps[1]
        a_hyp = hyps[2]
        rr = np.abs(x1 - x2)
        rqt = 1.0 + np.power(rr, 2.0) / (2.0 * a_hyp * (l_hyp ** 2.0))
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (rq_amp ** 2.0) / np.power(l_hyp, nn)
            efac = np.power(rqt, -a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                gfac = np.power(rqt, ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp, nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp, nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * 2.0 * rq_amp / np.power(l_hyp, nn)
            efac = np.power(rqt, -a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                gfac = np.power(rqt, ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp, nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp, nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (rq_amp ** 2.0) / np.power(l_hyp, nn)
            efac = np.power(rqt, -a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 3, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0, nn - ii + 2) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                gfac = np.power(rqt, ii) * spsp.gamma(a_hyp + float(nn) - float(ii) + 1.0) / (np.power(a_hyp, nn - ii + 1) * spsp.gamma(a_hyp))
                sfac = sfac + lfac * gfac * np.power(rr / l_hyp, nn - jj + 2)
            covm = afac * efac * sfac
        elif hder == 2:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (rq_amp ** 2.0) / np.power(l_hyp, nn)
            efac = np.power(rqt, -a_hyp - float(nn) - 1.0)
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                gfac = np.power(rqt, ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp, nn - ii) * spsp.gamma(a_hyp))
                pfac = (a_hyp - 2.0 * ii) / (a_hyp) * (rqt - 1.0) - float(nn - ii) / a_hyp - rqt * (np.log(rqt) + spsp.digamma(a_hyp + float(nn) - float(ii)) - spsp.digamma(a_hyp))
                sfac = sfac + cfac * gfac * pfac * np.power(rr / l_hyp, nn - jj)
            covm = afac * efac * sfac
        return covm


    def __init__(self, amp=1.0, ls=1.0, alpha=1.0):
        r'''
        Initializes the :code:`RQ_Kernel` instance.

        :kwarg amp: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

        :kwarg alpha: float. Hyperparameter representing degree of length scale mixing in model.

        :returns: none.
        '''

        hyps = np.zeros((3, ))
        if isinstance(amp, number_types) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Rational quadratic amplitude must be greater than 0.')
        if isinstance(ls, number_types) and float(ls) != 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Rational quadratic hyperparameter cannot equal 0.')
        if isinstance(alpha, number_types) and float(alpha) > 0.0:
            hyps[2] = float(alpha)
        else:
            raise ValueError('Rational quadratic alpha parameter must be greater than 0.')
        super().__init__('RQ', self.__calc_covm, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        ramp = float(hyps[0])
        rhp = float(hyps[1])
        ralp = float(hyps[2])
        kcopy = RQ_Kernel(ramp, rhp, ralp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Matern_HI_Kernel(_Kernel):
    r'''
    Matern Kernel with Half-Integer Order Parameter: Only differentiable in
    orders less than given order parameter, :code:`nu`. Allows fit to retain
    more features at expense of volatility, but effectively becomes
    equivalent to the square exponential kernel as :code:`nu` approaches
    infinity.

    The half-integer implentation allows for use of explicit simplifications
    of the derivatives, which greatly improves its speed.

    .. note::
  
        Recommended :code:`nu = 5/2` for second order differentiability
        while retaining maximum feature representation.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

    :kwarg nu: float. Constant value setting the volatility of the model, recommended value is 2.5.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        mat_amp = hyps[0]
        mat_hyp = hyps[1]
        nu = csts[0]
        if nu < np.abs(der):
            raise ValueError('Matern nu parameter must be greater than requested derivative order.')
        pp = int(nu)
        rr = np.abs(x1 - x2)
        mht = np.sqrt(2.0 * nu) * rr / mat_hyp
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (mat_amp ** 2.0) * np.power(np.sqrt(2.0 * nu) / mat_hyp, nn)
            efac = np.exp(-mht)
            spre = math.factorial(pp) / math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0, nn + 1):
                mfac = np.power(-1.0, nn - ii) * np.power(2.0, ii) * math.factorial(nn) / (math.factorial(ii) * math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0, pp - ii + 1):
                    ffac = spre * math.factorial(pp + zz) / (math.factorial(zz) * math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht, pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 0:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * 2.0 * mat_amp * np.power(np.sqrt(2.0 * nu) / mat_hyp, nn)
            efac = np.exp(-mht)
            spre = math.factorial(pp) / math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0, nn + 1):
                mfac = np.power(-1.0, nn - ii) * np.power(2.0, ii) * math.factorial(nn) / (math.factorial(ii) * math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0, pp - ii + 1):
                    ffac = spre * math.factorial(pp + zz) / (math.factorial(zz) * math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht, pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 1:
            afac = np.power(drdx1, dx1) * np.power(drdx2, dx2) * (mat_amp ** 2.0) * np.power(np.sqrt(2.0 * nu), nn) / np.power(mat_hyp, nn + 1)
            efac = np.exp(-mht)
            spre = math.factorial(pp) / math.factorial(2 * pp)
            ofac = np.zeros(rr.shape)
            for zz in np.arange(0, pp - nn):
                ffac = spre * math.factorial(pp + zz) / (math.factorial(zz) * math.factorial(pp - nn - zz - 1))
                ofac = ofac + ffac * np.power(2.0 * mht, pp - nn - zz - 1)
            tfac = -np.power(2.0, nn + 1) * ofac
            for ii in np.arange(0, nn + 1):
                mfac = np.power(-1.0, nn - ii) * np.power(2.0, ii) * math.factorial(nn) / (math.factorial(ii) * math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0, pp - ii + 1):
                    ffac = spre * math.factorial(pp + zz) / (math.factorial(zz) * math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht, pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        return covm


    def __init__(self, amp=0.1, ls=0.1, nu=2.5):
        r'''
        Initializes the :code:`Matern_HI_Kernel` instance.

        :kwarg amp: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

        :kwarg nu: float. Constant value setting the volatility of the model, recommended value is 2.5.

        :returns: none.
        '''

        hyps = np.zeros((2, ))
        csts = np.zeros((1, ))
        if isinstance(amp, number_types) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Matern amplitude hyperparameter must be greater than 0.')
        if isinstance(ls, number_types) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Matern length scale hyperparameter must be greater than 0.')
        if isinstance(nu, number_types) and float(nu) >= 0.0:
            csts[0] = float(int(nu)) + 0.5
        else:
            raise ValueError('Matern half-integer nu constant must be greater or equal to 0.')
        super().__init__('MH', self.__calc_covm, True, hyps, csts)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        mamp = float(hyps[0])
        mhp = float(hyps[1])
        nup = float(csts[0])
        kcopy = Matern_HI_Kernel(mamp, mhp, nup)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class NN_Kernel(_Kernel):
    r'''
    Neural Network Style Kernel: Implements a sigmoid covariance function similar
    to a perceptron (or neuron) in a neural network, good for strong discontinuities.

    .. warning::

        Suffers from high volatility, worse than the Matern kernel. Localization of the
        kernel variation to the features in data is not yet achieved. **Strongly
        recommended NOT to use**, as Gibbs kernel provides better localized feature
        selection but is limited to a pre-defined type instead of being general.

    :kwarg nna: float. Hyperparameter representing variability of model in y.

    :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

    :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        nn_amp = hyps[0]
        nn_off = hyps[1]
        nn_hyp = hyps[2]
        rr = np.abs(x1 - x2)
        pp = x1 * x2
        nnfac = 2.0 / np.pi
        nnn = 2.0 * ((nn_off ** 2.0) + (nn_hyp ** 2.0) * x1 * x2)
        nnd1 = 1.0 + 2.0 * ((nn_off ** 2.0) + (nn_hyp ** 2.0) * np.power(x1, 2.0))
        nnd2 = 1.0 + 2.0 * ((nn_off ** 2.0) + (nn_hyp ** 2.0) * np.power(x2, 2.0))
        chi = nnd1 * nnd2
        xi = chi - (nnn ** 2.0)
        covm = np.zeros(rr.shape)
        if der == 0:
            covm = (nn_amp ** 2.0) * nnfac * np.arcsin(nnn / np.power(chi, 0.5))
        elif der == 1:
            dpdx2 = x1
            dchidx2 = 4.0 * (nn_hyp ** 2.0) * x2 * nnd1
            nnk = 2.0 * (nn_hyp ** 2.0) / (chi * np.power(xi, 0.5))
            nnm = dpdx2 * chi - dchidx2 * nnn / (4.0 * (nn_hyp ** 2.0))
            covm = (nn_amp ** 2.0) * nnfac * nnk * nnm
        elif der == -1:
            dpdx1 = x2
            dchidx1 = 4.0 * (nn_hyp ** 2.0) * x1 * nnd2
            nnk = 2.0 * (nn_hyp ** 2.0) / (chi * np.power(xi, 0.5))
            nnm = dpdx1 * chi - dchidx1 * nnn / (4.0 * (nn_hyp ** 2.0))
            covm = (nn_amp ** 2.0) * nnfac * nnk * nnm
        elif der == 2 or der == -2:
            dpdx1 = x2
            dpdx2 = x1
            dchidx1 = 4.0 * (nn_hyp ** 2.0) * x1 * nnd2
            dchidx2 = 4.0 * (nn_hyp ** 2.0) * x2 * nnd1
            d2chi = 16.0 * (nn_hyp ** 4.0) * pp
            nnk = 2.0 * (nn_hyp ** 2.0) / (chi * np.power(xi, 0.5))
            nnt1 = chi * (1.0 + (nnn / xi) * (2.0 * (nn_hyp ** 2.0) * pp + d2chi / (8.0 * (nn_hyp ** 2.0))))
            nnt2 = (-0.5 * chi / xi) * (dpdx2 * dchidx1 + dpdx1 * dchidx2) 
            covm = (nn_amp ** 2.0) * nnfac * nnk * (nnt1 + nnt2)
        else:
            raise NotImplementedError(f'Derivatives of order 3 or higher not implemented in {self.name} kernel.')
        return covm


    def __init__(self,nna=1.0,nno=1.0,nnv=1.0):
        r'''
        Initializes the :code:`NN_Kernel` instance.

        :kwarg nna: float. Hyperparameter representing variability of model in y.

        :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

        :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.

        :returns: none.
        '''

        hyps = np.zeros((3, ))
        if isinstance(nna, number_types) and float(nna) > 0.0:
            hyps[0] = float(nna)
        else:
            raise ValueError('Neural network amplitude must be greater than 0.')
        if isinstance(nno, number_types):
            hyps[1] = float(nno)
        else:
            raise ValueError('Neural network offset parameter must be a real number.')
        if isinstance(nnv, number_types) and float(nnv) > 0.0:
            hyps[2] = float(nnv)
        else:
            raise ValueError('Neural network hyperparameter must be a real number.')
        super().__init__('NN', self.__calc_covm, False, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        nnamp = float(hyps[0])
        nnop = float(hyps[1])
        nnhp = float(hyps[2])
        kcopy = NN_Kernel(nnamp, nnop, nnhp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Gibbs_Kernel(_Kernel):
    r'''
    Gibbs Kernel: Implements a Gibbs covariance function with variable length
    scale defined by an externally-defined warping function.

    .. note::

        The warping function is stored in the variable, :code:`self._lfunc`,
        and must be an instance of the class :code:`_WarpingFunction`. This
         was enforced to ensure functionality of hyperparameter optimization.
        Developers are **strongly recommended** to use template
        :code:`_WarpingFunction` class when implementing new warping functions
        for this package!

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg wfunc: object. Warping function, as a :code:`_WarpingFunction` instance, representing the variability of model in x as a function of x.
    '''

    def __calc_covm(self, x1, x2, der=0, hder=None):
        r'''
        Implementation-specific covariance function.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        v_hyp = hyps[0]
        l_hyp1 = self._wfunc(x1, 0)
        l_hyp2 = self._wfunc(x2, 0)
        rr = x1 - x2
        ll = np.power(l_hyp1, 2.0) + np.power(l_hyp2, 2.0)
        mm = l_hyp1 * l_hyp2
        lder = int((int(np.abs(der)) + 1) / 2)
        hdermax = self._wfunc.hyperparameters.size
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = (v_hyp ** 2.0) * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                dlh1 = self._wfunc(x1, lder, ghder)
                dlh2 = self._wfunc(x2, lder, ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll, 2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll, 2.0) * dll
                covm = (v_hyp ** 2.0) * (c1 + c2) * np.exp(-np.power(rr, 2.0) / ll)
        elif der == 1:
            if hder is None:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2, lder)
                kfac = (v_hyp ** 2.0) * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll, 2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2, lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll, 2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2, lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll, 2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1, 0, ghder)
                dlh2 = self._wfunc(x2, 0, ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx2 = self._wfunc(x2, lder, ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll, 2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll, 2.0) * dll
                dkfac = (v_hyp ** 2.0) * (c1 + c2) * np.exp(-np.power(rr, 2.0) / ll)
                dt1 = ddldx2 / (2.0 * l_hyp2) - dldx2 * dlh2 / (2.0 * np.power(l_hyp2, 2.0))
                dt2 = -dlh2 * dldx2 / ll - l_hyp2 * ddldx2 / ll + l_hyp2 * dldx2 * dll / np.power(ll, 2.0)
                dt3 = (2.0 * dlh2 * dldx2 + 2.0 * l_hyp2 * ddldx2 - 4.0 * l_hyp2 * dldx2 * dll / ll) * np.power(rr / ll, 2.0)
                dt4 = drdx2 * 2.0 * rr * dll / np.power(ll, 2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == -1:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = (v_hyp ** 2.0) * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll, 2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1, lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll, 2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder >= 1 and hder <= 3:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1, lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll, 2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1, 0, ghder)
                dlh2 = self._wfunc(x2, 0, ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1, lder, ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll, 2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll, 2.0) * dll
                dkfac = (v_hyp ** 2.0) * (c1 + c2) * np.exp(-np.power(rr, 2.0) / ll)
                dt1 = ddldx1 / (2.0 * l_hyp1) - dldx1 * dlh1 / (2.0 * np.power(l_hyp1, 2.0))
                dt2 = -dlh1 * dldx1 / ll - l_hyp1 * ddldx1 / ll + l_hyp1 * dldx1 * dll / np.power(ll, 2.0)
                dt3 = (2.0 * dlh1 * dldx1 + 2.0 * l_hyp1 * ddldx1 - 4.0 * l_hyp1 * dldx1 * dll / ll) * np.power(rr / ll, 2.0)
                dt4 = drdx1 * 2.0 * rr * dll / np.power(ll, 2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == 2 or der == -2:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1, lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2, lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = (v_hyp ** 2.0) * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll, 4.0)
                d2 = -12.0 * mm * np.power(rr, 2.0) / np.power(ll, 3.0)
                d3 = 3.0 * mm / np.power(ll, 2.0)
                d4 = np.power(rr, 2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll, 2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll, 2.0) * (2.0 * np.power(rr, 2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr, 2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll, 4.0)
                d2 = -12.0 * mm * np.power(rr, 2.0) / np.power(ll, 3.0)
                d3 = 3.0 * mm / np.power(ll, 2.0)
                d4 = np.power(rr, 2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll, 2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll, 2.0) * (2.0 * np.power(rr, 2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1, lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2, lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                dlh1 = self._wfunc(x1, 0, ghder)
                dlh2 = self._wfunc(x2, 0, ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1, lder, ghder)
                ddldx2 = self._wfunc(x2, lder, ghder)
                ddd = ddldx1 * dldx2 + dldx1 * ddldx2
                dii = drdx1 * rr * ddldx2 / l_hyp2 - drdx1 * rr * dldx2 * dlh2 / np.power(l_hyp2, 2.0) + \
                      drdx2 * rr * ddldx1 / l_hyp1 - drdx2 * rr * dldx1 * dlh1 / np.power(l_hyp1, 2.0)
                djj = drdx1 * rr * ddldx2 / l_hyp2 + drdx1 * rr * dldx2 * dlh2 + \
                      drdx2 * rr * ddldx1 / l_hyp1 + drdx2 * rr * dldx1 * dlh1
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll, 2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll, 2.0) * dll
                kfac = (v_hyp ** 2.0) * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                dkfac = (v_hyp ** 2.0) * (c1 + c2) * np.exp(-np.power(rr, 2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll, 4.0)
                d2 = -12.0 * mm * np.power(rr, 2.0) / np.power(ll, 3.0)
                d3 = 3.0 * mm / np.power(ll, 2.0)
                d4 = np.power(rr, 2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dd1 = 4.0 * dmm * np.power(rr / ll, 4.0) - 16.0 * mm * dll * np.power(rr, 4.0) / np.power(ll, 5.0)
                dd2 = -12.0 * dmm * np.power(rr, 2.0) / np.power(ll, 3.0) + 36.0 * mm * dll * np.power(rr, 2.0) / np.power(ll, 4.0)
                dd3 = 3.0 * dmm / np.power(ll, 2.0) - 6.0 * mm * dll / np.power(ll, 3.0)
                dd4 = -(dll / ll + dmm / mm) * np.power(rr, 2.0) / (ll * mm)
                dd5 = dmm / (4.0 * np.power(mm, 2.0))
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                ddt = ddd * (d1 + d2 + d3 + d4 + d5) + dd * (dd1 + dd2 + dd3 + dd4 + dd5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll, 2.0)) - ii / ll
                djt1 = 6.0 * djj / np.power(ll, 2.0) - 12.0 * jj * dll / np.power(ll, 3.0)
                djt2 = -4.0 * djj * np.power(rr, 2.0) / np.power(ll, 3.0) + 12.0 * jj * dll * np.power(rr, 2.0) / np.power(ll, 4.0)
                djt3 = dii / ll - ii * dll / np.power(ll, 2.0)
                djt = djt1 + djt2 + djt3
                rt = 2.0 * drdx1 * drdx2 / np.power(ll, 2.0) * (2.0 * np.power(rr, 2.0) - ll)
                drt = -2.0 * drdx1 * drdx2 * (4.0 * np.power(rr, 2.0) / np.power(ll, 3.0) - 1.0 / np.power(ll, 2.0))
                covm = dkfac * (dt + jt + rt) + kfac * (ddt + djt + drt)
        else:
            raise NotImplementedError(f'Derivatives of order 3 or higher not implemented in {self.name} kernel.')
        return covm


    @property
    def wfuncname(self):
        r'''
        Returns the codename of the stored :code:`_WarpingFunction` instance.

        :returns: str. Codename of the stored :code:`_WarpingFunction` instance.
        '''

        # Ensure reconstruction failure if warping function is not properly defined
        wfname = '?'
        if isinstance(self._wfunc, _WarpingFunction):
            wfname = self._wfunc.name 
        else:
            warnings.warn(f'Gibbs_Kernel warping function is not a valid WarpingFunction object.')
        return wfname


    def evaluate_wfunc(self, xx, der=0, hder=None):
        r'''
        Evaluates the stored warping function at the specified values.

        :arg xx: array. Vector of x-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of x derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values, under the given derivative settings. Has the same dimensions as :code:`xx`.
        '''

        # Prevent catastrophic failure if warping function is not properly defined
        lsf = None
        if isinstance(self._wfunc, _WarpingFunction):
            lsf = self._wfunc(xx, der, hder)
        else:
            warnings.warn(f'Gibbs_Kernel warping function is not a valid WarpingFunction object.')
        return lsf


    def __init__(self, var=1.0, wfunc=None):
        r'''
        Initialize the :code:`Gibbs_Kernel` instance.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg wfunc: object. Warping function, as a :code:`_WarpingFunction` instance, representing the variability of model in x as a function of x.

        :returns: none.
        '''

        self._wfunc = None
        if isinstance(wfunc, _WarpingFunction):
            self._wfunc = copy.copy(wfunc)
        elif wfunc is None:
            self._wfunc = Constant_WarpingFunction(1.0e0)

        hyps = np.zeros((1, ))
        if isinstance(var, number_types):
            hyps[0] = float(var)
        else:
            raise ValueError('Amplitude hyperparameter must be a real number.')
        super().__init__('G', self.__calc_covm, True, hyps)


    @property
    def name(self):

        name = super().name
        if isinstance(self._wfunc, _WarpingFunction):
            name += f'w{self._wfunc.name}'
        return name


    @property
    def hyperparameters(self):

        val = super().hyperparameters
        if isinstance(self._wfunc, _WarpingFunction):
            val = np.hstack((val, self._wfunc.hyperparameters))
        return val


    @property
    def constants(self):

        val = super().constants
        if isinstance(self._wfunc, _WarpingFunction):
            val = np.hstack((val, self._wfunc.constants))
        return val


    @property
    def bounds(self):

        val = super().bounds
        if isinstance(self._wfunc, _WarpingFunction):
            wval = self._wfunc.bounds
            if wval is not None:
                val = np.hstack((val, wval)) if val is not None else wval
        return val


    @hyperparameters.setter
    def hyperparameters(self, theta):
        r'''
        Set the hyperparameters stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        '''

        userhyps = None
        if isinstance(theta, array_types):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError(f'{self.name} Kernel hyperparameters must be given as an array-like object.')
        if super().hyperparameters.size > 0:
            super(Gibbs_Kernel, self.__class__).hyperparameters.__set__(self, userhyps)
        if isinstance(self._wfunc, _WarpingFunction):
            nhyps = super().hyperparameters.size
            if nhyps < userhyps.size:
                self._wfunc.hyperparameters = userhyps[nhyps:]
        else:
            warnings.warn(f'{type(self).__name__} warping function is not a valid WarpingFunction instance.')


    @constants.setter
    def constants(self, consts):
        r'''
        Set the constants stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        '''

        usercsts = None
        if isinstance(consts, array_types):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError(f'{self.name} Kernel constants must be given as an array-like object.')
        if super().constants.size > 0:
            super(Gibbs_Kernel, self.__class__).constants.__set__(self,usercsts)
        if isinstance(self._wfunc, _WarpingFunction):
            ncsts = super().constants.size
            if ncsts < usercsts.size:
                self._wfunc.constants = usercsts[ncsts:]
        else:
            warnings.warn(f'{type(self).__name__} warping function is not a valid WarpingFunction object.')


    @bounds.setter
    def bounds(self, bounds):
        r'''
        Set the hyperparameter bounds stored in the :code:`Gibbs_Kernel` and stored :code:`_WarpingFunction` instances.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_Kernel` and :code:`_WarpingFunction` class implementations.

        :returns: none.
        '''

        userbnds = None
        if isinstance(bounds, array_types):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError(f'{self.name} Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.')
        if userbnds.shape[0] != 2:
            raise TypeError(f'{self.name} Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.')
        super(Gibbs_Kernel, self.__class__).bounds.__set__(self, userbnds)
        if isinstance(self._wfunc, _WarpingFunction):
            wbnds = super().bounds
            nbnds = wbnds.shape[1] if wbnds is not None else 0
            if nbnds < userbnds.shape[1]:
                self._wfunc.bounds = userbnds[:,nbnds:]
        else:
            warnings.warn(f'{type(self).__name__} warping function is not a valid WarpingFunction object.')


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        wfunc = copy.copy(self._wfunc)
        kcopy = Gibbs_Kernel(chp, wfunc)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class Constant_WarpingFunction(_WarpingFunction):
    r'''
    Constant Warping Function for Gibbs Kernel: effectively reduces Gibbs kernel to squared exponential kernel.
    
    :kwarg cv: float. Hyperparameter representing constant value which the warping function always evalutates to.
    '''

    def __calc_warp(self, zz, der=0, hder=None):
        r'''
        Implementation-specific warping function.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        c_hyp = hyps[0]
        warp = np.zeros(zz.shape)
        if der == 0:
            if hder is None:
                warp = c_hyp * np.ones(zz.shape)
            elif hder == 0:
                warp = np.ones(zz.shape)
        return warp


    def __init__(self, cv=1.0):
        r'''
        Initializes the :code:`Constant_WarpingFunction` instance.

        :kwarg cv: float. Hyperparameter representing constant value which warping function always evaluates to.

        :returns: none.
        '''

        hyps = np.zeros((1, ))
        if isinstance(cv, number_types):
            hyps[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super().__init__('C', self.__calc_warp, True, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        chp = float(hyps[0])
        kcopy = Constant_WarpingFunction(chp)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy



class IG_WarpingFunction(_WarpingFunction):
    r'''
    Inverse Gaussian Warping Function for Gibbs Kernel: localized variation of length-scale with variation limit.

    :kwarg lb: float. Hyperparameter representing base length scale.

    :kwarg gh: float. Hyperparameter representing height of Gaussian envelope adjusting the length scale.

    :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

    :kwarg gm: float. Constant indicating location of peak of Gaussian envelope adjusting the length scale.

    :kwarg mf: float. Constant indicating upper limit for height-to-base length scale ratio, to improve stability.
    '''

    def __calc_warp(self, zz, der=0, hder=None):
        r'''
        Implementation-specific warping function.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        base = hyps[0]
        amp = hyps[1]
        sig = hyps[2]
        mu = csts[0]
        maxfrac = csts[1]
        nn = int(np.abs(der))
        hh = amp if amp < (maxfrac * base) else maxfrac * base
        warp = np.ones(zz.shape) * base
        if hder is None:
            afac = -hh * np.exp(-np.power(zz - mu, 2.0) / (2.0 * (sig ** 2.0))) / np.power(sig, nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig, nn - jj)
            warp = base + afac * sfac if der == 0 else afac * sfac
        elif hder == 0:
            warp = np.ones(zz.shape) if der == 0 else np.zeros(zz.shape)
        elif hder == 1:
            afac = -np.exp(-np.power(zz - mu, 2.0) / (2.0 * (sig ** 2.0))) / np.power(sig, nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0, nn + 1, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0, nn - ii) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig, nn - jj)
            warp = afac * sfac
        elif hder == 2:
            afac = -hh * np.exp(-np.power(zz - mu, 2.0) / (2.0 * (sig ** 2.0))) / np.power(sig, nn + 1)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0, nn + 3, 2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0, nn - ii + 2) * math.factorial(nn) / (np.power(2.0, ii) * math.factorial(ii) * math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power((zz - mu) / sig, nn - jj + 2)
            warp = afac * sfac
        return warp


    def __init__(self, lb=1.0, gh=0.5, gs=1.0, gm=0.0, mf=0.6):
        r'''
        Initializes the :code:`IG_WarpingFunction` instance.

        :kwarg lb: float. Hyperparameter representing base length scale.

        :kwarg gh: float. Hyperparameter representing height of Gaussian envelope adjusting the length scale.

        :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

        :kwarg gm: float. Constant indicating location of peak of Gaussian envelope adjusting the length scale.

        :kwarg mf: float. Constant indicating upper limit for height-to-base length scale ratio, to improve stability.

        :returns: none.
        '''

        hyps = np.zeros((3, ))
        csts = np.zeros((2, ))
        if isinstance(lb, number_types) and float(lb) > 0.0:
            hyps[0] = float(lb)
        else:
            raise ValueError('Length scale function base hyperparameter must be greater than 0.')
        if isinstance(gh, number_types) and float(gh) > 0.0:
            hyps[1] = float(gh)
        else:
            raise ValueError('Length scale function minimum hyperparameter must be greater than 0.')
        if isinstance(gs, number_types) and float(gs) > 0.0:
            hyps[2] = float(gs)
        else:
            raise ValueError('Length scale function sigma hyperparameter must be greater than 0.')
        if isinstance(gm, number_types):
            csts[0] = float(gm)
        else:
            raise ValueError('Length scale function mu constant must be a real number.')
        if isinstance(mf, number_types) and float(mf) < 1.0:
            csts[1] = float(mf)
        else:
            raise ValueError('Length scale function minimum-to-base ratio limit must be less than 1.')
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = float(csts[1] * hyps[0])
        super().__init__('IG', self.__calc_warp, True, hyps, csts)


    @property
    def hyperparameters(self):

        return super().hyperparameters


    @property
    def constants(self):

        return super().constants


    @property
    def bounds(self):

        return super().bounds


    @hyperparameters.setter
    def hyperparameters(self, theta):
        r'''
        Set the hyperparameters stored in the :code:`_WarpingFunction` object. Specific implementation due to maximum fraction limit.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        super(IG_WarpingFunction, self.__class__).hyperparameters.__set__(self, theta)
        hyps = self.hyperparameters
        csts = self.constants
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = csts[1] * hyps[0]
            super(IG_WarpingFunction, self.__class__).hyperparameters.__set__(self, hyps)


    @constants.setter
    def constants(self, consts):
        r'''
        Set the constants stored in the :code:`_WarpingFunction` object. Specific implementation due to maximum fraction limit.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        super(IG_WarpingFunction, self.__class__).constants.__set__(self, consts)
        hyps = self.hyperparameters
        csts = self.constants
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = csts[1] * hyps[0]
            super(IG_WarpingFunction, self.__class__).hyperparameters.__set__(self, hyps)


    @bounds.setter
    def bounds(self, bounds):
        r'''
        Set the hyperparameter bounds stored in the :code:`_WarpingFunction` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        super(IG_WarpingFunction, self.__class__).bounds.__set__(self, bounds)
        if self._force_bounds:
            hyps = self.hyperparameters
            csts = self.constants
            if hyps[1] > (csts[1] * hyps[0]):
                hyps[1] = csts[1] * hyps[0]
                super(IG_WarpingFunction, self.__class__).hyperparameters.__set__(self, hyps)


    def __copy__(self):
        r'''
        Implementation-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: object. An exact duplicate of the current instance, which can be modified without affecting the original.
        '''

        hyps = self.hyperparameters
        csts = self.constants
        bnds = self.bounds
        lbhp = float(hyps[0])
        ghhp = float(hyps[1])
        gshp = float(hyps[2])
        gmc = float(csts[0])
        lrc = float(csts[1])
        kcopy = IG_WarpingFunction(lbhp, ghhp, gshp, gmc, lrc)
        kcopy.enforce_bounds(self._force_bounds)
        if bnds is not None:
            kcopy.bounds = bnds
        return kcopy

