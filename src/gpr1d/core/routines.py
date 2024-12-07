r'''
Routine class for Gaussian Process Regression fitting of 1D data with errorbars. Built in Python 3.x, adapted to be Python 2.x compatible.
  06/12/2024: No longer compatible with Python 2.x.

These classes were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET Contributors
    IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis, Boston, MA (2017)
    `<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

'''
#    Gaussian process theory: "Gaussian Processes for Machine Learning", C.E. Rasmussen and C.K.I. Williams (2006)

# Required imports
import warnings
import copy
import numpy as np
import scipy.linalg as spla
import scipy.stats as spst
from operator import itemgetter

from .definitions import number_types, array_types, default_dtype
from .kernels import _Kernel, _WarpingFunction

__all__ = [
    'GaussianProcessRegression1D',  # Main interpolation class
]


class GaussianProcessRegression1D():
    r'''
    Class containing variable containers, get/set functions, and fitting functions required to
    perform Gaussian process regressions on 1-dimensional data.

    .. note::

        This implementation requires the specific implementation of the :code:`_Kernel`
        template class provided within the same package!
    '''

    def __init__(self):
        r'''
        Defines the input and output containers used within the class, but they still requires instantiation.
        '''

        self._kk = None
        self._kb = None
        self._lp = 1.0
        self._xx = None
        self._xe = None
        self._yy = None
        self._ye = None
        self._dxx = None
        self._dyy = None
        self._dye = None
        self._eps = None
        self._opm = 'grad'
        self._opp = np.array([1.0e-5])
        self._dh = 1.0e-2
        self._lb = None
        self._ub = None
        self._cn = None
        self._ekk = None
        self._ekb = None
        self._elp = 6.0
        self._enr = None
        self._eeps = None
        self._eopm = 'grad'
        self._eopp = np.array([1.0e-5])
        self._edh = 1.0e-2
        self._ikk = None
        self._imax = 500
        self._xF = None
        self._estF = None
        self._barF = None
        self._varF = None
        self._dbarF = None
        self._dvarF = None
        self._lml = None
        self._nulllml = None
        self._barE = None
        self._varE = None
        self._dbarE = None
        self._dvarE = None
        self._varN = None
        self._dvarN = None
        self._gpxe = None
        self._gpye = None
        self._egpye = None
        self._nikk = None
        self._niekk = None
        self._fwarn = False
        self._opopts = ['grad', 'mom', 'nag', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']


    def __eq__(self, other):
        r'''
        Custom equality operator, only compares input data due to statistical
        variance of outputs.

        :arg other: object. Another :code:`GaussianProcessRegression1D` object.

        :returns: bool. Indicates whether the two objects have identical inputs.
        '''

        status = False
        if isinstance(other, GaussianProcessRegression1D):
            skk = self._kk.name == other._kk.name if self._kk is not None and other._kk is not None else self._kk == other._kk
            skb = np.all(np.isclose(self._kb, other._kb)) if self._kb is not None and other._kb is not None else np.all(np.atleast_1d(self._kb == other._kb))
            seps = np.isclose(self._eps, other._eps) if self._eps is not None and other._eps is not None else self._eps == other._eps
            sekk = self._ekk.name == other._ekk.name if self._ekk is not None and other._ekk is not None else self._ekk == other._ekk
            sekb = np.all(np.isclose(self._ekb, other._ekb)) if self._ekb is not None and other._ekb is not None else np.all(np.atleast_1d(self._ekb == other._ekb))
            seeps = np.isclose(self._eeps, other._eeps) if self._eeps is not None and other._eeps is not None else self._eeps == other._eeps
            sxx = np.all(np.isclose(self._xx, other._xx)) if self._xx is not None and other._xx is not None else np.all(np.atleast_1d(self._xx == other._xx))
            sxe = np.all(np.isclose(self._xe, other._xe)) if self._xe is not None and other._xe is not None else np.all(np.atleast_1d(self._xe == other._xe))
            syy = np.all(np.isclose(self._yy, other._yy)) if self._yy is not None and other._yy is not None else np.all(np.atleast_1d(self._yy == other._yy))
            sye = np.all(np.isclose(self._ye, other._ye)) if self._ye is not None and other._ye is not None else np.all(np.atleast_1d(self._ye == other._ye))
            sdxx = np.all(np.isclose(self._dxx, other._dxx)) if self._dxx is not None and other._dxx is not None else np.all(np.atleast_1d(self._dxx == other._dxx))
            sdyy = np.all(np.isclose(self._dyy, other._dyy)) if self._dyy is not None and other._dyy is not None else np.all(np.atleast_1d(self._dyy == other._dyy))
            sdye = np.all(np.isclose(self._dye, other._dye)) if self._dye is not None and other._dye is not None else np.all(np.atleast_1d(self._dye == other._dye))
            slb = np.isclose(self._lb, other._lb) if self._lb is not None and other._lb is not None else self._lb == other._lb
            sub = np.isclose(self._ub, other._ub) if self._ub is not None and other._ub is not None else self._ub == other._ub
            scn = np.isclose(self._cn, other._cn) if self._cn is not None and other._cn is not None else self._cn == other._cn
            #print(skk, skb, seps, sekk, sekb, seeps, sxx, sxe, sye, sdxx, sdyy, sdye, slb, sub, scn)
            status = (
                skk and skb and seps and sekk and sekb and seeps and
                sxx and sxe and syy and sye and sdxx and sdyy and sdye and
                np.isclose(self._lp,other._lp) and np.isclose(self._elp, other._elp) and
                slb and sub and scn and
                (self._opm == other._opm and np.all(np.isclose(self._opp, other._opp))) and
                (self._eopm == other._eopm and np.all(np.isclose(self._eopp, other._eopp)))
            )
            
        return status


    def __ne__(self, other):
        r'''
        Custom inequality operator, only compares input data due to statistical
        variance of outputs.

        :arg other: object. Another :code:`GaussianProcessRegression1D` object.

        :returns: bool. Indicates whether the two objects do not have identical inputs.
        '''

        return not self.__eq__(other)


    def set_kernel(self, kernel=None, kbounds=None, regpar=None):
        r'''
        Specify the kernel that the Gaussian process regression will be performed with.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility. (optional)

        :returns: none.
        '''

        if isinstance(kernel, _Kernel):
            self._kk = copy.copy(kernel)
            self._ikk = copy.copy(self._kk)
        if isinstance(self._kk, _Kernel):
            kh = np.log10(self._kk.hyperparameters)
            if isinstance(kbounds, array_types):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb.T) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._kb = kb
                self._kk.bounds = np.power(10.0, self._kb)
        if isinstance(regpar, number_types) and float(regpar) > 0.0:
            self._lp = float(regpar)


    def set_raw_data(self, xdata=None, ydata=None, xerr=None, yerr=None, dxdata=None, dydata=None, dyerr=None):
        r'''
        Specify the raw data that the Gaussian process regression will be performed on.
        Performs some consistency checks between the input raw data to ensure validity.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg xerr: array. Vector of x-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. Vector of x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. Vector of dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. Vector of dy/dx-errors of derivative data points to be included in fit. (optional)

        :returns: none.
        '''

        altered = False
        if isinstance(xdata, (list, tuple)) and len(xdata) > 0:
            self._xx = np.array(xdata).flatten()
            altered = True
        elif isinstance(xdata, np.ndarray) and xdata.size > 0:
            self._xx = xdata.flatten()
            altered = True
        if isinstance(xerr, (list, tuple)) and len(xerr) > 0:
            self._xe = np.array(xerr).flatten()
            altered = True
        elif isinstance(xerr, np.ndarray) and xerr.size > 0:
            self._xe = xerr.flatten()
            altered = True
        elif isinstance(xerr, str):
            self._xe = None
            altered = True
        if isinstance(ydata, (list, tuple)) and len(ydata) > 0:
            self._yy = np.array(ydata).flatten()
            altered = True
        elif isinstance(ydata, np.ndarray) and ydata.size > 0:
            self._yy = ydata.flatten()
            altered = True
        if isinstance(yerr, (list, tuple)) and len(yerr) > 0:
            self._ye = np.array(yerr).flatten()
            altered = True
        elif isinstance(yerr, np.ndarray) and yerr.size > 0:
            self._ye = yerr.flatten()
            altered = True
        elif isinstance(yerr, str):
            self._ye = None
            altered = True
        if isinstance(dxdata, (list, tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            self._dxx = temp.flatten()
            altered = True
        elif isinstance(dxdata, np.ndarray) and dxdata.size > 0:
            self._dxx = dxdata.flatten()
            altered = True
        elif isinstance(dxdata, str):
            self._dxx = None
            altered = True
        if isinstance(dydata, (list, tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            self._dyy = temp.flatten()
            altered = True
        elif isinstance(dydata, np.ndarray) and dydata.size > 0:
            self._dyy = dydata.flatten()
            altered = True
        elif isinstance(dydata, str):
            self._dyy = None
            altered = True
        if isinstance(dyerr, (list, tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            self._dye = temp.flatten()
            altered = True
        elif isinstance(dyerr, np.ndarray) and dyerr.size > 0:
            self._dye = dyerr.flatten()
            altered = True
        elif isinstance(dyerr, str):
            self._dye = None
            altered = True
        if altered:
            self._gpxe = None
            self._gpye = None
            self._egpye = None
            self._nikk = None
            self._niekk = None


    def set_conditioner(self, condnum=None, lbound=None, ubound=None):
        r'''
        Specify the parameters to ensure the condition number of the matrix is good,
        as well as set upper and lower bounds for the input data to be included.

        :kwarg condnum: float. Minimum allowable delta-x for input data before applying Gaussian blending to data points.

        :kwarg lbound: float. Minimum allowable y-value for input data, values below are omitted from fit procedure. (optional)

        :kwarg ubound: float. Maximum allowable y-value for input data, values above are omitted from fit procedure. (optional)

        :returns: none.
        '''

        if isinstance(condnum, number_types) and float(condnum) > 0.0:
            self._cn = float(condnum)
        elif isinstance(condnum, number_types) and float(condnum) <= 0.0:
            self._cn = None
        elif isinstance(condnum, str):
            self._cn = None
        if isinstance(lbound, number_types):
            self._lb = float(lbound)
        elif isinstance(lbound, str):
            self._lb = None
        if isinstance(ubound, number_types):
            self._ub = float(ubound)
        elif isinstance(ubound, str):
            self._ub = None


    def set_error_kernel(self, kernel=None, kbounds=None, regpar=None, nrestarts=None):
        r'''
        Specify the kernel that the Gaussian process regression on the error function
        will be performed with.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the error data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility. (optional)

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within :code:`kbounds`. (optional)

        :returns: none.
        '''

        altered = False
        if isinstance(kernel, _Kernel):
            self._ekk = copy.copy(kernel)
            altered = True
        if isinstance(self._ekk, _Kernel):
            kh = np.log10(self._ekk.hyperparameters)
            if isinstance(kbounds, array_types):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb.T) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._ekb = kb
                self._ekk.bounds = np.power(10.0, self._ekb)
                altered = True
        if isinstance(regpar, number_types) and float(regpar) > 0.0:
            self._elp = float(regpar)
            altered = True
        if isinstance(nrestarts, number_types):
            self._enr = int(nrestarts) if int(nrestarts) > 0 else 0
        if altered:
            self._gpxe = None
            self._gpye = None
            self._egpye = None
            self._nikk = None
            self._niekk = None


    def set_search_parameters(self, epsilon=None, method=None, spars=None, sdiff=None, maxiter=None):
        r'''
        Specify the search parameters that the Gaussian process regression will use.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include:
                       ['grad', 'mom', 'nag', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.
	              **Only** used if analytical implementation of kernel derivative is not present! (optional)

        :kwarg maxiter: int. Maximum number of iterations for hyperparameter optimization algorithm, default is 500. (optional)

        :returns: none.
        '''

        midx = None
        if isinstance(epsilon, number_types) and float(epsilon) > 0.0:
            self._eps = float(epsilon)
        elif isinstance(epsilon, number_types) and float(epsilon) <= 0.0:
            self._eps = None
        elif isinstance(epsilon, str):
            self._eps = None
        if isinstance(method, str):
            mstr = method.lower()
            if mstr in self._opopts:
                midx = self._opopts.index(mstr)
        elif isinstance(method, number_types) and int(method) >= 0 and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                self._opm = self._opopts[1]
                opp = np.array([1.0e-4, 0.9]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 2:
                self._opm = self._opopts[2]
                opp = np.array([1.0e-4, 0.9]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 3:
                self._opm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 4:
                self._opm = self._opopts[4]
                opp = np.array([1.0e-2, 0.9]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 5:
                self._opm = self._opopts[5]
                opp = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 6:
                self._opm = self._opopts[6]
                opp = np.array([2.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 7:
                self._opm = self._opopts[7]
                opp = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            else:
                self._opm = self._opopts[0]
                opp = np.array([1.0e-4]).flatten()
                for ii in np.arange(0, self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
        if isinstance(spars, (list, tuple)):
            for ii in np.arange(0, len(spars)):
                if ii < self._opp.size and isinstance(spars[ii], number_types):
                    self._opp[ii] = float(spars[ii])
        elif isinstance(spars, np.ndarray):
            for ii in np.arange(0, spars.size):
                if ii < self._opp.size and isinstance(spars[ii], number_types):
                    self._opp[ii] = float(spars[ii])
        if isinstance(sdiff, number_types) and float(sdiff) > 0.0:
            self._dh = float(sdiff)
        if isinstance(maxiter, number_types) and int(maxiter) > 0:
            self._imax = int(maxiter) if int(maxiter) > 50 else 50


    def set_error_search_parameters(self, epsilon=None, method=None, spars=None, sdiff=None):
        r'''
        Specify the search parameters that the Gaussian process regression will use for the error function.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include:
                       ['grad', 'mom', 'nag', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.
	              **Only** used if analytical implementation of kernel derivative is not present! (optional)

        :returns: none.
        '''

        emidx = None
        if isinstance(epsilon, number_types) and float(epsilon) > 0.0:
            self._eeps = float(epsilon)
        elif isinstance(epsilon, number_types) and float(epsilon) <= 0.0:
            self._eeps = None
        elif isinstance(epsilon, str):
            self._eeps = None
        if isinstance(method, str):
            mstr = method.lower()
            if mstr in self._opopts:
                emidx = self._opopts.index(mstr)
        elif isinstance(method, number_types) and int(method) >= 0 and int(method) < len(self._opopts):
            emidx = int(method)
        if emidx is not None:
            if emidx == 1:
                self._eopm = self._opopts[1]
                opp = np.array([1.0e-4, 0.9]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 2:
                self._eopm = self._opopts[2]
                opp = np.array([1.0e-4, 0.9]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 3:
                self._eopm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 4:
                self._eopm = self._opopts[4]
                opp = np.array([1.0e-2, 0.9]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 5:
                self._eopm = self._opopts[5]
                opp = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 6:
                self._eopm = self._opopts[6]
                opp = np.array([2.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 7:
                self._eopm = self._opopts[7]
                opp = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            else:
                self._eopm = self._opopts[0]
                opp = np.array([1.0e-4]).flatten()
                for ii in np.arange(0, self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
        if isinstance(spars, (list, tuple)):
            for ii in np.arange(0, len(spars)):
                if ii < self._eopp.size and isinstance(spars[ii], number_types):
                    self._eopp[ii] = float(spars[ii])
        elif isinstance(spars, np.ndarray):
            for ii in np.arange(0, spars.size):
                if ii < self._eopp.size and isinstance(spars[ii], number_types):
                    self._eopp[ii] = float(spars[ii])
        if isinstance(sdiff, number_types) and float(sdiff) > 0.0:
            self._edh = float(sdiff)


    def set_warning_flag(self, flag=True):
        r'''
        Specify the printing of runtime warnings within the
        hyperparameter optimization routine. The warnings are
        disabled by default but calling this function will
        enable them by default.

        :kwarg flag: bool. Flag to toggle display of warnings.

        :returns: none.
        '''

        self._fwarn = True if flag else False


    def reset_error_kernel(self):
        r'''
        Resets error kernel and associated settings to an empty
        state. Primarily used for setting up identical objects
        for comparison and testing purposes.

        :returns: none.
        '''

        self._ekk = None
        self._ekb = None
        self._elp = 6.0
        self._enr = None
        self._eeps = None
        self._eopm = 'grad'
        self._eopp = np.array([1.0e-5])
        self._edh = 1.0e-2
        self._gpxe = None
        self._gpye = None
        self._egpye = None
        self._nikk = None


    def get_raw_data(self):
        r'''
        Returns the input raw data passed in latest :code:`set_raw_data()` call,
        without any internal processing.

        :returns: (array, array, array, array, array, array, array).
            Vectors in order of x-values, y-values, x-errors, y-errors, derivative x-values, dy/dx-values, dy/dx-errors.
        '''

        rxx = copy.deepcopy(self._xx)
        ryy = copy.deepcopy(self._yy)
        rxe = copy.deepcopy(self._xe)
        rye = copy.deepcopy(self._ye)
        rdxx = copy.deepcopy(self._dxx)
        rdyy = copy.deepcopy(self._dyy)
        rdye = copy.deepcopy(self._dye)
        return (rxx, ryy, rxe, rye, rdxx, rdyy, rdye)


    def get_processed_data(self):
        r'''
        Returns the input data passed into the latest :code:`GPRFit()` call,
        including all internal processing performed by that call.

        .. note::

            If :code:`GPRFit()` was executed with :code:`nigp_flag = True`, then
            the raw x-error data is folded into the y-error. As such, this
            function only returns y-errors.

        :returns: (array, array, array, array, array, array, array).
            Vectors in order of x-values, y-values, y-errors, derivative x-values, dy/dx-values, dy/dx-errors.
        '''

        pxx = copy.deepcopy(self._xx)
        pyy = copy.deepcopy(self._yy)
        pye = copy.deepcopy(self._ye)
        if isinstance(pxx, np.ndarray) and isinstance(pyy, np.ndarray):
            rxe = self._xe if self._xe is not None else np.zeros(pxx.shape)
            rye = self._ye if self._gpye is None else self._gpye
            if rye is None:
                rye = np.zeros(pyy.shape)
            lb = -1.0e50 if self._lb is None else self._lb
            ub = 1.0e50 if self._ub is None else self._ub
            cn = 5.0e-3 if self._cn is None else self._cn
            (pxx, pxe, pyy, pye, nn) = self._condition_data(self._xx, rxe,self._yy, rye, lb, ub, cn)
        # Actually these should be conditioned as well (for next version?)
        dxx = copy.deepcopy(self._dxx)
        dyy = copy.deepcopy(self._dyy)
        dye = copy.deepcopy(self._dye)
        return (pxx, pyy, pye, dxx, dyy, dye)


    def get_gp_x(self):
        r'''
        Returns the x-values used in the latest :code:`GPRFit()` call.

        :returns: array. Vector of x-values corresponding to predicted y-values.
        '''

        return copy.deepcopy(self._xF)


    def get_gp_regpar(self):
        r'''
        Returns the regularization parameter value used in the latest :code:`GPRFit()` call.

        :returns: float. Regularization parameter value used in cost function evaluation.
        '''

        return self._lp


    def get_gp_error_regpar(self):
        r'''
        Returns the regularization parameter value used for error function fitting in the latest :code:`GPRFit()` call.

        :returns: float. Regularization parameter value used in cost function evaluation for error function fitting.
        '''

        return self._elp


    def get_gp_mean(self):
        r'''
        Returns the y-values computed in the latest :code:`GPRFit()` call.

        :returns: array. Vector of predicted y-values from fit.
        '''

        return copy.deepcopy(self._barF)


    # TODO: Place process noise fraction on GPRFit() level and remove the argument from these functions, currently introduces inconsistencies in statistics
    def get_gp_variance(self, noise_flag=True, noise_mult=None):
        r'''
        Returns the full covariance matrix of the y-values computed in the latest
        :code:`GPRFit()` call.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg noise_mult: float. Noise term multiplier to introduce known bias or covariance in data, must be greater than or equal to zero. (optional)

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        '''

        varF = copy.deepcopy(self._varF)
        if varF is not None and self._varN is not None and noise_flag:
            nfac = float(noise_mult) ** 2.0 if isinstance(noise_mult, number_types) and float(noise_mult) >= 0.0 else 1.0
            varF = varF + nfac * self._varN
        return varF


    def get_gp_std(self, noise_flag=True, noise_mult=None):
        r'''
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma error of fit.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors. (optional)

        :kwarg noise_mult: float. Noise term multiplier to introduce known bias or covariance in data, must be greater than or equal to zero. (optional)

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        '''

        sigF = None
        varF = self.get_gp_variance(noise_flag=noise_flag, noise_mult=noise_mult)
        if varF is not None:
            sigF = np.sqrt(np.diag(varF))
        return sigF


    def get_gp_drv_mean(self):
        r'''
        Returns the dy/dx-values computed in the latest :code:`GPRFit()` call.

        :returns: array. Vector of predicted dy/dx-values from fit, if requested in fit call.
        '''

        return copy.deepcopy(self._dbarF)


    def get_gp_drv_variance(self, noise_flag=True, process_noise_fraction=None):
        r'''
        Returns the full covariance matrix of the dy/dx-values computed in the latest
        :code:`GPRFit()` call.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: array. 2D meshgrid array containing full covariance matrix for predicted dy/dx-values from fit, if requested in fit call.
        '''

        dvarF = copy.deepcopy(self._dvarF)
        if dvarF is not None:
            dvar_numer = self.get_gp_variance(noise_flag=noise_flag, noise_mult=process_noise_fraction)
            dvar_denom = self.get_gp_variance(noise_flag=False)
            dvar_denom[dvar_denom == 0.0] = 1.0
            dvarmod = dvar_numer / dvar_denom
            if self._dvarN is not None and noise_flag:
                nfac = float(process_noise_fraction) ** 2.0 if isinstance(process_noise_fraction, number_types) and float(process_noise_fraction) >= 0.0 and float(process_noise_fraction) <= 1.0 else 1.0
                dvarF = dvarmod * dvarF + nfac * self._dvarN
            else:
                dvarF = dvarmod * dvarF
        return dvarF


    def get_gp_drv_std(self, noise_flag=True, process_noise_fraction=None):
        r'''
        Returns only the rooted diagonal elements of the covariance matrix of the 
        dy/dx-values computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma
        error of fit.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: array. 1D array containing 1 sigma errors of predicted dy/dx-values from fit, if requested in fit call.
        '''

        dsigF = None
        dvarF = self.get_gp_drv_variance(noise_flag=noise_flag, process_noise_fraction=process_noise_fraction)
        if dvarF is not None:
            dsigF = np.sqrt(np.diag(dvarF))
        return dsigF


    def get_gp_results(self, rtn_cov=False, noise_flag=True, process_noise_fraction=None):
        r'''
        Returns all common predicted values computed in the latest :code:`GPRFit()` call.

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead the 1 sigma errors. (optional)

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variances or errors. (optional)

        :kwarg process_noise_fraction: float. Specify split between process noise and observation noise in data, must be between zero and one. (optional)

        :returns: (array, array, array, array).
            Vectors in order of y-values, y-errors, dy/dx-values, dy/dx-errors.
        '''

        ra = self.get_gp_mean()
        rb = self.get_gp_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_std(noise_flag=noise_flag)
        rc = self.get_gp_drv_mean()
        rd = self.get_gp_drv_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_drv_std(noise_flag=noise_flag, process_noise_fraction=process_noise_fraction)
        return (ra, rb, rc, rd)


    def get_gp_lml(self):
        r'''
        Returns the log-marginal-likelihood of the latest :code:`GPRFit()` call.

        :returns: float. Log-marginal-likelihood value from fit.
        '''

        return self._lml


    def get_gp_null_lml(self):
        r'''
        Returns the log-marginal-likelihood for the null hypothesis, calculated by the latest :code:`GPRFit()` call.
        This value can be used to normalize the log-marginal-likelihood of the fit for a generalized goodness-of-fit metric.

        :returns: float. Log-marginal-likelihood value of null hypothesis.
        '''

        return self._nulllml


    def get_gp_r2(self):
        r'''
        Calculates the R-squared (coefficient of determination) using the results of the latest :code:`GPRFit()` call.

        :returns: float. R-squared value.
        '''
        r2 = None
        if self._xF is not None and self._estF is not None:
            myy = np.nanmean(self._yy)
            sstot = np.sum(np.power(self._yy - myy, 2.0))
            ssres = np.sum(np.power(self._yy - self._estF, 2.0))
            r2 = 1.0 - (ssres / sstot)
        return r2


    def get_gp_adjusted_r2(self):
        r'''
        Calculates the adjusted R-squared (coefficient of determination) using the results of the latest :code:`GPRFit()`
        call.

        :returns: float. Adjusted R-squared value.
        '''

        adjr2 = None
        if self._xF is not None and self._estF is not None:
            myy = np.nanmean(self._yy)
            sstot = np.sum(np.power(self._yy - myy, 2.0))
            ssres = np.sum(np.power(self._yy - self._estF, 2.0))
            kpars = np.hstack((self._kk.hyperparameters, self._kk.constants))
            adjr2 = 1.0 - (ssres / sstot) * (self._xx.size - 1.0) / (self._xx.size - kpars.size - 1.0)
        return adjr2


    def get_gp_generalized_r2(self):
        r'''
        Calculates the Cox and Snell pseudo R-squared (coefficient of determination) using the results of the latest
        :code:`GPRFit()` call.

        .. note:: This particular metric is for logistic regression and may not be fully applicable to generalized polynomial
                  regression. However, they are related here through the use of maximum likelihood optimization. Use with
                  extreme caution!!!

        :returns: float. Generalized pseudo R-squared value based on Cox and Snell methodology.
        '''

        genr2 = None
        if self._xF is not None:
            genr2 = 1.0 - np.exp(2.0 * (self._nulllml - self._lml) / self._xx.size)
        return genr2


    def get_gp_input_kernel(self):
        r'''
        Returns the original input kernel, with settings retained from before the
        hyperparameter optimization step.

        :returns: object. The original input :code:`_Kernel` instance, saved from the latest :code:`set_kernel()` call.
        '''

        return self._ikk


    def get_gp_kernel(self):
        r'''
        Returns the optimized kernel determined in the latest :code:`GPRFit()` call.

        :returns: object. The :code:`_Kernel` instance from the latest :code:`GPRFit()` call, including optimized hyperparameters if fit was performed.
        '''

        return self._kk


    def get_gp_kernel_details(self):
        r'''
        Returns the data needed to save the optimized kernel determined in the latest :code:`GPRFit()` call.

        :returns: (str, array, float).
            Kernel codename, vector of kernel hyperparameters and constants, regularization parameter.
        '''

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._kk, _Kernel):
            kname = self._kk.name
            kpars = np.hstack((self._kk.hyperparameters, self._kk.constants))
            krpar = self._lp
        return (kname, kpars, krpar)


    def get_gp_error_kernel(self):
        r'''
        Returns the optimized error kernel determined in the latest :code:`GPRFit()` call.

        :returns: object. The error :code:`_Kernel` instance from the latest :code:`GPRFit()` call, including optimized hyperparameters if fit was performed.
        '''

        return self._ekk


    def get_gp_error_kernel_details(self):
        r'''
        Returns the data needed to save the optimized error kernel determined in the latest :code:`GPRFit()` call.

        :returns: (str, array).
            Kernel codename, vector of kernel hyperparameters and constants, regularization parameter.
        '''

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._ekk, _Kernel):
            kname = self._ekk.name
            kpars = np.hstack((self._ekk.hyperparameters, self._ekk.constants))
            krpar = self._elp
        return (kname, kpars, krpar)


    def get_error_gp_mean(self):
        r'''
        Returns the fitted y-errors computed in the latest :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. Vector of predicted y-values from fit.
        '''

        return copy.deepcopy(self._barE)


    def get_error_gp_variance(self):
        r'''
        Returns the full covariance matrix of the fitted y-errors computed in the latest
        :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        '''

        return copy.deepcopy(self._varE)


    def get_error_gp_std(self):
        r'''
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest :code:`GPRFit()` call, corresponds to 1 sigma error of fit.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        '''

        sigE = None
        varE = self.get_error_gp_variance()
        if varE is not None:
            sigE = np.sqrt(np.diag(varE))
        return sigE


    def eval_error_function(self, xnew, enforce_positive=True):
        r'''
        Returns the error values used in heteroscedastic GPR, evaluated at the input x-values,
        using the error kernel determined in the latest :code:`GPRFit()` call.

        .. warning::

            These values represent information about the **input** y-error function,
            **not to be confused** with the y-errors of the GPR predictive
            distribution!!!

        :arg xnew: array. Vector of x-values at which the predicted error function should be evaluated at.

        :kwarg enforce_positive: bool. Returns of absolute values of the error function if :code:`True`.

        :returns: array. Vecotr of predicted y-errors from the fit using the error kernel.
        '''

        xn = None
        if isinstance(xnew, (list, tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew, np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        barE = None
        if xn is not None and self._gpye is not None and self._egpye is not None:
            barE = itemgetter(0)(self.__basic_fit(xn, kernel=self._ekk, ydata=self._gpye, yerr=self._egpye, epsilon='None'))
            if enforce_positive:
                barE = np.abs(barE)
        return barE


    def _gp_base_alg(self, xn, kk, lp, xx, yy, ye, dxx, dyy, dye, dd):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for 1-dimensional Gaussian process regression, with no
        idiot-proofing and no pre- or post-processing.

        .. note::

            It is **strongly recommended** that :code:`kk` be a :code:`_Kernel` instance as
            specified in this package but, within this function, it can essentially be any
            callable object which accepts the arguments :code:`(x1,x2,dd)`.

        :arg xn: array. Vector of x-values at which the fit will be evaluated.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given in 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg dd: int. Derivative order of output prediction.

        :returns: (array, array, float).
            Vector of predicted mean values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        '''

        # Set up the problem grids for calculating the required matrices from covf
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx, xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy, yyd)
        yed = dye if dflag else []
        yef = np.append(ye, yed)
        (x1, x2) = np.meshgrid(xx, xx)
        (x1h1, x2h1) = np.meshgrid(xx, xxd)
        (x1h2, x2h2) = np.meshgrid(xxd, xx)
        (x1d, x2d) = np.meshgrid(xxd, xxd)
        (xs1, xs2) = np.meshgrid(xn, xx)
        (xs1h, xs2h) = np.meshgrid(xn, xxd)
        (xt1, xt2) = np.meshgrid(xn, xn)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1, x2, der=0)
        KKh1 = kk(x1h1, x2h1, der=1)
        KKh2 = kk(x1h2, x2h2, der=-1)
        KKd = kk(x1d, x2d, der=2)
        KK = np.vstack((np.hstack((KKb, KKh2)), np.hstack((KKh1, KKd))))
        ksb = kk(xs1, xs2, der=-dd) if dd == 1 else kk(xs1, xs2, der=dd)
        ksh = kk(xs1h, xs2h, der=dd+1)
        ks = np.vstack((ksb, ksh))
        kt = kk(xt1, xt2, der=2*dd)
        kernel = KK + np.diag(yef ** 2.0)

        cholesky_flag = True
        if cholesky_flag:
            LL = spla.cholesky(kernel, lower=True)
            alpha = spla.cho_solve((LL, True), yf)
            kv = spla.cho_solve((LL, True), ks)
            #vv = np.dot(LL.T, spla.cho_solve((LL, True), ks))
            #kvs = np.dot(vv.T, vv)
            ldet = 2.0 * np.sum(np.log(np.diag(LL)))
        else:
            alpha = spla.solve(kernel, yf)
            kv = spla.solve(kernel, ks)
            #kvs = np.dot(ks.T, kv)
            ldet = np.log(spla.det(kernel))

        barF = np.dot(ks.T, alpha)          # Mean function
        varF = kt - np.dot(ks.T, kv)        # Variance of mean function

        # Log-marginal-likelihood provides an indication of how statistically well the fit describes the training data
        #    1st term: Describes the goodness of fit for the given data
        #    2nd term: Penalty for complexity / simplicity of the covariance function
        #    3rd term: Penalty for the size of given data set
        lml = -0.5 * np.dot(yf.T, alpha) - 0.5 * lp * ldet - 0.5 * xf.size * np.log(2.0 * np.pi)

        # Log-marginal-likelihood of the null hypothesis (constant at mean value),
        # can be used as a normalization factor for general goodness-of-fit metric
        zfilt = (np.abs(yef) >= 1.0e-10)
        yft = [0.0]
        yeft = [0.0]
        if np.any(zfilt):
            yft = np.power(yf[zfilt] / yef[zfilt], 2.0)
            yeft = 2.0 * np.log(yef[zfilt])
        lmlz = -0.5 * np.sum(yft) - 0.5 * lp * np.sum(yeft) - 0.5 * xf.size * np.log(2.0 * np.pi)

        return (barF, varF, lml, lmlz)


    def _gp_brute_deriv1(self, xn, kk, lp, xx, yy, ye):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for brute-force first-order derivative of 1-dimensional Gaussian process regression.
        **Not recommended for production runs**, but useful for testing custom :code:`_Kernel` class implementations
        which have hard-coded derivative calculations.

        :arg xn: array. Vector of x-values at which the fit will be evaluated.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :returns: (array, array, float).
            Vector of predicted mean derivative values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        '''

        # Set up the problem grids for calculating the required matrices from covf
        (x1, x2) = np.meshgrid(xx, xx)
        (xs1, xs2) = np.meshgrid(xx, xn)
        (xt1, xt2) = np.meshgrid(xn, xn)
        # Set up predictive grids with slight offset in x1 and x2, forms corners of a box around original xn point
        step = np.amin(np.abs(np.diff(xn)))
        xnl = xn - step * 0.5e-3            # The step is chosen intelligently to be smaller than smallest dxn
        xnu = xn + step * 0.5e-3
        (xl1, xl2) = np.meshgrid(xx, xnl)
        (xu1, xu2) = np.meshgrid(xx, xnu)
        (xll1, xll2) = np.meshgrid(xnl, xnl)
        (xlu1, xlu2) = np.meshgrid(xnu, xnl)
        (xuu1, xuu2) = np.meshgrid(xnu, xnu)

        KK = kk(x1, x2)
        LL = spla.cholesky(KK + np.diag(ye ** 2.0),lower=True)
        alpha = spla.cho_solve((LL, True), yy)
        # Approximation of first derivative of covf (df/dxn1)
        ksl = kk(xl1, xl2)
        ksu = kk(xu1, xu2)
        dks = (ksu.T - ksl.T) / (step * 1.0e-3)
        dvv = np.dot(LL.T, spla.cho_solve((LL, True), dks))
        # Approximation of second derivative of covf (d^2f/dxn1 dxn2)
        ktll = kk(xll1, xll2)
        ktlu = kk(xlu1, xlu2)
        ktul = ktlu.T
        ktuu = kk(xuu1, xuu2)
        dktl = (ktlu - ktll) / (step * 1.0e-3)
        dktu = (ktuu - ktul) / (step * 1.0e-3)
        ddkt = (dktu - dktl) / (step * 1.0e-3)
        barF = np.dot(dks.T, alpha)          # Mean function
        varF = ddkt - np.dot(dvv.T, dvv)     # Variance of mean function
        lml = -0.5 * np.dot(yy.T, alpha) - lp * np.sum(np.log(np.diag(LL))) - 0.5 * xx.size * np.log(2.0 * np.pi)

        return (barF, varF, lml)


    def _gp_brute_grad_lml(self, kk, lp, xx, yy, ye, dxx, dyy, dye, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for brute-force computation of gradient of log-marginal-likelihood with respect to the
        hyperparameters in logarithmic space. Result must be divided by :code:`ln(10) * theta` in order to have the
        gradient with respect to the hyperparameters in linear space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg dh: float. Step size in hyperparameter space used in derivative approximation.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        '''

        xn = np.array([0.0])
        theta = np.log10(kk.hyperparameters)
        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0, theta.size):
            testkk = copy.copy(kk)
            theta_in = theta.copy()
            theta_in[ii] = theta[ii] - 0.5 * dh
            testkk.hyperparameters = np.power(10.0, theta_in)
            llml = itemgetter(2)(self._gp_base_alg(xn, testkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            theta_in[ii] = theta[ii] + 0.5 * dh
            testkk.hyperparameters = np.power(10.0, theta_in)
            ulml = itemgetter(2)(self._gp_base_alg(xn, testkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            gradlml[ii] = (ulml - llml) / dh

        return gradlml


    def _gp_grad_lml(self, kk, lp, xx, yy, ye, dxx, dyy, dye):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Bare-bones algorithm for computation of gradient of log-marginal-likelihood with respect to the hyperparameters
        in linear space. Result must be multiplied by :code:`ln(10) * theta` in order to have the gradient with respect
        to the hyperparameters in logarithmic space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        '''

        # Set up the problem grids for calculating the required matrices from covf
        theta = kk.hyperparameters
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx, xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy, yyd)
        yed = dye if dflag else []
        yef = np.append(ye, yed)
        (x1, x2) = np.meshgrid(xx, xx)
        (x1h1, x2h1) = np.meshgrid(xx, xxd)
        (x1h2, x2h2) = np.meshgrid(xxd, xx)
        (x1d, x2d) = np.meshgrid(xxd, xxd)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1, x2, der=0)
        KKh1 = kk(x1h1, x2h1, der=1)
        KKh2 = kk(x1h2, x2h2, der=-1)
        KKd = kk(x1d, x2d, der=2)
        KK = np.vstack((np.hstack((KKb, KKh2)), np.hstack((KKh1, KKd))))
        kernel = KK + np.diag(yef ** 2.0)

        cholesky_flag = True
        if cholesky_flag:
            LL = spla.cholesky(kernel, lower=True)
            alpha = spla.cho_solve((LL, True), yf)
        else:
            alpha = spla.solve(kernel, yf)

        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0, theta.size):
            HHb = kk(x1, x2, der=0, hder=ii)
            HHh1 = kk(x1h1, x2h1, der=1, hder=ii)
            HHh2 = kk(x1h2, x2h2, der=-1, hder=ii)
            HHd = kk(x1d, x2d, der=2, hder=ii)
            HH = np.vstack((np.hstack((HHb, HHh2)), np.hstack((HHh1, HHd))))
            PP = np.dot(alpha.T, HH)
            if cholesky_flag:
                QQ = spla.cho_solve((LL, True), HH)
            else:
                QQ = spla.solve(kernel, HH)
            gradlml[ii] = 0.5 * np.dot(PP, alpha) - 0.5 * lp * np.sum(np.diag(QQ))

        return gradlml


    def _gp_grad_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Gradient ascent hyperparameter optimization algorithm, searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            theta_step = eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on gradient ascent search.')
        return (newkk, lmlnew)


    def _gp_momentum_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, gam, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Gradient ascent hyperparameter optimization algorithm with momentum, searches hyperparameters
        in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on momentum gradient ascent search.')
        return (newkk, lmlnew)


    def _gp_nesterov_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, gam, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Nesterov-accelerated gradient ascent hyperparameter optimization algorithm with momentum,
        searches hyperparameters in log-space. Effectively makes prediction of the next step and
        uses that with back-correction factor as the current update.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        if newkk.is_hderiv_implemented():
            # Hyperparameter derivatives computed in linear space
            gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
            gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_base)
        else:
            gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
        theta_step = eta * gradlml
        theta_old = theta_base.copy()
        theta_new = theta_old + theta_step
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        while dlml > eps and icount < self._imax:
            newkk.hyperparameters = np.power(10.0,theta_new)
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on Nesterov gradient ascent search.')
        return (newkk, lmlnew)


    def _gp_adagrad_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive gradient ascent hyperparameter optimization algorithm, searches hyperparameters
        in log-space. Suffers from extremely aggressive step modification due to continuous
        accumulation of denominator term, recommended to use :code:`adadelta` algorithm.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx` if given.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx` if given.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-2.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        gold = np.zeros(theta_base.shape)
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            gnew = gold + np.power(gradlml, 2.0)
            theta_step = eta * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive gradient ascent search.')
        return (newkk, lmlnew)


    def _gp_adadelta_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, gam, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive gradient ascent hyperparameter optimization algorithm with decaying accumulation
        window, searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Initial guess for gain factor on gradient to define next step, recommended 1.0e-2.

        :arg gam: float. Forgetting factor on accumulated gradient term, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        etatemp = np.ones(theta_base.shape) * eta
        told = theta_step.copy()
        gold = np.zeros(theta_base.shape)
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            gnew = gam * gold + (1.0 - gam) * np.power(gradlml, 2.0)
            theta_step = etatemp * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            tnew = gam * told + (1.0 - gam) * np.power(theta_step, 2.0)
            etatemp = np.sqrt(tnew + 1.0e-8)
            told = tnew
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on decaying adaptive gradient ascent search.')
        return (newkk, lmlnew)


    def _gp_adam_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, b1, b2, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive moment estimation hyperparameter optimization algorithm, searches hyperparameters
        in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml, 2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml, 2.0)
            theta_step = eta * (mnew / (1.0 - (b1 ** (icount + 1)))) / (np.sqrt(vnew / (1.0 - (b2 ** (icount + 1)))) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive moment estimation search.')
        return (newkk, lmlnew)


    def _gp_adamax_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, b1, b2, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Adaptive moment estimation hyperparameter optimization algorithm with l-infinity, searches
        hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 2.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml, 2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml, 2.0)
            unew = b2 * vnew if vold is None else np.nanmax([b2 * vold, np.abs(gradlml)], axis=0)
            theta_step = eta * (mnew / (1.0 - (b1 ** (icount + 1)))) / unew
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on adaptive moment l-infinity search.')
        return (newkk, lmlnew)


    def _gp_nadam_optimizer(self, kk, lp, xx, yy, ye, dxx, dyy, dye, eps, eta, b1, b2, dh):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Nesterov-accelerated adaptive moment estimation hyperparameter optimization algorithm,
        searches hyperparameters in log-space.

        .. note::

            The optimizer is limited to :code:`self._imax` attempts to achieve the desired convergence
            criteria. A message generated when the maximum number of iterations is reached without
            desired convergence, but this does not mean that the resulting fit is necessarily poor.

        :arg kk: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as :code:`dxx`.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit, assumed to be given as 1 sigma. Must have same dimensions as :code:`dxx`.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient, recommended 1.0e-2. **Only** applicable if brute-force derivative is used.

        :returns: (object, float).
            Final :code:`_Kernel` instance resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        '''

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = np.log10(newkk.hyperparameters)
        gradlml = np.zeros(theta_base.shape)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        while dlml > eps and icount < self._imax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self._gp_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0, theta_old)
            else:
                gradlml = self._gp_brute_grad_lml(newkk, lp, xx, yy, ye, dxx, dyy, dye, dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml, 2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml, 2.0)
            theta_step = eta * (mnew / (1.0 - (b1 ** (icount + 1))) + (1.0 - b1) * gradlml / (1.0 - (b1 ** (icount + 1)))) / (np.sqrt(vnew / (1.0 - b2)) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.hyperparameters = np.power(10.0, theta_new)
            lmlnew = itemgetter(2)(self._gp_base_alg(xn, newkk, lp, xx, yy, ye, dxx, dyy, dye, 0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == self._imax:
            print('   Maximum number of iterations performed on Nesterov adaptive moment search.')
        return (newkk, lmlnew)


    def _condition_data(self, xx, xe, yy, ye, lb, ub, cn):
        r'''
        **INTERNAL FUNCTION** - Use main call functions!!!

        Conditions the input data to remove data points which are too close together, as
        defined by the user, and data points that are outside user-defined bounds.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg xe: array. Vector of x-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as :code:`xx`.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as :code:`xx`.

        :arg lb: float. Minimum allowable y-value for input data, values below are omitted from fit procedure.

        :arg ub: float. Maximum allowable y-value for input data, values above are omitted from fit procedure.

        :arg cn: float. Minimum allowable delta-x for input data before applying Gaussian blending.

        :returns: (array, array, array, array, array).
            Vectors in order of conditioned x-values, conditioned x-errors, conditioned y-values, conditioned y-errors,
            number of data points blended into corresponding index.
        '''

        good = np.all([np.invert(np.isnan(xx)), np.invert(np.isnan(yy)), np.isfinite(xx), np.isfinite(yy)], axis=0)
        xe = xe[good] if xe.size == xx.size else np.full(xx[good].shape, xe[0])
        ye = ye[good] if ye.size == yy.size else np.full(yy[good].shape, ye[0])
        xx = xx[good]
        yy = yy[good]
        xsc = np.nanmax(np.abs(xx)) if np.nanmax(np.abs(xx)) > 1.0e3 else 1.0   # Scaling avoids overflow when squaring
        ysc = np.nanmax(np.abs(yy)) if np.nanmax(np.abs(yy)) > 1.0e3 else 1.0   # Scaling avoids overflow when squaring
        xx = xx / xsc
        xe = xe / xsc
        yy = yy / ysc
        ye = ye / ysc
        nn = np.array([])
        cxx = np.array([])
        cxe = np.array([])
        cyy = np.array([])
        cye = np.array([])
        for ii in np.arange(0, xx.size):
            if yy[ii] >= lb and yy[ii] <= ub:
                fflag = False
                for jj in np.arange(0, cxx.size):
                    if np.abs(cxx[jj] - xx[ii]) < cn and not fflag:
                        cxe[jj] = np.sqrt(((cxe[jj] ** 2.0) * nn[jj] + (xe[ii] ** 2.0) + (cxx[jj] ** 2.0) * nn[jj] + (xx[ii] ** 2.0)) / (nn[jj] + 1.0) - ((cxx[jj] * nn[jj] + xx[ii]) / ((nn[jj] + 1.0)) ** 2.0))
                        cxx[jj] = (cxx[jj] * nn[jj] + xx[ii]) / (nn[jj] + 1.0)
                        cye[jj] = np.sqrt(((cye[jj] ** 2.0) * nn[jj] + (ye[ii] ** 2.0) + (cyy[jj] ** 2.0) * nn[jj] + (yy[ii] ** 2.0)) / (nn[jj] + 1.0) - ((cyy[jj] * nn[jj] + yy[ii]) / ((nn[jj] + 1.0)) ** 2.0))
                        cyy[jj] = (cyy[jj] * nn[jj] + yy[ii]) / (nn[jj] + 1.0)
                        nn[jj] = nn[jj] + 1.0
                        fflag = True
                if not fflag:
                    nn = np.hstack((nn, 1.0))
                    cxx = np.hstack((cxx, xx[ii]))
                    cxe = np.hstack((cxe, xe[ii]))
                    cyy = np.hstack((cyy, yy[ii]))
                    cye = np.hstack((cye, ye[ii]))
        cxx = cxx * xsc
        cxe = cxe * xsc
        cyy = cyy * ysc
        cye = cye * ysc
        return (cxx, cxe, cyy, cye, nn)


    def __basic_fit(
        self,
        xnew,
        kernel=None,
        regpar=None,
        xdata=None,
        ydata=None,
        yerr=None,
        dxdata=None,
        dydata=None,
        dyerr=None,
        epsilon=None,
        method=None,
        spars=None,
        sdiff=None,
        do_drv=False,
        rtn_cov=False
    ):
        r'''
        **RESTRICTED ACCESS FUNCTION** - Can be called externally for testing if user is familiar with algorithm.

        Basic GP regression fitting routine, **recommended** to call this instead of the bare-bones functions
        as this applies additional input checking.

        .. note::

            This function does **not** strictly use class data and does **not** store results inside the class
            either!!! This is done to allow this function to be used as a minimal working version and also as
            a standalone test for new :code:`_Kernel`, :code:`_OperatorKernel` and :code:`_WarpingFunction`
            class implementations.

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. Vector of x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. Vector of dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. Vector of dy/dx-errors of derivative data points to be included in fit, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable. (optional)

        :kwarg method: str or int. Hyperparameter optimization algorithm selection. Choices include::
                       [:code:`grad`, :code:`mom`, :code:`nag`, :code:`adagrad`, :code:`adadelta`, :code:`adam`, :code:`adamax`, :code:`nadam`] or their respective indices in the list.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method. (optional)

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2. (optional)

        :kwarg do_drv: bool. Set as true to predict the derivative of the fit instead of the fit. (optional)

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors. (optional)

        :returns: (array, array, float, object).
            Vector of predicted mean values, vector or matrix of predicted errors, log-marginal-likelihood of fit
            including the regularization component, final :code:`_Kernel` instance with optimized hyperparameters if performed.
        '''

        xn = None
        kk = copy.copy(self._kk)
        lp = self._lp
        xx = copy.deepcopy(self._xx)
        yy = copy.deepcopy(self._yy)
        ye = copy.deepcopy(self._ye) if self._gpye is None else copy.deepcopy(self._gpye)
        dxx = copy.deepcopy(self._dxx)
        dyy = copy.deepcopy(self._dyy)
        dye = copy.deepcopy(self._dye)
        eps = self._eps
        opm = self._opm
        opp = copy.deepcopy(self._opp)
        dh = self._dh
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        midx = None
        if isinstance(xnew, (list, tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew, np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel, _Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar, number_types) and float(regpar) > 0.0:
            lp = float(regpar)
        if isinstance(xdata, (list, tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata, np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata, (list, tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata, np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr, (list, tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr, np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        elif isinstance(yerr, str):
            ye = None
        if isinstance(dxdata, (list, tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            dxx = temp.flatten()
        elif isinstance(dxdata, np.ndarray) and dxdata.size > 0:
            dxx = dxdata.flatten()
        elif isinstance(dxdata, str):
            dxx = None
        if isinstance(dydata, (list, tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            dyy = temp.flatten()
        elif isinstance(dydata, np.ndarray) and dydata.size > 0:
            dyy = dydata.flatten()
        elif isinstance(dydata, str):
            dyy = None
        if isinstance(dyerr, (list, tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp, item) if item is not None else np.append(temp, np.nan)
            dye = temp.flatten()
        elif isinstance(dyerr, np.ndarray) and dyerr.size > 0:
            dye = dyerr.flatten()
        elif isinstance(dyerr, str):
            dye = None
        if isinstance(epsilon, number_types) and float(epsilon) > 0.0:
            eps = float(epsilon)
        elif isinstance(epsilon, number_types) and float(epsilon) <= 0.0:
            eps = None
        elif isinstance(epsilon, str):
            eps = None
        if isinstance(method, str):
            mstr = method.lower()
            if mstr in self._opopts:
                midx = self._opopts.index(mstr)
        elif isinstance(method, number_types) and int(method) >= 0 and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                opm = self._opopts[1]
                oppt = np.array([1.0e-5, 0.9]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 2:
                opm = self._opopts[2]
                oppt = np.array([1.0e-5, 0.9]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 3:
                opm = self._opopts[3]
                oppt = np.array([1.0e-2]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 4:
                opm = self._opopts[4]
                oppt = np.array([1.0e-2, 0.9]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 5:
                opm = self._opopts[5]
                oppt = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 6:
                opm = self._opopts[6]
                oppt = np.array([2.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 7:
                opm = self._opopts[7]
                oppt = np.array([1.0e-3, 0.9, 0.999]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            else:
                opm = self._opopts[0]
                oppt = np.array([1.0e-5]).flatten()
                for ii in np.arange(0, opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
        if isinstance(spars, (list, tuple)):
            for ii in np.arange(0, len(spars)):
                if ii < opp.size and isinstance(spars[ii], number_types):
                    opp[ii] = float(spars[ii])
        elif isinstance(spars, np.ndarray):
            for ii in np.arange(0, spars.size):
                if ii < opp.size and isinstance(spars[ii], number_types):
                    opp[ii] = float(spars[ii])
        if isinstance(sdiff, number_types) and float(sdiff) > 0.0:
            dh = float(sdiff)

        barF = None
        errF = None
        lml = None
        lmlz = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk, _Kernel):
            # Remove all data and associated data that contain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx, xe, yy, ye, nn) = self._condition_data(xx, xe, yy, ye, lb, ub, cn)
            myy = np.mean(yy)
            yy = yy - myy
            sc = np.nanmax(np.abs(yy))
            if sc == 0.0:
                sc = 1.0
            yy = yy / sc
            ye = ye / sc
            dnn = None
            if dxx is not None and dyy is not None and dxx.size == dyy.size:
                if dye is None:
                    dye = np.array([0.0])
                dxe = np.array([0.0])
                (dxx, dxe, dyy, dye, dnn) = self._condition_data(dxx, dxe, dyy, dye, -1.0e50, 1.0e50, cn)
                dyy = dyy / sc
                dye = dye / sc
            dd = 1 if do_drv else 0
            nkk = copy.copy(kk)
            if eps is not None and not do_drv:
                if opm == 'mom' and opp.size > 1:
                    (nkk, lml) = self._gp_momentum_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], dh)
                elif opm == 'nag' and opp.size > 1:
                    (nkk, lml) = self._gp_nesterov_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], dh)
                elif opm == 'adagrad' and opp.size > 0:
                    (nkk, lml) = self._gp_adagrad_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], dh)
                elif opm == 'adadelta' and opp.size > 1:
                    (nkk, lml) = self._gp_adadelta_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], dh)
                elif opm == 'adam' and opp.size > 2:
                    (nkk, lml) = self._gp_adam_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], opp[2], dh)
                elif opm == 'adamax' and opp.size > 2:
                    (nkk, lml) = self._gp_adamax_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], opp[2], dh)
                elif opm == 'nadam' and opp.size > 2:
                    (nkk, lml) = self._gp_nadam_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], opp[1], opp[2], dh)
                elif opm == 'grad' and opp.size > 0:
                    (nkk, lml) = self._gp_grad_optimizer(nkk, lp, xx, yy, ye, dxx, dyy, dye, eps, opp[0], dh)
            (barF, varF, lml, lmlz) = self._gp_base_alg(xn, nkk, lp, xx, yy, ye, dxx, dyy, dye, dd)
            barF = barF * sc if do_drv else barF * sc + myy
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF, errF, lml, lmlz, nkk)


    def __brute_derivative(
        self,
        xnew,
        kernel=None,
        regpar=None,
        xdata=None,
        ydata=None,
        yerr=None,
        rtn_cov=False
    ):
        r'''
        **RESTRICTED ACCESS FUNCTION** - Can be called externally for testing if user is familiar with algorithm.

        Brute-force numerical GP regression derivative routine, **recommended** to call this instead of bare-bones
        functions above. Kept for ability to convince user of validity of regular GP derivative, but can also be
        wildly wrong on some data due to numerical errors.

        .. note::

            *Recommended* to use derivative flag on :code:`__basic_fit()` function, as it was tested to be
            more robust, provided the input kernels are properly defined.

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. Vector of x-values of data points to be fitted.

        :kwarg ydata: array. Vector of y-values of data points to be fitted.

        :kwarg yerr: array. Vector of y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors. (optional)

        :returns: (array, array, float).
            Vector of predicted dy/dx-values, vector or matrix of predicted dy/dx-errors, log-marginal-likelihood of fit
            including the regularization component.
        '''

        xn = None
        kk = self._kk
        lp = self._lp
        xx = self._xx
        yy = self._yy
        ye = self._ye if self._gpye is None else self._gpye
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        if isinstance(xnew, (list, tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew, np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel, _Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar, number_types) and float(regpar) > 0.0:
            self._lp = float(regpar)
        if isinstance(xdata, (list, tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata, np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata, (list, tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata, np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr, (list, tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr, np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        if ye is None and yy is not None:
            ye = np.zeros(yy.shape)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk, _Kernel):
            # Remove all data and associated data that conatain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx, xe, yy, ye, nn) = self._condition_data(xx, xe, yy, ye, lb, ub, cn)
            myy = np.mean(yy)
            yy = yy - myy
            sc = np.nanmax(np.abs(yy))
            if sc == 0.0:
                sc = 1.0
            yy = yy / sc
            ye = ye / sc
            (barF, varF, lml) = self._gp_brute_deriv1(xn, kk, lp, xx, yy, ye)
            barF = barF * sc
            varF = varF * (sc ** 2.0)
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF, errF, lml)


    def make_HSGP_errors(self):
        r'''
        Calculates a vector of modified y-errors based on GPR fit of input y-errors,
        for use inside a heteroscedastic GPR execution.

        .. note::

            This function is automatically called inside :code:`GPRFit()` when the
            :code:`hsgp_flag` argument is :code:`True`. For this reason, this function
            automatically stores all results within the appropriate class variables.

        :returns: none.
        '''

        if isinstance(self._ekk, _Kernel) and self._ye is not None and self._yy.size == self._ye.size:
            elml = None
            ekk = None
            xntest = np.array([0.0])
            ye = copy.deepcopy(self._ye) if self._gpye is None else copy.deepcopy(self._gpye)
            aye = np.full(ye.shape,np.nanmax([0.2 * np.mean(np.abs(ye)), 1.0e-3 * np.nanmax(np.abs(self._yy))]))
#            dye = copy.deepcopy(self._dye)
#            adye = np.full(dye.shape, np.nanmax([0.2 * np.mean(np.abs(dye)), 1.0e-3 * np.nanmax(np.abs(self._dyy))])) if dye is not None else None
#            if adye is not None:
#                adye[adye < 1.0e-2] = 1.0e-2
            if self._ekk.bounds is not None and self._eeps is not None and self._egpye is None:
                elp = self._elp
                ekk = copy.copy(self._ekk)
                ekkvec = []
                elmlvec = []
                try:
                    (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
                        xntest,
                        kernel=ekk,
                        regpar=elp,
                        ydata=ye,
                        yerr=aye,
                        dxdata='None',
                        dydata='None',
                        dyerr='None',
                        epsilon=self._eeps,
                        method=self._eopm,
                        spars=self._eopp,
                        sdiff=self._edh
                    ))
#                    (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
#                        xntest,
#                        kernel=ekk,
#                        regpar=elp,
#                        ydata=ye,
#                        yerr=aye,
#                        dydata=dye,
#                        dyerr=adye,
#                        epsilon=self._eeps,
#                        method=self._eopm,
#                        spars=self._eopp,
#                        sdiff=self._edh
#                    ))
                    ekkvec.append(copy.copy(ekk))
                    elmlvec.append(elml)
                except (ValueError, np.linalg.linalg.LinAlgError):
                    ekkvec.append(None)
                    elmlvec.append(np.nan)
                for jj in np.arange(0, self._enr):
                    ekb = np.log10(self._ekk.bounds)
                    etheta = np.abs(ekb[1, :] - ekb[0, :]).flatten() * np.random.random_sample((ekb.shape[1], )) + np.nanmin(ekb, axis=0).flatten()
                    ekk.hyperparameters = np.power(10.0, etheta)
                    try:
                        (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
                            xntest,
                            kernel=ekk,
                            regpar=elp,
                            ydata=ye,
                            yerr=aye,
                            dxdata='None',
                            dydata='None',
                            dyerr='None',
                            epsilon=self._eeps,
                            method=self._eopm,
                            spars=self._eopp,
                            sdiff=self._edh
                        ))
#                        (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
#                            xntest,
#                            kernel=ekk,
#                            regpar=elp,
#                            ydata=ye,
#                            yerr=aye,
#                            dydata=dye,
#                            dyerr=adye,
#                            epsilon=self._eeps,
#                            method=self._eopm,
#                            spars=self._eopp,
#                            sdiff=self._edh
#                        ))
                        ekkvec.append(copy.copy(ekk))
                        elmlvec.append(elml)
                    except (ValueError, np.linalg.linalg.LinAlgError):
                        ekkvec.append(None)
                        elmlvec.append(np.nan)
                eimaxv = np.where(elmlvec == np.nanmax(elmlvec))[0]
                if len(eimaxv) > 0:
                    eimax = eimaxv[0]
                    ekk = ekkvec[eimax]
                    self._ekk = copy.copy(ekkvec[eimax])
                else:
                    raise ValueError('None of the error fit attempts converged. Please change error kernel settings and try again.')
            elif self._eeps is not None and self._egpye is None:
                elp = self._elp
                ekk = copy.copy(self._ekk)
                (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
                    xntest,
                    kernel=ekk,
                    regpar=elp,
                    ydata=ye,
                    yerr=aye,
                    dxdata='None',
                    dydata='None',
                    dyerr='None',
                    epsilon=self._eeps,
                    method=self._eopm,
                    spars=self._eopp,
                    sdiff=self._edh
                ))
#                (elml, ekk) = itemgetter(2, 4)(self.__basic_fit(
#                    xntest,
#                    kernel=ekk,
#                    regpar=elp,
#                    ydata=ye,
#                    yerr=aye,
#                    dydata=dye,
#                    dyerr=adye,
#                    epsilon=self._eeps,
#                    method=self._eopm,
#                    spars=self._eopp,
#                    sdiff=self._edh
#                ))
                self._ekk = copy.copy(ekk)
            if isinstance(self._ekk, _Kernel):
                epsx = 1.0e-8 * (np.nanmax(self._xx) - np.nanmin(self._xx)) if self._xx.size > 1 else 1.0e-8
                xntest = self._xx.copy() + epsx
                tgpye = itemgetter(0)(self.__basic_fit(
                    xntest,
                    kernel=self._ekk,
                    regpar=self._elp,
                    ydata=ye,
                    yerr=aye,
                    dxdata='None',
                    dydata='None',
                    dyerr='None',
                    epsilon='None'
                ))
#                self._gpye = itemgetter(0)(self.__basic_fit(
#                    xntest,
#                    kernel=self._ekk,
#                    regpar=self._elp,
#                    ydata=ye,
#                    yerr=aye,
#                    dydata=dye,
#                    dyerr=adye,
#                    epsilon='None'
#                ))
                self._gpye = np.abs(tgpye)
                self._egpye = aye.copy()
        else:
            raise ValueError('Check input y-errors to make sure they are valid.')


    def make_NIGP_errors(self, nrestarts=0, hsgp_flag=False):
        r'''
        Calculates a vector of modified y-errors based on input x-errors and a test model
        gradient, for use inside a noisy input GPR execution.

        .. note::

            This function is automatically called inside :code:`GPRFit()` when the
            :code:`nigp_flag` argument is :code:`True`. For this reason, this function
            automatically stores all results within the appropriate class variables.

        .. warning::

            This function does not iterate until the test model derivatives and the actual
            fit derivatives are self-consistent! Although this would be the most rigourous
            implementation, it was decided that this approximation was good enough for the
            uses of the current implementation. (v >= 1.0.1)

        .. warning::

            The results of this function may be washed away by the heteroscedastic GP
            implementation due to the fact that the y-error modifications are included
            when fitting the error kernel. This can be addressed in the future by
            separating the contributions due to noisy input and due to heteroscedastic
            GP, with a separate noise kernel for each.

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within the provided hyperparameter bounds. (optional)

        :kwarg hsgp_flag: bool. Indicates Gaussian Process regression fit with variable y-errors. (optional)

        :returns: none.
        '''

        # Check inputs
        nr = 0
        if isinstance(nrestarts, number_types) and int(nrestarts) > 0:
            nr = int(nrestarts)

        if isinstance(self._kk, _Kernel) and self._xe is not None and self._xx.size == self._xe.size:
            nlml = None
            nkk = None
            xntest = np.array([0.0])
            if not isinstance(self._nikk, _Kernel):
                if self._kk.bounds is not None and nr > 0:
                    tkk = copy.copy(self._kk)
                    kkvec = []
                    lmlvec = []
                    try:
                        (tlml, tkk) = itemgetter(2, 4)(self.__basic_fit(
                            xntest,
                            kernel=tkk
                        ))
                        kkvec.append(copy.copy(tkk))
                        lmlvec.append(tlml)
                    except ValueError:
                        kkvec.append(None)
                        lmlvec.append(np.nan)
                    for ii in np.arange(0, nr):
#                        kb = self._kb
                        kb = np.log10(self._kk.bounds)
                        theta = np.abs(kb[1, :] - kb[0, :]).flatten() * np.random.random_sample((kb.shape[1], )) + np.nanmin(kb, axis=0).flatten()
                        tkk.hyperparameters = np.power(10.0, theta)
                        try:
                            (tlml, tkk) = itemgetter(2, 4)(self.__basic_fit(
                                xntest,
                                kernel=tkk
                            ))
                            kkvec.append(copy.copy(tkk))
                            lmlvec.append(tlml)
                        except ValueError:
                            kkvec.append(None)
                            lmlvec.append(np.nan)
                    imax = np.where(lmlvec == np.nanmax(lmlvec))[0][0]
                    (nlml, nkk) = itemgetter(2, 4)(self.__basic_fit(
                        xntest,
                        kernel=kkvec[imax],
                        epsilon='None'
                    ))
                else:
                    (nlml, nkk) = itemgetter(2, 4)(self.__basic_fit(
                        xntest
                    ))
            else:
                nkk = copy.copy(self._nikk)
            if isinstance(nkk, _Kernel):
                self._nikk = copy.copy(nkk)
                epsx = 1.0e-8 * (np.nanmax(self._xx) - np.nanmin(self._xx)) if self._xx.size > 1 else 1.0e-8
                xntest = self._xx.copy() + epsx
                dbarF = itemgetter(0)(self.__basic_fit(
                    xntest,
                    kernel=nkk,
                    do_drv=True
                ))
#                cxe = self._xe.copy()
#                cxe[np.isnan(cxe)] = 0.0
#                self._gpxe = np.abs(cxe * dbarF)
                cxe = self._xe.copy()
                cye = self._ye.copy() if self._gpye is None else self._gpye.copy()
                nfilt = np.any([np.isnan(cxe), np.isnan(cye)], axis=0)
                cxe[nfilt] = 0.0
                cye[nfilt] = 0.0
                self._gpye = np.sqrt((cye ** 2.0) + ((cxe * dbarF) ** 2.0))
                self._egpye = np.full(cye.shape, np.nanmax([0.2 * np.mean(np.abs(self._gpye)), 1.0e-3 * np.nanmax(np.abs(self._yy))])) if not hsgp_flag else self._egpye
        else:
            raise ValueError('Check input x-errors to make sure they are valid.')


    def GPRFit(
        self,
        xnew,
        hsgp_flag=True,
        nigp_flag=False,
        nrestarts=None
    ):
        r'''
        Main GP regression fitting routine, **recommended** to call this after using set functions, instead of the
        :code:`__basic_fit()` function, as this adapts the method based on inputs, performs 1st derivative and
        saves output to class variables.

        - Includes implementation of Monte Carlo kernel restarts within the user-defined bounds, via nrestarts argument
        - Includes implementation of Heteroscedastic Output Noise, requires setting of error kernel before fitting
            For details, see article: K. Kersting, 'Most Likely Heteroscedastic Gaussian Process Regression' (2007)
        - Includes implementation of Noisy-Input Gaussian Process (NIGP) assuming Gaussian x-error, via nigp_flag argument
            For details, see article: A. McHutchon, C.E. Rasmussen, 'Gaussian Process Training with Input Noise' (2011)

        :arg xnew: array. Vector of x-values at which the predicted fit will be evaluated.

        :kwarg hsgp_flag: bool. Set as true to perform Gaussian Process regression fit with proper propagation of y-errors. Default is :code:`True`. (optional)

        :kwarg nigp_flag: bool. Set as true to perform Gaussian Process regression fit with proper propagation of x-errors. Default is :code:`False`. (optional)

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter values within the provided hyperparameter bounds. (optional)

        :returns: none.
        '''
        # Check inputs
        xn = None
        nr = 0
        if isinstance(xnew, (list, tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew, np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(nrestarts, number_types) and int(nrestarts) > 0:
            nr = int(nrestarts)
        if xn is None:
            raise ValueError('A valid vector of prediction x-points must be given.')
        oxn = copy.deepcopy(xn)

        if not self._fwarn:
            warnings.filterwarnings('ignore', category=RuntimeWarning)

        barF = None
        varF = None
        lml = None
        lmlz = None
        nkk = None
        estF = None
        if nigp_flag:
            self.make_NIGP_errors(nr, hsgp_flag=hsgp_flag)
        if hsgp_flag:
            self.make_HSGP_errors()
        if self._gpye is None:
            hsgp_flag = False
            nigp_flag = False
            self._gpye = copy.deepcopy(self._ye)
            self._egpye = None

        # These loops adjust overlapping values between raw data vector and requested prediction vector, to avoid NaN values in final prediction
        if self._xx is not None:
            epsx = 1.0e-6 * (np.nanmax(xn) - np.nanmin(xn)) if xn.size > 1 else 1.0e-6 * (np.nanmax(self._xx) - np.nanmin(self._xx))
            for xi in np.arange(0, xn.size):
                for rxi in np.arange(0, self._xx.size):
                    if xn[xi] == self._xx[rxi]:
                        xn[xi] = xn[xi] + epsx
        if self._dxx is not None:
            epsx = 1.0e-6 * (np.nanmax(xn) - np.nanmin(xn)) if xn.size > 1 else 1.0e-6 * (np.nanmax(self._dxx) - np.nanmin(self._dxx))
            for xi in np.arange(0, xn.size):
                for rxi in np.arange(0, self._dxx.size):
                    if xn[xi] == self._dxx[rxi]:
                        xn[xi] = xn[xi] + epsx

        if self._egpye is not None:
            edye = np.full(self._dye.shape, np.nanmax([0.2 * np.mean(np.abs(self._dye)), 1.0e-3 * np.nanmax(np.abs(self._dyy))])) if self._dye is not None else None
            if edye is not None:
                edye[edye < 1.0e-2] = 1.0e-2
            (self._barE, self._varE) = itemgetter(0, 1)(self.__basic_fit(
                xn,
                kernel=self._ekk,
                ydata=self._gpye,
                yerr=self._egpye,
                dxdata='None',
                dydata='None',
                dyerr='None',
                epsilon='None',
                rtn_cov=True
            ))
            (self._dbarE, self._dvarE) = itemgetter(0, 1)(self.__basic_fit(
                xn,
                kernel=self._ekk,
                ydata=self._gpye,
                yerr=self._egpye,
                dxdata='None',
                dydata='None',
                dyerr='None',
                do_drv=True,
                rtn_cov=True
            ))
#            (self._barE, self._varE) = itemgetter(0, 1)(self.__basic_fit(
#                xn,
#                kernel=self._ekk,
#                ydata=self._gpye,
#                yerr=self._egpye,
#                dydata=self._dye,
#                dyerr=edye,
#                epsilon='None',
#                rtn_cov=True
#            ))
#            (self._dbarE, self._dvarE) = itemgetter(0, 1)(self.__basic_fit(
#                xn,
#                kernel=self._ekk,
#                ydata=self._gpye,
#                yerr=self._egpye,
#                dydata=self._dye,
#                dyerr=edye,
#                do_drv=True,
#                rtn_cov=True
#            ))

#            nxn = np.linspace(np.nanmin(xn), np.nanmax(xn), 1000)
#            ddx = np.nanmin(np.diff(nxn)) * 1.0e-2
#            xnl = nxn - 0.5 * ddx
#            xnu = nxn + 0.5 * ddx
#            dbarEl = itemgetter(0)(self.__basic_fit(
#                xnl,
#                kernel=self._ekk,
#                ydata=self._gpye,
#                yerr=self._egpye,
#                dxdata='None',
#                dydata='None',
#                dyerr='None',
#                do_drv=True
#            ))
#            dbarEu = itemgetter(0)(self.__basic_fit(
#                xnu,
#                kernel=self._ekk,
#                ydata=self._gpye,
#                yerr=self._egpye,
#                dxdata='None',
#                dydata='None',
#                dyerr='None',
#                do_drv=True
#            ))
#            ddbarEt = np.abs(dbarEu - dbarEl) / ddx
#            nsum = 50
#            ddbarE = np.zeros(xn.shape)
#            for nx in np.arange(0, xn.size):
#                ivec = np.where(nxn >= xn[nx])[0][0]
#                nbeg = nsum - (ivec + 1) if (ivec + 1) < nsum else 0
#                nend = nsum - (nxn.size - ivec - 1) if (nxn.size - ivec - 1) < nsum else 0
#                temp = None
#                if nbeg > 0:
#                    vbeg = np.full((nbeg, ), ddbarEt[0])
#                    temp = np.hstack((vbeg, ddbarEt[:ivec+nsum+1]))
#                    ddbarE[nx] = float(np.mean(temp))
#                elif nend > 0:
#                    vend = np.full((nend, ), ddbarEt[-1]) if nend > 0 else np.array([])
#                    temp = np.hstack((ddbarEt[ivec-nsum:], vend))
#                    ddbarE[nx] = float(np.mean(temp))
#                else:
#                    ddbarE[nx] = float(np.mean(ddbarEt[ivec-nsum:ivec+nsum+1]))
#            self._ddbarE = ddbarE.copy()
        else:
            self._gpye = np.full(xn.shape, np.sqrt(np.nanmean(np.power(self._ye, 2.0)))) if self._ye is not None else None
            self._barE = copy.deepcopy(self._gpye) if self._gpye is not None else None
            self._varE = np.zeros(xn.shape) if self._barE is not None else None
            self._dbarE = np.zeros(xn.shape) if self._barE is not None else None
            self._dvarE = np.zeros(xn.shape) if self._barE is not None else None
#            self._ddbarE = np.zeros(xn.shape) if self._barE is not None else None

        if isinstance(self._kk, _Kernel) and self._kk.bounds is not None and nr > 0:
            xntest = np.array([0.0])
            tkk = copy.copy(self._kk)
            kkvec = []
            lmlvec = []
            try:
                (tlml, tkk) = itemgetter(2, 4)(self.__basic_fit(
                    xntest,
                    kernel=tkk
                ))
                kkvec.append(copy.copy(tkk))
                lmlvec.append(tlml)
            except (ValueError, np.linalg.linalg.LinAlgError):
                kkvec.append(None)
                lmlvec.append(np.nan)
            for ii in np.arange(0, nr):
#                kb = self._kb
                kb = np.log10(self._kk.bounds)
                theta = np.abs(kb[1, :] - kb[0, :]).flatten() * np.random.random_sample((kb.shape[1], )) + np.nanmin(kb, axis=0).flatten()
                tkk.hyperparameters = np.power(10.0, theta)
                try:
                    (tlml, tkk) = itemgetter(2, 4)(self.__basic_fit(
                        xntest,
                        kernel=tkk
                    ))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except (ValueError, np.linalg.linalg.LinAlgError):
                    kkvec.append(None)
                    lmlvec.append(np.nan)
            imaxv = np.where(lmlvec == np.nanmax(lmlvec))[0]
            if len(imaxv) > 0:
                imax = imaxv[0]
                (barF, varF, lml, lmlz, nkk) = self.__basic_fit(
                    xn,
                    kernel=kkvec[imax],
                    epsilon='None',
                    rtn_cov=True
                )
                estF = itemgetter(0)(self.__basic_fit(
                    self._xx + 1.0e-10,
                    kernel=kkvec[imax],
                    epsilon='None',
                    rtn_cov=True
                ))
            else:
                raise ValueError('None of the fit attempts converged. Please adjust kernel settings and try again.')
        elif isinstance(self._kk, _Kernel):
            (barF, varF, lml, lmlz, nkk) = self.__basic_fit(
                xn,
                rtn_cov=True
            )
            estF = itemgetter(0)(self.__basic_fit(
                self._xx + 1.0e-10,
                kernel=nkk,
                epsilon='None',
                rtn_cov=True
            ))

        if barF is not None and isinstance(nkk, _Kernel):
            self._xF = copy.deepcopy(oxn)
            self._barF = copy.deepcopy(barF)
            self._varF = copy.deepcopy(varF) if varF is not None else None
            self._estF = copy.deepcopy(estF) if estF is not None else None
            self._lml = lml
            self._nulllml = lmlz
            self._kk = copy.copy(nkk) if isinstance(nkk, _Kernel) else None
            (dbarF, dvarF) = itemgetter(0, 1)(self.__basic_fit(
                xn,
                do_drv=True,
                rtn_cov=True
            ))
            self._dbarF = copy.deepcopy(dbarF) if dbarF is not None else None
            self._dvarF = copy.deepcopy(dvarF) if dvarF is not None else None
            self._varN = np.diag(np.power(self._barE, 2.0)) if self._barE is not None else np.diag(np.zeros(self._xF.shape))
            self._dvarN = np.diag(np.power(self._dbarE, 2.0)) if self._dbarE is not None else np.diag(np.zeros(self._xF.shape))

            # It seems that the second derivative term is not necessary, should be used to refine the mathematics!
#            ddfac = copy.deepcopy(self._ddbarE) if self._ddbarE is not None else 0.0
#            self._dvarN = np.diag(2.0 * (np.power(self._dbarE, 2.0) + np.abs(self._barE * ddfac))) if self._dbarE is not None else None
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings('default', category=RuntimeWarning)


    def sample_GP(
        self,
        nsamples,
        actual_noise=False,
        without_noise=False,
        simple_out=False
    ):
        r'''
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        :code:`GPRFit()` method.

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance as actual Gaussian noise. Only operates on diagonal elements. (optional)

        :kwarg without_noise: bool. Specifies complete exclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation. (optional)

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest :code:`GPRFit()`. If :code:`simple_out = True`, row 0 is the mean and row 1 is the 1 sigma error.
        '''

        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples, number_types) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings('ignore', category=RuntimeWarning)

        samples = None
        if ns > 0:
            noise_flag = actual_noise if not without_noise else False
            mu = self.get_gp_mean()
            var = self.get_gp_variance(noise_flag=noise_flag)
            mult_flag = not actual_noise if not without_noise else False
            mult = self.get_gp_std(noise_flag=mult_flag) / self.get_gp_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu, cov=var, size=ns)
            samples = mult * (samples - mu) + mu
            if samples is not None and simple_out:
                mean = np.nanmean(samples, axis=0)
                std = np.nanstd(samples, axis=0)
                samples = np.vstack((mean, std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings('default', category=RuntimeWarning)

        return samples


    def sample_GP_derivative(
        self,
        nsamples,
        actual_noise=False,
        without_noise=False,
        simple_out=False
    ):
        r'''
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        :code:`GPRFit()` method.

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance as actual Gaussian noise. Only operates on diagonal elements. (optional)

        :kwarg without_noise: bool. Specifies complete exclusion of noise term in returned variance. Only operates on diagonal elements. (optional)

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation. (optional)

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest :code:`GPRFit()`. If :code:`simple_out = True`, row 0 is the mean and row 1 is the 1 sigma error.
        '''

        # Check instantiation of output class variables
        if self._xF is None or self._dbarF is None or self._dvarF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples, number_types) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings('ignore', category=RuntimeWarning)

        samples = None
        if ns > 0:
            noise_flag = actual_noise if not without_noise else False
            mu = self.get_gp_drv_mean()
            var = self.get_gp_drv_variance(noise_flag=noise_flag)
            mult_flag = not actual_noise if not without_noise else False
            mult = self.get_gp_drv_std(noise_flag=mult_flag) / self.get_gp_drv_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu, cov=var, size=ns)
            samples = mult * (samples - mu) + mu
            if samples is not None and simple_out:
                mean = np.nanmean(samples, axis=0)
                std = np.nanstd(samples, axis=0)
                samples = np.vstack((mean, std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings('default', category=RuntimeWarning)

        return samples


    def MCMC_posterior_sampling(self,nsamples):
        r'''
        Performs Monte Carlo Markov chain based posterior analysis over hyperparameters,
        using the log-marginal-likelihood as the acceptance criterion.

        .. warning::

            This function is suspected to be **incorrect** as currently coded! It should
            use data likelihood from model as the acceptance criterion instead of the
            log-marginal-likelihood. However, MCMC analysis was found only to be
            necessary when using non-Gaussian likelihoods and priors, otherwise the
            result is effectively equivalent to maximization of the LML. (v >= 1.0.1)

        :arg nsamples: int. Number of samples to draw from the posterior distribution.

        :returns: (array, array, array, array).
            Matrices containing rows of predicted y-values, predicted y-errors, predicted dy/dx-values,
            predicted dy/dx-errors with each having a number of rows equal to the number of samples.
        '''
        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to use MCMC posterior sampling.')

        # Check inputs
        ns = 0
        if isinstance(nsamples, number_types) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings('ignore', category=RuntimeWarning)

        sbarM = None
        ssigM = None
        sdbarM = None
        sdsigM = None
        if isinstance(self._kk, _Kernel) and ns > 0:
            olml = self._lml
            otheta = np.log10(self._kk.hyperparameters)
            tlml = olml
            theta = otheta.copy()
            step = np.ones(theta.shape)
            flagvec = [True] * theta.size
            for ihyp in np.arange(0, theta.size):
                xntest = np.array([0.0])
                iflag = flagvec[ihyp]
                while iflag:
                    tkk = copy.copy(self._kk)
                    theta_step = np.zeros(theta.shape)
                    theta_step[ihyp] = step[ihyp]
                    theta_new = theta + theta_step
                    tkk.hyperparameters = np.power(10.0, theta_new)
                    ulml = None
                    try:
                        ulml = itemgetter(2)(self.__basic_fit(
                            xntest,
                            kernel=tkk,
                            epsilon='None'
                        ))
                    except (ValueError, np.linalg.linalg.LinAlgError):
                        ulml = tlml - 3.0
                    theta_new = theta - theta_step
                    tkk.hyperparameters = np.power(10.0, theta_new)
                    llml = None
                    try:
                        llml = itemgetter(2)(self.__basic_fit(
                            xntest,
                            kernel=tkk,
                            epsilon='None'
                        ))
                    except (ValueError, np.linalg.linalg.LinAlgError):
                        llml = tlml - 3.0
                    if (ulml - tlml) >= -2.0 or (llml - tlml) >= -2.0:
                        iflag = False
                    else:
                        step[ihyp] = 0.5 * step[ihyp]
                flagvec[ihyp] = iflag
            nkk = copy.copy(self._kk)
            for ii in np.arange(0, ns):
                theta_prop = theta.copy()
                accept = False
                xntest = np.array([0.0])
                nlml = tlml
                jj = 0
                kk = 0
                while not accept:
                    jj = jj + 1
                    rstep = np.random.normal(0.0, 0.5 * step)
                    theta_prop = theta_prop + rstep
                    nkk.hyperparameters = np.power(10.0, theta_prop)
                    try:
                        nlml = itemgetter(2)(self.__basic_fit(
                            xntest,
                            kernel=nkk,
                            epsilon='None'
                        ))
                        if (nlml - tlml) > 0.0:
                            accept = True
                        else:
                            accept = True if np.power(10.0, nlml - tlml) >= np.random.uniform() else False
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        accept = False
                    if jj > 100:
                        step = 0.9 * step
                        jj = 0
                        kk = kk + 1
                    if kk > 100:
                        theta_prop = otheta.copy()
                        tlml = olml
                        kk = 0
                tlml = nlml
                theta = theta_prop.copy()
                xn = self._xF.copy()
                nkk.hyperparameters = np.power(10.0, theta)
                (barF, sigF, tlml, tlmlz, nkk) = self.__basic_fit(
                    xn,
                    kernel=nkk,
                    epsilon='None'
                )
                sbarM = barF.copy() if sbarM is None else np.vstack((sbarM, barF))
                ssigM = sigF.copy() if ssigM is None else np.vstack((ssigM, sigF))
                (dbarF, dsigF) = itemgetter(0,1)(self.__basic_fit(
                    xn,
                    kernel=nkk,
                    epsilon='None',
                    do_drv=True
                ))
                sdbarM = dbarF.copy() if sdbarM is None else np.vstack((sdbarM, dbarF))
                sdsigM = dsigF.copy() if sdsigM is None else np.vstack((sdsigM, dsigF))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings('default', category=RuntimeWarning)

        return (sbarM, ssigM, sdbarM, sdsigM)



class SimplifiedGaussianProcessRegression1D(GaussianProcessRegression1D):
    r'''
    A simplified version of the main :code:`GaussianProcessRegression1D`
    class with pre-defined settings, only requiring the bare necessities
    to use for fitting. Although this class allows an entry point akin
    to typical :mod:`scipy` classes, it is provided primarily to be
    used as a template for implementations meant to simplify the GPR1D
    experience for the average user.

    .. note::

        Optimization of hyperparameters is only performed *once* using
        settings at the time of the *first* call! All subsequent calls
        use the results of the first optimization, regardless of its
        quality or convergence status.

    :arg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in the fit procedure.

    :arg xdata: array. Vector of x-values corresponding to data to be fitted.

    :arg ydata: array. Vector of y-values corresponding to data to be fitted. Must be the same shape as :code:`xdata`.

    :arg yerr: array. Vector of y-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`.

    :kwarg xerr: array. Vector of x-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`. (optional)

    :kwarg kernel_bounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

    :kwarg reg_par: float. Parameter adjusting penalty on kernel complexity. (optional)

    :kwarg epsilon: float. Convergence criterion on change in log-marginal-likelihood. (optional)

    :kwarg num_restarts: int. Number of kernel restarts. (optional)

    :kwarg hyp_opt_gain: float. Gain value on the hyperparameter optimizer, expert use only. (optional)
    '''

    def __init__(
        self,
        kernel,
        xdata,
        ydata,
        yerr,
        xerr=None,
        kernel_bounds=None,
        reg_par=1.0,
        epsilon=1.0e-2,
        num_restarts=0,
        hyp_opt_gain=1.0e-2,
        include_noise=True
    ):
        r'''
        Defines customized :code:`GaussianProcessRegression1D` instance with
        a pre-defined common settings for both data fit and error fit. Input
        parameters reduced only to essentials and most crucial knobs for
        fine-tuning.

        :arg kernel: object. The covariance function, as a :code:`_Kernel` instance, to be used in the fit procedure.

        :arg xdata: array. Vector of x-values corresponding to data to be fitted.

        :arg ydata: array. Vector of y-values corresponding to data to be fitted. Must be the same shape as :code:`xdata`.

        :arg yerr: array. Vector of y-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`.

        :kwarg xerr: array. Optional vector of x-errors corresponding to data to be fitted, assumed to be given as 1 sigma. Must be the same shape as :code:`xdata`. (optional)

        :kwarg kernel_bounds: array. 2D array with rows being hyperparameters and columns being :code:`[lower,upper]` bounds. (optional)

        :arg reg_par: float. Parameter adjusting penalty on kernel complexity. (optional)

        :kwarg epsilon: float. Convergence criterion on change in log-marginal-likelihood. (optional)

        :kwarg num_restarts: int. Number of kernel restarts. (optional)

        :kwarg hyp_opt_gain: float. Gain value on the hyperparameter optimizer, expert use only. (optional)

        :kwarg include_noise: bool. Specifies inclusion of Gaussian noise term in returned variance. Only operates on diagonal elements. (optional)

        :returns: none.
        '''
        super().__init__()
        self._nrestarts = num_restarts

        self.set_raw_data(xdata=xdata, ydata=ydata, yerr=yerr, xerr=xerr)

        eps = 'none' if not isinstance(epsilon, number_types) else epsilon
        sg = hyp_opt_gain if isinstance(hyp_opt_gain, number_types) else 1.0e-1
        self.set_kernel(kernel=kernel, kbounds=kernel_bounds, regpar=reg_par)
        self.set_search_parameters(epsilon=epsilon, method='adam', spars=[sg, 0.4, 0.8])

        self._perform_heterogp = False if isinstance(yerr, number_types) or self._ye is None else True
        self._perform_nigp = False if self._xe is None else True
        self._include_noise = True if include_noise else False

        if self._perform_heterogp:
            error_length = 5.0 * (np.nanmax(self._xx) - np.nanmin(self._xx)) / float(self._xx.size) if self._xx is not None else 5.0e-1
            error_kernel = RQ_Kernel(5.0e-1, error_length, 3.0e1)
            error_kernel_hyppar_bounds = np.atleast_2d([[1.0e-1, error_length * 0.25, 1.0e1], [1.0e0, error_length * 4.0, 5.0e1]])
            self.set_error_kernel(kernel=error_kernel, kbounds=error_kernel_hyppar_bounds, regpar=5.0, nrestarts=0)
            self.set_error_search_parameters(epsilon=1.0e-1, method='adam', spars=[1.0e-2, 0.4, 0.8])
        elif isinstance(yerr, number_types):
            ye = np.full(self._yy.shape, yerr)
            self.set_raw_data(yerr=ye)


    def __call__(self, xnew):
        r'''
        Defines a simplified fitting execution, *only* performs
        optimization on the *first* call. Subsequent calls
        merely evaluate the optimized fit at the input x-values.

        :arg xnew: array. Vector of x-values corresponding to points at which the GPR results should be evaulated.

        :returns: tuple.
                  Mean of GPR predictive distribution, ie. the fit ;
                  Standard deviation of mean, given as 1 sigma ;
                  Mean derivative of GPR predictive disstribution, ie. the derivative of the fit ;
                  Standard deviation of mean derivative, given as 1 sigma.
        '''
        nrestarts = self._nrestarts
        if self._xF is not None:
            nrestarts = None
            self.set_search_parameters(epsilon='none')
            self.set_error_search_parameters(epsilon='none')
        self.GPRFit(xnew, hsgp_flag=self._perform_heterogp, nigp_flag=self._perform_nigp, nrestarts=nrestarts)
        return self.get_gp_results(noise_flag=self._include_noise)


    def sample(self, xnew, derivative=False):
        r'''
        Provides a more intuitive function for sampling the
        predictive distribution. Only provides one sample
        per call, unlike the more complex function in the
        main class.

        :arg xnew: array. Vector of x-values corresponding to points where GPR results should be evaulated at.

        :kwarg derivative: bool. Flag to indicate sampling of fit derivative instead of the fit. (optional)

        :returns: array. Vector of y-values corresponding to a random sample of the GPR predictive distribution.
        '''
        self.__call__(xnew)
        remove_noise = not self._include_noise
        output = self.sample_GP(1, actual_noise=False, without_noise=remove_noise) if not derivative else self.sample_GP_derivative(1, actual_noise=False, without_noise=remove_noise)
        return output

