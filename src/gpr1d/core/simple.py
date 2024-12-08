r'''
Some helpful functions and classes for reproducability and user-friendliness

These functions were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET Contributors
    IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis, Boston, MA (2017)
    `<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

'''
# Required imports
import numpy as np

from .definitions import number_types, array_types
from .kernels import RQ_Kernel
from .routines import GaussianProcessRegression1D

__all__ = [
    'SimplifiedGaussianProcessRegression1D',  # Simplified fitting routine wrapper
]


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

