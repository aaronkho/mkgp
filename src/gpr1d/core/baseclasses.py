r'''
Base kernel classes for Gaussian Process Regression fitting of 1D data with errorbars. Built in Python 3.x, adapted to be Python 2.x compatible.
  06/12/2024: No longer compatible with Python 2.x.

These classes were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET Contributors
    IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis, Boston, MA (2017)
    `<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

'''
#    Kernel theory: "Gaussian Process for Machine Learning", C.E. Rasmussen, C.K.I. Williams (2006)

# Required imports
import warnings
import numpy as np

from .definitions import number_types, array_types, default_dtype


class _Kernel():
    r'''
    Base class to be inherited by **ALL** kernel implementations in order for type checks to succeed.
    Type checking done with :code:`isinstance(<object>,<this_module>._Kernel)`.

    Ideology:

    - :code:`self._fname` is a string, designed to provide an easy way to check the kernel instance type.
    - :code:`self._function` contains the covariance function, k, along with **at least** dk/dx1, dk/dx2, and d^2k/dx1dx2.
    - :code:`self._hyperparameters` contains free variables that are designed to vary in logarithmic-space.
    - :code:`self._constants` contains free variables that should not be changed during parameter searches, or true constants.
    - :code:`self._bounds` contains the bounds of the free variables to be used in randomized kernel restart algorithms.

    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly **NOT** recommended unless you are familiar with how these structures work and their interdependencies.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    :kwarg name: str. Codename of :code:`_Kernel` class implementation.

    :kwarg func: callable. Covariance function of :code:`_Kernel` class implementation.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg hyps: array. Hyperparameters to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

    :kwarg csts: array. Constants to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

    :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_Kernel` class implementation. (optional)

    :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_Kernel` class implementation. (optional)
    '''

    def __init__(
        self,
        name=None,
        func=None,
        hderf=False,
        hyps=None,
        csts=None,
        htags=None,
        ctags=None
    ):
        r'''
        Initializes the :code:`_Kernel` instance.

        .. note::

            Nothing is done with the :code:`htags` and :code:`ctags` arguments currently. (v >= 1.0.1)

        :kwarg name: str. Codename of :code:`_Kernel` class implementation.

        :kwarg func: callable. Covariance function of :code:`_Kernel` class implementation.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

        :kwarg csts: array. Constants to be stored in the :code:`_Kernel` instance, ordered according to the specific :code:`_Kernel` class implementation.

        :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_Kernel` class implementation. (optional)

        :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_Kernel` class implementation. (optional)

        :returns: none.
        '''

        self._fname = name
        self._function = func if callable(func) else None
        self._hyperparameters = np.array(hyps, dtype=default_dtype).flatten() if isinstance(hyps, array_types) else None
        self._constants = np.array(csts, dtype=default_dtype).flatten() if isinstance(csts, array_types) else None
        self._hyp_lbounds = None
        self._hyp_ubounds = None
        self._hderflag = hderf
        self._force_bounds = False


    def __call__(self, x1, x2, der=0, hder=None):
        r'''
        Default class call function, evaluates the stored covariance function at the input values.

        :arg x1: array. Meshgrid of x_1-values at which to evaulate the covariance function.

        :arg x2: array. Meshgrid of x_2-values at which to evaulate the covariance function.

        :kwarg der: int. Order of x derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the covariance function, requires explicit implementation. (optional)

        :returns: array. Covariance function evaluations at input value pairs using the given derivative settings. Has the same dimensions as :code:`x1` and :code:`x2`.
        '''

        k_out = None
        if callable(self._function):
            xt1 = None
            xt2 = None
            dert = 0
            hdert = None
            if isinstance(x1, number_types):
                xt1 = np.array(np.atleast_2d(x1), dtype=default_dtype)
            elif isinstance(x1, array_types):
                xt1 = np.array(np.atleast_2d(x1), dtype=default_dtype)
            if isinstance(x2, number_types):
                xt2 = np.array(np.atleast_2d(x2), dtype=default_dtype)
            elif isinstance(x2, array_types):
                xt2 = np.array(np.atleast_2d(x2), dtype=default_dtype)
            if isinstance(der, number_types):
                dert = int(der)
            if isinstance(hder, number_types):
                hdert = int(hder)
            if isinstance(xt1, np.ndarray) and isinstance(xt2, np.ndarray):
                k_out = self._function(xt1, xt2, dert, hdert)
            else:
                raise TypeError(f'Arguments x1 and x2 must be a 2D-array-like object.')
        else:
            raise NotImplementedError(f'Covariance function of {self.name} Kernel object not yet defined.')
        return k_out


    def __eq__(self,other):
        r'''
	Custom equality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_Kernel` class instance.

        :returns: bool. Indicates whether the two objects are equal to each other.
        '''

        status = False
        if isinstance(other, _Kernel):
            if self.name == other.name:
                shyp = np.all(np.isclose(self.hyperparameters, other.hyperparameters))
                scst = np.all(np.isclose(self.constants, other.constants))
                status = self.hyperparameters.size == other.hyperparameters.size and self.constants.size == other.constants.size and shyp and scst
        return status


    def __ne__(self, other):
        r'''
	Custom inequality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_Kernel` class instance.

        :returns: bool. Indicates whether the two objects are not equal to each other.
        '''

        return not self.__eq__(other)


    def enforce_bounds(self, value=True):
        r'''
        Sets a flag to enforce the given hyperparameter bounds.

        :kwarg value: bool. Boolean value to set the flag.

        :returns: none.
        '''

        self._force_bounds = True if value else False


    @property
    def name(self):
        r'''
        Returns the codename of the :code:`_Kernel` instance.

        :returns: str. Codename of the :code:`_Kernel` instance.
        '''

        return f'{self._fname}'


    @property
    def hyperparameters(self):
        r'''
        Return the hyperparameters stored in the :code:`_Kernel` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_Kernel` class implementation.
        '''

        val = np.array([])
        if isinstance(self._hyperparameters, np.ndarray):
            val = self._hyperparameters.copy()
        return val


    @property
    def constants(self):
        r'''
        Return the constants stored in the :code:`_Kernel` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_Kernel` class implementation.
        '''

        val = np.array([])
        if isinstance(self._constants, np.ndarray):
            val = self._constants.copy()
        return val


    @property
    def bounds(self):
        r'''
        Return the hyperparameter search bounds stored in the :code:`_Kernel` instance.

        :returns: array. Hyperparameter lower/upper bounds list, ordered according to the specific :code:`_Kernel` class implementation.
        '''

        val = None
        if isinstance(self._hyp_lbounds, np.ndarray) and isinstance(self._hyp_ubounds, np.ndarray) and self._hyp_lbounds.shape == self._hyp_ubounds.shape:
            val = np.vstack((self._hyp_lbounds.flatten(), self._hyp_ubounds.flatten()))
        return val


    def is_hderiv_implemented(self):
        r'''
        Checks if the explicit hyperparameter derivative is implemented in the :code:`_Kernel` class implementation.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        '''

        return self._hderflag


    @hyperparameters.setter
    def hyperparameters(self, theta):
        r'''
        Set the hyperparameters stored in the :code:`_Kernel` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_Kernel` implementation.

        :returns: none.
        '''

        userhyps = None
        if isinstance(theta, array_types):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError(f'{self.name} Kernel hyperparameters must be given as an array-like object.')
        nhyps = self._hyperparameters.size
        if nhyps > 0:
            if userhyps.size >= nhyps:
                if self._force_bounds and isinstance(self._hyp_lbounds, np.ndarray) and self._hyp_lbounds.size == nhyps:
                    htemp = userhyps[:nhyps]
                    lcheck = (htemp < self._hyp_lbounds)
                    htemp[lcheck] = self._hyp_lbounds[lcheck]
                    userhyps[:nhyps] = htemp
                if self._force_bounds and isinstance(self._hyp_ubounds, np.ndarray) and self._hyp_ubounds.size == nhyps:
                    htemp = userhyps[:nhyps]
                    ucheck = (htemp > self._hyp_ubounds)
                    htemp[ucheck] = self._hyp_ubounds[ucheck]
                    userhyps[:nhyps] = htemp
                self._hyperparameters = np.array(userhyps[:nhyps], dtype=default_dtype)
            else:
                raise ValueError(f'{self.name} Kernel hyperparameters must contain at least {nhyps} elements.')
        else:
            warnings.warn(f'{self.name} Kernel instance has no hyperparameters.', stacklevel=2)


    @constants.setter
    def constants(self,consts):
        r'''
        Set the constants stored in the :code:`_Kernel` instance.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_Kernel` class implementation.

        :returns: none.
        '''

        usercsts = None
        if isinstance(consts, array_types):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError(f'{self.name} Kernel constants must be given as an array-like object.')
        ncsts = self._constants.size
        if ncsts > 0:
            if usercsts.size >= ncsts:
                self._constants = np.array(usercsts[:ncsts], dtype=default_dtype)
            else:
                raise ValueError(f'{self.name} Kernel constants must contain at least {ncsts} elements.')
        else:
            warnings.warn(f'{self.name} Kernel instance has no constants.', stacklevel=2)


    @bounds.setter
    def bounds(self,bounds):
        r'''
        Set the hyperparameter bounds stored in the :code:`_Kernel` instance.

        :arg bounds: 2D array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_Kernel` class implementation.

        :returns: none.
        '''

        userbnds = None
        if isinstance(bounds, array_types):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError(f'{self.name} Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.')
        if userbnds.shape[0] != 2:
            raise TypeError(f'{self.name} Kernel bounds must be given as a 2D-array-like object with exactly 2 rows.')
        nhyps = self._hyperparameters.size
        if nhyps > 0:
            if userbnds.shape[1] >= nhyps:
                self._hyp_lbounds = np.array(userbnds[0, :nhyps], dtype=default_dtype)
                self._hyp_ubounds = np.array(userbnds[1, :nhyps], dtype=default_dtype)
                if self._force_bounds:
                    self.hyperparameters = self._hyperparameters.copy()
            else:
                raise ValueError(f'{self.name} Kernel bounds must be a 2D-array-like object with exactly 2 rows and at least {nhyps} elements per row.')
        else:
            warnings.warn(f'{self.name} Kernel instance has no hyperparameters to set bounds for.', stacklevel=2)



class _OperatorKernel(_Kernel):
    r'''
    Base operator class to be inherited by **ALL** operator kernel implementations for custom get/set functions.
    Type checking done with :code:`isinstance(<object>,<this_module>._OperatorKernel)` if needed.

    Ideology:

    - :code:`self._kernel_list` is a Python list of :code:`_Kernel` instances on which the specified operation will be performed

    Get/set functions adjusted to call get/set functions of each constituent kernel instead of using its own
    attributes, which are mostly left as :code:`None`.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    :kwarg name: str. Codename of :code:`_OperatorKernel` class implementation.

    :kwarg func: callable. Covariance function of :code:`_OperatorKernel` class implementation, ideally an operation on provided :code:`_Kernel` instances.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg klist: array. List of :code:`_Kernel` instances to be operated on by the :code:`_OperatorKernel` instance, input order determines the order of parameter lists.
    '''

    def __init__(
        self,
        name='Op',
        func=None,
        hderf=False,
        klist=None
    ):
        r'''
        Initializes the :code:`_OperatorKernel` instance.

        :kwarg name: str. Codename of :code:`_OperatorKernel` class implementation.

        :kwarg func: callable. Covariance function of :code:`_OperatorKernel` class implementation, ideally an operation on provided :code:`_Kernel` instances.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg klist: array. List of :code:`_Kernel` instances to be operated on by the :code:`_OperatorKernel` instance, input order determines the order of parameter lists.

        :returns: none.
        '''

        super().__init__(name, func, hderf)
        self._kernel_list = klist if klist is not None else []


    @property
    def name(self):
        r'''
        Returns the codename of the :code:`_OperatorKernel` instance.

        :returns: str. Codename of the :code:`_OperatorKernel` instance.
        '''

        kname = '-'.join([f'{kk.name}' for kk in self._kernel_list])
        return f'{self._fname}({kname})'


    @property
    def basename(self):
        r'''
        Returns the base codename of the :code:`_OperatorKernel` instance.

        :returns: str. Base codename of the :code:`_OperatorKernel` instance.
        '''

        return f'{self._fname}'


    @property
    def hyperparameters(self):
        r'''
        Return the hyperparameters of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        '''

        val = np.array([])
        for kk in self._kernel_list:
            val = np.hstack((val, kk.hyperparameters))
        return val


    @property
    def constants(self):
        r'''
        Return the constants of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        '''

        val = np.array([])
        for kk in self._kernel_list:
            val = np.hstack((val, kk.constants))
        return val


    @property
    def bounds(self):
        r'''
        Return the hyperparameter bounds of all the :code:`_Kernel` instances stored within the :code:`_OperatorKernel` instance.

        :returns: array. Hyperparameter lower/upper bounds list, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.
        '''

        val = None
        for kk in self._kernel_list:
            kval = kk.bounds
            if kval is not None:
                val = np.hstack((val, kval)) if val is not None else kval.copy()
        return val


    @hyperparameters.setter
    def hyperparameters(self, theta):
        r'''
        Set the hyperparameters stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        '''

        userhyps = None
        if isinstance(theta, array_types):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError(f'{self.name} OperatorKernel hyperparameters must be given as an array-like object.')
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userhyps.size >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.hyperparameters.size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.hyperparameters = theta[ndone:]
                        else:
                            kk.hyperparameters = theta[ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError(f'{self.name} OperatorKernal hyperparameters must contain at least {nhyps} elements.')
        else:
            warnings.warn(f'{self.name} OperatorKernel instance has no hyperparameters.', stacklevel=2)


    @constants.setter
    def constants(self, consts):
        r'''
        Set the constants stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        '''

        usercsts = None
        if isinstance(consts, array_types):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError(f'{self.name} OperatorKernel constants must be given as an array-like object.')
        ncsts = self.constants.size
        if ncsts > 0:
            if usercsts.size >= ncsts:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.constants.size
                    if nhere != ndone:
                        if nhere == ncsts:
                            kk.constants = consts[ndone:]
                        else:
                            kk.constants = consts[ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError(f'{self.name} OperatorKernel constants must contain at least {ncsts} elements.')
        else:
            warnings.warn(f'{self.name} OperatorKernel instance has no constants.', stacklevel=2)


    @bounds.setter
    def bounds(self, bounds):
        r'''
        Set the hyperparameter bounds stored in all the :code:`_Kernel` instances within the :code:`_OperatorKernel` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_OperatorKernel` class implementation and the current :code:`self._kernel_list` instance.

        :returns: none.
        '''

        userbnds = None
        if isinstance(bounds, array_types):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError(f'{self.name} OperatorKernel bounds must be given as a 2d-array-like object with exactly 2 rows.')
        if userbnds.shape[0] != 2:
            raise TypeError(f'{self.name} OperatorKernel bounds must be given as a 2d-array-like object with exactly 2 rows.')
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userbnds.shape[1] >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.hyperparameters.size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.bounds = userbnds[:, ndone:]
                        else:
                            kk.bounds = userbnds[:, ndone:nhere]
                        ndone = nhere
            else:
                raise ValueError(f'{self.name} OperatorKernel bounds must be a 2D-array-like object with exactly 2 rows and contain at least {nhyps} elements per row.')
        else:
            warnings.warn(f'{self.name} OperatorKernel instance has no hyperparameters to set bounds for.', stacklevel=2)



class _WarpingFunction():
    r'''
    Base class to be inherited by **ALL** warping function implementations in order for type checks to succeed.
    Type checking done with :code:`isinstance(<object>,<this_module>._WarpingFunction)`.

    Ideology:

    - :code:`self._fname` is a string, designed to provide an easy way to check the warping function instance type.
    - :code:`self._function` contains the warping function, l, along with *at least* dl/dz and d^2l/dz^2.
    - :code:`self._hyperparameters` contains free variables that are designed to vary in logarithmic-space.
    - :code:`self._constants` contains free variables that should not be changed during hyperparameter optimization, or true constants.
    - :code:`self._bounds` contains the bounds of the free variables to be used in randomized kernel restarts.

    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly **NOT** recommended unless you are familiar with how these structures work and their interdependencies.

    .. warning::

        Get/set functions not yet programmed as actual attributes! May required some code restructuring to incorporate
        this though. (v >= 1.0.1)

    .. note::

        The usage of the variable z in the documentation is simply to emphasize the generality of the object. In actuality,
        it is the same as x within the :code:`_Kernel` base class.

    :kwarg name: str. Codename of :code:`_WarpingFunction` class implementation.

    :kwarg func: callable. Warping function of :code:`_WarpingFunction` class implementation.

    :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

    :kwarg hyps: array. Hyperparameters to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

    :kwarg csts: array. Constants to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

    :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

    :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)
    '''

    def __init__(
        self,
        name='Warp',
        func=None,
        hderf=False,
        hyps=None,
        csts=None,
        htags=None,
        ctags=None
    ):
        r'''
        Initializes the :code:`_WarpingFunction` instance.

        .. note::

            Nothing is done with the :code:`htags` and :code:`ctags` arguments currently. (v >= 1.0.1)

        :kwarg name: str. Codename of :code:`_WarpingFunction` class implementation.

        :kwarg func: callable. Warping function of :code:`_WarpingFunction` class implementation.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives within :code:`func` for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :kwarg csts: array. Constants to be stored in the :code:`_WarpingFunction` instance, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :kwarg htags: array. Names of hyperparameters to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

        :kwarg ctags: array. Names of constants to be stored as indices in the :code:`_WarpingFunction` class implementation. (optional)

        :returns: none.
        '''

        self._fname = name
        self._function = func if func is not None else None
        self._hyperparameters = np.array(hyps, dtype=default_dtype) if isinstance(hyps, array_types) else None
        self._constants = np.array(csts, dtype=default_dtype) if isinstance(csts, array_types) else None
        self._hyp_lbounds = None
        self._hyp_ubounds = None
        self._hderflag = hderf
        self._force_bounds = False


    def __call__(self, zz, der=0, hder=None):
        r'''
        Default class call function, evaluates the stored warping function at the input values.

        :arg zz: array. Vector of z-values at which to evaulate the warping function, can be 1D or 2D depending on application.

        :kwarg der: int. Order of z derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :kwarg hder: int. Order of hyperparameter derivative with which to evaluate the warping function, requires explicit implementation. (optional)

        :returns: array. Warping function evaluations at input values using the given derivative settings. Has the same dimensions as :code:`zz`.
        '''

        k_out = None
        if self._function is not None:
            k_out = self._function(zz, der, hder)
        else:
            raise NotImplementedError('Warping function not yet defined.')
        return k_out


    def __eq__(self, other):
        r'''
	Custom equality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_WarpingFunction` class instance.

        :returns: bool. Indicates whether the two objects are equal to each other.
        '''

        status = False
        if isinstance(other, _WarpingFunction):
            if self.name == other.name:
                shyp = np.all(np.isclose(self.hyperparameters, other.hyperparameters))
                scst = np.all(np.isclose(self.constants, other.constants))
                status = self.hyperparameters.size == other.hyperparameters.size and self.constants.size == other.constants.size and shyp and scst
        return status


    def __ne__(self, other):
        r'''
	Custom inequality operator. Compares name, hyperparameters and constants.

        :arg other: obj. Any other :code:`_WarpingFunction` class instance.

        :returns: bool. Indicates whether the two objects are not equal to each other.
        '''

        return not self.__eq__(other)


    def enforce_bounds(self, value=True):
        r'''
        Sets a flag to enforce the given hyperparameter bounds.

        :kwarg value: bool. Boolean value to set the flag.

        :returns: none.
        '''

        self._force_bounds = True if value else False


    @property
    def name(self):
        r'''
        Returns the codename of the :code:`_WarpingFunction` instance.

        :returns: str. Codename of the :code:`_WarpingFunction` instance.
        '''

        return f'{self._fname}'


    @property
    def hyperparameters(self):
        r'''
        Return the hyperparameters stored in the :code:`_WarpingFunction` instance.

        :returns: array. Hyperparameter list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        '''

        val = np.array([])
        if self._hyperparameters is not None:
            val = self._hyperparameters.copy()
        return val


    @property
    def constants(self):
        r'''
        Return the constants stored in the :code:`_WarpingFunction` instance.

        :returns: array. Constant list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        '''

        val = np.array([])
        if self._constants is not None:
            val = self._constants.copy()
        return val


    @property
    def bounds(self):
        r'''
        Return the hyperparameter search bounds stored in the :code:`_WarpingFunction` instance.

        :returns: array. Hyperparameter bounds list, ordered according to the specific :code:`_WarpingFunction` class implementation.
        '''

        val = None
        if isinstance(self._hyp_lbounds, np.ndarray) and isinstance(self._hyp_ubounds, np.ndarray) and self._hyp_lbounds.shape == self._hyp_ubounds.shape:
            val = np.vstack((self._hyp_lbounds.flatten(), self._hyp_ubounds.flatten()))
        return val


    def is_hderiv_implemented(self):
        r'''
        Checks if the explicit hyperparameter derivative is implemented in this :code:`_WarpingFunction` class implementation.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        '''

        return self._hderflag


    @hyperparameters.setter
    def hyperparameters(self, theta):
        r'''
        Set the hyperparameters stored in the :code:`_WarpingFunction` instance.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        userhyps = None
        if isinstance(theta, array_types):
            userhyps = np.array(theta).flatten()
        else:
            raise TypeError(f'{self.name} WarpingFunction hyperparameters must be given as an array-like object.')
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userhyps.size >= nhyps:
                if self._force_bounds and isinstance(self._hyp_lbounds, np.ndarray) and self._hyp_lbounds.size == nhyps:
                    htemp = userhyps[:nhyps]
                    lcheck = (htemp < self._hyp_lbounds)
                    htemp[lcheck] = self._hyp_lbounds[lcheck]
                    userhyps[:nhyps] = htemp
                if self._force_bounds and isinstance(self._hyp_ubounds, np.ndarray) and self._hyp_ubounds.size == nhyps:
                    htemp = userhyps[:nhyps]
                    ucheck = (htemp > self._hyp_ubounds)
                    htemp[ucheck] = self._hyp_ubounds[ucheck]
                    userhyps[:nhyps] = htemp
                self._hyperparameters = np.array(userhyps[:nhyps], dtype=default_dtype)
            else:
                raise ValueError(f'{self.name} WarpingFunction hyperparameters must contain at least {nhyps} elements.')
        else:
            warnings.warn(f'{self.name} WarpingFunction instance has no hyperparameters.', stacklevel=2)


    @constants.setter
    def constants(self, consts):
        r'''
        Set the constants stored in the :code:`_WarpingFunction` object.

        :arg consts: array. Constant list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        usercsts = None
        if isinstance(consts, array_types):
            usercsts = np.array(consts).flatten()
        else:
            raise TypeError(f'{self.name} WarpingFunction constants must be given as an array-like object.')
        ncsts = self.constants.size
        if ncsts > 0:
            if usercsts.size >= ncsts:
                self._constants = np.array(usercsts[:ncsts], dtype=default_dtype)
            else:
                raise ValueError(f'{self.name} WarpingFunction constants must contain at least {ncsts} elements.')
        else:
            warnings.warn(f'{self.name} WarpingFunction instance has no constants.', stacklevel=2)


    @bounds.setter
    def bounds(self,bounds):
        r'''
        Set the hyperparameter bounds stored in the :code:`_WarpingFunction` instance.

        :arg bounds: array. Hyperparameter lower/upper bound list to be stored, ordered according to the specific :code:`_WarpingFunction` class implementation.

        :returns: none.
        '''

        userbnds = None
        if isinstance(bounds, array_types):
            userbnds = np.atleast_2d(bounds)
        else:
            raise TypeError(f'{self.name} WarpingFunction bounds must be given as a 2D-array-like object with exactly 2 rows.')
        if userbnds.shape[0] != 2:
            raise TypeError(f'{self.name} WarpingFunction bounds must be given as a 2D-array-like object with exactly 2 rows.')
        nhyps = self.hyperparameters.size
        if nhyps > 0:
            if userbnds.shape[1] >= nhyps:
                self._hyp_lbounds = np.array(userbnds[0, :nhyps], dtype=default_dtype)
                self._hyp_ubounds = np.array(userbnds[1, :nhyps], dtype=default_dtype)
                if self._force_bounds:
                    self.hyperparameters = self._hyperparameters.copy()
            else:
                raise ValueError(f'{self.name} WarpingFunction bounds must be a 2D-array-like object with exactly 2 rows and contain at least {nhyps} elements per row.')
        else:
            warnings.warn(f'{self.name} WarpingFunction instance has no hyperparameters to set bounds for.', stacklevel=2)

