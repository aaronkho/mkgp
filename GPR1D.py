"""
Classes for Gaussian Process Regression fitting of 1D data with errorbars.

These classes were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET,
in
IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis (Boston, MA, 2017),
`<https://nucleus.iaea.org/sites/fusionportal/Shared Documents/Fusion Data Processing 2nd/31.05/Ho.pdf>`_

"""
#    Kernel theory: "Gaussian Process for Machine Learning", C.E. Rasmussen, C.K.I. Williams (2006)
#    Gaussian process theory: "Gaussian Processes for Machine Learning", C.E. Rasmussen and C.K.I. Williams (2006)

# Required imports
import warnings
import re
import copy
import numpy as np
import scipy.special as spsp
import scipy.linalg as spla
import scipy.stats as spst
from operator import itemgetter

np_itypes = (np.int8,np.int16,np.int32,np.int64)
np_utypes = (np.uint8,np.uint16,np.uint32,np.uint64)
np_ftypes = (np.float16,np.float32,np.float64)

__all__ = ['Sum_Kernel', 'Product_Kernel', 'Symmetric_Kernel',  # Kernel operator classes
           'Constant_Kernel', 'Noise_Kernel', 'Linear_Kernel', 'Poly_Order_Kernel', 'SE_Kernel', 'RQ_Kernel',
           'Matern_HI_Kernel', 'NN_Kernel', 'Gibbs_Kernel',  # Kernel classes
           'Constant_WarpingFunction', 'IG_WarpingFunction',  # Warping function classes for Gibbs Kernel
           'KernelConstructor', 'KernelReconstructor',  # Kernel construction functions
           'GaussianProcessRegression1D']  # Main interpolation class


class _Kernel(object):
    """
    Base class   *** to be inherited by ALL kernel implementations in order for type checks to succeed ***
        Type checking done with:     isinstance(<obj>,<this_module>._Kernel)
    Ideology: fname is a string, designed to provide an easy way to check the kernel object type
              function contains the covariance function, k, along with dk/dx1, dk/dx2, and d^2k/dx1dx2
              hyperparameters contains free variables that vary in logarithmic-space
              constants contains "free" variables that should not be changed during parameter searches, or true constants
              bounds contains the bounds of the free variables to be used in MC hyperparameter searches
    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly NOT recommended unless you are familiar with how these structures work and their interdependencies.
    """

    def __init__(self,name="None",func=None,hderf=False,hyps=None,csts=None,htags=None,ctags=None):
        """
        Initializes the Kernel object.

        :kwarg name: str. Codename of Kernel object.

        :kwarg func: callable. Evaluation function of Kernel object.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the Kernel object.

        :kwarg csts: array. Constants to be stored in the Kernel object.

        :kwarg htags: array. Optional names of hyperparameters to be stored in the Kernel object.

        :kwarg ctags: array. Optional names of constants to be stored in the Kernel object.
        """

        self._fname = name
        self._function = func if callable(func) else None
        self._hyperparameters = np.array(copy.deepcopy(hyps)).flatten() if hyps is not None else None
        self._constants = np.array(copy.deepcopy(csts)).flatten() if csts is not None else None
        self._bounds = None
        self._hderflag = hderf


    def __call__(self,x1,x2,der=0,hder=None):
        """
        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        k_out = None
        if callable(self._function):
            k_out = self._function(x1,x2,der,hder)
        else:
            raise NotImplementedError('Kernel function not yet defined.')
        return k_out


    def get_name(self):
        """
        Returns the codename of the Kernel object.

        :returns: str. Kernel codename.
        """

        return self._fname


    def get_hyperparameters(self,log=False):
        """
        Return the hyperparameters stored in the Kernel object.

        :kwarg log: bool. Returns values as log10(values).

        :returns: array. Hyperparameter list.
        """

        val = np.array([])
        if self._hyperparameters is not None:
            val = np.log10(self._hyperparameters) if log else self._hyperparameters
        return val


    def get_constants(self):
        """
        Return the constants stored in the Kernel object.

        :returns: array. Constant list.
        """

        val = np.array([])
        if self._constants is not None:
            val = self._constants
        return val


    def get_bounds(self,log=False):
        """
        Return the hyperparameter search bounds stored in the Kernel object.

        :returns: array. Bounds list.
        """

        val = None
        if self._bounds is not None:
            val = np.log10(self._bounds) if log else self._bounds
        return val


    def is_hderiv_implemented(self):
        """
        Checks if the explicit hyperparameter derivative is implemented in this Kernel object.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        """

        return self._hderflag


    def set_hyperparameters(self,theta,log=False):
        """
        Set the hyperparameters stored in the Kernel object.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific Kernel object.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        uhyps = None
        if isinstance(theta,(list,tuple)):
            uhyps = np.array(theta).flatten()
        elif isinstance(theta,np.ndarray):
            uhyps = theta.flatten()
        else:
            raise TypeError('Argument theta must be an array-like object.')
        if log:
            uhyps = np.power(10.0,uhyps)
        if self._hyperparameters is not None:
            if uhyps.size >= self._hyperparameters.size:
                if self._bounds is not None and self._bounds.shape[1] == self._hyperparameters.size:
                    htemp = uhyps[:self._hyperparameters.size]
                    lbounds = self._bounds[0,:].flatten()
                    lcheck = (htemp < lbounds)
                    htemp[lcheck] = lbounds[lcheck]
                    ubounds = self._bounds[1,:].flatten()
                    ucheck = (htemp > ubounds)
                    htemp[ucheck] = ubounds[ucheck]
                    uhyps[:self._hyperparameters.size] = htemp
                self._hyperparameters = uhyps[:self._hyperparameters.size]
            else:
                raise ValueError('Argument theta must contain at least %d elements.' % (self._hyperparameters.size))
        else:
            raise AttributeError('Kernel object has no hyperparameters.')


    def set_constants(self,consts):
        """
        Set the constants stored in the Kernel object.

        :arg consts: array. Constant list to be stored, ordered according to the specific Kernel object.

        :returns: none.
        """

        ucsts = None
        if isinstance(consts,(list,tuple)):
            ucsts = np.array(consts).flatten()
        elif isinstance(consts,np.ndarray):
            ucsts = consts.flatten()
        else:
            raise TypeError('Argument consts must be an array-like object.')
        if self._constants is not None:
            if ucsts.size >= self._constants.size:
                self._constants = ucsts[:self._constants.size]
            else:
                raise ValueError('Argument consts must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('Kernel object has no constants.')


    def set_bounds(self,lbounds,ubounds,log=False):
        """
        Set the hyperparameter bounds stored in the Kernel object.

        :arg lbounds: array. Hyperparameter lower bound list to be stored, ordered according to the specific Kernel object.

        :arg ubounds: array. Hyperparameter upper bound list to be stored, ordered according to the specific Kernel object.

        :kwarg log: bool. Indicates that lbounds and ubounds are passed in as log10(lbounds) and log10(ubounds).

        :returns: none.
        """

        ubnds = None
        if isinstance(lbounds,(list,tuple)):
            ubnds = np.array(lbounds).flatten()
        elif isinstance(lbounds,np.ndarray):
            ubnds = lbounds.flatten()
        else:
            raise TypeError('Argument lbounds must be an array-like object.')
        if isinstance(ubounds,(list,tuple)) and len(ubounds) == ubnds.size:
            ubnds = np.vstack((ubnds,np.array(ubounds).flatten()))
        elif isinstance(ubounds,np.ndarray) and ubounds.size == ubnds.size:
            ubnds = np.vstack((ubnds,ubounds.flatten()))
        else:
            raise TypeError('Argument ubounds must be an array-like object and have dimensions equal to lbounds.')
        if log:
            ubnds = np.power(10.0,ubnds)
        if self._hyperparameters is not None:
            if ubnds.shape[1] >= self._hyperparameters.size:
                self._bounds = ubnds[:,:self._hyperparmaeters.size]
                htemp = self._hyperparameters.copy()
                lbounds = self._bounds[0,:].flatten()
                lcheck = (htemp < lbounds)
                htemp[lcheck] = lbounds[lcheck]
                ubounds = self._bounds[1,:].flatten()
                ucheck = (htemp > ubounds)
                htemp[ucheck] = ubounds[ucheck]
                self._hyperparameters = copy.deepcopy(htemp)
            else:
                raise ValueError('Arguments lbounds and ubounds must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('Kernel object has no hyperparameters to set bounds for.')



class _OperatorKernel(_Kernel):
    """
    Base operator class   *** To be inherited by ALL operator kernel implementations for obtain custom get/set functions ***
        Type checking done with:     isinstance(<obj>,<this_module>._OperatorKernel) if needed
    Ideology: kernel_list is a Python list of Kernel objects on which the specified operation will be performed on
    Get/set functions adjusted to call get/set functions of each constituent kernel instead of using its own properties,
    which are mostly left as None.
    """
    def __init__(self,name="None",func=None,hderf=False,klist=None):
        """
        Initializes the OperatorKernel object.

        :kwarg name: str. Codename of OperatorKernel object.

        :kwarg func: callable. Evaluation function of OperatorKernel object.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives for optimization algorithms.

        :kwarg klist: array. List of Kernel objects to be operated on by the OperatorKernel object.

        :returns: none.
        """
        super(_OperatorKernel,self).__init__(name,func,hderf)
        self._kernel_list = klist if klist is not None else []


    def get_hyperparameters(self,log=False):
        """
        Return the hyperparameters stored in all the Kernel objects within the OperatorKernel object.

        :kwarg log: bool. Returns values as log10(values).

        :returns: array. Hyperparameter list.
        """

        val = np.array([])
        for kk in self._kernel_list:
            val = np.append(val,kk.get_hyperparameters(log=log))
        return val


    def get_constants(self):
        """
        Return the constants stored in all the Kernel objects within the OperatorKernel object.

        :returns: array. Constant list.
        """

        val = np.array([])
        for kk in self._kernel_list:
            val = np.append(val,kk.get_constants())
        return val


    def get_bounds(self,log=False):
        """
        Return the hyperparameter bounds stored in all the Kernel objects within the OperatorKernel object.

        :kwarg log: bool. Returns values as log10(values).

        :returns: array. Hyperparameter bounds list.
        """

        val = np.array([])
        for kk in self._kernel_list:
            kval = kk.get_bounds(log=log)
            if kval is not None:
                val = np.append(val,kval)
        if val.size == 0:
            val = None
        return val


    def set_hyperparameters(self,theta,log=False):
        """
        Set the hyperparameters stored in all the Kernel objects within the OperatorKernel object.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific OperatorKernel object.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        uhyps = None
        if isinstance(theta,(list,tuple)):
            uhyps = np.array(theta).flatten()
        elif isinstance(theta,np.ndarray):
            uhyps = theta.flatten()
        else:
            raise TypeError('Argument theta must be an array-like object.')
        if log:
            uhyps = np.power(10.0,uhyps)
        nhyps = self.get_hyperparameters().size
        if nhyps > 0:
            if uhyps.size >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.get_hyperparameters().size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.set_hyperparameters(theta[ndone:],log=log)
                        else:
                            kk.set_hyperparameters(theta[ndone:nhere],log=log)
                        ndone = nhere
            else:
                raise ValueError('Argument theta must contain at least %d elements.' % (nhyps))
        else:
            raise AttributeError('Kernel object has no hyperparameters.')


    def set_constants(self,consts):
        """
        Set the constants stored in all the Kernel objects within the OperatorKernel object.

        :arg consts: array. Constant list to be stored, ordered according to the specific OperatorKernel object.

        :returns: none.
        """

        ucsts = None
        if isinstance(consts,(list,tuple)):
            ucsts = np.array(consts).flatten()
        elif isinstance(consts,np.ndarray):
            ucsts = consts.flatten()
        else:
            raise TypeError('Argument consts must be an array-like object.')
        ncsts = self.get_constants().size
        if ncsts > 0:
            if ucsts.size >= ncsts:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.get_constants().size
                    if nhere != ndone:
                        if nhere == ncsts:
                            kk.set_constants(consts[ndone:])
                        else:
                            kk.set_constants(consts[ndone:nhere])
                        ndone = nhere
            else:
                raise ValueError('Argument consts must contain at least %d elements.' % (ncsts))
        else:
            raise AttributeError('Kernel object has no constants.')


    def set_bounds(self,lbounds,ubounds,log=False):
        """
        Set the hyperparameter bounds stored in all the Kernel objects within the OperatorKernel object.

        :arg lbounds: array. Hyperparameter lower bound list to be stored, ordered according to the specific OperatorKernel object.

        :arg ubounds: array. Hyperparameter upper bound list to be stored, ordered according to the specific OperatorKernel object.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        ubnds = None
        if isinstance(lbounds,(list,tuple)):
            ubnds = np.array(lbounds).flatten()
        elif isinstance(lbounds,np.ndarray):
            ubnds = lbounds.flatten()
        else:
            raise TypeError('Argument lbounds must be an array-like object.')
        if isinstance(ubounds,(list,tuple)) and len(ubounds) == ubnds.size:
            ubnds = np.vstack((ubnds,np.array(ubounds).flatten()))
        elif isinstance(ubounds,np.ndarray) and ubounds.size == ubnds.size:
            ubnds = np.vstack((ubnds,ubounds.flatten()))
        else:
            raise TypeError('Argument ubounds must be an array-like object and have dimensions equal to lbounds.')
        if log:
            ubnds = np.power(10.0,ubnds)
        nhyps = self.get_hyperparameters().size
        if nhyps > 0:
            if ubnds.shape[1] >= nhyps:
                ndone = 0
                for kk in self._kernel_list:
                    nhere = ndone + kk.get_hyperparameters().size
                    if nhere != ndone:
                        if nhere == nhyps:
                            kk.set_bounds(ubnds[0,ndone:],ubnds[1,ndone:],log=log)
                        else:
                            kk.set_bounds(ubnds[0,ndone:nhere],ubnds[1,ndone:nhere],log=log)
                        ndone = nhere
            else:
                raise ValueError('Arguments lbounds and ubounds must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('Kernel object has no hyperparameters to set bounds for.')



class _WarpingFunction(object):
    """
    Base class   *** to be inherited by ALL warping function implementations in order for type checks to succeed ***
        Type checking done with:     isinstance(<obj>,<this_module>.WarpingFunction)
    Ideology: fname is a string, designed to provide an easy way to check the warping function object type
              function contains the warping function, l, along with dl/dz and d^2l/dz^2
              hyperparameters contains free variables that vary in logarithmic-space
              constants contains "free" variables that should not be changed during parameter searches, or true constants
              bounds contains the bounds of the free variables to be used in MC hyperparameter searches
    Get/set functions already given, but as always in Python, all functions can be overridden by specific implementation.
    This is strongly NOT recommended unless you are familiar with how these structures work and their interdependencies.
    .. note:: The usage of the variable z in the documentation is simply to emphasize the generality of the object.
    """

    def __init__(self,name="None",func=None,hderf=False,hyps=None,csts=None):
        """
        Initializes the WarpingFunction object.

        :kwarg name: str. Codename of WarpingFunction object.

        :kwarg func: callable. Evaluation function of WarpingFunction object.

        :kwarg hderf: bool. Indicates availability of analytical hyperparameter derivatives for optimization algorithms.

        :kwarg hyps: array. Hyperparameters to be stored in the WarpingFunction object.

        :kwarg csts: array. Constants to be stored in the WarpingFunction object.

        :returns: none.
        """

        self._fname = name
        self._function = func if func is not None else None
        self._hyperparameters = copy.deepcopy(hyps) if hyps is not None else None
        self._constants = copy.deepcopy(csts) if csts is not None else None
        self._bounds = None
        self._hderflag = hderf


    def __call__(self,zz,der=0,hder=None):
        """
        :arg zz: array. Values to evaulate the warping function at, can be 1-D or 2-D depending on application.

        :kwarg der: int. Order of z derivative to evaluate the warping function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the warping function at, requires explicit implementation.

        :returns: array. Warping function evaluations at input values, same dimensions as zz.
        """

        k_out = None
        if self._function is not None:
            k_out = self._function(zz,der,hder)
        else:
            raise NotImplementedError('Warping function not yet defined.')
        return k_out


    def get_name(self):
        """
        Returns the codename of the WarpingFunction object.

        :returns: str. WarpingFunction codename.
        """

        return self._fname


    def get_hyperparameters(self,log=False):
        """
        Return the hyperparameters stored in the WarpingFunction object.

        :kwarg log: bool. Returns values as log10(values).

        :returns: array. Hyperparameter list.
        """

        val = np.array([])
        if self._hyperparameters is not None:
            val = np.log10(self._hyperparameters) if log else self._hyperparameters
        return val


    def get_constants(self):
        """
        Return the constants stored in the WarpingFunction object.

        :returns: array. Constant list.
        """

        val = np.array([])
        if self._constants is not None:
            val = self._constants
        return val


    def get_bounds(self,log=False):
        """
        Return the hyperparameter search bounds stored in the WarpingFunction object.

        :returns: array. Bounds list.
        """

        val = None
        if self._bounds is not None:
            val = np.log10(self._bounds) if log else self._bounds
        return val


    def is_hderiv_implemented(self):
        """
        Checks if the explicit hyperparameter derivative is implemented in this WarpingFunction object.

        :returns: bool. True if explicit hyperparameter derivative is implemented.
        """

        return self._hderflag


    def set_hyperparameters(self,theta,log=False):
        """
        Set the hyperparameters stored in the WarpingFunction object.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific WarpingFunction object.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        uhyps = None
        if isinstance(theta,(list,tuple)):
            uhyps = np.array(theta).flatten()
        elif isinstance(theta,np.ndarray):
            uhyps = theta.flatten()
        else:
            raise TypeError('Argument theta must be an array-like object.')
        if log:
            uhyps = np.power(10.0,uhyps)
        if self._hyperparameters is not None:
            if uhyps.size >= self._hyperparameters.size:
                if self._bounds is not None and self._bounds.shape[1] == self._hyperparameters.size:
                    htemp = uhyps[:self._hyperparameters.size]
                    lbounds = self._bounds[0,:].flatten()
                    lcheck = (htemp < lbounds)
                    htemp[lcheck] = lbounds[lcheck]
                    ubounds = self._bounds[1,:].flatten()
                    ucheck = (htemp > ubounds)
                    htemp[ucheck] = ubounds[ucheck]
                    uhyps[:self._hyperparameters.size] = htemp
                self._hyperparameters = uhyps[:self._hyperparameters.size]
            else:
                raise ValueError('Argument theta must contain at least %d elements.' % (self._hyperparameters.size))
        else:
            raise AttributeError('WarpingFunction object has no hyperparameters.')


    def set_constants(self,consts):
        """
        Set the constants stored in the WarpingFunction object.

        :arg consts: array. Constant list to be stored, ordered according to the specific WarpingFunction object.

        :returns: none.
        """

        ucsts = None
        if isinstance(consts,(list,tuple)):
            ucsts = np.array(consts).flatten()
        elif isinstance(consts,np.ndarray):
            ucsts = consts.flatten()
        else:
            raise TypeError('Argument consts must be an array-like object.')
        if self._constants is not None:
            if ucsts.size >= self._constants.size:
                self._constants = ucsts[:self._constants.size]
            else:
                raise ValueError('Argument consts must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('WarpingFunction object has no constants.')


    def set_bounds(self,lbounds,ubounds,log=False):
        """
        Set the hyperparameter bounds stored in the WarpingFunction object.

        :arg lbounds: array. Hyperparameter lower bound list to be stored, ordered according to the specific WarpingFunction object.

        :arg ubounds: array. Hyperparameter upper bound list to be stored, ordered according to the specific WarpingFunction object.

        :kwarg log: bool. Indicates that lbounds and ubounds are passed in as log10(lbounds) and log10(ubounds).

        :returns: none.
        """

        ubnds = None
        if isinstance(lbounds,(list,tuple)):
            ubnds = np.array(lbounds).flatten()
        elif isinstance(lbounds,np.ndarray):
            ubnds = lbounds.flatten()
        else:
            raise TypeError('Argument lbounds must be an array-like object.')
        if isinstance(ubounds,(list,tuple)) and len(ubounds) == ubnds.size:
            ubnds = np.vstack((ubnds,np.array(ubounds).flatten()))
        elif isinstance(ubounds,np.ndarray) and ubounds.size == ubnds.size:
            ubnds = np.vstack((ubnds,ubounds.flatten()))
        else:
            raise TypeError('Argument ubounds must be an array-like object and have dimensions equal to lbounds.')
        if log:
            ubnds = np.power(10.0,ubnds)
        if self._hyperparameters is not None:
            if ubnds.shape[1] >= self._hyperparameters.size:
                self._bounds = ubnds[:,:self._hyperparmaeters.size]
                htemp = self._hyperparameters.copy()
                lbounds = self._bounds[0,:].flatten()
                lcheck = (htemp < lbounds)
                htemp[lcheck] = lbounds[lcheck]
                ubounds = self._bounds[1,:].flatten()
                ucheck = (htemp > ubounds)
                htemp[ucheck] = ubounds[ucheck]
                self._hyperparameters = copy.deepcopy(htemp)
            else:
                raise ValueError('Arguments lbounds and ubounds must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('WarpingFunction object has no hyperparameters to set bounds for.')



# ****************************************************************************************************************************************
# ------- Place ALL custom kernel implementations BELOW ----------------------------------------------------------------------------------
# ****************************************************************************************************************************************

class Sum_Kernel(_OperatorKernel):
    """
    Sum Kernel: Implements the sum of two (or more) Kernel objects.

    :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be added together, minimum of 2.

    :kwarg klist: list. Kernel objects to be added together, minimum of 2.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        covm = np.NaN if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp)
            if ihyp is not None:
                hyps = np.array(kk.get_hyperparameters())
                nhyps = hyps.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the Sum_Kernel object. Adapted to be Python 2.x compatible.

        :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be added together, minimum of 2.

        :kwarg klist: list. Kernel objects to be added together, minimum of 2.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        name = "None"
        if len(args) >= 2 and isinstance(args[0],_Kernel) and isinstance(args[1],_Kernel):
            name = ""
            for kk in args:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        elif isinstance(klist,list) and len(klist) >= 2 and isinstance(klist[0],_Kernel) and isinstance(klist[1],_Kernel):
            name = ""
            for kk in klist:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel objects.')
        super(Sum_Kernel,self).__init__("Sum("+name+")",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Sum_Kernel(klist=kcopy_list)
        return kcopy



class Product_Kernel(_OperatorKernel):
    """
    Product Kernel: Implements the product of two (or more) Kernel objects.

    :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be multiplied together, minimum of 2.

    :kwarg klist: list. Kernel objects to be multiplied together, minimum of 2.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        covm = np.NaN if self._kernel_list is None else np.zeros(x1.shape)
        nks = len(self._kernel_list)
        dermat = np.atleast_2d([0] * nks)
        sd = int(np.sign(der))
        for ii in np.arange(0,int(np.abs(der))):
            for jj in np.arange(1,nks):
                deradd = dermat.copy()
                dermat = np.vstack((dermat,deradd))
            for row in np.arange(0,dermat.shape[0]):
                rem = row % nks
                fac = (row - rem) / (nks**int(np.abs(der)))
                idx = int((rem + fac) % nks)
                dermat[row,idx] = dermat[row,idx] + 1
        oddfilt = (np.mod(dermat,2) != 0)
        dermat[oddfilt] = sd * dermat[oddfilt]
        for row in np.arange(0,dermat.shape[0]):
            ihyp = hder
            covterm = np.ones(x1.shape)
            for col in np.arange(0,dermat.shape[1]):
                kk = self._kernel_list[col]
                covterm = covterm * kk(x1,x2,int(dermat[row,col]),ihyp)
                if ihyp is not None:
                    hyps = np.array(kk.get_hyperparameters())
                    nhyps = hyps.size
                    ihyp = ihyp - nhyps
            covm = covm + covterm
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the Product_Kernel object. Adapted to be Python 2.x compatible.

        :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be multiplied together, minimum of 2.

        :kwarg klist: list. Kernel objects to be multiplied together, minimum of 2.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        name = "None"
        if len(args) >= 2 and isinstance(args[0],_Kernel) and isinstance(args[1],_Kernel):
            name = ""
            for kk in args:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        elif isinstance(klist,list) and len(klist) >= 2 and isinstance(klist[0],_Kernel) and isinstance(klist[1],_Kernel):
            name = ""
            for kk in klist:
                if isinstance(kk,_Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel objects.')
        super(Product_Kernel,self).__init__("Prod("+name+")",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Product_Kernel(klist=kcopy_list)
        return kcopy



class Symmetric_Kernel(_OperatorKernel):
    """
    1D Symmetric Kernel: Enforces even symmetry about zero for any given Kernel object (only uses first Kernel argument, though it accepts many).
    This is really only useful if you wish to rigourously infer data on other side of axis of symmetry without assuming the data
    can just be flipped or if data exists on other side but a symmetric solution is desired. *** NOT FULLY TESTED! ***

    :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be made symmetric, minimum of 2.

    :kwarg klist: list. Kernel object to be made symmetric, maximum of 1.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        covm = np.NaN if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp) + kk(-x1,x2,der,ihyp)      # Not sure if division by 2 is necessary to conserve covm
            if ihyp is not None:
                hyps = np.array(kk.get_hyperparameters())
                nhyps = hyps.size
                ihyp = ihyp - nhyps
        return covm


    def __init__(self,*args,**kwargs):
        """
        Initializes the Symmetric_Kernel object. Adapted to be Python 2.x compatible.

        :arg \*args: obj. Any number of Kernel objects arguments, separated by commas, representing Kernel objects to be made symmetric, minimum of 2.

        :kwarg klist: list. Kernel object to be made symmetric, maximum of 1.

        :returns: none.
        """

        klist = kwargs.get("klist")
        uklist = []
        name = "None"
        if len(args) >= 1 and isinstance(args[0],_Kernel):
            name = ""
            if len(args) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = args[0]
            uklist.append(kk)
            name = name + kk.get_name()
        elif isinstance(klist,list) and len(klist) >= 1 and isinstance(klist[0],_Kernel):
            name = ""
            if len(klist) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = klist[0]
            uklist.append(kk)
            name = name + kk.get_name()
        else:
            raise TypeError('Arguments to Symmetric_Kernel must be Kernel objects.')
        super(Symmetric_Kernel,self).__init__("Sym("+name+")",self.__calc_covm,True,uklist)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Symmetric_Kernel(klist=kcopy_list)
        return kcopy



class Constant_Kernel(_Kernel):
    """
    Constant Kernel: always evaluates to a constant value, regardless of x1 and x2.
    .. warning:: Note that this is NOT INHERENTLY A VALID COVARIANCE FUNCTION, as it yields singular covariance matrices!
    However, it provides a nice way to add bias to any other kernel. (is this even true?!?)

    :kwarg cv: float. Constant value which kernel always evaluates to.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        c_hyp = self._constants[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = c_hyp * np.ones(rr.shape)
        return covm


    def __init__(self,cv=1.0):
        """
        Initializes the Constant_Kernel object.

        :kwarg cv: float. Constant value which kernel always evaluates to.

        :returns: none.
        """

        csts = np.zeros((1,))
        if isinstance(cv,(float,int,np_itypes,np_utypes,np_ftypes)):
            csts[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super(Constant_Kernel,self).__init__("C",self.__calc_covm,True,None,csts)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._constants[0])
        kcopy = Constant_Kernel(chp)
        return kcopy



class Noise_Kernel(_Kernel):
    """
    Noise Kernel: adds a user-defined degree of expected noise in the data / measurement process.
    .. warning:: Note that this is NOT THE SAME as measurement error, which should be applied externally in GP!!!

    :kwarg nv: float. Hyperparameter representing the noise level.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        n_hyp = self._hyperparameters[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm[rr == 0.0] = n_hyp**2.0
            elif hder == 0:
                covm[rr == 0.0] = 2.0 * n_hyp
#       Applied second derivative of Kronecker delta, assuming it is actually a Gaussian centred on rr = 0 with small width ss
#       Surprisingly provides good variance estimate but issues with enforcing derivative constraints (needs more work!)
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


    def __init__(self,nv=1.0):
        """
        Initializes the Noise_Kernel object.

        :kwarg nv: float. Hyperparameter representing the noise level.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(nv,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(nv)
        else:
            raise ValueError('Noise hyperparameter must be a real number.')
        super(Noise_Kernel,self).__init__("n",self.__calc_covm,True,hyps)

    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        nhp = float(self._hyperparameters[0])
        kcopy = Noise_Kernel(nhp)
        return kcopy



class Linear_Kernel(_Kernel):
    """
    Linear Kernel: Applies linear regression ax + b, where b = 0, can be multiplied with itself
    for higher order pure polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. a.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        v_hyp = self._hyperparameters[0]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * pp
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = v_hyp**2.0 * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = v_hyp**2.0 * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = v_hyp**2.0 * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self,var=1.0):
        """
        Initializes the Linear_Kernel object.

        :kwarg var: float. Hyperparameter multiplying linear component of model.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        super(Linear_Kernel,self).__init__("L",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._hyperparameters[0])
        kcopy = Linear_Kernel(chp)
        return kcopy



class Poly_Order_Kernel(_Kernel):
    """
    Polynomial Order Kernel: Applies linear regression ax + b, where b != 0, can be multiplied with
    itself for higher order polynomials.

    :kwarg var: float. Hyperparameter multiplying linear component of model, ie. a.

    :kwarg cst: float. Hyperparameter added to linear component of model, ie. b.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        v_hyp = self._hyperparameters[0]
        b_hyp = self._hyperparameters[1]
        pp = x1 * x2
        covm = np.zeros(pp.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * pp + b_hyp**2.0
            elif hder == 0:
                covm = 2.0 * v_hyp * pp
            elif hder == 1:
                covm = b_hyp * np.ones(pp.shape)
        elif der == 1:
            dpdx2 = x1
            if hder is None:
                covm = v_hyp**2.0 * dpdx2
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx2
        elif der == -1:
            dpdx1 = x2
            if hder is None:
                covm = v_hyp**2.0 * dpdx1
            elif hder == 0:
                covm = 2.0 * v_hyp * dpdx1
        elif der == 2 or der == -2:
            if hder is None:
                covm = v_hyp**2.0 * np.ones(pp.shape)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.ones(pp.shape)
        return covm


    def __init__(self,var=1.0,cst=1.0):
        """
        Initializes the Poly_Order_Kernel object.

        :kwarg var: float. Hyperparameter multiplying linear component of model.

        :kwarg cst: float. Hyperparameter added to linear component of model.

        :returns: none.
        """

        hyps = np.zeros((2,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Multiplicative hyperparameter must be greater than 0.')
        if isinstance(cst,(float,int,np_itypes,np_utypes,np_ftypes)) and float(cst) > 0.0:
            hyps[1] = float(cst)
        else:
            raise ValueError('Additive hyperparameter must be greater than 0.')
        super(Poly_Order_Kernel,self).__init__("P",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._hyperparameters[0])
        cst = float(self._hyperparameters[1])
        kcopy = Poly_Order_Kernel(chp,cst)
        return kcopy



class SE_Kernel(_Kernel):
    """
    Square Exponential Kernel: Infinitely differentiable (ie. extremely smooth) covariance function.

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        v_hyp = self._hyperparameters[0]
        l_hyp = self._hyperparameters[1]
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
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * v_hyp**2.0 / np.power(l_hyp,nn)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * v_hyp / np.power(l_hyp,nn)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - jj) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * v_hyp**2.0 / np.power(l_hyp,nn + 1)
            efac = np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power(rr / l_hyp,nn - jj + 2)
            covm = afac * efac * sfac
        return covm


    def __init__(self,var=1.0,ls=1.0):
        """
        Initializes the SE_Kernel object.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter represeting variability of model in x, ie. length scale.

        :returns: none.
        """

        hyps = np.zeros((2,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        if isinstance(ls,(float,int,np_itypes,np_utypes,np_ftypes)) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Length scale hyperparameter must be greater than 0.')
        super(SE_Kernel,self).__init__("SE",self.__calc_covm,True,hyps)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._hyperparameters[0])
        shp = float(self._hyperparameters[1])
        kcopy = SE_Kernel(chp,shp)
        return kcopy



class RQ_Kernel(_Kernel):
    """
    Rational Quadratic Kernel: Also infinitely differentiable, but provides higher tolerance for steep slopes.
    Acts as infinite sum of SE kernels for a_hyp < 20, otherwise effectively identical to SE as a_hyp -> infinity.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

    :kwarg alpha: float. Hyperparameter representing degree of length scale mixing in model.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        rq_amp = self._hyperparameters[0]
        l_hyp = self._hyperparameters[1]
        a_hyp = self._hyperparameters[2]
        rr = np.abs(x1 - x2)
        rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
        drdx1 = np.sign(x1 - x2)
        drdx1[drdx1 == 0] = 1.0
        drdx2 = np.sign(x2 - x1)
        drdx2[drdx2 == 0] = -1.0
        nn = int(np.abs(der))
        dx1 = int(nn / 2) + 1 if (der % 2) != 0 and der < 0 else int(nn / 2)
        dx2 = int(nn / 2) + 1 if (der % 2) != 0 and der > 0 else int(nn / 2)

        covm = np.zeros(rr.shape)
        if hder is None:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * rq_amp / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                sfac = sfac + cfac * gfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn))
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii) + 1.0) / (np.power(a_hyp,nn - ii + 1) * spsp.gamma(a_hyp))
                sfac = sfac + lfac * gfac * np.power(rr / l_hyp,nn - jj + 2)
            covm = afac * efac * sfac
        elif hder == 2:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * rq_amp**2.0 / np.power(l_hyp,nn)
            efac = np.power(rqt,-a_hyp - float(nn) - 1.0)
            sfac = np.zeros(rr.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                gfac = np.power(rqt,ii) * spsp.gamma(a_hyp + float(nn) - float(ii)) / (np.power(a_hyp,nn - ii) * spsp.gamma(a_hyp))
                pfac = (a_hyp - 2.0 * ii) / (a_hyp) * (rqt - 1.0) - float(nn - ii) / a_hyp - rqt * (np.log(rqt) + spsp.digamma(a_hyp + float(nn) - float(ii)) - spsp.digamma(a_hyp))
                sfac = sfac + cfac * gfac * pfac * np.power(rr / l_hyp,nn - jj)
            covm = afac * efac * sfac
        return covm


    def __init__(self,amp=1.0,ls=1.0,alpha=1.0):
        """
        Initializes the RQ_Kernel object.

        :param amp: float. Hyperparameter representing variability of model in y.

        :param ls: float. Hyperparameter representing variability of model in x, ie. base length scale.

        :param alpha: float. Hyperparameter representing degree of length scale mixing in model.

        :returns: none.
        """

        hyps = np.zeros((3,))
        if isinstance(amp,(float,int,np_itypes,np_utypes,np_ftypes)) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Rational quadratic amplitude must be greater than 0.')
        if isinstance(ls,(float,int,np_itypes,np_utypes,np_ftypes)) and float(ls) != 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Rational quadratic hyperparameter cannot equal 0.')
        if isinstance(alpha,(float,int,np_itypes,np_utypes,np_ftypes)) and float(alpha) > 0.0:
            hyps[2] = float(alpha)
        else:
            raise ValueError('Rational quadratic alpha parameter must be greater than 0.')
        super(RQ_Kernel,self).__init__("RQ",self.__calc_covm,True,hyps)

    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        ramp = float(self._hyperparameters[0])
        rhp = float(self._hyperparameters[1])
        ralp = float(self._hyperparameters[2])
        kcopy = RQ_Kernel(ramp,rhp,ralp)
        return kcopy



class Matern_HI_Kernel(_Kernel):
    """
    Matern Kernel with Half-Integer nu: Only differentiable in orders less than given nu, allows fit to retain more features at expense of volatility.
    The half-integer implentation allows for use of explicit simplifications of the derivatives, which greatly improves its speed.

    Recommended nu: 5/2 for second order differentiability while retaining maximum feature representation, becomes SE Kernel with nu -> infinity.

    :kwarg amp: float. Hyperparameter representing variability of model in y.

    :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

    :kwarg nu: float. Constant value setting the volatility of the model, recommended valie is 2.5.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        mat_amp = self._hyperparameters[0]
        mat_hyp = self._hyperparameters[1]
        nu = self._constants[0]
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
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * mat_amp**2.0 * np.power(np.sqrt(2.0 * nu) / mat_hyp,nn)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 0:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * 2.0 * mat_amp * np.power(np.sqrt(2.0 * nu) / mat_hyp,nn)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            tfac = np.zeros(rr.shape)
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        elif hder == 1:
            afac = np.power(drdx1,dx1) * np.power(drdx2,dx2) * mat_amp**2.0 * np.power(np.sqrt(2.0 * nu),nn) / np.power(mat_hyp,nn + 1)
            efac = np.exp(-mht)
            spre = np.math.factorial(pp) / np.math.factorial(2 * pp)
            ofac = np.zeros(rr.shape)
            for zz in np.arange(0,pp - nn):
                ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - nn - zz - 1))
                ofac = ofac + ffac * np.power(2.0 * mht,pp - nn - zz - 1)
            tfac = -np.power(2.0,nn + 1) * ofac
            for ii in np.arange(0,nn + 1):
                mfac = np.power(-1.0,nn - ii) * np.power(2.0,ii) * np.math.factorial(nn) / (np.math.factorial(ii) * np.math.factorial(nn - ii))
                sfac = np.zeros(rr.shape)
                for zz in np.arange(0,pp - ii + 1):
                    ffac = spre * np.math.factorial(pp + zz) / (np.math.factorial(zz) * np.math.factorial(pp - ii - zz))
                    sfac = sfac + ffac * np.power(2.0 * mht,pp - ii - zz)
                tfac = tfac + mfac * sfac
            covm = afac * efac * tfac
        return covm

    def __init__(self,amp=0.1,ls=0.1,nu=2.5):
        """
        Initializes the Matern_HI_Kernel object.

        :kwarg amp: float. Hyperparameter representing variability of model in y.

        :kwarg ls: float. Hyperparameter representing variability of model in x, ie. length scale.

        :kwarg nu: float. Constant value setting the volatility of the model, recommended value is 2.5.

        :returns: none.
        """

        hyps = np.zeros((2,))
        csts = np.zeros((1,))
        if isinstance(amp,(float,int)) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Matern amplitude hyperparameter must be greater than 0.')
        if isinstance(ls,(float,int)) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Matern length scale hyperparameter must be greater than 0.')
        if isinstance(nu,(float,int)) and float(nu) >= 0.0:
            csts[0] = float(int(nu)) + 0.5
        else:
            raise ValueError('Matern half-integer nu constant must be greater or equal to 0.')
        super(Matern_HI_Kernel,self).__init__("MH",self.__calc_covm,True,hyps,csts)

    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        mamp = float(self._hyperparameters[0])
        mhp = float(self._hyperparameters[1])
        nup = float(self._constants[0])
        kcopy = Matern_HI_Kernel(mamp,mhp,nup)
        return kcopy



class NN_Kernel(_Kernel):
    """
    Neural Network Style Kernel: implements a sigmoid covariance function similar to a perceptron in a neural network, good for strong discontinuities.
    .. note:: Suffers from high volatility like the Matern kernel, have not figured out how to localize impact of kernel to the features in data.

    :kwarg nna: float. Hyperparameter representing variability of model in y.

    :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

    :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        nn_amp = self._hyperparameters[0]
        nn_off = self._hyperparameters[1]
        nn_hyp = self._hyperparameters[2]
        rr = np.abs(x1 - x2)
        pp = x1 * x2
        nnfac = 2.0 / np.pi
        nnn = 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x1 * x2)
        nnd1 = 1.0 + 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x1**2.0)
        nnd2 = 1.0 + 2.0 * (nn_off**2.0 + nn_hyp**2.0 * x2**2.0)
        chi = nnd1 * nnd2
        xi = chi - nnn**2.0
        covm = np.zeros(rr.shape)
        if der == 0:
            covm = nn_amp**2.0 * nnfac * np.arcsin(nnn / np.power(chi,0.5))
        elif der == 1:
            dpdx2 = x1
            dchidx2 = 4.0 * nn_hyp**2.0 * x2 * nnd1
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnm = dpdx2 * chi - dchidx2 * nnn / (4.0 * nn_hyp**2.0)
            covm = nn_amp**2.0 * nnfac * nnk * nnm
        elif der == -1:
            dpdx1 = x2
            dchidx1 = 4.0 * nn_hyp**2.0 * x1 * nnd2
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnm = dpdx1 * chi - dchidx1 * nnn / (4.0 * nn_hyp**2.0)
            covm = nn_amp**2.0 * nnfac * nnk * nnm
        elif der == 2 or der == -2:
            dpdx1 = x2
            dpdx2 = x1
            dchidx1 = 4.0 * nn_hyp**2.0 * x1 * nnd2
            dchidx2 = 4.0 * nn_hyp**2.0 * x2 * nnd1
            d2chi = 16.0 * nn_hyp**4.0 * pp
            nnk = 2.0 * nn_hyp**2.0 / (chi * np.power(xi,0.5))
            nnt1 = chi * (1.0 + (nnn / xi) * (2.0 * nn_hyp**2.0 * pp + d2chi / (8.0 * nn_hyp**2.0)))
            nnt2 = (-0.5 * chi / xi) * (dpdx2 * dchidx1 + dpdx1 * dchidx2) 
            covm = nn_amp**2.0 * nnfac * nnk * (nnt1 + nnt2)
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not implemented in '+self.get_name()+' kernel.')
        return covm


    def __init__(self,nna=1.0,nno=1.0,nnv=1.0):
        """
        Initializes the NN_Kernel object.

        :kwarg nna: float. Hyperparameter representing variability of model in y.

        :kwarg nno: float. Hyperparameter representing offset of the sigmoid from the origin.

        :kwarg nnv: float. Hyperparameter representing variability of model in x, ie. length scale.

        :returns: none.
        """

        hyps = np.zeros((3,))
        if isinstance(nna,(float,int)) and float(nna) > 0.0:
            hyps[0] = float(nna)
        else:
            raise ValueError('Neural network amplitude must be greater than 0.')
        if isinstance(nno,(float,int)):
            hyps[1] = float(nno)
        else:
            raise ValueError('Neural network offset parameter must be a real number.')
        if isinstance(nnv,(float,int)) and float(nnv) > 0.0:
            hyps[2] = float(nnv)
        else:
            raise ValueError('Neural network hyperparameter must be a real number.')
        super(NN_Kernel,self).__init__("NN",self.__calc_covm,False,hyps)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        nnamp = float(self._hyperparameters[0])
        nnop = float(self._hyperparameters[1])
        nnhp = float(self._hyperparameters[2])
        kcopy = NN_Kernel(nnamp,nnop,nnhp)
        return kcopy



class Gibbs_Kernel(_Kernel):
    """
    Gibbs Kernel: implements a Gibbs covariance function with variable length scale defined by a warping
                  function self._lfunc, which can be any function which produces only positive values
                  and has an implementation of its first derivative via the "der" argument in its call command.

    :kwarg var: float. Hyperparameter representing variability of model in y.

    :kwarg wfunc: obj. WarpingFunction object representing the variability of model in x as a function of x.
    """

    def __calc_covm(self,x1,x2,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg x1: array. Meshgrid of x_1 values to evaulate the kernel function at.

        :arg x2: array. Meshgrid of x_2 values to evaulate the kernel function at.

        :kwarg der: int. Order of x derivative to evaluate the kernel function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the kernel function at, requires explicit implementation.

        :returns: array. Kernel evaluations at (x_1,x_2) pairs, same dimensions as x1 and x2.
        """

        v_hyp = self._hyperparameters[0]
        l_hyp1 = self._wfunc(x1,0)
        l_hyp2 = self._wfunc(x2,0)
        rr = x1 - x2
        ll = np.power(l_hyp1,2.0) + np.power(l_hyp2,2.0)
        mm = l_hyp1 * l_hyp2
        lder = int((int(np.abs(der)) + 1) / 2)
        hdermax = self._wfunc.get_hyperparameters().size
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                dlh1 = self._wfunc(x1,lder,ghder)
                dlh2 = self._wfunc(x2,lder,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                covm = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
        elif der == 1:
            if hder is None:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx2 = self._wfunc(x2,lder,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx2 / (2.0 * l_hyp2) - dldx2 * dlh2 / (2.0 * np.power(l_hyp2,2.0))
                dt2 = -dlh2 * dldx2 / ll - l_hyp2 * ddldx2 / ll + l_hyp2 * dldx2 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh2 * dldx2 + 2.0 * l_hyp2 * ddldx2 - 4.0 * l_hyp2 * dldx2 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx2 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == -1:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder >= 1 and hder <= 3:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1,lder,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx1 / (2.0 * l_hyp1) - dldx1 * dlh1 / (2.0 * np.power(l_hyp1,2.0))
                dt2 = -dlh1 * dldx1 / ll - l_hyp1 * ddldx1 / ll + l_hyp1 * dldx1 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh1 * dldx1 + 2.0 * l_hyp1 * ddldx1 - 4.0 * l_hyp1 * dldx1 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx1 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == 2 or der == -2:
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder == 0:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                covm = kfac * (dt + jt + rt)
            elif hder > 0 and hder <= hdermax:
                ghder = hder - 1
                drdx1 = np.ones(rr.shape)
                dldx1 = self._wfunc(x1,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._wfunc(x2,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                dlh1 = self._wfunc(x1,0,ghder)
                dlh2 = self._wfunc(x2,0,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._wfunc(x1,lder,ghder)
                ddldx2 = self._wfunc(x2,lder,ghder)
                ddd = ddldx1 * dldx2 + dldx1 * ddldx2
                dii = drdx1 * rr * ddldx2 / l_hyp2 - drdx1 * rr * dldx2 * dlh2 / np.power(l_hyp2,2.0) + \
                      drdx2 * rr * ddldx1 / l_hyp1 - drdx2 * rr * dldx1 * dlh1 / np.power(l_hyp1,2.0)
                djj = drdx1 * rr * ddldx2 / l_hyp2 + drdx1 * rr * dldx2 * dlh2 + \
                      drdx2 * rr * ddldx1 / l_hyp1 + drdx2 * rr * dldx1 * dlh1
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                d1 = 4.0 * mm * np.power(rr / ll,4.0)
                d2 = -12.0 * mm * np.power(rr,2.0) / np.power(ll,3.0)
                d3 = 3.0 * mm / np.power(ll,2.0)
                d4 = np.power(rr,2.0) / (ll * mm)
                d5 = -1.0 / (4.0 * mm)
                dd1 = 4.0 * dmm * np.power(rr / ll,4.0) - 16.0 * mm * dll * np.power(rr,4.0) / np.power(ll,5.0)
                dd2 = -12.0 * dmm * np.power(rr,2.0) / np.power(ll,3.0) + 36.0 * mm * dll * np.power(rr,2.0) / np.power(ll,4.0)
                dd3 = 3.0 * dmm / np.power(ll,2.0) - 6.0 * mm * dll / np.power(ll,3.0)
                dd4 = -(dll / ll + dmm / mm) * np.power(rr,2.0) / (ll * mm)
                dd5 = dmm / (4.0 * np.power(mm,2.0))
                dt = dd * (d1 + d2 + d3 + d4 + d5)
                ddt = ddd * (d1 + d2 + d3 + d4 + d5) + dd * (dd1 + dd2 + dd3 + dd4 + dd5)
                jt = jj / ll * (6.0 / ll - 4.0 * np.power(rr / ll,2.0)) - ii / ll
                djt1 = 6.0 * djj / np.power(ll,2.0) - 12.0 * jj * dll / np.power(ll,3.0)
                djt2 = -4.0 * djj * np.power(rr,2.0) / np.power(ll,3.0) + 12.0 * jj * dll * np.power(rr,2.0) / np.power(ll,4.0)
                djt3 = dii / ll - ii * dll / np.power(ll,2.0)
                djt = djt1 + djt2 + djt3
                rt = 2.0 * drdx1 * drdx2 / np.power(ll,2.0) * (2.0 * np.power(rr,2.0) - ll)
                drt = -2.0 * drdx1 * drdx2 * (4.0 * np.power(rr,2.0) / np.power(ll,3.0) - 1.0 / np.power(ll,2.0))
                covm = dkfac * (dt + jt + rt) + kfac * (ddt + djt + drt)
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not implemented in '+self.get_name()+' kernel.')
        return covm


    def get_wfunc_name():
        """
        Returns the codename of the stored WarpingFunction object.

        :returns: str. WarpingFunction codename, returns UD if function is somehow not a WarpingFunction object.
        """

        wname = ""
        try:
            wname = self._wfunc.get_name()
        except AttributeError:
            wname = "UD"         # Stands for user-defined
        return wname


    def evaluate_wfunc(self,xx,der=0,hder=None):
        """
        Evaluates the stored WarpingFunction object at the specified values.

        :arg xx: array. Values to evaulate the warping function at, can be 1-D or 2-D depending on application.

        :kwarg der: int. Order of x derivative to evaluate the warping function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the warping function at, requires explicit implementation.

        :returns: array. Warping function evaluations at input values, same dimensions as xx.
        """

        lsf = self._wfunc(xx,der,hder)
        return lsf


    def __init__(self,var=1.0,wfunc=None):
        """
        Initialize the Gibbs_Kernel object.

        :kwarg var: float. Hyperparameter representing variability of model in y.

        :kwarg wfunc: obj. WarpingFunction object representing the variability of model in x as a function of x.

        :returns: none.
        """

        self._wfunc = None
        if isinstance(wfunc,_WarpingFunction):
            self._wfunc = copy.copy(wfunc)
        elif wfunc is None:
            self._wfunc = Constant_WarpingFunction(1.0e0)
        wfname = self._wfunc.get_name()
        wfhyps = self._wfunc.get_hyperparameters()
        wfcsts = self._wfunc.get_constants()

        hyps = np.zeros((1,))
        if isinstance(var,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(var)
        else:
            raise ValueError('Amplitude hyperparameter must be a real number.')
        hyps = np.hstack((hyps,wfhyps))
        csts = wfcsts.copy() if wfcsts.size > 0 else None 
        super(Gibbs_Kernel,self).__init__("Gw"+wfname,self.__calc_covm,True,hyps,csts)


    def set_hyperparameters(self,theta,log=False):
        """
        Set the hyperparameters stored in the Gibbs_Kernel and stored WarpingFunction objects.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific Kernel and WarpingFunction objects.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        super(Gibbs_Kernel,self).set_hyperparameters(theta,log)
        if self._hyperparameters.size > 1:
            self._wfunc.set_hyperparameters(self._hyperparameters[1:])


    def set_constants(self,consts):
        """
        Set the constants stored in the Gibbs_Kernel and stored WarpingFunction objects.

        :arg const: array. Constant list to be stored, ordered according to the specific Kernel and WarpingFunction objects.

        :returns: none.
        """

        super(Gibbs_Kernel,self).set_constants(consts)
        if self._constants.size > 0:
            self._wfunc.set_constants(self._constants)


    def set_bounds(self,lbounds,ubounds,log=False):
        """
        Set the hyperparameter bounds stored in the Gibbs_Kernel and stored WarpingFunction objects.

        :arg lbounds: array. Hyperparameter lower bound list to be stored, ordered according to the specific Kernel and WarpingFunction objects.

        :arg ubounds: array. Hyperparameter upper bound list to be stored, ordered according to the specific Kernel and WarpingFunction objects.

        :kwarg log: bool. Indicates that lbounds and ubounds are passed in as log10(lbounds) and log10(ubounds).

        :returns: none.
        """

        super(Gibbs_Kernel,self).set_bounds(lbounds,ubounds,log)
        if self._hyperparameters.size > 1:
            self._wfunc.set_hyperparameters(self._hyperparameters[1:])        


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._hyperparameters[0])
        wfunc = self._wfunc
        kcopy = Gibbs_Kernel(chp,wfunc)
        return kcopy



class Constant_WarpingFunction(_WarpingFunction):
    """
    Constant Warping Function for Gibbs Kernel: effectively reduces Gibbs kernel to squared-exponential kernel.
    
    :kwarg cv: float. Hyperparameter representing constant value which the warping function always evalutates to.
    """

    def __calc_warp(self,zz,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg zz: array. Values to evaulate the warping function at, can be 1-D or 2-D depending on application.

        :kwarg der: int. Order of z derivative to evaluate the warping function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the warping function at, requires explicit implementation.

        :returns: array. Warping function evaluations at input values, same dimensions as zz.
        """

        c_hyp = self._hyperparameters[0]
        warp = np.zeros(zz.shape)
        if der == 0:
            if hder is None:
                warp = c_hyp * np.ones(zz.shape)
            elif hder == 0:
                warp = np.ones(zz.shape)
        return warp


    def __init__(self,cv=1.0):
        """
        Initializes the Constant_WarpingFunction object.

        :kwarg cv: float. Hyperparameter representing constant value which warping function always evaluates to.

        :returns: none.
        """

        hyps = np.zeros((1,))
        if isinstance(cv,(float,int,np_itypes,np_utypes,np_ftypes)):
            hyps[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        super(Constant_WarpingFunction,self).__init__("C",self.__calc_warp,True,hyps)


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        chp = float(self._hyperparameters[0])
        kcopy = Constant_WarpingFunction(chp)
        return kcopy



class IG_WarpingFunction(_WarpingFunction):
    """
    Inverse Gaussian Warping Function for Gibbs Kernel: localized variation of length-scale with variation limit.

    :kwarg lb: float. Hyperparameter representing base length scale.

    :kwarg gh: float. Hyperparameter representing height of Gaussian envelope adjusting the length scale.

    :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

    :kwarg gm: float. Constant indicating location of peak of Gaussian envelope adjusting the length scale.

    :kwarg mf: float. Constant indicating lower limit for height-to-base ratio, to improve stability.
    """

    def __calc_warp(self,zz,der=0,hder=None):
        """
        Object-specific calculation function.

        :arg zz: array. Values to evaulate the warping function at, can be 1-D or 2-D depending on application.

        :kwarg der: int. Order of z derivative to evaluate the warping function at, requires explicit implementation.

        :kwarg hder: int. Order of hyperparameter derivative to evaluate the warping function at, requires explicit implementation.

        :returns: array. Warping function evaluations at input values, same dimensions as zz.
        """

        base = self._hyperparameters[0]
        amp = self._hyperparameters[1]
        sig = self._hyperparameters[2]
        mu = self._constants[0]
        maxfrac = self._constants[1]
        nn = int(np.abs(der))
        hh = amp if amp < (maxfrac * base) else maxfrac * base
        warp = np.ones(zz.shape) * base
        if hder is None:
            afac = -hh * np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig,nn - jj)
            warp = base + afac * sfac if der == 0 else afac * sfac
        elif hder == 0:
            warp = np.ones(zz.shape) if der == 0 else np.zeros(zz.shape)
        elif hder == 1:
            afac = -np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 1,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                cfac = np.power(-1.0,nn - ii) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj))
                sfac = sfac + cfac * np.power((zz - mu) / sig,nn - jj)
            warp = afac * sfac
        elif hder == 2:
            afac = -hh * np.exp(-np.power(zz - mu,2.0) / (2.0 * sig**2.0)) / np.power(sig,nn + 1)
            sfac = np.zeros(zz.shape)
            for jj in np.arange(0,nn + 3,2):
                ii = int(jj / 2)                # Note that jj = 2 * ii  ALWAYS!
                dfac = np.power(-1.0,nn - ii + 2) * np.math.factorial(nn) / (np.power(2.0,ii) * np.math.factorial(ii) * np.math.factorial(nn - jj + 2))
                lfac = dfac * ((nn + 2.0) * (nn + 1.0) - float(jj))
                sfac = sfac + lfac * np.power((zz - mu) / sig,nn - jj + 2)
            warp = afac * sfac
        return warp


    def __init__(self,lb=1.0,gh=0.5,gs=1.0,gm=0.0,mf=0.6):
        """
        Initializes the IG_WarpingFunction object.

        :kwarg lb: float. Hyperparameter representing base length scale.

        :kwarg gh: float. Hyperparameter representing minimal length scale.

        :kwarg gs: float. Hyperparameter indicating width of Gaussian envelope adjusting the length scale.

        :kwarg gm: float. Constant indicating location of minimal length scale.

        :kwarg mf: float. Constant indicating lower limit for ratio of minimal-to-base length scale, to improve stability.

        :returns: none.
        """

        hyps = np.zeros((3,))
        csts = np.zeros((2,))
        if isinstance(lb,(float,int,np_itypes,np_utypes,np_ftypes)) and float(lb) > 0.0:
            hyps[0] = float(lb)
        else:
            raise ValueError('Length scale function base hyperparameter must be greater than 0.')
        if isinstance(gh,(float,int,np_itypes,np_utypes,np_ftypes)) and float(gh) > 0.0:
            hyps[1] = float(gh)
        else:
            raise ValueError('Length scale function minimum hyperparameter must be greater than 0.')
        if isinstance(gs,(float,int,np_itypes,np_utypes,np_ftypes)) and float(gs) > 0.0:
            hyps[2] = float(gs)
        else:
            raise ValueError('Length scale function sigma hyperparameter must be greater than 0.')
        if isinstance(gm,(float,int,np_itypes,np_utypes,np_ftypes)):
            csts[0] = float(gm)
        else:
            raise ValueError('Length scale function mu constant must be a real number.')
        if isinstance(mf,(float,int,np_itypes,np_utypes,np_ftypes)) and float(mf) < 1.0:
            csts[1] = float(mf)
        else:
            raise ValueError('Length scale function minimum-to-base ratio limit must be less than 1.')
        if hyps[1] > (csts[1] * hyps[0]):
            hyps[1] = float(csts[1] * hyps[0])
        super(IG_WarpingFunction,self).__init__("IG",self.__calc_warp,True,hyps,csts)


    def set_hyperparameters(self,theta,log=False):
        """
        Set the hyperparameters stored in the WarpingFunction object. Specific implementation due to maximum fraction limit.

        :arg theta: array. Hyperparameter list to be stored, ordered according to the specific WarpingFunction object.

        :kwarg log: bool. Indicates that theta is passed in as log10(theta).

        :returns: none.
        """

        super(IG_WarpingFunction,self).set_hyperparameters(theta,log)
        base = self._hyperparameters[0]
        amp = self._hyperparameters[1]
        maxfrac = self._constants[1]
        if amp > (maxfrac * base):
            self._hyperparameters[1] = maxfrac * base


    def set_constants(self,consts):
        """
        Set the constants stored in the WarpingFunction object. Specific implementation due to maximum fraction limit.

        :arg consts: array. Constant list to be stored, ordered according to the specific WarpingFunction object.

        :returns: none.
        """

        super(IG_WarpingFunction,self).set_constants(consts)
        base = self._hyperparameters[0]
        amp = self._hyperparameters[1]
        maxfrac = self._constants[1]
        if amp > (maxfrac * base):
            self._hyperparameters[1] = maxfrac * base


    def __copy__(self):
        """
        Object-specific copy function, needed for robust hyperparameter optimization routine.

        :returns: obj. An exact duplicate of the current object, which can be modified without affecting the original.
        """

        lbhp = float(self._hyperparameters[0])
        ghhp = float(self._hyperparameters[1])
        gshp = float(self._hyperparameters[2])
        gmc = float(self._constants[0])
        lrc = float(self._constants[1])
        kcopy = IG_WarpingFunction(lbhp,ghhp,gshp,gmc,lrc)
        return kcopy



class GaussianProcessRegression1D(object):
    """
    Class containing variable containers, get/set functions, and fitting functions required to perform a 1-dimensional GPR fit.
    .. note:: This implementation requires the specific implementation of the Kernel class, provided in the same file!
    """

    def __init__(self):
        """
        Defines the input and output containers used within the class, still requires instantiation.
        """

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
        self._xF = None
        self._barF = None
        self._varF = None
        self._dbarF = None
        self._dvarF = None
        self._lml = None
        self._eflag = False
        self._barE = None
        self._varE = None
        self._dbarE = None
        self._dvarE = None
        self._varN = None
        self._dvarN = None
        self._nye = None
        self._fwarn = False
        self._opopts = ['grad','mom','nag','adagrad','adadelta','adam','adamax','nadam']


    def set_kernel(self,kernel=None,kbounds=None,regpar=None):
        """
        Specify the kernel that the Gaussian process regression will be performed with.

        :kwarg kernel: Kernel object. The covariance function to be used in fitting the data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being [lower,upper] bounds.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :returns: none.
        """

        if isinstance(kernel,_Kernel):
            self._kk = copy.copy(kernel)
            self._ikk = copy.copy(self._kk)
        if isinstance(self._kk,_Kernel):
            kh = self._kk.get_hyperparameters(log=True)
            if isinstance(kbounds,(list,tuple,np.ndarray)):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb.T) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._kb = kb
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            self._lp = float(regpar)


    def set_raw_data(self,xdata=None,ydata=None,xerr=None,yerr=None,dxdata=None,dydata=None,dyerr=None):
        """
        Specify the raw data that the Gaussian process regression will be performed on.
        Performs some consistency checks between the input raw data to ensure validity.

        :kwarg xdata: array. x-values of data points to be fitted.

        :kwarg ydata: array. y-values of data points to be fitted.

        :kwarg xerr: array. x-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg yerr: array. y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. dy/dx-errors of derivative data points to be included in fit. (optional)

        :returns: none.
        """

        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            self._xx = np.array(xdata).flatten()
            self._eflag = False
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            self._xx = xdata.flatten()
            self._eflag = False
        if isinstance(xerr,(list,tuple)) and len(xerr) > 0:
            self._xe = np.array(xerr).flatten()
        elif isinstance(xerr,np.ndarray) and xerr.size > 0:
            self._xe = xerr.flatten()
        elif isinstance(xerr,str):
            self._xe = None
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            self._yy = np.array(ydata).flatten()
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            self._yy = ydata.flatten()
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            self._ye = np.array(yerr).flatten()
            self._eflag = False
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            self._ye = yerr.flatten()
            self._eflag = False
        elif isinstance(yerr,str):
            self._ye = None
            self._eflag = False
        if isinstance(dxdata,(list,tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dxx = temp.flatten()
        elif isinstance(dxdata,np.ndarray) and dxdata.size > 0:
            self._dxx = dxdata.flatten()
        elif isinstance(dxdata,str):
            self._dxx = None
        if isinstance(dydata,(list,tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dyy = temp.flatten()
        elif isinstance(dydata,np.ndarray) and dydata.size > 0:
            self._dyy = dydata.flatten()
        elif isinstance(dydata,str):
            self._dyy = None
        if isinstance(dyerr,(list,tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self._dye = temp.flatten()
        elif isinstance(dyerr,np.ndarray) and dyerr.size > 0:
            self._dye = dyerr.flatten()
        elif isinstance(dyerr,str):
            self._dye = None


    def set_conditioner(self,condnum=None,lbound=None,ubound=None):
        """
        Specify the parameters to ensure the condition number of the matrix is good,
        as well as set upper and lower bounds for the input data to be included.

        :kwarg condnum: float. Minimum allowable delta-x for input data before applying Gaussian blending.

        :kwarg lbound: float. Minimum allowable y-value for input data, values below are omitted from fit procedure.

        :kwarg ubound: float. Maximum allowable y-value for input data, values above are omitted from fit procedure.

        :returns: none.
        """

        if isinstance(condnum,(float,int,np_itypes,np_utypes,np_ftypes)) and float(condnum) > 0.0:
            self._cn = float(condnum)
        elif isinstance(condnum,(float,int,np_itypes,np_utypes,np_ftypes)) and float(condnum) <= 0.0:
            self._cn = None
        elif isinstance(condnum,str):
            self._cn = None
        if isinstance(lbound,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._lb = float(lbound)
        elif isinstance(lbound,str):
            self._lb = None
        if isinstance(ubound,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._ub = float(ubound)
        elif isinstance(ubound,str):
            self._ub = None


    def set_error_kernel(self,kernel=None,kbounds=None,regpar=None,nrestarts=None):
        """
        Specify the kernel that the Gaussian process regression on the error function 
        will be performed with.

        :kwarg kernel: Kernel object. The covariance function to be used in fitting the error data with Gaussian process regression.

        :kwarg kbounds: array. 2D array with rows being hyperparameters and columns being [lower,upper] bounds.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter vlaues within kbounds argument.

        :returns: none.
        """

        if isinstance(kernel,_Kernel):
            self._ekk = copy.copy(kernel)
            self._eflag = False
        if isinstance(self._ekk,_Kernel):
            kh = self._ekk.get_hyperparameters(log=True)
            if isinstance(kbounds,(list,tuple,np.ndarray)):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb.T) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self._ekb = kb
                self._eflag = False
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            self._elp = float(regpar)
            self._eflag = False
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)):
            self._enr = int(nrestarts) if int(nrestarts) > 0 else 0


    def set_search_parameters(self,epsilon=None,method=None,spars=None,sdiff=None):
        """
        Specify the search parameters that the Gaussian process regression will use.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method.

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.

        :returns: none.
        """

        midx = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            self._eps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            self._eps = None
        elif isinstance(epsilon,str):
            self._eps = None
        if isinstance(method,str):
            mstr = method.lower()
            for mm in np.arange(0,len(self._opopts)):
                if re.match(mstr,self._opopts[mm],flags=re.IGNORECASE):
                    midx = mm
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                self._opm = self._opopts[1]
                opp = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 2:
                self._opm = self._opopts[2]
                opp = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 3:
                self._opm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 4:
                self._opm = self._opopts[4]
                opp = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 5:
                self._opm = self._opopts[5]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 6:
                self._opm = self._opopts[6]
                opp = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            elif midx == 7:
                self._opm = self._opopts[7]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
            else:
                self._opm = self._opopts[0]
                opp = np.array([1.0e-5]).flatten()
                for ii in np.arange(0,self._opp.size):
                    if ii < opp.size:
                        opp[ii] = self._opp[ii]
                self._opp = opp.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < self._opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._opp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < self._opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._opp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            self._dh = float(sdiff)


    def set_error_search_parameters(self,epsilon=None,method=None,spars=None,sdiff=None):
        """
        Specify the search parameters that the Gaussian process regression will use for the error function.
        Performs some consistency checks on input values to ensure validity.

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg method: str or int. Hyperparameter optimization algorithm selection.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method.

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.

        :returns: none.
        """

        emidx = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            self._eeps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            self._eeps = None
        elif isinstance(epsilon,str):
            self._eeps = None
        if isinstance(method,str):
            mstr = method.lower()
            for mm in np.arange(0,len(self._opopts)):
                if re.match(mstr,self._opopts[mm],flags=re.IGNORECASE):
                    emidx = mm
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) < len(self._opopts):
            emidx = int(method)
        if emidx is not None:
            if emidx == 1:
                self._eopm = self._opopts[1]
                opp = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 2:
                self._eopm = self._opopts[2]
                opp = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 3:
                self._eopm = self._opopts[3]
                opp = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 4:
                self._eopm = self._opopts[4]
                opp = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 5:
                self._eopm = self._opopts[5]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 6:
                self._eopm = self._opopts[6]
                opp = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            elif emidx == 7:
                self._eopm = self._opopts[7]
                opp = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
            else:
                self._eopm = self._opopts[0]
                opp = np.array([1.0e-5]).flatten()
                for ii in np.arange(0,self._eopp.size):
                    if ii < opp.size:
                        opp[ii] = self._eopp[ii]
                self._eopp = opp.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < self._eopp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._eopp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < self._eopp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    self._eopp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            self._edh = float(sdiff)


    def set_warning_flag(self,flag=True):
        """
        Specify the printing of runtime warnings within the
        hyperparameter optimization routine. The warnings are
        disabled by default.

        :kwarg flag: bool. Flag to toggle display of warnings.

        :returns: none.
        """

        self._fwarn = True if flag else False


    def get_raw_data(self,conditioned=False):
        """
        Returns the input raw data passed in latest set_raw_data() call,
        with or without omissions / filters / conditioning.
        If set_conditioner() not called, uses default settings for conditioner.

        :kwarg conditioned: bool. Perform data conditioning before returning values to see data actually used by fitting routine.

        :returns: (array, array, array, array, array, array, array).
            Vectors in order of x-values, y-values, x-errors, y-errors, derivative x-values, dy/dx-values, dy/dx-errors.
        """

        cxx = self._xx
        cyy = self._yy
        cxe = self._xe
        cye = self._ye
        dxx = self._dxx
        dyy = self._dyy
        dye = self._dye
        if conditioned:
            lb = -1.0e50 if self._lb is None else self._lb
            ub = 1.0e50 if self._ub is None else self._ub
            cn = 5.0e-3 if self._cn is None else self._cn
            (cxx,cyy,cxe,cye,nn) = self._condition_data(cxx,cyy,cxe,cye,lb,ub,cn)
        return (cxx,cyy,cxe,cye,dxx,dyy,dye)


    def get_gp_x(self):
        """
        Returns the x-values used in the latest GPRFit() call.

        :returns: array. Vector of x-values corresponding to predicted y-values.
        """

        return self._xF


    def get_gp_mean(self):
        """
        Returns the y-values computed in the latest GPRFit() call.

        :returns: array. Predicted y-values from fit.
        """

        return self._barF


    def get_gp_variance(self,noise_flag=True):
        """
        Returns the full covariance matrix of the y-values computed in the latest
        GPRFit() call.

        :param noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements.

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        """

        varF = self._varF
        if varF is not None and self._varN is not None and noise_flag:
            varF = varF + self._varN
        return varF


    def get_gp_std(self,noise_flag=True):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest GPRFit() call, corresponds to 1 sigma error of fit.

        :param noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors.

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        """

        sigF = None
        varF = self.get_gp_variance(noise_flag=noise_flag)
        if varF is not None:
            sigF = np.sqrt(np.diag(varF))
        return sigF


    def get_gp_drv_mean(self):
        """
        Returns the dy/dx-values computed in the latest GPRFit() call.

        :returns: array. Predicted dy/dx-values from fit, if requested in fit call.
        """

        return self._dbarF


    def get_gp_drv_variance(self,noise_flag=True):
        """
        Returns the full covariance matrix of the dy/dx-values computed in the latest
        GPRFit() call.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements.

        :returns: array. 2D meshgrid array containing full covariance matrix for predicted dy/dx-values from fit.
        """

        dvarF = self._dvarF
        dvarmod = self.get_gp_variance(noise_flag=noise_flag) / self.get_gp_variance(noise_flag=False)
        if dvarF is not None and self._dvarN is not None and noise_flag:
            dvarF = dvarF + self._dvarN
        dvarF = dvarF * dvarmod
        return dvarF


    def get_gp_drv_std(self,noise_flag=True):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the 
        dy/dx-values computed in the latest GPRFit() call, corresponds to 1 sigma
        error of fit.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned 1 sigma errors.

        :returns: array. 1D array containing 1 sigma errors of predicted dy/dx-values from fit.
        """

        dsigF = None
        dvarF = self.get_gp_drv_variance(noise_flag=noise_flag)
        if dvarF is not None:
            dsigF = np.sqrt(np.diag(dvarF))
        return dsigF


    def get_gp_results(self,rtn_cov=False,noise_flag=True):
        """
        Returns all common predicted values computed in the latest GPRFit() call.

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead the 1 sigma errors.

        :kwarg noise_flag: bool. Specifies inclusion of noise term in returned variances or errors.

        :returns: (array, array, array, array).
            Vectors in order of y-values, y-errors, dy/dx-values, dy/dx-errors.
        """

        ra = self.get_gp_mean()
        rb = self.get_gp_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_std(noise_flag=noise_flag)
        rc = self.get_gp_drv_mean()
        rd = self.get_gp_drv_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_drv_std(noise_flag=noise_flag)
        return (ra,rb,rc,rd)


    def get_gp_lml(self):
        """
        Returns the log-marginal-likelihood of the latest GPRFit() call.

        :returns: float. Log-marginal-likelihood value from fit.
        """

        return self._lml


    def get_gp_input_kernel(self):
        """
        Returns the original input kernel, with settings retained from before the 
        hyperparameter optimization step.

        :returns: obj. The original input Kernel object from latest set_kernel() call.
        """

        return self._ikk


    def get_gp_kernel(self):
        """
        Returns the optimized kernel determined in the latest GPRFit() call.

        :returns: obj. The Kernel object from the latest GPRFit() call, including optimized hyperparameters if performed.
        """

        return self._kk


    def get_gp_kernel_details(self):
        """
        Returns the data needed to save the optimized kernel determined in the latest GPRFit() call.

        :returns: (str, array, float).
            The Kernel object codename, vector of kernel hyperparameters and constants, regularization parameter.
        """

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._kk,_Kernel):
            kname = self._kk.get_name()
            kpars = np.hstack((self._kk.get_hyperparameters(log=False),self._kk.get_constants()))
            krpar = self._lp
        return (kname,kpars,krpar)


    def get_error_kernel(self):
        """
        Returns the optimized error kernel determined in the latest GPRFit() call.

        :returns: obj. The error Kernel object from the latest GPRFit() call, including optimized hyperparameters if performed.
        """

        return self._ekk


    def get_gp_error_kernel_details(self):
        """
        Returns the data needed to save the optimized error kernel determined in the latest GPRFit() call.

        :returns: (str, array).
            The error Kernel object codename, vector of kernel hyperparameters and constants, regularization parameter.
        """

        kname = None
        kpars = None
        krpar = None
        if isinstance(self._ekk,_Kernel):
            kname = self._ekk.get_name()
            kpars = np.hstack((self._ekk.get_hyperparameters(log=False),self._ekk.get_constants()))
            krpar = self._elp
        return (kname,kpars,krpar)


    def get_error_gp_mean(self):
        """
        Returns the y-errors computed in the latest GPRFit() call.

        :returns: array. Predicted y-values from fit.
        """

        return self._barE


    def get_error_gp_variance(self):
        """
        Returns the full covariance matrix of the y-errors computed in the latest
        GPRFit() call.

        :returns: array. 2D meshgrid array containing full covariance matrix of predicted y-values from fit.
        """

        return self._varE


    def get_error_gp_std(self):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest GPRFit() call, corresponds to 1 sigma error of fit.

        :returns: array. 1D array containing 1 sigma errors of predicted y-values from fit.
        """

        sigE = None
        varE = self.get_error_gp_variance()
        if varE is not None:
            sigE = np.sqrt(np.diag(varE))
        return sigE


    def get_error_function(self,xnew):
        """
        Returns the error values used in heteroscedastic GPR, evaluated at the input x-values,
        using the error kernel determined in the latest GPRFit() call.

        :arg xnew: array. Vector of x-values at which the predicted error function should be evaluated at.

        :returns: array. Predicted y-errors from the fit using the error kernel.
        """

        xn = None
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        ye = self._ye if self._nye is None else self._nye
        barE = None
        if xn is not None and ye is not None and self._eflag:
            barE = itemgetter(0)(self.__basic_fit(xn,kernel=self._ekk,ydata=ye,yerr=0.1*ye,epsilon='None'))
        return barE


    def __gp_base_alg(self,xn,kk,lp,xx,yy,ye,dxx,dyy,dye,dd):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Bare-bones algorithm for gaussian process regression, no idiot-proofing, no pre- or post-processing

        Note that it is recommended that kk be a Kernel object as specified in this file
        but it can be, in essence, any passed object which can be called with arguments:
            (x1,x2,derivative_order) with (x1,x2) being a meshgrid
        and returns an array with shape of x1 and/or x2 (these must have identical shape).

        :arg xn: array. Vector of x-values which the fit will be evaluated at.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dd: int. Derivative order of output prediction.

        :returns: (array, array, float).
            Vector of predicted mean values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx,xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy,yyd)
        yed = dye if dflag else []
        yef = np.append(ye,yed)
        (x1,x2) = np.meshgrid(xx,xx)
        (x1h1,x2h1) = np.meshgrid(xx,xxd)
        (x1h2,x2h2) = np.meshgrid(xxd,xx)
        (x1d,x2d) = np.meshgrid(xxd,xxd)
        (xs1,xs2) = np.meshgrid(xn,xx)
        (xs1h,xs2h) = np.meshgrid(xn,xxd)
        (xt1,xt2) = np.meshgrid(xn,xn)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1,x2,der=0)
        KKh1 = kk(x1h1,x2h1,der=1)
        KKh2 = kk(x1h2,x2h2,der=-1)
        KKd = kk(x1d,x2d,der=2)
        KK = np.vstack((np.hstack((KKb,KKh2)),np.hstack((KKh1,KKd))))
        LL = spla.cholesky(KK + np.diag(yef**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yf)
        ksb = kk(xs1,xs2,der=-dd) if dd == 1 else kk(xs1,xs2,der=dd)
        ksh = kk(xs1h,xs2h,der=dd+1)
        ks = np.vstack((ksb,ksh))
        vv = np.dot(LL.T,spla.cho_solve((LL,True),ks))
        kt = kk(xt1,xt2,der=2*dd)
        barF = np.dot(ks.T,alpha)          # Mean function
        varF = kt - np.dot(vv.T,vv)        # Variance of mean function

        # Log-marginal-likelihood provides an indication of how statistically well the fit describes the training data
        #    1st term: Describes the goodness of fit for the given data
        #    2nd term: Penalty for complexity / simplicity of the covariance function
        #    3rd term: Penalty for the size of given data set
        lml = -0.5 * np.dot(yf.T,alpha) - lp * np.sum(np.log(np.diag(LL))) - 0.5 * xf.size * np.log(2.0 * np.pi)

        return (barF,varF,lml)


    def __gp_brute_deriv1(self,xn,kk,lp,xx,yy,ye):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Bare-bones algorithm for brute-force first-order derivative of Gaussian process regression (single input dimension).
        Not recommended for production runs, but useful for testing custom Kernel objects which have hard-coded derivative calculations.

        :arg xn: array. Vector of x-values which the fit will be evaluated at.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :returns: (array, array, float).
            Vector of predicted mean derivative values, matrix of predicted variances and covariances,
            log-marginal-likelihood of prediction including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        (x1,x2) = np.meshgrid(xx,xx)
        (xs1,xs2) = np.meshgrid(xx,xn)
        (xt1,xt2) = np.meshgrid(xn,xn)
        # Set up predictive grids with slight offset in x1 and x2, forms corners of a box around original xn point
        step = np.amin(np.abs(np.diff(xn)))
        xnl = xn - step * 0.5e-3        # The step is chosen intelligently to be smaller than smallest dxn
        xnu = xn + step * 0.5e-3
        (xl1,xl2) = np.meshgrid(xx,xnl)
        (xu1,xu2) = np.meshgrid(xx,xnu)
        (xll1,xll2) = np.meshgrid(xnl,xnl)
        (xlu1,xlu2) = np.meshgrid(xnu,xnl)
        (xuu1,xuu2) = np.meshgrid(xnu,xnu)

        KK = kk(x1,x2)
        LL = spla.cholesky(KK + np.diag(ye**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yy)
        # Approximation of first derivative of covf (df/dxn1)
        ksl = kk(xl1,xl2)
        ksu = kk(xu1,xu2)
        dks = (ksu.T - ksl.T) / (step * 1.0e-3)
        dvv = np.dot(LL.T,spla.cho_solve((LL,True),dks))
        # Approximation of second derivative of covf (d^2f/dxn1 dxn2)
        ktll = kk(xll1,xll2)
        ktlu = kk(xlu1,xlu2)
        ktul = ktlu.T
        ktuu = kk(xuu1,xuu2)
        dktl = (ktlu - ktll) / (step * 1.0e-3)
        dktu = (ktuu - ktul) / (step * 1.0e-3)
        ddkt = (dktu - dktl) / (step * 1.0e-3)
        barF = np.dot(dks.T,alpha)          # Mean function
        varF = ddkt - np.dot(dvv.T,dvv)     # Variance of mean function
        lml = -0.5 * np.dot(yy.T,alpha) - lp * np.sum(np.log(np.diag(LL))) - 0.5 * xx.size * np.log(2.0 * np.pi)

        return (barF,varF,lml)


    def __gp_brute_grad_lml(self,kk,lp,xx,yy,ye,dxx,dyy,dye,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Bare-bones algorithm for brute-force computation of gradient of log-marginal-likelihood with respect to the
        hyperparameters in logarithmic space.
        Result must be divided by ln(10) * theta in order to have the gradient with respect to the hyperparameters
        in linear space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dh: float. Step size in hyperparameter space used in derivative approximation.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        """

        xn = np.array([0.0])
        theta = kk.get_hyperparameters(log=True)
        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0,theta.size):
            testkk = copy.copy(kk)
            theta_in = theta.copy()
            theta_in[ii] = theta[ii] - 0.5 * dh
            testkk.set_hyperparameters(theta_in,log=True)
            llml = itemgetter(2)(self.__gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            theta_in[ii] = theta[ii] + 0.5 * dh
            testkk.set_hyperparameters(theta_in,log=True)
            ulml = itemgetter(2)(self.__gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            gradlml[ii] = (ulml - llml) / dh

        return gradlml


    def __gp_grad_lml(self,kk,lp,xx,yy,ye,dxx,dyy,dye):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Bare-bones algorithm for computation of gradient of log-marginal-likelihood with respect to the hyperparameters
        in linear space.
        Result must be multiplied by ln(10) * theta in order to have the gradient with respect to the hyperparameters
        in logarithmic space.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :returns: array. Vector of log-marginal-likelihood derivatives with respect to the hyperparameters including the regularization component.
        """

        # Set up the problem grids for calculating the required matrices from covf
        theta = kk.get_hyperparameters()
        dflag = True if dxx is not None and dyy is not None and dye is not None else False
        xxd = dxx if dflag else []
        xf = np.append(xx,xxd)
        yyd = dyy if dflag else []
        yf = np.append(yy,yyd)
        yed = dye if dflag else []
        yef = np.append(ye,yed)
        (x1,x2) = np.meshgrid(xx,xx)
        (x1h1,x2h1) = np.meshgrid(xx,xxd)
        (x1h2,x2h2) = np.meshgrid(xxd,xx)
        (x1d,x2d) = np.meshgrid(xxd,xxd)

        # Algorithm, see theory (located in book specified at top of file) for details
        KKb = kk(x1,x2,der=0)
        KKh1 = kk(x1h1,x2h1,der=1)
        KKh2 = kk(x1h2,x2h2,der=-1)
        KKd = kk(x1d,x2d,der=2)
        KK = np.vstack((np.hstack((KKb,KKh2)),np.hstack((KKh1,KKd))))
        LL = spla.cholesky(KK + np.diag(yef**2.0),lower=True)
        alpha = spla.cho_solve((LL,True),yf)
        gradlml = np.zeros(theta.shape).flatten()
        for ii in np.arange(0,theta.size):
            HHb = kk(x1,x2,der=0,hder=ii)
            HHh1 = kk(x1h1,x2h1,der=1,hder=ii)
            HHh2 = kk(x1h2,x2h2,der=-1,hder=ii)
            HHd = kk(x1d,x2d,der=2,hder=ii)
            HH = np.vstack((np.hstack((HHb,HHh2)),np.hstack((HHh1,HHd))))
            PP = np.dot(alpha.T,HH)
            QQ = spla.cho_solve((LL,True),HH)
            gradlml[ii] = 0.5 * np.dot(PP,alpha) - 0.5 * lp * np.sum(np.diag(QQ))

        return gradlml


    def __gp_grad_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Gradient ascent hyperparameter optimization algorithm, searches hyperparameters in log-space.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
#                gradlml = 1.0e1 * gradlml
            theta_step = eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on gradient ascent search.')
        return (newkk,lmlnew)


    def __gp_momentum_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Gradient ascent hyperparameter optimization algorithm with momentum, searches hyperparameters in log-space
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :arg gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on momentum gradient ascent search.')
        return (newkk,lmlnew)


    def __gp_nesterov_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Nesterov-accelerated gradient ascent hyperparameter optimization algorithm with momentum, searches hyperparameters in log-space.
           Effectively makes prediction of the next step and uses that with back-correction factor as the current update.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :param eps: float. Desired convergence criteria.

        :param eta: float. Gain factor on gradient to define next step, recommended 1.0e-5.

        :param gam: float. Momentum factor multiplying previous step, recommended 0.9.

        :param dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        if newkk.is_hderiv_implemented():
            # Hyperparameter derivatives computed in linear space
            gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
            gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_base)
        else:
            gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
        theta_step = eta * gradlml
        theta_old = theta_base.copy()
        theta_new = theta_old + theta_step
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            newkk.set_hyperparameters(theta_new,log=True)
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            theta_step = gam * theta_step + eta * gradlml
            theta_new = theta_old + theta_step   # Only called ascent since step is added here, not subtracted
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on Nesterov-accelerated gradient ascent search.')
        return (newkk,lmlnew)


    def __gp_adagrad_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Adaptive gradient ascent hyperparameter optimization algorithm, searches hyperparameters in log-space.
           Suffers from extremely aggressive step modification due to continuous accumulation of denominator term, recommended to use AdaDelta.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-2.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        gold = np.zeros(theta_base.shape)
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
#                gradlml = -gradlml
            gnew = gold + np.power(gradlml,2.0)
            theta_step = eta * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on adaptive gradient ascent search.')
        return (newkk,lmlnew)


    def __gp_adadelta_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,gam,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Adaptive gradient ascent hyperparameter optimization algorithm with decaying accumulation window, searches hyperparameters in log-space.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Initial guess for gain factor on gradient to define next step, recommended 1.0e-2.

        :arg gam: float. Forgetting factor on accumulated gradient term, recommended 0.9.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        etatemp = np.ones(theta_base.shape) * eta
        told = theta_step.copy()
        gold = np.zeros(theta_base.shape)
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
#                gradlml = -gradlml
            gnew = gam * gold + (1.0 - gam) * np.power(gradlml,2.0)
            theta_step = etatemp * gradlml / np.sqrt(gnew + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            tnew = gam * told + (1.0 - gam) * np.power(theta_step,2.0)
            etatemp = np.sqrt(tnew + 1.0e-8)
            told = tnew
            gold = gnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on decaying adaptive gradient ascent search.')
        return (newkk,lmlnew)


    def __gp_adam_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Adaptive moment estimation hyperparameter optimization algorithm, searches hyperparameters in log-space.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1))) / (np.sqrt(vnew / (1.0 - b2**(icount + 1))) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on adaptive moment estimation search.')
        return (newkk,lmlnew)


    def __gp_adamax_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Adaptive moment estimation hyperparameter optimization algorithm with l-infinity, searches hyperparameters in log-space.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 2.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to approximate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            unew = b2 * vnew if vold is None else np.nanmax([b2 * vold,np.abs(gradlml)],axis=0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1))) / unew
            theta_new = theta_old + theta_step
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on adaptive moment estimation search.')
        return (newkk,lmlnew)


    def __gp_nadam_optimizer(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,eta,b1,b2,dh):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Nesterov-accelerated adaptive moment estimation hyperparameter optimization algorithm, searches hyperparameters in log-space.
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria.
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad.

        :arg kk: callable. Any object which can be called with 2 arguments and optional derivative order argument, returning the covariance matrix.

        :arg lp: float. Regularization parameter, larger values effectively enforce smoother / flatter fits.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg dxx: array. Vector of x-values of derivative data to be included in fit. Set to an empty list to specify no data.

        :arg dyy: array. Vector of dy-values of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg dye: array. Vector of dy-errors of derivative data to be included in fit. Must have same dimensions as dxx.

        :arg eps: float. Desired convergence criteria.

        :arg eta: float. Gain factor on gradient to define next step, recommended 1.0e-3.

        :arg b1: float. Forgetting factor on gradient term, recommended 0.9.

        :arg b2: float. Forgetting factor on second moment of gradient term, recommended 0.999.

        :arg dh: float. Step size used to calculate the gradient (only applicable if brute-force derivative is used), recommended 1.0e-2.

        :returns: (obj, float).
            Final Kernel object resulting from hyperparameter optimization of LML, final log-marginal-likelihood including the regularization component.
        """

        # Set up required data for performing the search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        theta_step = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = eps + 1.0
        mold = None
        vold = None
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
                # Hyperparameter derivatives computed in linear space
                gradlml_lin = self.__gp_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye)
                gradlml = gradlml_lin * np.log(10.0) * np.power(10.0,theta_old)
            else:
                gradlml = self.__gp_brute_grad_lml(newkk,lp,xx,yy,ye,dxx,dyy,dye,dh)
            mnew = gradlml if mold is None else b1 * mold + (1.0 - b1) * gradlml
            vnew = np.power(gradlml,2.0) if vold is None else b2 * vold + (1.0 - b2) * np.power(gradlml,2.0)
            theta_step = eta * (mnew / (1.0 - b1**(icount + 1)) + (1.0 - b1) * gradlml / (1.0 - b1**(icount + 1))) / (np.sqrt(vnew / (1.0 - b2)) + 1.0e-8)
            theta_new = theta_old + theta_step
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            mold = mnew
            vold = vnew
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on adaptive moment estimation search.')
        return (newkk,lmlnew)


    def __condition_data(self,xx,xe,yy,ye,lb,ub,cn):
        """
        INTERNAL FUNCTION - USE MAIN CALL FUNCTIONS

        Conditions the input data to remove data points which are too close together, as
        defined by the user, and data points that are outside user-defined bounds.

        :arg xx: array. Vector of x-values of data to be fitted.

        :arg xe: array. Vector of x-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg yy: array. Vector of y-values of data to be fitted. Must have same dimensions as xx.

        :arg ye: array. Vector of y-errors of data to be fitted, assumed to be given as 1 sigma. Must have same dimensions as xx.

        :arg lb: float. Minimum allowable y-value for input data, values below are omitted from fit procedure.

        :arg ub: float. Maximum allowable y-value for input data, values above are omitted from fit procedure.

        :arg cn: float. Minimum allowable delta-x for input data before applying Gaussian blending.

        :returns: (array, array, array, array, array).
            Vectors in order of conditioned x-values, conditioned x-errors, conditioned y-values, conditioned y-errors,
            number of data points blended into corresponding index.
        """

        good = np.all([np.invert(np.isnan(xx)),np.invert(np.isnan(yy)),np.isfinite(xx),np.isfinite(yy)],axis=0)
        xe = xe[good] if xe.size == xx.size else np.full(xx[good].shape,xe[0])
        ye = ye[good] if ye.size == yy.size else np.full(yy[good].shape,ye[0])
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
        for ii in np.arange(0,xx.size):
            if yy[ii] >= lb and yy[ii] <= ub:
                fflag = False
                for jj in np.arange(0,cxx.size):
                    if np.abs(cxx[jj] - xx[ii]) < cn and not fflag:
                        cxe[jj] = np.sqrt((cxe[jj]**2.0 * nn[jj] + xe[ii]**2.0 + cxx[jj]**2.0 * nn[jj] + xx[ii]**2.0) / (nn[jj] + 1.0) - ((cxx[jj] * nn[jj] + xx[ii]) / (nn[jj] + 1.0))**2.0)
                        cxx[jj] = (cxx[jj] * nn[jj] + xx[ii]) / (nn[jj] + 1.0)
                        cye[jj] = np.sqrt((cye[jj]**2.0 * nn[jj] + ye[ii]**2.0 + cyy[jj]**2.0 * nn[jj] + yy[ii]**2.0) / (nn[jj] + 1.0) - ((cyy[jj] * nn[jj] + yy[ii]) / (nn[jj] + 1.0))**2.0)
                        cyy[jj] = (cyy[jj] * nn[jj] + yy[ii]) / (nn[jj] + 1.0)
                        nn[jj] = nn[jj] + 1.0
                        fflag = True
                if not fflag:
                    nn = np.hstack((nn,1.0))
                    cxx = np.hstack((cxx,xx[ii]))
                    cxe = np.hstack((cxe,xe[ii]))
                    cyy = np.hstack((cyy,yy[ii]))
                    cye = np.hstack((cye,ye[ii]))
        cxx = cxx * xsc
        cxe = cxe * xsc
        cyy = cyy * ysc
        cye = cye * ysc
        return (cxx,cxe,cyy,cye,nn)


    def __basic_fit(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,dxdata=None,dydata=None,dyerr=None,epsilon=None,method=None,spars=None,sdiff=None,do_drv=False,rtn_cov=False):
        """
        RESTRICTED ACCESS FUNCTION - Can be called externally for testing if user is familiar with algorithm.

        Basic GP regression fitting routine, RECOMMENDED to call this instead of the bare-bones functions
        as this applies additional input checking. Note that this function does NOT strictly use class data!!!

        :arg xnew: array. x-values at which the predicted fit will be evaluated at.

        :kwarg kernel: Kernel object. The covariance function to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. x-values of data points to be fitted.

        :kwarg ydata: array. y-values of data points to be fitted.

        :kwarg yerr: array. y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg dxdata: array. x-values of derivative data points to be included in fit. (optional)

        :kwarg dydata: array. dy/dx-values of derivative data points to be included in fit. (optional)

        :kwarg dyerr: array. dy/dx-errors of derivative data points to be included in fit. (optional)

        :kwarg epsilon: float. Convergence criteria for optimization algorithm, set negative to disable.

        :kwarg spars: array. Parameters for hyperparameter optimization algorithm, defaults depend on chosen method.

        :kwarg sdiff: float. Step size for hyperparameter derivative approximations in optimization algorithms, default is 1.0e-2.

        :kwarg do_drv: bool. Set as true to predict the derivative of the fit instead of the fit.

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors.

        :returns: (array, array, float, obj).
            Vector of predicted mean values, vector or matrix of predicted errors, log-marginal-likelihood of fit
            including the regularization component, final Kernel object with optimized hyperparameters if performed.
        """

        xn = None
        kk = self._kk
        lp = self._lp
        xx = self._xx
        yy = self._yy
        ye = self._ye if self._nye is None else self._nye
        dxx = self._dxx
        dyy = self._dyy
        dye = self._dye
        eps = self._eps
        opm = self._opm
        opp = self._opp
        dh = self._dh
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        midx = None
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,_Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar,(float,int,np_itypes,np_utypes,np_ftypes)) and float(regpar) > 0.0:
            self._lp = float(regpar)
        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        elif isinstance(yerr,str):
            ye = None
        if isinstance(dxdata,(list,tuple)) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dxx = temp.flatten()
        elif isinstance(dxdata,np.ndarray) and dxdata.size > 0:
            dxx = dxdata.flatten()
        elif isinstance(dxdata,str):
            dxx = None
        if isinstance(dydata,(list,tuple)) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dyy = temp.flatten()
        elif isinstance(dydata,np.ndarray) and dydata.size > 0:
            dyy = dydata.flatten()
        elif isinstance(dydata,str):
            dyy = None
        if isinstance(dyerr,(list,tuple)) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dye = temp.flatten()
        elif isinstance(dyerr,np.ndarray) and dyerr.size > 0:
            dye = dyerr.flatten()
        elif isinstance(dyerr,str):
            dye = None
        if isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) > 0.0:
            eps = float(epsilon)
        elif isinstance(epsilon,(float,int,np_itypes,np_utypes,np_ftypes)) and float(epsilon) <= 0.0:
            eps = None
        elif isinstance(epsilon,str):
            eps = None
        if isinstance(method,str):
            mstr = method.lower()
            for mm in np.arange(0,len(self._opopts)):
                if re.match(mstr,self._opopts[mm],flags=re.IGNORECASE):
                    midx = mm
        elif isinstance(method,(float,int,np_itypes,np_utypes,np_ftypes)) and int(method) < len(self._opopts):
            midx = int(method)
        if midx is not None:
            if midx == 1:
                opm = self._opopts[1]
                oppt = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 2:
                opm = self._opopts[2]
                oppt = np.array([1.0e-5,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 3:
                opm = self._opopts[3]
                oppt = np.array([1.0e-2]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 4:
                opm = self._opopts[4]
                oppt = np.array([1.0e-2,0.9]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 5:
                opm = self._opopts[5]
                oppt = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 6:
                opm = self._opopts[6]
                oppt = np.array([2.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            elif midx == 7:
                opm = self._opopts[7]
                oppt = np.array([1.0e-3,0.9,0.999]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
            else:
                opm = self._opopts[0]
                oppt = np.array([1.0e-5]).flatten()
                for ii in np.arange(0,opp.size):
                    if ii < oppt.size:
                        oppt[ii] = opp[ii]
                opp = oppt.copy()
        if isinstance(spars,(list,tuple)):
            for ii in np.arange(0,len(spars)):
                if ii < opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    opp[ii] = float(spars[ii])
        elif isinstance(spars,np.ndarray):
            for ii in np.arange(0,spars.size):
                if ii < opp.size and isinstance(spars[ii],(float,int,np_itypes,np_utypes,np_ftypes)):
                    opp[ii] = float(spars[ii])
        if isinstance(sdiff,(float,int,np_itypes,np_utypes,np_ftypes)) and float(sdiff) > 0.0:
            dh = float(sdiff)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,_Kernel):
            # Remove all data and associated data that contain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx,xe,yy,ye,nn) = self.__condition_data(xx,xe,yy,ye,lb,ub,cn)
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
                (dxx,dxe,dyy,dye,dnn) = self.__condition_data(dxx,dxe,dyy,dye,-1.0e50,1.0e50,cn)
                dyy = dyy / sc
                dye = dye / sc
            dd = 1 if do_drv else 0
            nkk = copy.copy(kk)
            if eps is not None and not do_drv:
                if opm == 'mom' and opp.size > 1:
                    (nkk,lml) = self.__gp_momentum_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'nag' and opp.size > 1:
                    (nkk,lml) = self.__gp_nesterov_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'adagrad' and opp.size > 0:
                    (nkk,lml) = self.__gp_adagrad_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],dh)
                elif opm == 'adadelta' and opp.size > 1:
                    (nkk,lml) = self.__gp_adadelta_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],dh)
                elif opm == 'adam' and opp.size > 2:
                    (nkk,lml) = self.__gp_adam_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'adamax' and opp.size > 2:
                    (nkk,lml) = self.__gp_adamax_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'nadam' and opp.size > 2:
                    (nkk,lml) = self.__gp_nadam_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],opp[1],opp[2],dh)
                elif opm == 'grad' and opp.size > 0:
                    (nkk,lml) = self.__gp_grad_optimizer(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,opp[0],dh)
            (barF,varF,lml) = self.__gp_base_alg(xn,nkk,lp,xx,yy,ye,dxx,dyy,dye,dd)
            barF = barF * sc if do_drv else barF * sc + myy
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF,errF,lml,nkk)


    def __brute_derivative(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,rtn_cov=False):
        """
        RESTRICTED ACCESS FUNCTION - Can be called externally for testing if user is familiar with algorithm.

        Brute-force numerical GP regression derivative routine, RECOMMENDED to call this instead of bare-bones functions above
        Kept for ability to convince user of validity of regular GP derivative, but can also be wildly wrong on some data due to numerical errors
        RECOMMENDED to use derivative flag on __basic_fit() function, as it was tested and seems to be more robust, provided kernels are properly defined

        :arg xnew: array. x-values at which the predicted fit will be evaluated at.

        :kwarg kernel: Kernel object. The covariance function to be used in fitting the data with Gaussian process regression.

        :kwarg regpar: float. Regularization parameter, multiplies penalty term for kernel complexity to reduce volatility.

        :kwarg xdata: array. x-values of data points to be fitted.

        :kwarg ydata: array. y-values of data points to be fitted.

        :kwarg yerr: array. y-errors of data points to be fitted, assumed to be Gaussian noise specified at 1 sigma. (optional)

        :kwarg rtn_cov: bool. Set as true to return the full predicted covariance matrix instead of the 1 sigma errors.

        :returns: (array, array, float).
            Vector of predicted dy/dx-values, vector or matrix of predicted dy/dx-errors, log-marginal-likelihood of fit
            including the regularization component.
        """

        xn = None
        kk = self._kk
        lp = self._lp
        xx = self._xx
        yy = self._yy
        ye = self._ye if self._nye is None else self._nye
        lb = -1.0e50 if self._lb is None else self._lb
        ub = 1.0e50 if self._ub is None else self._ub
        cn = 5.0e-3 if self._cn is None else self._cn
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,_Kernel):
            kk = copy.copy(kernel)
        if isinstance(regpar,(float,int)) and float(regpar) > 0.0:
            self._lp = float(regpar)
        if isinstance(xdata,(list,tuple)) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif isinstance(xdata,np.ndarray) and xdata.size > 0:
            xx = xdata.flatten()
        if isinstance(ydata,(list,tuple)) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif isinstance(ydata,np.ndarray) and ydata.size > 0:
            yy = ydata.flatten()
        if isinstance(yerr,(list,tuple)) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif isinstance(yerr,np.ndarray) and yerr.size > 0:
            ye = yerr.flatten()
        if ye is None and yy is not None:
            ye = np.zeros(yy.shape)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,_Kernel):
            # Remove all data and associated data that conatain NaNs
            if ye is None:
                ye = np.array([0.0])
            xe = np.array([0.0])
            (xx,xe,yy,ye,nn) = self.__condition_data(xx,xe,yy,ye,lb,ub,cn)
            myy = np.mean(yy)
            yy = yy - myy
            sc = np.nanmax(np.abs(yy))
            if sc == 0.0:
                sc = 1.0
            yy = yy / sc
            ye = ye / sc
            (barF,varF,lml) = self.__gp_brute_deriv1(xn,kk,lp,xx,yy,ye)
            barF = barF * sc
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF,errF,lml)


    def make_NIGP_errors(self,nrestarts=0):
        """
        Automatically-called function which returns a vector of modified y-errors based
        on input x-errors and a test model gradient. Note that this function does not
        iterate until the test model derivatives and actual fit derivatives are self-
        consistent!

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter vlaues within kbounds argument.

        :returns: none.
        """

        # Check inputs
        nr = 0
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nrestarts) > 0:
            nr = int(nrestarts)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        if self._kk is not None and self._xe is not None and self._xx.size == self._xe.size:
            barF = None
            nkk = None
            xntest = np.array([0.0])
            if self._kb is not None and nr > 0:
                kkvec = []
                lmlvec = []
                tkk = copy.copy(self._kk)
                try:
                    (tlml,tkk) = itemgetter(2,3)(self.__basic_fit(xntest))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except ValueError:
                    kkvec.append(None)
                    lmlvec.append(np.NaN)
                for ii in np.arange(0,nr):
                    theta = np.abs(self._kb[:,1] - self._kb[:,0]).flatten() * np.random.random_sample((self._kb.shape[0],)) + np.nanmin(self._kb,axis=1).flatten()
                    tkk.set_hyperparameters(theta,log=True)
                    try:
                        (tlml,tkk) = itemgetter(2,3)(self.__basic_fit(xntest,kernel=tkk))
                        kkvec.append(copy.copy(tkk))
                        lmlvec.append(tlml)
                    except ValueError:
                        kkvec.append(None)
                        lmlvec.append(np.NaN)
                imax = np.where(lmlvec == np.nanmax(lmlvec))[0][0]
                (barF,nkk) = itemgetter(0,3)(self.__basic_fit(xntest,kernel=kkvec[imax],epsilon=-1.0))
            else:
                (barF,nkk) = itemgetter(0,3)(self.__basic_fit(xntest))
            if barF is not None and isinstance(nkk,_Kernel):
                xntest = self._xx.copy() + 1.0e-8
                dbarF = itemgetter(0)(self.__basic_fit(xntest,kernel=nkk,do_drv=True))
                nfilt = np.any([np.isnan(self._xe),np.isnan(self._ye)],axis=0)
                cxe = self._xe.copy()
                cxe[nfilt] = 0.0
                print(dbarF,cxe)
                cye = self._ye.copy()
                cye[nfilt] = 0.0
                self._nye = np.sqrt(cye**2.0 + (cxe * dbarF)**2.0)
        else:
            raise ValueError('Check input x-errors to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)


    def GPRFit(self,xnew,nigp_flag=False,nrestarts=None):
        """
        Main GP regression fitting routine, RECOMMENDED to call this after using set functions instead of __basic_fit()
        as this adapts the method based on inputs, performs 1st derivative and saves output to class variables

        - Includes implementation of Monte Carlo kernel restarts within the user-defined bounds, via nrestarts argument
        - Includes implementation of Heteroscedastic Output Noise, requires setting of error kernel before fitting
            For details, see article: K. Kersting, 'Most Likely Heteroscedastic Gaussian Process Regression' (2007)
        - Includes implementation of Noisy-Input Gaussian Process (NIGP) assuming Gaussian x-error, via nigp_flag argument
            For details, see article: A. McHutchon, C.E. Rasmussen, 'Gaussian Process Training with Input Noise' (2011)

        Developer note: Should this iterate until predicted derivative is consistent with the one
                        used to model impact of input noise?

        :arg xnew: array. x-values at which the predicted fit will be evaluated at.

        :kwarg nigp_flag: bool. Set as true to perform Gaussian Process regression fit accounting for input x-errors.

        :kwarg nrestarts: int. Number of kernel restarts using uniform randomized hyperparameter vlaues within kbounds argument.

        :returns: none.
        """
        # Check inputs
        xn = None
        nr = 0
        if isinstance(xnew,(list,tuple)) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif isinstance(xnew,np.ndarray) and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(nrestarts,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nrestarts) > 0:
            nr = int(nrestarts)
        if xn is None:
            raise ValueError('A valid vector of prediction x-points must be given.')

        barF = None
        varF = None
        lml = None
        nkk = None
        self._nye = None
        if nigp_flag:
            self.make_NIGP_errors(nr)
        hscflag = True if self._ye is not None else False

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        if self._kk is not None and self._kb is not None and nr > 0:
            xntest = np.array([0.0])
            kkvec = []
            lmlvec = []
            tkk = copy.copy(self._kk)
            try:
                (tlml,tkk) = itemgetter(2,3)(self.__basic_fit(xntest))
                kkvec.append(copy.copy(tkk))
                lmlvec.append(tlml)
            except (ValueError,np.linalg.linalg.LinAlgError):
                kkvec.append(None)
                lmlvec.append(np.NaN)
            for ii in np.arange(0,nr):
                theta = np.abs(self._kb[:,1] - self._kb[:,0]).flatten() * np.random.random_sample((self._kb.shape[0],)) + np.nanmin(self._kb,axis=1).flatten()
                tkk.set_hyperparameters(theta,log=True)
                try:
                    (tlml,tkk) = itemgetter(2,3)(self.__basic_fit(xntest,kernel=tkk))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except (ValueError,np.linalg.linalg.LinAlgError):
                    kkvec.append(None)
                    lmlvec.append(np.NaN)
            imaxv = np.where(lmlvec == np.nanmax(lmlvec))[0]
            if len(imaxv) > 0:
                imax = imaxv[0]
                (barF,varF,lml,nkk) = self.__basic_fit(xn,kernel=kkvec[imax],epsilon='None',rtn_cov=True)
            else:
                raise ValueError('None of the fit attempts converged. Please adjust kernel settings and try again.')
        else:
            (barF,varF,lml,nkk) = self.__basic_fit(xn,rtn_cov=True)

        if barF is not None and isinstance(nkk,_Kernel):
            barE = None
            varE = None
            dbarE = None
            dvarE = None
            ddbarE = None
            if hscflag:
                xntest = np.array([0.0])
                ye = self._ye.copy() if self._nye is None else self._nye.copy()
                if isinstance(self._ekk,_Kernel) and self._ekb is not None and not self._eflag and self._eeps is not None:
                    elp = self._elp
                    ekk = copy.copy(self._ekk)
                    ekkvec = []
                    elmlvec = []
                    try:
                        (elml,ekk) = itemgetter(2,3)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                        ekkvec.append(copy.copy(ekk))
                        elmlvec.append(elml)
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ekkvec.append(None)
                        elmlvec.append(np.NaN)
                    for jj in np.arange(0,self._enr):
                        etheta = np.abs(self._ekb[:,1] - self._ekb[:,0]).flatten() * np.random.random_sample((self._ekb.shape[0],)) + np.nanmin(self._ekb,axis=1).flatten()
                        ekk.set_hyperparameters(etheta,log=True)
                        try:
                            (elml,ekk) = itemgetter(2,3)(self.__basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                            ekkvec.append(copy.copy(ekk))
                            elmlvec.append(elml)
                        except (ValueError,np.linalg.linalg.LinAlgError):
                            ekkvec.append(None)
                            elmlvec.append(np.NaN)
                    eimaxv = np.where(elmlvec == np.nanmax(elmlvec))[0]
                    if len(eimaxv) > 0:
                        eimax = eimaxv[0]
                        self._ekk = copy.copy(ekkvec[eimax])
                        self._eflag = True
                    else:
                        raise ValueError('None of the error fit attempts converged. Please change error kernel settings and try again.')
                elif not self._eflag and self._eeps is not None:
                    ekk = Noise_Kernel(float(np.mean(ye)))
                    (elml,ekk) = itemgetter(2,3)(self.__basic_fit(xntest,kernel=ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=self._eeps,method=self._eopm,spars=self._eopp,sdiff=self._edh))
                    self._ekk = copy.copy(ekk)
                    self._eflag = True
                (barE,varE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon='None',rtn_cov=True))
                if barE is not None:
                    (dbarE,dvarE) = itemgetter(0,1)(self.__basic_fit(xn,kernel=self._ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',do_drv=True,rtn_cov=True))
                    nxn = np.linspace(np.nanmin(xn),np.nanmax(xn),1000)
                    ddx = np.nanmin(np.diff(nxn)) * 1.0e-2
                    xnl = nxn - 0.5 * ddx
                    xnu = nxn + 0.5 * ddx
                    dbarEl = itemgetter(0)(self.__basic_fit(xnl,kernel=self._ekk,ydata=ye,yerr=0.1*ye,do_drv=True))
                    dbarEu = itemgetter(0)(self.__basic_fit(xnu,kernel=self._ekk,ydata=ye,yerr=0.1*ye,do_drv=True))
                    ddbarEt = np.abs(dbarEu - dbarEl) / ddx
                    nsum = 50
                    ddbarE = np.zeros(xn.shape)
                    for nx in np.arange(0,xn.size):
                        ivec = np.where(nxn >= xn[nx])[0][0]
                        nbeg = nsum - (ivec + 1) if (ivec + 1) < nsum else 0
                        nend = nsum - (nxn.size - ivec - 1) if (nxn.size - ivec - 1) < nsum else 0
                        temp = None
                        if nbeg > 0:
                            vbeg = np.full((nbeg,),ddbarEt[0])
                            temp = np.hstack((vbeg,ddbarEt[:ivec+nsum+1]))
                            ddbarE[nx] = float(np.mean(temp))
                        elif nend > 0:
                            vend = np.full((nend,),ddbarEt[-1]) if nend > 0 else np.array([])
                            temp = np.hstack((ddbarEt[ivec-nsum:],vend))
                            ddbarE[nx] = float(np.mean(temp))
                        else:
                            ddbarE[nx] = float(np.mean(ddbarEt[ivec-nsum:ivec+nsum+1]))

            self._xF = xn.copy()
            self._barF = barF.copy()
            self._varF = varF.copy() if varF is not None else None
            self._barE = barE.copy() if barE is not None else None
            self._varE = varE.copy() if varE is not None else None
            self._varN = np.diag(np.power(barE,2.0)) if barE is not None else None
            self._lml = lml
            self._kk = copy.copy(nkk) if isinstance(nkk,_Kernel) else None
            (dbarF,dvarF) = itemgetter(0,1)(self.__basic_fit(xn,do_drv=True,rtn_cov=True))
            self._dbarF = dbarF.copy() if dbarF is not None else None
            self._dvarF = dvarF.copy() if dvarF is not None else None
            self._dbarE = dbarE.copy() if dbarE is not None else None
            self._dvarE = dvarE.copy() if dvarE is not None else None
#            self._dvarN = dvarF + np.diag(np.power(dbarE,2.0)) if dvarF is not None and dbarE is not None else None
#            ddfac = np.sqrt(np.mean(np.power(ddbarE,2.0))) if ddbarE is not None else 0.0
            ddfac = ddbarE.copy() if ddbarE is not None else 0.0
            self._dvarN = np.diag(2.0 * (np.power(dbarE,2.0) + np.abs(barE * ddfac))) if barE is not None and dbarE is not None else None
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)


    def sample_GP(self,nsamples,actual_noise=False,simple_out=False):
        """
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        GPRFit() method.

        :arg nsamples: int. Number of samples to perform.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements.

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation.

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest GPRFit(). If simple_out, row 0 is the mean and row 1 is the 1 sigma error.
        """

        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        samples = None
        if ns > 0:
            mu = self.get_gp_mean()
            var = self.get_gp_variance(noise_flag=actual_noise)
            mult_flag = not actual_noise
            mult = self.get_gp_std(noise_flag=mult_flag) / self.get_gp_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu,cov=var,size=ns)
            samples = mult * (samples - mu) + mu
#            for ii in np.arange(0,ns):
#                syy = np.random.multivariate_normal(mu,var)
#                syy = spst.multivariate_normal.rvs(mean=mu,cov=var,size=1)
#                sample = mult * (syy - mu) + mu
#                samples = syy.copy() if samples is None else np.vstack((samples,sample))
            if samples is not None and simple_out:
                mean = np.mean(samples,axis=0)
                std = np.std(samples,axis=0)
                samples = np.vstack((mean,std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return samples


    def sample_GP_derivative(self,nsamples,actual_noise=False,simple_out=False):
        """
        Samples Gaussian process posterior on data for predictive functions.
        Can be used by user to check validity of mean and variance outputs of
        GPRFit() method.

        :arg nsamples: int. Number of samples to perform.

        :kwarg actual_noise: bool. Specifies inclusion of noise term in returned variance. Only operates on diagonal elements.

        :kwarg simple_out: bool. Set as true to average over all samples and return only the mean and standard deviation.

        :returns: array. Rows containing sampled fit evaluated at xnew used in latest GPRFit(). If simple_out, row 0 is the mean and row 1 is the 1 sigma error.
        """

        # Check instantiation of output class variables
        if self._xF is None or self._dbarF is None or self._dvarF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        samples = None
        if ns > 0:
            mu = self.get_gp_drv_mean()
            var = self.get_gp_drv_variance(noise_flag=actual_noise)
            mult_flag = not actual_noise
            mult = self.get_gp_drv_std(noise_flag=mult_flag) / self.get_gp_drv_std(noise_flag=False)
            samples = spst.multivariate_normal.rvs(mean=mu,cov=var,size=ns)
            samples = mult * (samples - mu) + mu
#            for ii in np.arange(0,ns):
#                syy = np.random.multivariate_normal(mu,var)
#                syy = spst.multivariate_normal.rvs(mean=mu,cov=var,size=1)
#                sample = mult * (syy - mu) + mu
#                samples = syy.copy() if samples is None else np.vstack((samples,sample))
            if samples is not None and simple_out:
                mean = np.mean(samples,axis=0)
                std = np.std(samples,axis=0)
                samples = np.vstack((mean,std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return samples


    def MCMC_posterior_sampling(self,nsamples):   # NOT RECOMMENDED, not fully tested
        """
        Performs Monte Carlo Markov chain based posterior analysis over hyperparameters, using LML as the likelihood

        .. warning::

            This function is INCORRECT as coded, should use data likelihood from model instead of LML as the
            acceptance criterion. However, MCMC analysis is only necessary when using non-Gaussian
            likelihoods, otherwise the result is equivalent to maximization of LML

        :arg nsamples: int. Number of samples to perform.

        :returns: (array, array, array, array).
            Matrices containing rows of predicted y-values, predicted y-errors, predicted dy/dx-values,
            predicted dy/dx-errors with each having a number of rows equal to the number of samples.
        """
        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to use MCMC posterior sampling.')

        # Check inputs
        ns = 0
        if isinstance(nsamples,(float,int,np_itypes,np_utypes,np_ftypes)) and int(nsamples) > 0:
            ns = int(nsamples)

        if not self._fwarn:
            warnings.filterwarnings("ignore",category=RuntimeWarning)

        sbarM = None
        ssigM = None
        sdbarM = None
        sdsigM = None
        if isinstance(self._kk,_Kernel) and ns > 0:
            olml = self._lml
            otheta = self._kk.get_hyperparameters(log=True)
            tlml = olml
            theta = otheta.copy()
            step = np.ones(theta.shape)
            flagvec = [True] * theta.size
            for ihyp in np.arange(0,theta.size):
                xntest = np.array([0.0])
                iflag = flagvec[ihyp]
                while iflag:
                    tkk = copy.copy(self._kk)
                    theta_step = np.zeros(theta.shape)
                    theta_step[ihyp] = step[ihyp]
                    theta_new = theta + theta_step
                    tkk.set_hyperparameters(theta_new,log=True)
                    ulml = None
                    try:
                        ulml = itemgetter(2)(self.__basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ulml = tlml - 3.0
                    theta_new = theta - theta_step
                    tkk.set_hyperparameters(theta_new,log=True)
                    llml = None
                    try:
                        llml = itemgetter(2)(self.__basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        llml = tlml - 3.0
                    if (ulml - tlml) >= -2.0 or (llml - tlml) >= -2.0:
                        iflag = False
                    else:
                        step[ihyp] = 0.5 * step[ihyp]
                flagvec[ihyp] = iflag
            nkk = copy.copy(self._kk)
            for ii in np.arange(0,ns):
                theta_prop = theta.copy()
                accept = False
                xntest = np.array([0.0])
                nlml = tlml
                jj = 0
                kk = 0
                while not accept:
                    jj = jj + 1
                    rstep = np.random.normal(0.0,0.5*step)
                    theta_prop = theta_prop + rstep
                    nkk.set_hyperparameters(theta_prop,log=True)
                    try:
                        nlml = itemgetter(2)(self.__basic_fit(xntest,kernel=nkk,epsilon='None'))
                        if (nlml - tlml) > 0.0:
                            accept = True
                        else:
                            accept = True if np.power(10.0,nlml - tlml) >= np.random.uniform() else False
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
                nkk.set_hyperparameters(theta,log=True)
                (barF,sigF,tlml,nkk) = self.__basic_fit(xn,kernel=nkk,epsilon='None')
                sbarM = barF.copy() if sbarM is None else np.vstack((sbarM,barF))
                ssigM = sigF.copy() if ssigM is None else np.vstack((ssigM,sigF))
                (dbarF,dsigF) = itemgetter(0,1)(self.__basic_fit(xn,kernel=nkk,epsilon='None',do_drv=True))
                sdbarM = dbarF.copy() if sdbarM is None else np.vstack((sdbarM,dbarF))
                sdsigM = dsigF.copy() if sdsigM is None else np.vstack((sdsigM,dsigF))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')

        if not self._fwarn:
            warnings.filterwarnings("default",category=RuntimeWarning)

        return (sbarM,ssigM,sdbarM,sdsigM)


# ****************************************************************************************************************************************
# ------- Some helpful functions for reproducability and user-friendliness ---------------------------------------------------------------
# ****************************************************************************************************************************************


def KernelConstructor(name):
    """
    Function to construct a kernel solely based on the kernel codename.
    .. note:: OperatorKernel objects now use encapsulating round brackets to specify their constituents

    :param name: str. The codename of the desired Kernel object.
    """

    kernel = None
    if isinstance(name,str):
        m = re.search(r'^(.*?)\((.*)\)$',name)
        if m:
            links = m.group(2).split('-')
            names = []
            bflag = False
            rname = ''
            for jj in np.arange(0,len(links)):
                rname = links[jj] if not bflag else rname + '-' + links[jj]
                if re.search('\(',links[jj]):
                    bflag = True
                if re.search('\)',links[jj]):
                    bflag = False
                if not bflag:
                    names.append(rname)
            kklist = []
            for ii in np.arange(0,len(names)):
                kklist.append(KernelConstructor(names[ii]))
            if re.search('^Sum$',m.group(1)):
                kernel = Sum_Kernel(klist=kklist)
            elif re.search('^Prod$',m.group(1)):
                kernel = Product_Kernel(klist=kklist)
            elif re.search('^Sym$',m.group(1)):
                kernel = Symmetric_Kernel(klist=kklist)
        else:
            if re.match('^C$',name):
                kernel = Constant_Kernel()
            elif re.match('^n$',name):
                kernel = Noise_Kernel()
            elif re.match('^L$',name):
                kernel = Linear_Kernel()
            elif re.match('^P$',name):
                kernel = Poly_Order_Kernel()
            elif re.match('^SE$',name):
                kernel = SE_Kernel()
            elif re.match('^RQ$',name):
                kernel = RQ_Kernel()
            elif re.match('^MH$',name):
                kernel = Matern_HI_Kernel()
            elif re.match('^NN$',name):
                kernel = NN_Kernel()
            elif re.match('^Gw',name):
                wname = re.search('^Gw(.*)$',name).group(1)
                wfunc = None
                if re.match('^C$',wname):
                    wfunc = Constant_WarpingFunction()
                elif re.match('^IG$',wname):
                    wfunc = IG_WarpingFunction()
                kernel = Gibbs_Kernel(wfunc=wfunc)
    return kernel


def KernelReconstructor(name,pars=None,log=False):
    """
    Function to reconstruct any kernel from its kernel codename and parameter list,
    useful for saving only necessary data to represent a GPR1D object.

    :param name: str. The codename of the desired Kernel object.

    :param pars: array. The hyperparameter and constant values to be stored in the Kernel object, order determined by the Kernel.

    :param log: bool. Indicates that pars was passed in as log10(pars).
    """

    kernel = KernelConstructor(name)
    pvec = None
    if isinstance(pars,(list,tuple)):
        pvec = np.array(pars).flatten()
    elif isinstance(pars,np.ndarray):
        pvec = pars.flatten()
    if isinstance(kernel,_Kernel) and pvec is not None:
        nhyp = kernel.get_hyperparameters().size
        ncst = kernel.get_constants().size
        if pvec.size >= nhyp:
            theta = pvec[:nhyp] if pvec.size > nhyp else pvec.copy()
            kernel.set_hyperparameters(theta,log=log)
        if ncst > 0 and pvec.size >= (nhyp + ncst):
            csts = pvec[nhyp:nhyp+ncst] if pvec.size > (nhyp + ncst) else pvec[nhyp:]
            kernel.set_constants(csts)
    return kernel
