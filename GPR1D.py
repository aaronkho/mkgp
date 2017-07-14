# Custom covariance functions / kernels for GPR
# Developer: Aaron Ho -- 09/01/2017
#    Kernel theory: "Gaussian Process for Machine Learning", C.E. Rasmussen, C.K.I. Williams (2006)
# Edited: Aaron Ho -- 13/02/2017          cleaned up script, added comment for ideology and per kernel implementation
# Edited: Aaron Ho -- 27/02/2017          removed parameters field and included hyperparameter derivatives for LML maximization (not complete)
# Edited: Aaron Ho -- 13/03/2017          addition of Gibbs kernel with Gaussian length scale function
# Edited: Aaron Ho -- 12/04/2017          addition of adjusted linear kernel for use in polynomial regression

# Functions to test /use Gaussian Process module
# Developer: Aaron Ho -- 09/01/2017
#    Gaussian process theory: "Gaussian Processes for Machine Learning", C.E. Rasmussen and C.K.I. Williams (2006)
# Edited: Aaron Ho -- 13/02/2017          cleaned up scripts, added general comments for each function, no metacode comments yet
# Edited: Aaron Ho -- 27/02/2017          removed parameters field, seemed redundant and useless
# Edited: Aaron Ho -- 13/03/2017          adjusted Gram matrix calculation to not use simple negative transpose, arg for x values of derivative data added
# Edited: Aaron Ho -- 16/03/2017          added regularization parameter to complexity term of log-marginal-likelihood
# Edited: Aaron Ho -- 24/04/2017          added implementation of NIGP for normally distributed input noise
# Edited: Aaron Ho -- 19/05/2017          added implementation of Monte Carlo randomized restarts, to avoid local maxima

# Merged kernel specification file and GPR functions into a single file (for integartion into OMFIT)
# Developer: Aaron Ho -- 01/06/2017
# Edited: Aaron Ho -- 05/06/2017          consolidated GPR functions into GPR1D class, restructured GPR and NIGPR with MC into single function
# Edited: Aaron Ho -- 07/06/2017          added docstrings to all classes / functions
# Edited: Aaron Ho -- 13/06/2017          added more strict implementation of heteroscedasticity (variable y errors in x) and data conditioner

# Required imports
import re
import copy
import numpy as np
import scipy.linalg as spla
from operator import itemgetter
import matplotlib.pyplot as plt

class Kernel():
    """
    Base class   *** to be inherited by ALL kernel implementations in order for type checks to succeed ***
        Type checking done with:     isinstance(<obj>,<this_module>.Kernel)
    Ideology: fname is a string, designed to provide an easy way to check the kernel object type
              function contains the covariance function, k, along with dk/dx1, dk/dx2, and d^2k/dx1dx2
              hyperparameters contains free variables that vary in logarithmic-space
              constants contains "free" variables that should not be changed during parameter searches, or true constants
              bounds contains the bounds of the free variables to be used in parameter searches   *** NOT YET IMPLEMENTED!!! ***
    Get and set functions already given, but all functions can be overridden by specific implementation, NOT recommended
    """
# Commented lines indicate the required statements to re-include parameters field

#    def __init__(self,name="None",func=None,hderf=False,hyps=None,pars=None,csts=None):
    def __init__(self,name="None",func=None,hderf=False,hyps=None,csts=None):
        self._fname = name
        self._function = func if func is not None else None
        self._hyperparameters = copy.deepcopy(hyps) if hyps is not None else None
#        self._parameters = copy.deepcopy(pars) if pars is not None else None
        self._constants = copy.deepcopy(csts) if csts is not None else None
        self._bounds = None
#        self._hderflag = hderf
        self._hderflag = False     # d/dhyp capability turned off, further testing required!

    def __call__(self,x1,x2,der=0,hder=None):
        k_out = None
        if self._function is not None:
            k_out = self._function(x1,x2,der,hder)
        else:
            raise NotImplementedError('Kernel function not yet defined.')
        return k_out

    def get_name(self):
        return self._fname

    def get_hyperparameters(self,log=False):
        val = np.array([])
        if self._hyperparameters is not None:
            val = np.log10(self._hyperparameters) if log else self._hyperparameters
        return val

#    def get_parameters(self):
#        val = np.array([])
#        if self._parameters is not None:
#            val = self._parameters
#        return val

    def get_constants(self):
        val = np.array([])
        if self._constants is not None:
            val = self._constants
        return val

    def get_bounds(self,log=False):
        val = None
        if self._bounds is not None:
            val = np.log10(self._bounds) if log else self._bounds
        return val

    def is_hderiv_implemented(self):
        return self._hderflag

    def set_hyperparameters(self,theta,log=False):
        uhyps = None
        if type(theta) in (list,tuple):
            uhyps = np.array(theta).flatten()
        elif type(theta) is np.ndarray:
            uhyps = theta.flatten()
        else:
            raise TypeError('Argument theta must be an array-like object.')
        if log:
            uhyps = np.power(10.0,uhyps)
        if self._hyperparameters is not None:
            if uhyps.size >= self._hyperparameters.size:
                self._hyperparameters = uhyps[:self._hyperparameters.size]
            else:
                raise ValueError('Argument theta must contain at least %d elements.' % (self._hyperparameters.size))
        else:
            raise AttributeError('Kernel object has no hyperparameters.')

#    def set_parameters(self,params):
#        upars = None
#        if type(params) in (list,tuple):
#            upars = np.array(params).flatten()
#        elif type(params) is np.ndarray:
#            upars = params.flatten()
#        else:
#            raise TypeError('Argument params must be an array-like object.')
#        if self._parameters is not None:
#            if upars.size >= self._parameters.size:
#                self._parameters = upars[:self._parameters.size]
#            else:
#                raise ValueError('Argument params must contain at least %d elements.' % (self._parameters.size))
#        else:
#            raise AttributeError('Kernel object has no parameters.')

    def set_constants(self,consts):
        ucsts = None
        if type(consts) in (list,tuple):
            ucsts = np.array(consts).flatten()
        elif type(consts) is np.ndarray:
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
        ubnds = None
        if type(lbounds) in (list,tuple):
            ubnds = np.array(lbounds).flatten()
        elif type(lbounds) is np.ndarray:
            ubnds = lbounds.flatten()
        else:
            raise TypeError('Argument lbounds must be an array-like object.')
        if type(ubounds) in (list,tuple) and len(ubounds) == ubnds.size:
            ubnds = np.vstack((ubnds,np.array(ubounds).flatten()))
        elif type(ubounds) is np.ndarray and ubounds.size == ubnds.size:
            ubnds = np.vstack((ubnds,ubounds.flatten()))
        else:
            raise TypeError('Argument ubounds must be an array-like object and have dimensions equal to lbounds.')
        if log:
            ubnds = np.power(10.0,ubnds)
        if self._hyperparameters is not None:
            if ubnds.shape[1] >= self._hyperparameters.size:
                self._bounds = ubnds[:,:self._hyperparmaeters.size]
            else:
                raise ValueError('Arguments lbounds and ubounds must contain at least %d elements.' % (self._constants.size))
        else:
            raise AttributeError('Kernel object has no hyperparameters to set bounds for.')


class OperatorKernel(Kernel):
    """
    Base operator class   *** To be inherited by ALL operator kernel implementations for obtain custom get/set functions ***
    Ideology: get/set functions adjusted to call get/set functions of each constituent kernel
    """
    def __init__(self,name="None",func=None,hderf=False,klist=None):
        self._kernel_list = klist if klist is not None else []
        Kernel.__init__(self,name,func,hderf)

    def get_name(self):
        return self._fname

    def get_hyperparameters(self,log=False):
        val = np.array([])
        for kk in self._kernel_list:
            val = np.append(val,kk.get_hyperparameters(log=log))
        return val

#    def get_parameters(self):
#        val = np.array([])
#        for kk in self._kernel_list:
#            val = np.append(val,kk.get_parameters())
#        return val

    def get_constants(self):
        val = np.array([])
        for kk in self._kernel_list:
            val = np.append(val,kk.get_constants())
        return val

    def get_bounds(self,log=False):
        val = np.array([])
        for kk in self._kernel_list:
            kval = kk.get_bounds(log=log)
            if kval is not None:
                val = np.append(val,kval)
        if val.size == 0:
            val = None
        return val

    def set_hyperparameters(self,theta,log=False):
        uhyps = None
        if type(theta) in (list,tuple):
            uhyps = np.array(theta).flatten()
        elif type(theta) is np.ndarray:
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

#    def set_parameters(self,params):
#        upars = None
#        if type(params) in (list,tuple):
#            upars = np.array(params).flatten()
#        elif type(params) is np.ndarray:
#            upars = params.flatten()
#        else:
#            raise TypeError('Argument params must be an array-like object.')
#        npars = self.get_parameters().size
#        if npars > 0:
#            if upars.size >= npars:
#                ndone = 0
#                for kk in self._kernel_list:
#                    nhere = ndone + kk.get_parameters().size
#                    if nhere != ndone:
#                        if nhere == npars:
#                            kk.set_parameters(params[ndone:])
#                        else:
#                            kk.set_parameters(params[ndone:nhere])
#                        ndone = nhere
#            else:
#                raise ValueError('Argument params must contain at least %d elements.' % (npars))
#        else:
#            raise AttributeError('Kernel object has no parameters.')

    def set_constants(self,consts):
        ucsts = None
        if type(consts) in (list,tuple):
            ucsts = np.array(consts).flatten()
        elif type(consts) is np.ndarray:
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
        ubnds = None
        if type(lbounds) in (list,tuple):
            ubnds = np.array(lbounds).flatten()
        elif type(lbounds) is np.ndarray:
            ubnds = lbounds.flatten()
        else:
            raise TypeError('Argument lbounds must be an array-like object.')
        if type(ubounds) in (list,tuple) and len(ubounds) == ubnds.size:
            ubnds = np.vstack((ubnds,np.array(ubounds).flatten()))
        elif type(ubounds) is np.ndarray and ubounds.size == ubnds.size:
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


# ****************************************************************************************************************************************
# ------- Place ALL custom kernel implementations BELOW ----------------------------------------------------------------------------------
# ****************************************************************************************************************************************

class Sum_Kernel(OperatorKernel):
    """
    Sum Kernel: Implements the sum of two (or more) Kernel objects
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        covm = np.NaN if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp)
            if ihyp is not None:
                hyps = np.array(kk.get_hyperparameters())
                nhyps = hyps.size
#                pars = np.array(kk.get_parameters())
#                npars = pars.size
                ihyp = ihyp - nhyps # - npars
        return covm

    def __init__(self,*args,klist=None):
        uklist = []
        name = "None"
        if len(args) >= 2 and isinstance(args[0],Kernel) and isinstance(args[1],Kernel):
            name = ""
            for kk in args:
                if isinstance(kk,Kernel):
                    uklist.append(kk)
                    name = name + kk.get_name()
        elif type(klist) is list and len(klist) >= 2 and isinstance(klist[0],Kernel) and isinstance(klist[1],Kernel):
            name = ""
            for kk in klist:
                if isinstance(kk,Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel objects.')
        OperatorKernel.__init__(self,"Sum_"+name,self.__calc_covm,True,uklist)

    def __copy__(self):
        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Sum_Kernel(klist=kcopy_list)
        return kcopy

class Product_Kernel(OperatorKernel):
    """
    Product Kernel: Implements the product of two (or more) Kernel objects
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
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
#                    pars = np.array(kk.get_parameters())
#                    npars = pars.size
                    ihyp = ihyp - nhyps # - npars
            covm = covm + covterm
        return covm

    def __init__(self,*args,klist=None):
        uklist = []
        name = "None"
        if len(args) >= 2 and isinstance(args[0],Kernel) and isinstance(args[1],Kernel):
            name = ""
            for kk in args:
                if isinstance(kk,Kernel):
                    uklist.append(kk)
                    name = name + kk.get_name()
        elif type(klist) is list and len(klist) >= 2 and isinstance(klist[0],Kernel) and isinstance(klist[1],Kernel):
            name = ""
            for kk in klist:
                if isinstance(kk,Kernel):
                    uklist.append(kk)
                    name = name + "-" + kk.get_name() if name else kk.get_name()
        else:
            raise TypeError('Arguments to Sum_Kernel must be Kernel objects.')
        OperatorKernel.__init__(self,"Prod_"+name,self.__calc_covm,True,uklist)

    def __copy__(self):
        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Product_Kernel(klist=kcopy_list)
        return kcopy


class Symmetric_Kernel(OperatorKernel):
    """
    1D Symmetric Kernel: Enforces even symmetry about zero for any given Kernel object (only uses first Kernel argument, though it accepts many)
    This is really only useful if you wish to rigourously infer data on other side of axis of symmetry without assuming the data
    can just be flipped or if data on other side is present but require GP to return symmetric solution *** NOT TESTED! ***
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        covm = np.NaN if self._kernel_list is None else np.zeros(x1.shape)
        ihyp = hder
        for kk in self._kernel_list:
            covm = covm + kk(x1,x2,der,ihyp) + kk(-x1,x2,der,ihyp)      # Not sure if division by 2 is necessary to conserve covm
            if ihyp is not None:
                hyps = np.array(kk.get_hyperparameters())
                nhyps = hyps.size
#                pars = np.array(kk.get_parameters())
#                npars = pars.size
                ihyp = ihyp - nhyps # - npars
        return covm

    def __init__(self,*args,klist=None):
        uklist = []
        name = "None"
        if len(args) >= 1 and isinstance(args[0],Kernel):
            name = ""
            if len(args) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = args[0]
            uklist.append(kk)
            name = name + kk.get_name()
        elif type(klist) is list and len(klist) >= 1 and isinstance(klist[0],Kernel):
            name = ""
            if len(klist) >= 2:
                print("Only the first kernel argument is used in Symmetric_Kernel class, use other operators first.")
            kk = klist[0]
            uklist.append(kk)
            name = name + kk.get_name()
        else:
            raise TypeError('Arguments to Symmetric_Kernel must be Kernel objects.')
        OperatorKernel.__init__(self,"Sym_"+name,self.__calc_covm,True,uklist)

    def __copy__(self):
        kcopy_list = []
        for kk in self._kernel_list:
            kcopy_list.append(copy.copy(kk))
        kcopy = Symmetric_Kernel(klist=kcopy_list)
        return kcopy


class Constant_Kernel(Kernel):
    """
    Constant Kernel: always evaluates to a constant value, regardless of x1 and x2
    Note that this is NOT INHERENTLY A VALID COVARIANCE FUNCTION, as it yields singular covariance matrices!
    However, it provides a nice way to add bias to any other kernel (is this even true?!?)
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        c_hyp = self._constants[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = c_hyp * np.ones(rr.shape)
        return covm

    def __init__(self,cv=1.0):
        csts = np.zeros((1,))
        if type(cv) in (float,int):
            csts[0] = float(cv)
        else:
            raise ValueError('Constant value must be a real number.')
        Kernel.__init__(self,"C",self.__calc_covm,True,None,csts)

    def __copy__(self):
        chp = float(self._constants[0])
        kcopy = Constant_Kernel(chp)
        return kcopy


class Noise_Kernel(Kernel):
    """
    Noise Kernel: adds a user-defined degree of expected noise in the data / measurement process
    Note that this is NOT THE SAME as measurement error, which should be applied externally in GP!!!
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        n_hyp = self._hyperparameters[0]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm[rr == 0.0] = n_hyp**2.0
            elif hder == 0:
                covm[rr == 0.0] = 2.0 * n_hyp
# Applied second derivative of Kronecker delta, assuming n_hyp is actually a Gaussian centred on rr = 0 with width ss
# Surprisingly provides good variance estimate but issues with enforcing derivative constraints (need more work!)
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
        hyps = np.zeros((1,))
        if type(nv) in (float,int):
            hyps[0] = float(nv)
        else:
            raise ValueError('Noise hyperparameter must be a real number.')
        Kernel.__init__(self,"n",self.__calc_covm,True,hyps)

    def __copy__(self):
        nhp = float(self._hyperparameters[0])
        kcopy = Noise_Kernel(nhp)
        return kcopy


class Linear_Kernel(Kernel):
    """
    Linear Kernel: Applies linear regression (b = 0), can be multiplied with itself for higher order pure polynomials
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
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
        hyps = np.zeros((1,))
        if type(var) in (float,int) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        Kernel.__init__(self,"L",self.__calc_covm,True,hyps)

    def __copy__(self):
        chp = float(self._hyperparameters[0])
        kcopy = Linear_Kernel(chp)
        return kcopy


class Poly_Order_Kernel(Kernel):
    """
    Polynomial Order Kernel: Applies linear regression (b != 0), can be multiplied with itself for higher order polynomials
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
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
        hyps = np.zeros((2,))
        if type(var) in (float,int) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Multiplicative hyperparameter must be greater than 0.')
        if type(cst) in (float,int) and float(cst) > 0.0:
            hyps[1] = float(cst)
        else:
            raise ValueError('Additive hyperparameter must be greater than 0.')
        Kernel.__init__(self,"P",self.__calc_covm,True,hyps)

    def __copy__(self):
        chp = float(self._hyperparameters[0])
        cst = float(self._hyperparameters[1])
        kcopy = Poly_Order_Kernel(chp,cst)
        return kcopy


class SE_Kernel(Kernel):
    """
    Square Exponential Kernel: Infinitely differentiable (ie. extremely smooth) covariance function
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        v_hyp = self._hyperparameters[0]
        l_hyp = self._hyperparameters[1]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            if hder is None:
                covm = v_hyp**2.0 * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 0:
                covm = 2.0 * v_hyp * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 1:
                covm = np.power(rr,2.0) * v_hyp**2.0 / (l_hyp**3.0) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
        elif der == 1:
            drdx2 = np.sign(x2 - x1)
            if hder is None:
                covm = -drdx2 * v_hyp**2.0 * rr / (l_hyp**2.0) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 0:
                covm = -drdx2 * 2.0 * v_hyp * rr / (l_hyp**2.0) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 1:
                hdt1 = -2.0 * v_hyp**2.0 * rr / (l_hyp**3.0)
                hdt2 = v_hyp**4.0 * np.power(rr,3.0) / (l_hyp**5.0)
                covm = -drdx2 * (hdt1 + hdt2) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
        elif der == -1:
            drdx1 = np.sign(x1 - x2)
            if hder is None:
                covm = -drdx1 * v_hyp**2.0 * rr / (l_hyp**2.0) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 0:
                covm = -drdx1 * 2.0 * v_hyp * rr / (l_hyp**2.0) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 1:
                hdt1 = -2.0 * v_hyp**2.0 * rr / (l_hyp**3.0)
                hdt2 = v_hyp**4.0 * np.power(rr,3.0) / (l_hyp**5.0)
                covm = -drdx1 * (hdt1 + hdt2) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
        elif der == 2 or der == -2:
            drdx1 = np.sign(x1 - x2)
            drdx1[drdx1==0] = 1.0
            drdx2 = np.sign(x2 - x1)
            drdx2[drdx2==0] = -1.0
            if hder is None:
                cc = 1.0 / (l_hyp**2.0) - np.power(rr,2.0) / (l_hyp**4.0)
                covm = -drdx1 * drdx2 * v_hyp**2.0 * cc * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 0:
                cc = 1.0 / (l_hyp**2.0) - np.power(rr,2.0) / (l_hyp**4.0)
                covm = -drdx1 * drdx2 * 2.0 * v_hyp * cc * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
            elif hder == 1:
                cc = 1.0 / (l_hyp**2.0) - np.power(rr,2.0) / (l_hyp**4.0)
                hdt1 = v_hyp**2.0 * (4.0 * np.power(rr,2.0) / (l_hyp**5.0) - 2.0 / (l_hyp**3.0))
                hdt2 = v_hyp**2.0 * cc * np.power(rr,2.0) / (l_hyp**3.0)
                covm = -drdx1 * drdx2 * (hdt1 + hdt2) * np.exp(-np.power(rr,2.0) / (2.0 * l_hyp**2.0))
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not implemented in '+self.get_name()+' kernel.')
        return covm

    def __init__(self,var=1.0,ls=1.0):
        hyps = np.zeros((2,))
        if type(var) in (float,int) and float(var) > 0.0:
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be greater than 0.')
        if type(ls) in (float,int) and float(ls) > 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Length scale hyperparameter must be greater than 0.')
        Kernel.__init__(self,"SE",self.__calc_covm,True,hyps)

    def __copy__(self):
        chp = float(self._hyperparameters[0])
        shp = float(self._hyperparameters[1])
        kcopy = SE_Kernel(chp,shp)
        return kcopy


class RQ_Kernel(Kernel):
    """
    Rational Quadratic Kernel: Also infinitely differentiable, but provides higher tolerance for steep slopes
    Acts as infinite sum of SE kernels for a_hyp < 20, otherwise effectively identical to SE as a_hyp -> infinity
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        rq_amp = self._hyperparameters[0]
        l_hyp = self._hyperparameters[1]
        a_hyp = self._hyperparameters[2]
        rr = np.abs(x1 - x2)
        covm = np.zeros(rr.shape)
        if der == 0:
            rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
            if hder is None:
                rqfac = np.power(rqt,-a_hyp)
                covm = rq_amp**2.0 * rqfac
            elif hder == 0:
                rqfac = np.power(rqt,-a_hyp)
                covm = 2.0 * rq_amp * rqfac
            elif hder == 1:
                rqfac = -np.power(rr,2.0) / (a_hyp * l_hyp**3.0) * np.power(rqt,-a_hyp - 1.0)
                covm = rq_amp**2.0 * rqfac
            elif hder == 2:
                rqfac = (np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt)) * np.power(rqt,-a_hyp)
                covm = rq_amp**2.0 * rqfac
        elif der == 1:
            drdx2 = np.sign(x2 - x1)
            rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
            if hder is None:
                rqfac = (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx2 * rq_amp**2.0 * rqfac
            elif hder == 0:
                rqfac = (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx2 * 2.0 * rq_amp * rqfac
            elif hder == 1:
                at1 = (a_hyp + 1.0) / a_hyp
                hdt1 = -2.0 * rr / (l_hyp**3.0)
                hdt2 = np.power(rr,3.0) / (l_hyp**5.0 * rqt)
                rqfac = (hdt1 + at1 * hdt2) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx2 * rq_amp**2.0 * rqfac
            elif hder == 2:
                at1 = (a_hyp + 1.0) / a_hyp
                hdt = np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt) / at1
                rqfac = at1 * hdt * (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx2 * rq_amp**2.0 * rqfac
        elif der == -1:
            drdx1 = np.sign(x1 - x2)
            rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
            if hder is None:
                rqfac = (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx1 * rq_amp**2.0 * rqfac
            elif hder == 0:
                rqfac = (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx1 * 2.0 * rq_amp * rqfac
            elif hder == 1:
                at1 = (a_hyp + 1.0) / a_hyp
                hdt1 = -2.0 * rr / (l_hyp**3.0)
                hdt2 = np.power(rr,3.0) / (l_hyp**5.0 * rqt)
                rqfac = (hdt1 + at1 * hdt2) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx1 * rq_amp**2.0 * rqfac
            elif hder == 2:
                at1 = (a_hyp + 1.0) / a_hyp
                hdt = np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt) / at1
                rqfac = at1 * hdt * (rr / l_hyp**2.0) * np.power(rqt,-a_hyp - 1.0)
                covm = -drdx1 * rq_amp**2.0 * rqfac
        elif der == 2 or der == -2:
            drdx1 = np.sign(x1 - x2)
            drdx1[drdx1==0] = 1.0
            drdx2 = np.sign(x2 - x1)
            drdx2[drdx2==0] = -1.0
            rqt = 1.0 + np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0)
            if hder is None:
                at1 = (a_hyp + 1.0) / a_hyp
                rqt1 = rqt / (l_hyp**2.0)
                rqt2 = -np.power(rr,2.0) / (l_hyp**4.0)
                rqfac = (rqt1 + at1 * rqt2) * np.power(rqt,-a_hyp - 2.0)
                covm = -drdx1 * drdx2 * rq_amp**2.0 * rqfac
            elif hder == 0:
                at1 = (a_hyp + 1.0) / a_hyp
                rqt1 = rqt / (l_hyp**2.0)
                rqt2 = -np.power(rr,2.0) / (l_hyp**4.0)
                rqfac = (rqt1 + at1 * rqt2) * np.power(rqt,-a_hyp - 2.0)
                covm = -drdx1 * drdx2 * 2.0 * rq_amp * rqfac
            elif hder == 1:
                at1 = (a_hyp + 1.0) / a_hyp
                at2 = (a_hyp + 2.0) / a_hyp
                hdt1 = -2.0 * rqt / (l_hyp**3.0)
                hdt2 = 5.0 * np.power(rr,2.0) / (l_hyp**5.0)
                hdt3 = -np.power(rr,4.0) / (l_hyp**7.0 * rqt)
                rqfac = (hdt1 + at1 * hdt2 + at1 * at2 * hdt3) * np.power(rqt,-a_hyp - 2.0)
                covm = -drdx1 * drdx2 * rq_amp**2.0 * rqfac
            elif hder == 2:
                at1 = (a_hyp + 1.0) / a_hyp
                at2 = (a_hyp + 2.0) / a_hyp
                at3 = (a_hyp + 3.0) / a_hyp
                hdt1a = np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt) / at1
                hdt1 = -at1 * hdt1a * 2.0 / (l_hyp**3.0) * rqt
                hdt2a = -1.0 / (a_hyp**2.0)
                hdt2b = np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt) / at2
                hdt2 = (hdt2a + at1 * at2 * hdt2b) * 5.0 * np.power(rr,2.0) / (l_hyp**5.0)
                hdt3a = -(4.0 / (a_hyp**3.0) + 3.0 / (a_hyp**2.0))
                hdt3b = np.power(rr,2.0) / (2.0 * a_hyp * l_hyp**2.0 * rqt) - np.log(rqt) / at3
                hdt3 = -(hdt3a + at1 * at2 * at3 * hdt2b) * np.power(rr,4.0) / (l_hyp**7.0 * rqt)
                rqfac = (hdt1 + hdt2 + hdt3) * np.power(rqt,-a_hyp - 2.0)
                covm = -drdx1 * drdx2 * rq_amp**2.0 * rqfac
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not yet implemented in '+self.get_name()+' kernel.')
        return covm

    def __init__(self,amp=1.0,ls=1.0,alpha=1.0):
        hyps = np.zeros((3,))
        if type(amp) in (float,int) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Rational quadratic amplitude must be greater than 0.')
        if type(ls) in (float,int) and float(ls) != 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Rational quadratic hyperparameter cannot equal 0.')
        if type(alpha) in (float,int) and float(alpha) > 0.0:
            hyps[2] = float(alpha)
        else:
            raise ValueError('Rational quadratic alpha parameter must be greater than 0.')
        Kernel.__init__(self,"RQ",self.__calc_covm,True,hyps)

    def __copy__(self):
        ramp = float(self._hyperparameters[0])
        rhp = float(self._hyperparameters[1])
        ralp = float(self._hyperparameters[2])
        kcopy = RQ_Kernel(ramp,rhp,ralp)
        return kcopy


class Matern_HI_Kernel(Kernel):
    """
    Matern Kernel with Half-Integer nu: Only differentiable in orders less than given nu, allows fit to retain more features at expense of volatility
    The half-integer implentation allows for use of explicit simplifications of the derivatives, which greatly improves its speed
    Recommended nu: 5/2 for second order differentiability while retaining maximum feature representation, becomes SE Kernel with nu -> infinity
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        mat_amp = self._hyperparameters[0]
        mat_hyp = self._hyperparameters[1]
        nu = self._constants[0]
        if nu < np.abs(der):
            raise ValueError('Matern nu parameter must be greater than requested derivative order.')
        nn = int(nu)
        rr = np.abs(x1 - x2)
        mpre = np.power(2.0,float(nn)) * np.math.factorial(nn) / np.math.factorial(2 * nn)
        zz = np.sqrt(2.0 * nu) * rr / mat_hyp
        covm = np.zeros(rr.shape)
        if der == 0:
            msum = 0.0
            for jj in np.arange(0,nn+1):
                sfac = np.math.factorial(nn + jj) / (np.power(2.0,float(jj)) * np.math.factorial(jj) * np.math.factorial(nn - jj))
                ssum = 0.0
                for mm in np.arange(0,der+1):
                    if (nn - jj - mm) >= 0:
                       dcmb = np.power(-1.0,der + mm) * np.math.factorial(der) / (np.math.factorial(mm) * np.math.factorial(der - mm))
                       dfac = np.math.factorial(nn - jj) / np.math.factorial(nn - jj - mm)
                       ssum = ssum + dcmb * dfac * np.power(zz,float(nn - jj - mm))
                msum = msum + sfac * ssum
            covm = mat_amp**2.0 * mpre * np.exp(-zz) * msum
        elif der == 1:
            drdx2 = np.sign(x2 - x1)
            msum = 0.0
            for jj in np.arange(0,nn+1):
                sfac = np.math.factorial(nn + jj) / (np.power(2.0,float(jj)) * np.math.factorial(jj) * np.math.factorial(nn - jj))
                ssum = 0.0
                for mm in np.arange(0,der+1):
                    if (nn - jj - mm) >= 0:
                       dcmb = np.power(-1.0,der + mm) * np.math.factorial(der) / (np.math.factorial(mm) * np.math.factorial(der - mm))
                       dfac = np.math.factorial(nn - jj) / np.math.factorial(nn - jj - mm)
                       ssum = ssum + dcmb * dfac * np.power(zz,float(nn - jj - mm))
                msum = msum + sfac * ssum
            covm = mat_amp**2.0 * drdx2 * mpre * np.exp(-zz) * msum
        elif der == -1:
            drdx1 = np.sign(x1 - x2)
            msum = 0.0
            for jj in np.arange(0,nn+1):
                sfac = np.math.factorial(nn + jj) / (np.power(2.0,float(jj)) * np.math.factorial(jj) * np.math.factorial(nn - jj))
                ssum = 0.0
                for mm in np.arange(0,der+1):
                    if (nn - jj - mm) >= 0:
                       dcmb = np.power(-1.0,der + mm) * np.math.factorial(der) / (np.math.factorial(mm) * np.math.factorial(der - mm))
                       dfac = np.math.factorial(nn - jj) / np.math.factorial(nn - jj - mm)
                       ssum = ssum + dcmb * dfac * np.power(zz,float(nn - jj - mm))
                msum = msum + sfac * ssum
            covm = mat_amp**2.0 * drdx1 * mpre * np.exp(-zz) * msum
        elif der == 2 or der == -2:
            drdx1 = np.sign(x1 - x2)
            drdx1[drdx1==0] = 1.0
            drdx2 = np.sign(x2 - x1)
            drdx2[drdx2==0] = -1.0
            msum = 0.0
            for jj in np.arange(0,nn+1):
                sfac = np.math.factorial(nn + jj) / (np.power(2.0,float(jj)) * np.math.factorial(jj) * np.math.factorial(nn - jj))
                ssum = 0.0
                for mm in np.arange(0,der+1):
                    if (nn - jj - mm) >= 0:
                       dcmb = np.power(-1.0,der + mm) * np.math.factorial(der) / (np.math.factorial(mm) * np.math.factorial(der - mm))
                       dfac = np.math.factorial(nn - jj) / np.math.factorial(nn - jj - mm)
                       ssum = ssum + dcmb * dfac * np.power(zz,float(nn - jj - mm))
                msum = msum + sfac * ssum
            covm = mat_amp**2.0 * drdx1 * drdx2 * mpre * np.exp(-zz) * msum
        else:
            raise NotImplementedError('Derivatives of order 3 or higher not yet implemented in '+self.get_name()+' kernel.')
        return covm

    def __init__(self,amp=0.1,ls=0.1,nu=2.5):
        hyps = np.zeros((2,))
        csts = np.zeros((1,))
        if type(amp) in (float,int) and float(amp) > 0.0:
            hyps[0] = float(amp)
        else:
            raise ValueError('Matern amplitude hyperparameter must be greater than 0.')
        if type(ls) in (float,int) and float(ls) != 0.0:
            hyps[1] = float(ls)
        else:
            raise ValueError('Matern hyperparameter cannot equal 0.')
        if type(nu) in (float,int) and float(nu) >= 0.0:
            csts[0] = float(int(nu)) + 0.5
        else:
            raise ValueError('Matern half-integer nu constant must be greater or equal to 0.')
        Kernel.__init__(self,"MH",self.__calc_covm,False,hyps,csts)

    def __copy__(self):
        mamp = float(self._hyperparameters[0])
        mhp = float(self._hyperparameters[1])
        nup = float(self._constants[0])
        kcopy = Matern_HI_Kernel(mamp,mhp,nup)
        return kcopy


class NN_Kernel(Kernel):
    """
    Neural Network Style Kernel: implements a sigmoid covariance function similar to a perceptron in a neural network, good for strong discontinuities
    User note: Suffers from high volatility like the Matern kernel, have not figured out how to localize impact of kernel to the features in data
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
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
        hyps = np.zeros((3,))
        if type(nna) in (float,int) and float(nna) > 0.0:
            hyps[0] = float(nna)
        else:
            raise ValueError('Neural network amplitude must be greater than 0.')
        if type(nno) in (float,int):
            hyps[1] = float(nno)
        else:
            raise ValueError('Neural network offset parameter must be a real number.')
        if type(nnv) in (float,int):
            hyps[2] = float(nnv)
        else:
            raise ValueError('Neural network hyperparameter must be a real number.')
        Kernel.__init__(self,"NN",self.__calc_covm,False,hyps)

    def __copy__(self):
        nnamp = float(self._hyperparameters[0])
        nnop = float(self._hyperparameters[1])
        nnhp = float(self._hyperparameters[2])
        kcopy = NN_Kernel(nnamp,nnop,nnhp)
        return kcopy


class GSE_GL_Kernel(Kernel):
    """
    Gibbs Kernel with Gaussian Length Scale Function: implements a Gibbs covariance function with variable length scale
    User note: This implementation uses a Gaussian function to define the length scale, but in practice,
               the function handle self._lfunc can be replaced to any function which produces only positive values
               and has an implementation of its first derivative via the "der" argument in the call command
    """
    def __calc_covm(self,x1,x2,der=0,hder=None):
        v_hyp = self._hyperparameters[0]
        lb_hyp = self._hyperparameters[1]
        lp_hyp = self._hyperparameters[2]
        lm_hyp = self._constants[0]
        ls_hyp = self._hyperparameters[3]
        l_hyp1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp)
        l_hyp2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp)
        rr = x1 - x2
        ll = np.power(l_hyp1,2.0) + np.power(l_hyp2,2.0)
        mm = l_hyp1 * l_hyp2
        covm = np.zeros(rr.shape)
        if der == 0:
            lder = 0
            if hder is None:
                covm = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder == 0:
                covm = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
            elif hder >= 1 and hder <= 3:
                ghder = hder - 1
                dlh1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                dlh2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                covm = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
        elif der == 1:
            lder = 1
            if hder is None:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder >= 1 and hder <= 3:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                dlh1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                dlh2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx2 / (2.0 * l_hyp2) - dldx2 * dlh2 / (2.0 * np.power(l_hyp2,2.0))
                dt2 = -dlh2 * dldx2 / ll - l_hyp2 * ddldx2 / ll + l_hyp2 * dldx2 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh2 * dldx2 + 2.0 * l_hyp2 * ddldx2 - 4.0 * l_hyp2 * dldx2 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx2 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == -1:
            lder = 1
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                kfac = v_hyp**2.0 * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx1 / (2.0 * l_hyp1)
                t2 = -l_hyp1 * dldx1 / ll
                t3 = 2.0 * l_hyp1 * dldx1 * np.power(rr / ll,2.0)
                t4 = -drdx1 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder == 0:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                covm = kfac * (t1 + t2 + t3 + t4)
            elif hder >= 1 and hder <= 3:
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der)
                kfac = 2.0 * v_hyp * np.sqrt(2.0 * mm / ll) * np.exp(-np.power(rr,2.0) / ll)
                t1 = dldx2 / (2.0 * l_hyp2)
                t2 = -l_hyp2 * dldx2 / ll
                t3 = 2.0 * l_hyp2 * dldx2 * np.power(rr / ll,2.0)
                t4 = -drdx2 * 2.0 * rr / ll
                dlh1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der,ghder)
                dlh2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der,ghder)
                c1 = np.sqrt(ll / (8.0 * mm)) * (2.0 * dmm / ll - 2.0 * mm * dll / np.power(ll,2.0))
                c2 = np.sqrt(2.0 * mm / ll) * np.power(rr / ll,2.0) * dll
                dkfac = v_hyp**2.0 * (c1 + c2) * np.exp(-np.power(rr,2.0) / ll)
                dt1 = ddldx2 / (2.0 * l_hyp2) - dldx2 * dlh2 / (2.0 * np.power(l_hyp2,2.0))
                dt2 = -dlh2 * dldx2 / ll - l_hyp2 * ddldx2 / ll + l_hyp2 * dldx2 * dll / np.power(ll,2.0)
                dt3 = (2.0 * dlh2 * dldx2 + 2.0 * l_hyp2 * ddldx2 - 4.0 * l_hyp2 * dldx2 * dll / ll) * np.power(rr / ll,2.0)
                dt4 = drdx2 * 2.0 * rr * dll / np.power(ll,2.0)
                covm = dkfac * (t1 + t2 + t3 + t4) + kfac * (dt1 + dt2 + dt3 + dt4)
        elif der == 2 or der == -2:
            lder = 1
            if hder is None:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
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
                dldx1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
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
            elif hder >= 1 and hder <= 3:
                drdx1 = np.ones(rr.shape)
                dldx1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                drdx2 = -np.ones(rr.shape)
                dldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder)
                dd = dldx1 * dldx2
                ii = drdx1 * rr * dldx2 / l_hyp2 + drdx2 * rr * dldx1 / l_hyp1
                jj = drdx1 * rr * dldx2 * l_hyp2 + drdx2 * rr * dldx1 * l_hyp1
                dlh1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der,ghder)
                dlh2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der,ghder)
                dmm = dlh1 * l_hyp2 + l_hyp1 * dlh2
                dll = 2.0 * dlh1 + 2.0 * dlh2
                ddldx1 = self._lfunc(x1,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
                ddldx2 = self._lfunc(x2,lb_hyp,lp_hyp,lm_hyp,ls_hyp,lder,ghder)
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

    def __gauss_ls(self,x,base=1.0,amp=1.0,mu=0.0,sig=1.0,der=0,ghder=None):
        maxfrac = 0.6
        hh = amp if amp < (maxfrac * base) else maxfrac * base
        ls = np.ones(x.shape) * base
        if der == 0:
            if ghder is None:
                ls = base - hh * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 0:
                ls = np.ones(x.shape)
            elif ghder == 1:
                ls = -np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 2:
                ls = -hh * (np.power(x - mu,2.0) / sig**3.0) * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
        elif der == 1 or der == -1:
            if ghder is None:
                ls = hh * (x - mu) / sig**2.0 * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 0:
                ls = np.zeros(x.shape)
            elif ghder == 1:
                ls = (x - mu) / sig**2.0 * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 2:
                t1 = -2.0 * (x - mu) / sig
                t2 = np.power((x - mu) / sig,3.0)
                ls = (t1 + t2) * hh / sig**2.0 * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
        elif der == 2 or der == -2:
            if ghder is None:
                t1 = 1.0
                t2 = -np.power((x - mu) / sig,2.0) 
                ls = (t1 + t2) * hh / sig**2.0 * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 0:
                ls = np.zeros(x.shape)
            elif ghder == 1:
                t1 = 1.0
                t2 = -np.power((x - mu) / sig,2.0)
                ls = (t1 + t2) / sig**2.0 * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
            elif ghder == 2:
                t1 = -2.0 / sig**3.0
                t2 = 4.0 * np.power(x - mu,2.0) / sig**5.0
                t3 = (1.0 - np.power((x - mu) / sig,2.0)) * np.power(x - mu,2.0) / sig**3.0
                ls = (t1 + t2 + t3) * np.exp(-np.power(x - mu,2.0) / (2.0 * sig**2.0))
        return ls

    def evaluate_lsf(self,xx,der=0):
        v_hyp = self._hyperparameters[0]
        lb_hyp = self._hyperparameters[1]
        lp_hyp = self._hyperparameters[2]
        lm_hyp = self._constants[0]
        ls_hyp = self._hyperparameters[3]
        lsf = self._lfunc(xx,lb_hyp,lp_hyp,lm_hyp,ls_hyp,der)
        return lsf

    def set_lsf_mu(self,lm=1.0):
        if type(lm) in (float,int):
            self._constants[0] = float(lm)
        else:
            raise ValueError('Length scale function exponential mu hyperparameter must be a real number.')

    def __init__(self,var=1.0,lb=1.0,gh=0.5,lm=0.0,lsig=1.0):
        self._lfunc = self.__gauss_ls
        hyps = np.zeros((4,))
        csts = np.zeros((1,))
        if type(var) in (float,int):
            hyps[0] = float(var)
        else:
            raise ValueError('Constant hyperparameter must be a real number.')
        if type(lb) in (float,int) and float(lb) > 0.0:
            hyps[1] = float(lb)
        else:
            raise ValueError('Length scale function base hyperparameter must be greater than 0.')
        if type(gh) in (float,int) and float(gh) > 0.0:
            hyps[2] = float(gh)
        else:
            raise ValueError('Length scale function peak hyperparameter must be greater than 0.')
        if type(lm) in (float,int):
            csts[0] = float(lm)
        else:
            raise ValueError('Length scale function exponential mu hyperparameter must be a real number.')
        if type(lsig) in (float,int) and float(lsig) > 0.0:
            hyps[3] = float(lsig)
        else:
            raise ValueError('Length scale function sigma hyperparameter must be greater than 0.')
        Kernel.__init__(self,"GGL",self.__calc_covm,True,hyps,csts)

    def __copy__(self):
        chp = float(self._hyperparameters[0])
        lbhp = float(self._hyperparameters[1])
        lphp = float(self._hyperparameters[2])
        lmhp = float(self._constants[0])
        lshp = float(self._hyperparameters[3])
        kcopy = GSE_GL_Kernel(chp,lbhp,lphp,lmhp,lshp)
        return kcopy


def Kernel_Constructor(name):
    """
    Function to construct a basic kernel solely based on the kernel codename
    """
    kernel = None
    if type(name) is str:
        if re.match('C',name):
            kernel = Constant_Kernel()
        elif re.match('n',name):
            kernel = Noise_Kernel()
        elif re.match('L',name):
            kernel = Linear_Kernel()
        elif re.match('P',name):
            kernel = Poly_Order_Kernel()
        elif re.match('SE',name):
            kernel = SE_Kernel()
        elif re.match('RQ',name):
            kernel = RQ_Kernel()
        elif re.match('MH',name):
            kernel = Matern_HI_Kernel()
        elif re.match('NN',name):
            kernel = NN_Kernel()
        elif re.match('GGL',name):
            kernel = GSE_GL_Kernel()
    return kernel


def Recursive_Kernel_Constructor(name):
    """
    Function to construct a complex kernel solely based on the kernel codename
    """
    kernel = None
    if type(name) is str:
        m = re.search(r'^(.*?)_(.*)$',name)
        if m:
            names = m.group(2).split('-')
            kklist = []
            for ii in np.arange(0,len(names)):
                kklist.append(Recursive_Kernel_Constructor(names[ii]))
            if re.search('Sum',m.group(1)):
                kernel = Sum_Kernel(klist=kklist)
            elif re.search('Prod',m.group(1)):
                kernel = Product_Kernel(klist=kklist)
            elif re.search('Sym',m.group(1)):
                kernel = Symmetric_Kernel(klist=kklist)
        else:
            kernel = Kernel_Constructor(name)
    return kernel


def Kernel_Reconstructor(name,pars=None,log=False):
    """
    Function to reconstruct any kernel from its kernel codename and parameter list,
    useful for saving only necessary data to represent a GPR1D object
    """
    kernel = Recursive_Kernel_Constructor(name)
    pvec = None
    if type(pars) in (list,tuple):
        pvec = np.array(pars).flatten()
    elif type(pars) is np.ndarray:
        pvec = pars.flatten()
    if isinstance(kernel,Kernel) and pvec is not None:
        nhyp = kernel.get_hyperparameters().size
        ncst = kernel.get_constants().size
        if pvec.size >= nhyp:
            theta = pvec[:nhyp] if pvec.size > nhyp else pvec.copy()
            kernel.set_hyperparameters(theta,log=log)
        if ncst > 0 and pvec.size >= (nhyp + ncst):
            csts = pvec[nhyp:nhyp+ncst] if pvec.size > (nhyp + ncst) else pvec[nhyp:]
            kernel.set_constants(csts)
    return kernel


class GPR1D():
    """
    Class containing variable containers, get/set functions, and fitting functions required to perform a 1-dimensional GPR fit
    User note: This implementation requires the specific implementation of the Kernel class, provided in the same file!
    """
    def __init__(self):
        """
        Defines the input and output containers used within the class, requires instantiation
        """
        self.kk = None
        self.kb = None
        self.lp = 1.0
        self.xx = None
        self.xe = None
        self.yy = None
        self.ye = None
        self.dxx = None
        self.dyy = None
        self.dye = None
        self.eps = None
        self.slh = 0.005
        self.dlh = 0.01
        self.lb = None
        self.ub = None
        self.cn = None
        self.ekk = None
        self.ekb = None
        self.elp = None
        self.enr = 5
        self.esflag = True
        self._ikk = None
        self._xF = None
        self._barF = None
        self._varF = None
        self._dbarF = None
        self._dvarF = None
        self._lml = None
        self._eflag = False
        self._varN = None
        self._dvarN = None
        self._nye = None


    def set_kernel(self,kernel=None,kbounds=None,regpar=None):
        """
        Specify the kernel that the Gaussian process regression will be performed with
        """
        if isinstance(kernel,Kernel):
            self.kk = copy.copy(kernel)
            self._ikk = copy.copy(self.kk)
        if isinstance(self.kk,Kernel):
            kh = self.kk.get_hyperparameters(log=True)
            if type(kbounds) in (list,tuple,np.ndarray):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb.T) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self.kb = kb
        if type(regpar) in (float,int) and float(regpar) > 0.0:
            self.lp = float(regpar)


    def set_raw_data(self,xdata=None,ydata=None,xerr=None,yerr=None,dxdata=None,dydata=None,dyerr=None):
        """
        Specify the raw data that the Gaussian process regression will be performed on
        Performs some consistency checks between the input raw data to ensure validity
        """
        if type(xdata) in (list,tuple) and len(xdata) > 0:
            self.xx = np.array(xdata).flatten()
            self._eflag = False
        elif type(xdata) is np.ndarray and xdata.size > 0:
            self.xx = xdata.flatten()
            self._eflag = False
        if type(xerr) in (list,tuple) and len(xerr) > 0:
            self.xe = np.array(xerr).flatten()
        elif type(xerr) is np.ndarray and xerr.size > 0:
            self.xe = xerr.flatten()
        elif type(xerr) is str:
            self.xe = None
        if type(ydata) in (list,tuple) and len(ydata) > 0:
            self.yy = np.array(ydata).flatten()
        elif type(ydata) is np.ndarray and ydata.size > 0:
            self.yy = ydata.flatten()
        if type(yerr) in (list,tuple) and len(yerr) > 0:
            self.ye = np.array(yerr).flatten()
            self._eflag = False
        elif type(yerr) is np.ndarray and yerr.size > 0:
            self.ye = yerr.flatten()
            self._eflag = False
        elif type(yerr) is str:
            self.ye = None
            self._eflag = False
        if type(dxdata) in (list,tuple) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self.dxx = temp.flatten()
        elif type(dxdata) is np.ndarray and dxdata.size > 0:
            self.dxx = dxdata.flatten()
        elif type(dxdata) is str:
            self.dxx = None
        if type(dydata) in (list,tuple) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self.dyy = temp.flatten()
        elif type(dydata) is np.ndarray and dydata.size > 0:
            self.dyy = dydata.flatten()
        elif type(dydata) is str:
            self.dyy = None
        if type(dyerr) in (list,tuple) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            self.dye = temp.flatten()
        elif type(dyerr) is np.ndarray and dyerr.size > 0:
            self.dye = dyerr.flatten()
        elif type(dyerr) is str:
            self.dye = None


    def set_conditioner(self,condnum=None,lbound=None,ubound=None):
        """
        Specify the parameters to ensure the condition number of the matrix is good,
        as well as set upper and lower bounds for the input data to be included
        """
        if type(condnum) in (float,int) and condnum > 0.0:
            self.cn = float(condnum)
        elif type(condnum) in (float,int) and condnum <= 0.0:
            self.cn = None
        elif type(condnum) is str:
            self.cn = None
        if type(lbound) in (float,int):
            self.lb = float(lbound)
        elif type(lbound) is str:
            self.lb = None
        if type(ubound) in (float,int):
            self.ub = float(ubound)
        elif type(ubound) is str:
            self.ub = None


    def set_error_kernel(self,kernel=None,kbounds=None,regpar=None,nrestarts=None,searchflag=True):
        """
        Specify the kernel that the Gaussian process regression on the error function 
        will be performed with
        """
        if isinstance(kernel,Kernel):
            self.ekk = copy.copy(kernel)
            self._eflag = False
        if isinstance(self.ekk,Kernel):
            kh = self.ekk.get_hyperparameters(log=True)
            if type(kbounds) in (list,tuple,np.ndarray):
                kb = np.atleast_2d(kbounds)
                if np.any(np.isnan(kb.flatten())) or np.any(np.invert(np.isfinite(kb.flatten()))) or np.any(kb.flatten() <= 0.0) or len(kb.shape) > 2:
                    kb = None
                elif kb.shape[0] == 2:
                    kb = np.log10(kb.T) if kb.shape[1] == kh.size else None
                elif kb.shape[1] == 2:
                    kb = np.log10(kb) if kb.shape[0] == kh.size else None
                else:
                    kb = None
                self.ekb = kb
                self._eflag = False
        if type(regpar) in (float,int) and float(regpar) > 0.0:
            self.elp = float(regpar)
            self._eflag = False
        if type(nrestarts) in (float,int):
            self.enr = int(nrestarts) if int(nrestarts) > 0 else 0
        self.esflag = True if searchflag else False


    def set_search_parameters(self,epsilon=None,sgain=None,sdiff=None):
        """
        Specify the search parameters that the Gaussian process regression will use
        Performs some consistency checks on input values to ensure validity
        """
        if type(epsilon) in (float,int) and float(epsilon) > 0.0:
            self.eps = float(epsilon)
        elif type(epsilon) in (float,int) and float(epsilon) <= 0.0:
            self.eps = None
        elif type(epsilon) is str:
            self.eps = None
        if type(sgain) in (float,int) and float(sgain) > 0.0:
            self.slh = float(sgain)
        if type(sdiff) in (float,int) and float(sdiff) > 0.0:
            self.dlh = float(sdiff)


    def get_gp_x(self):
        """
        Returns the x-values used in the latest GPRFit() call
        """
        return self._xF


    def get_gp_mean(self):
        """
        Returns the y-values computed in the latest GPRFit() call
        """
        return self._barF


    def get_gp_variance(self,noise_flag=True):
        """
        Returns the full covariance matrix of the y-values computed in the latest
        GPRFit() call
        """
        varF = self._varF
        if varF is not None and self._varN is not None and noise_flag:
            varF = varF + self._varN
        return varF


    def get_gp_std(self,noise_flag=True):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the y-values
        computed in the latest GPRFit() call
        """
        sigF = None
        varF = self.get_gp_variance(noise_flag=noise_flag)
        if varF is not None:
            sigF = np.sqrt(np.diag(varF))
        return sigF


    def get_gp_drv_mean(self):
        """
        Returns the dy/dx-values computed in the latest GPRFit() call
        """
        return self._dbarF


    def get_gp_drv_variance(self,noise_flag=True):
        """
        Returns the full covariance matrix of the dy/dx-values computed in the latest
        GPRFit() call
        """
        dvarF = self._dvarF
        if dvarF is not None and self._dvarN is not None and noise_flag:
            dvarF = dvarF + self._dvarN
        return dvarF


    def get_gp_drv_std(self,noise_flag=True):
        """
        Returns only the rooted diagonal elements of the covariance matrix of the 
        dy/dx-values computed in the latest GPRFit() call
        """
        dsigF = None
        dvarF = self.get_gp_drv_variance(noise_flag=noise_flag)
        if dvarF is not None:
            dsigF = np.sqrt(np.diag(dvarF))
        return dsigF


    def get_gp_results(self,rtn_cov=False,noise_flag=True):
        """
        Returns tuple of (y-values,y-errors,dy/dx-values,dy/dx-errors) computed in
        the latest GPRFit() call
        """
        ra = self.get_gp_mean()
        rb = self.get_gp_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_std(noise_flag=noise_flag)
        rc = self.get_gp_drv_mean()
        rd = self.get_gp_drv_variance(noise_flag=noise_flag) if rtn_cov else self.get_gp_drv_std(noise_flag=noise_flag)
        return (ra,rb,rc,rd)


    def get_gp_lml(self):
        """
        Returns the log-marginal-likelihood of the latest GPRFit() call
        """
        return self._lml


    def get_gp_input_kernel(self):
        """
        Returns the original input kernel, with settings retained from before the 
        hyperparameter optimization step
        """
        return self._ikk


    def get_gp_kernel(self):
        """
        Returns the optimized kernel determined in the latest GPRFit() call
        """
        return self.kk


    def get_gp_kernel_details(self):
        """
        Returns tuple of (name,hyperparameters and constants) of the optimized kernel
        determined in the latest GPRFit() call
        """
        kname = None
        kpars = None
        if isinstance(self.kk,Kernel):
            kname = self.kk.get_name()
            kpars = np.hstack((self.kk.get_hyperparameters(log=False),self.kk.get_constants()))
        return (kname,kpars)


    def get_error_kernel(self):
        """
        Returns the optimized error kernel determined in the latest GPRFit() call
        """
        return self.ekk


    def get_gp_error_kernel_details(self):
        """
        Returns tuple of (name,hyperparameters and constants) of the optimized error kernel
        determined in the latest GPRFit() call
        """
        kname = None
        kpars = None
        if isinstance(self.ekk,Kernel):
            kname = self.ekk.get_name()
            kpars = np.hstack((self.ekk.get_hyperparameters(log=False),self.ekk.get_constants()))
        return (kname,kpars)


    def get_error_function(self,xnew):
        """
        Returns the error values used in heteroscedastic GPR, evaluated at the input x-values,
        using the error kernel determined in the latest GPRFit() call
        """
        xn = None
        if type(xnew) in (list,tuple) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif type(xnew) is np.ndarray and xnew.size > 0:
            xn = xnew.flatten()
        ye = self.ye if self._nye is None else self._nye
        barE = None
        if xn is not None and ye is not None and self._eflag:
            barE = itemgetter(0)(self.basic_fit(xn,kernel=self.ekk,ydata=ye,yerr=0.1*ye,epsilon='None'))
        return barE


    def __gp_base_alg(self,xn,kk,lp,xx,yy,ye,dxx,dyy,dye,dd):
        """
        Bare-bones algorithm for gaussian process regression, no idiot-proofing, no pre- or post-processing
        Note that it is recommended that covf be a Kernel object from kernfunc module
        but it can be, in essence, any passed object which can be called with arguments:
            (x1,x2,derivative_order) with (x1,x2) being a meshgrid
        and returns an array with shape of x1 and/or x2 (these should have identical shape)
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
        Bare-bones algorithm for brute-force first-order derivative of gaussian process regression (single input dimension)
        Not recommended for large training sets, but useful for testing custom Kernel objects which have hard-coded derivative calculations
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


    def __gp_grad_ascent(self,kk,lp,xx,yy,ye,dxx,dyy,dye,eps,slh,dlh):
        """
        Gradient ascent hyperparameter searching algorithm, searches hyperparameters in log-space and parameters in linear-space
        Note that it is currently limited to 500 attempts to achieve the desired convergence criteria
        It may be important to adjust these depending on the data given to the GP
            eps = desired convergence criteria
            dlh = the step size used to calculate the gradient
            slh = the gain factor on gradient to choose next step
        Message generated when the max. iterations is reached without desired convergence, though result is not necessarily bad
        """
        # Set up the required data for performing the gradient ascent search
        xn = np.array([0.0])    # Reduction of prediction vector for speed bonus
        newkk = copy.copy(kk)
        theta_base = kk.get_hyperparameters(log=True)
        gradtheta = np.zeros(theta_base.shape)
        theta_old = theta_base.copy()
        lmlold = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
        lmlnew = 0.0
        dlml = np.abs(lmlold - lmlnew)
        icount = 0
        itermax = 500
        while dlml > eps and icount < itermax:
            if newkk.is_hderiv_implemented():
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

                # Algorithm, see theory (located in book specified at top of file) for details
                KKb = newkk(x1,x2,der=0)
                KKh1 = newkk(x1h1,x2h1,der=1)
                KKh2 = newkk(x1h2,x2h2,der=-1)
                KKd = newkk(x1d,x2d,der=2)
                KK = np.vstack((np.hstack((KKb,KKh2)),np.hstack((KKh1,KKd))))
                LL = spla.cholesky(KK + np.diag(yef**2.0),lower=True)
                alpha = spla.cho_solve((LL,True),yf)
                for ii in np.arange(0,theta_base.size):
                    HHb = newkk(x1,x2,der=0,hder=ii)
                    HHh1 = newkk(x1h1,x2h1,der=1,hder=ii)
                    HHh2 = newkk(x1h2,x2h2,der=-1,hder=ii)
                    HHd = newkk(x1d,x2d,der=2,hder=ii)
                    HH = np.vstack((np.hstack((HHb,HHh2)),np.hstack((HHh1,HHd))))
                    PP = np.dot(alpha.T,HH)
                    QQ = spla.cho_solve((LL,True),HH)
                    dlml = 0.5 * np.dot(PP,alpha) - 0.5 * np.sum(np.diag(QQ)) 
                    gradtheta[ii] = dlml
            else:
                for ii in np.arange(0,theta_base.size):
                    testkk = copy.copy(kk)
                    theta_in = theta_old.copy()
                    theta_in[ii] = theta_old[ii] - 0.5 * dlh
                    testkk.set_hyperparameters(theta_in,log=True)
                    llml = itemgetter(2)(self.__gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
                    theta_in[ii] = theta_old[ii] + 0.5 * dlh
                    testkk.set_hyperparameters(theta_in,log=True)
                    ulml = itemgetter(2)(self.__gp_base_alg(xn,testkk,lp,xx,yy,ye,dxx,dyy,dye,0))
                    gradtheta[ii] = (ulml - llml) / dlh
            theta_new = theta_old + slh * gradtheta
            newkk.set_hyperparameters(theta_new,log=True)
            lmlnew = itemgetter(2)(self.__gp_base_alg(xn,newkk,lp,xx,yy,ye,dxx,dyy,dye,0))
            dlml = np.abs(lmlold - lmlnew)
            theta_old = theta_new.copy()
            lmlold = lmlnew
            icount = icount + 1
        if icount == itermax:
            print('   Maximum number of iterations performed on gradient ascent search.')
        return (newkk,lmlnew)


    def __condition_data(self,xx,xe,yy,ye,lb,ub,cn):
        """
        Conditions the input data to remove data points which are too close together, as
        defined by the user, and data points that are outside user-defined bounds
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


    def basic_fit(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,dxdata=None,dydata=None,dyerr=None,epsilon=None,sgain=None,sdiff=None,do_drv=False,rtn_cov=False):
        """
        Basic GP regression fitting routine, RECOMMENDED to call this instead of the bare-bones functions
        as this applies additional input checking. Note that this function does NOT strictly use class data!!!
        """
        xn = None
        kk = self.kk
        lp = self.lp
        xx = self.xx
        yy = self.yy
        ye = self.ye if self._nye is None else self._nye
        dxx = self.dxx
        dyy = self.dyy
        dye = self.dye
        eps = self.eps
        slh = self.slh
        dlh = self.dlh
        lb = -1.0e50 if self.lb is None else self.lb
        ub = 1.0e50 if self.ub is None else self.ub
        cn = 5.0e-3 if self.cn is None else self.cn
        if type(xnew) in (list,tuple) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif type(xnew) is np.ndarray and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,Kernel):
            kk = copy.copy(kernel)
        if type(regpar) in (float,int) and float(regpar) > 0.0:
            self.lp = float(regpar)
        if type(xdata) in (list,tuple) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif type(xdata) is np.ndarray and xdata.size > 0:
            xx = xdata.flatten()
        if type(ydata) in (list,tuple) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif type(ydata) is np.ndarray and ydata.size > 0:
            yy = ydata.flatten()
        if type(yerr) in (list,tuple) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif type(yerr) is np.ndarray and yerr.size > 0:
            ye = yerr.flatten()
        elif type(yerr) is str:
            ye = None
        if type(dxdata) in (list,tuple) and len(dxdata) > 0:
            temp = np.array([])
            for item in dxdata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dxx = temp.flatten()
        elif type(dxdata) is np.ndarray and dxdata.size > 0:
            dxx = dxdata.flatten()
        elif type(dxdata) is str:
            dxx = None
        if type(dydata) in (list,tuple) and len(dydata) > 0:
            temp = np.array([])
            for item in dydata:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dyy = temp.flatten()
        elif type(dydata) is np.ndarray and dydata.size > 0:
            dyy = dydata.flatten()
        elif type(dydata) is str:
            dyy = None
        if type(dyerr) in (list,tuple) and len(dyerr) > 0:
            temp = np.array([])
            for item in dyerr:
                temp = np.append(temp,item) if item is not None else np.append(temp,np.NaN)
            dye = temp.flatten()
        elif type(dyerr) is np.ndarray and dyerr.size > 0:
            dye = dyerr.flatten()
        elif type(dyerr) is str:
            dye = None
        if type(epsilon) in (float,int) and float(epsilon) > 0.0:
            eps = float(epsilon)
        elif type(epsilon) in (float,int) and float(epsilon) <= 0.0:
            eps = None
        elif type(epsilon) is str:
            eps = None
        if type(sgain) in (float,int) and float(sgain) > 0.0:
            slh = float(sgain)
        if type(sdiff) in (float,int) and float(sdiff) > 0.0:
            dlh = float(sdiff)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,Kernel):
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
                (nkk,lml) = self.__gp_grad_ascent(nkk,lp,xx,yy,ye,dxx,dyy,dye,eps,slh,dlh)
            (barF,varF,lml) = self.__gp_base_alg(xn,nkk,lp,xx,yy,ye,dxx,dyy,dye,dd)
            barF = barF * sc if do_drv else barF * sc + myy
            varF = varF * sc**2.0
            errF = varF if rtn_cov else np.sqrt(np.diag(varF))
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')
        return (barF,errF,lml,nkk)


    def brute_derivative(self,xnew,kernel=None,regpar=None,xdata=None,ydata=None,yerr=None,rtn_cov=False):
        """
        Brute-force numerical GP regression derivative routine, RECOMMENDED to call this instead of bare-bones functions above
        Kept for ability to convince user of validity of regular GP derivative, but can also be wildly wrong on some data due to numerical errors
        RECOMMENDED to use derivative flag on basic_fit() function, as it was tested and seems to be more robust, provided kernels are properly defined
        """
        xn = None
        kk = self.kk
        lp = self.lp
        xx = self.xx
        yy = self.yy
        ye = self.ye if self._nye is None else self._nye
        lb = -1.0e50 if self.lb is None else self.lb
        ub = 1.0e50 if self.ub is None else self.ub
        cn = 5.0e-3 if self.cn is None else self.cn
        if type(xnew) in (list,tuple) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif type(xnew) is np.ndarray and xnew.size > 0:
            xn = xnew.flatten()
        if isinstance(kernel,Kernel):
            kk = copy.copy(kernel)
        if type(regpar) in (float,int) and float(regpar) > 0.0:
            self.lp = float(regpar)
        if type(xdata) in (list,tuple) and len(xdata) > 0:
            xx = np.array(xdata).flatten()
        elif type(xdata) is np.ndarray and xdata.size > 0:
            xx = xdata.flatten()
        if type(ydata) in (list,tuple) and len(ydata) > 0:
            yy = np.array(ydata).flatten()
        elif type(ydata) is np.ndarray and ydata.size > 0:
            yy = ydata.flatten()
        if type(yerr) in (list,tuple) and len(yerr) > 0:
            ye = np.array(yerr).flatten()
        elif type(yerr) is np.ndarray and yerr.size > 0:
            ye = yerr.flatten()
        if ye is None and yy is not None:
            ye = np.zeros(yy.shape)

        barF = None
        errF = None
        lml = None
        nkk = None
        if xx is not None and yy is not None and xx.size == yy.size and xn is not None and isinstance(kk,Kernel):
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


    def make_NIGP_errors(self,nrestarts):
        """
        Automatically-called function which returns a vector of modified y-errors based
        on input x-errors and a test model gradient. Note that this function does not
        iterate until the test model derivatives and actual fit derivatives are self-
        consistent!
        """
        # Check inputs
        nr = 0
        if type(nrestarts) in (float,int) and int(nrestarts) > 0:
            nr = int(nrestarts)

        if self.kk is not None and self.xe is not None and self.xx.size == self.xe.size:
            barF = None
            nkk = None
            xntest = np.array([0.0])
            if self.kb is not None and nr > 0:
                kkvec = []
                lmlvec = []
                tkk = copy.copy(self.kk)
                try:
                    (tlml,tkk) = itemgetter(2,3)(self.basic_fit(xntest))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except ValueError:
                    kkvec.append(None)
                    lmlvec.append(np.NaN)
                for ii in np.arange(0,nr):
                    theta = np.abs(self.kb[:,1] - self.kb[:,0]).flatten() * np.random.random_sample((self.kb.shape[0],)) + np.nanmin(self.kb,axis=1).flatten()
                    tkk.set_hyperparameters(theta,log=True)
                    try:
                        (tlml,tkk) = itemgetter(2,3)(self.basic_fit(xntest,kernel=tkk))
                        kkvec.append(copy.copy(tkk))
                        lmlvec.append(tlml)
                    except ValueError:
                        kkvec.append(None)
                        lmlvec.append(np.NaN)
                imax = np.where(lmlvec == np.nanmax(lmlvec))[0][0]
                (barF,nkk) = itemgetter(0,3)(self.basic_fit(xntest,kernel=kkvec[imax],epsilon=-1.0))
            else:
                (barF,nkk) = itemgetter(0,3)(self.basic_fit(xntest))
            if barF is not None and isinstance(nkk,Kernel):
                xntest = self.xx.copy() + 1.0e-8
                dbarF = itemgetter(0)(self.basic_fit(xntest,kernel=nkk,do_drv=True))
                nfilt = np.any([np.isnan(self.xe),np.isnan(self.ye)],axis=0)
                cxe = self.xe
                cxe[nfilt] = 0.0
                cye = self.ye
                cye[nfilt] = 0.0
                self._nye = np.sqrt(cye**2.0 + cxe * dbarF**2.0)
        else:
            raise ValueError('Check input x-errors to make sure they are valid.')


    def GPRFit(self,xnew,nigp_flag=False,nrestarts=None):
        """
        Main GP regression fitting routine, RECOMMENDED to call this after using set functions instead of basic_fit()
        as this adapts the method based on inputs, performs 1st derivative and saves output to class variables

        Includes implementation of Monte Carlo kernel restarts within the user-defined bounds, via nrestarts argument
        Includes implementation of Heteroscedastic Output Noise, requires setting of error kernel before fitting
            For details, see article: K. Kersting, 'Most Likely Heteroscedastic Gaussian Process Regression' (2007)
        Includes implementation of Noisy-Input Gaussian Process (NIGP) assuming Gaussian x-error, via nigp_flag argument
            For details, see article: A. McHutchon, C.E. Rasmussen, 'Gaussian Process Training with Input Noise' (2011)
            Developer note: Should this iterate until predicted derivative is consistent with the one
                            used to model impact of input noise?
        """
        # Check inputs
        xn = None
        nr = 0
        if type(xnew) in (list,tuple) and len(xnew) > 0:
            xn = np.array(xnew).flatten()
        elif type(xnew) is np.ndarray and xnew.size > 0:
            xn = xnew.flatten()
        if type(nrestarts) in (float,int) and int(nrestarts) > 0:
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
        hscflag = True if self.ye is not None else False

        if self.kk is not None and self.kb is not None and nr > 0:
            xntest = np.array([0.0])
            kkvec = []
            lmlvec = []
            tkk = copy.copy(self.kk)
            try:
                (tlml,tkk) = itemgetter(2,3)(self.basic_fit(xntest))
                kkvec.append(copy.copy(tkk))
                lmlvec.append(tlml)
            except (ValueError,np.linalg.linalg.LinAlgError):
                kkvec.append(None)
                lmlvec.append(np.NaN)
            for ii in np.arange(0,nr):
                theta = np.abs(self.kb[:,1] - self.kb[:,0]).flatten() * np.random.random_sample((self.kb.shape[0],)) + np.nanmin(self.kb,axis=1).flatten()
                tkk.set_hyperparameters(theta,log=True)
                try:
                    (tlml,tkk) = itemgetter(2,3)(self.basic_fit(xntest,kernel=tkk))
                    kkvec.append(copy.copy(tkk))
                    lmlvec.append(tlml)
                except (ValueError,np.linalg.linalg.LinAlgError):
                    kkvec.append(None)
                    lmlvec.append(np.NaN)
            imaxv = np.where(lmlvec == np.nanmax(lmlvec))[0]
            if len(imaxv) > 0:
                imax = imaxv[0]
                (barF,varF,lml,nkk) = self.basic_fit(xn,kernel=kkvec[imax],epsilon='None',rtn_cov=True)
            else:
                raise ValueError('None of the fit attempts converged. Please adjust kernel settings and try again.')
        else:
            (barF,varF,lml,nkk) = self.basic_fit(xn,rtn_cov=True)

        if barF is not None and isinstance(nkk,Kernel):
            barE = None
            dbarE = None
            ddbarE = None
            if hscflag:
                xntest = np.array([0.0])
                ye = self.ye if self._nye is None else self._nye
                if isinstance(self.ekk,Kernel) and self.ekb is not None and not self._eflag and self.esflag:
                    elp = self.elp if self.elp is not None else 6.0
                    ekk = copy.copy(self.ekk)
                    ekkvec = []
                    elmlvec = []
                    try:
                        (elml,ekk) = itemgetter(2,3)(self.basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=1.0e-3))
                        ekkvec.append(copy.copy(ekk))
                        elmlvec.append(elml)
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ekkvec.append(None)
                        elmlvec.append(np.NaN)
                    for jj in np.arange(0,self.enr):
                        etheta = np.abs(self.ekb[:,1] - self.ekb[:,0]).flatten() * np.random.random_sample((self.ekb.shape[0],)) + np.nanmin(self.ekb,axis=1).flatten()
                        ekk.set_hyperparameters(etheta,log=True)
                        try:
                            (elml,ekk) = itemgetter(2,3)(self.basic_fit(xntest,kernel=ekk,regpar=elp,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=1.0e-3))
                            ekkvec.append(copy.copy(ekk))
                            elmlvec.append(elml)
                        except (ValueError,np.linalg.linalg.LinAlgError):
                            ekkvec.append(None)
                            elmlvec.append(np.NaN)
                    eimaxv = np.where(elmlvec == np.nanmax(elmlvec))[0]
                    if len(eimaxv) > 0:
                        eimax = eimaxv[0]
                        self.ekk = copy.copy(ekkvec[eimax])
                        self._eflag = True
                    else:
                        raise ValueError('None of the error fit attempts converged. Please change error kernel settings and try again.')
                elif not self._eflag and self.esflag:
                    ekk = Noise_Kernel(float(np.mean(ye)))
                    (elml,ekk) = itemgetter(2,3)(self.basic_fit(xntest,kernel=ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon=1.0e-3))
                    self.ekk = copy.copy(ekk)
                    self._eflag = True
                barE = itemgetter(0)(self.basic_fit(xn,kernel=self.ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',epsilon='None'))
                if barE is not None:
                    dbarE = itemgetter(0)(self.basic_fit(xn,kernel=self.ekk,ydata=ye,yerr=0.1*ye,dxdata='None',dydata='None',dyerr='None',do_drv=True))
                    nxn = np.linspace(np.nanmin(xn),np.nanmax(xn),1000)
                    ddx = np.nanmin(np.diff(nxn)) * 1.0e-2
                    xnl = nxn - 0.5 * ddx
                    xnu = nxn + 0.5 * ddx
                    dbarEl = itemgetter(0)(self.basic_fit(xnl,kernel=self.ekk,ydata=ye,yerr=0.1*ye,do_drv=True))
                    dbarEu = itemgetter(0)(self.basic_fit(xnu,kernel=self.ekk,ydata=ye,yerr=0.1*ye,do_drv=True))
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
            self._varN = np.diag(np.power(barE,2.0)) if barE is not None else None
            self._lml = lml
            self.kk = copy.copy(nkk) if isinstance(nkk,Kernel) else None
            (dbarF,dvarF) = itemgetter(0,1)(self.basic_fit(xn,do_drv=True,rtn_cov=True))
            self._dbarF = dbarF.copy() if dbarF is not None else None
            self._dvarF = dvarF.copy() if dvarF is not None else None
#            self._dvarN = dvarF + np.diag(np.power(dbarE,2.0)) if dvarF is not None and dbarE is not None else None
#            ddfac = np.sqrt(np.mean(np.power(ddbarE,2.0))) if ddbarE is not None else 0.0
            ddfac = np.abs(ddbarE)
            self._dvarN = np.diag(2.0 * (np.power(dbarE,2.0) + np.abs(barE * ddfac))) if barE is not None and dbarE is not None else None
        else:
            raise ValueError('Check GP inputs to make sure they are valid.')


    def sample_GP(self,nsamples,noise_flag=False,simple_out=False):
        """
        Samples Gaussian process posterior for predictive functions, returns n samples
        Can be used by user to check validity of mean and variance outputs of GPRFit()
        """
        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to sample the GP.')

        # Check inputs
        ns = 0
        if type(nsamples) in (float,int) and int(nsamples) > 0:
            ns = int(nsamples)

        samples = None
        if ns > 0:
            mu = self.get_gp_mean()
            var = self.get_gp_variance(noise_flag=noise_flag)
            for ii in np.arange(0,ns):
                syy = np.random.multivariate_normal(mu,var)
                samples = syy.copy() if samples is None else np.vstack((samples,syy))
            if samples is not None and simple_out:
                mean = np.mean(samples,axis=0)
                std = np.std(samples,axis=0)
                samples = np.vstack((mean,std))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')
        return samples


    def MCMC_posterior_sampling(self,nsamples):
        """
        Performs Monte Carlo Markov chain based posterior analysis over hyperparameters, using LML as the likelihood
        """
        # Check instantiation of output class variables
        if self._xF is None or self._barF is None or self._varF is None:
            raise ValueError('Run GPRFit() before attempting to use MCMC posterior sampling.')

        # Check inputs
        ns = 0
        if type(nsamples) in (float,int) and int(nsamples) > 0:
            ns = int(nsamples)
        
        sbarM = None
        ssigM = None
        sdbarM = None
        sdsigM = None
        if isinstance(self.kk,Kernel) and ns > 0:
            olml = self._lml
            otheta = self.kk.get_hyperparameters(log=True)
            tlml = olml
            theta = otheta.copy()
            step = np.ones(theta.shape)
            flagvec = [True] * theta.size
            for ihyp in np.arange(0,theta.size):
                xntest = np.array([0.0])
                iflag = flagvec[ihyp]
                while iflag:
                    tkk = copy.copy(self.kk)
                    theta_step = np.zeros(theta.shape)
                    theta_step[ihyp] = step[ihyp]
                    theta_new = theta + theta_step
                    tkk.set_hyperparameters(theta_new,log=True)
                    ulml = None
                    try:
                        ulml = itemgetter(2)(self.basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        ulml = tlml - 3.0
                    theta_new = theta - theta_step
                    tkk.set_hyperparameters(theta_new,log=True)
                    llml = None
                    try:
                        llml = itemgetter(2)(self.basic_fit(xntest,kernel=tkk,epsilon='None'))
                    except (ValueError,np.linalg.linalg.LinAlgError):
                        llml = tlml - 3.0
                    if (ulml - tlml) >= -2.0 or (llml - tlml) >= -2.0:
                        iflag = False
                    else:
                        step[ihyp] = 0.5 * step[ihyp]
                flagvec[ihyp] = iflag
            nkk = copy.copy(self.kk)
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
                        nlml = itemgetter(2)(self.basic_fit(xntest,kernel=nkk,epsilon='None'))
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
                (barF,sigF,tlml,nkk) = self.basic_fit(xn,kernel=nkk,epsilon='None')
                sbarM = barF.copy() if sbarM is None else np.vstack((sbarM,barF))
                ssigM = sigF.copy() if ssigM is None else np.vstack((ssigM,sigF))
                (dbarF,dsigF) = itemgetter(0,1)(self.basic_fit(xn,kernel=nkk,epsilon='None',do_drv=True))
                sdbarM = dbarF.copy() if sdbarM is None else np.vstack((sdbarM,dbarF))
                sdsigM = dsigF.copy() if sdsigM is None else np.vstack((sdsigM,dsigF))
        else:
            raise ValueError('Check inputs to sampler to make sure they are valid.')
        return (sbarM,ssigM,sdbarM,sdsigM)
