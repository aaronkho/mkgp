r'''
Some helpful functions and classes for reproducability and user-friendliness
'''

# Required imports
import re
import numpy as np

from .definitions import number_types, array_types, default_dtype
from .kernels import (
    _Kernel,
    Constant_Kernel,
    Noise_Kernel,
    Linear_Kernel,
    Poly_Order_Kernel,
    SE_Kernel,
    RQ_Kernel,
    Matern_HI_Kernel,
    NN_Kernel,
    Gibbs_Kernel,
    Sum_Kernel,
    Product_Kernel,
    Symmetric_Kernel,
    Constant_WarpingFunction,
    IG_WarpingFunction,
)

__all__ = [
    'KernelConstructor', 'KernelReconstructor',  # Kernel construction functions
]


def KernelConstructor(name, dtype=None):
    r'''
    Function to construct a kernel solely based on the kernel codename.

    .. note::

        All :code:`_OperatorKernel` class implementations should use encapsulating
        round brackets to specify their constituents. (v >= 1.0.1)

    :arg name: str. The codename of the desired :code:`_Kernel` instance.

    :returns: object. The desired :code:`_Kernel` instance with default parameters. Returns :code:`None` if given kernel codename was invalid.
    '''

    kernel = None
    if isinstance(name, str):
        m = re.search(r'^(.*?)\((.*)\)$',name)
        if m:
            links = m.group(2).split('-')
            names = []
            bflag = False
            rname = ''
            for jj in range(len(links)):
                rname = links[jj] if not bflag else rname + '-' + links[jj]
                if re.search(r'\(', links[jj]):
                    bflag = True
                if re.search(r'\)', links[jj]):
                    bflag = False
                if not bflag:
                    names.append(rname)
            kklist = []
            for ii in range(len(names)):
                kklist.append(KernelConstructor(names[ii], dtype=dtype))
            if re.search(r'^Sum$', m.group(1)):
                kernel = Sum_Kernel(klist=kklist, dtype=dtype)
            elif re.search(r'^Prod$', m.group(1)):
                kernel = Product_Kernel(klist=kklist, dtype=dtype)
            elif re.search(r'^Sym$', m.group(1)):
                kernel = Symmetric_Kernel(klist=kklist, dtype=dtype)
        else:
            if re.search(r'^C$', name):
                kernel = Constant_Kernel(dtype=dtype)
            elif re.search(r'^n$', name):
                kernel = Noise_Kernel(dtype=dtype)
            elif re.search(r'^L$', name):
                kernel = Linear_Kernel(dtype=dtype)
            elif re.search(r'^P$', name):
                kernel = Poly_Order_Kernel(dtype=dtype)
            elif re.search(r'^SE$', name):
                kernel = SE_Kernel(dtype=dtype)
            elif re.search(r'^RQ$', name):
                kernel = RQ_Kernel(dtype=dtype)
            elif re.search(r'^MH$', name):
                kernel = Matern_HI_Kernel(dtype=dtype)
            elif re.search(r'^NN$', name):
                kernel = NN_Kernel(dtype=dtype)
            elif re.search(r'^Gw', name):
                wname = re.search(r'^Gw(.*)$', name).group(1)
                wfunc = None
                if re.search(r'^C$', wname):
                    wfunc = Constant_WarpingFunction(dtype=dtype)
                elif re.search(r'^IG$', wname):
                    wfunc = IG_WarpingFunction(dtype=dtype)
                kernel = Gibbs_Kernel(wfunc=wfunc, dtype=dtype)
    return kernel


def KernelReconstructor(name, pars=None, dtype=None):
    r'''
    Function to reconstruct any :code:`_Kernel` instance from its codename and parameter list,
    useful for saving only necessary data to represent a :code:`GaussianProcessRregression1D`
    instance.

    .. note::

        All :code:`_OperatorKernel` class implementations should use encapsulating
        round brackets to specify their constituents. (v >= 1.0.1)

    :arg name: str. The codename of the desired :code:`_Kernel` instance.

    :kwarg pars: array. The hyperparameter and constant values to be stored in the :code:`_Kernel` instance, order determined by the specific :code:`_Kernel` class implementation. (optional)

    :returns: object. The desired :code:`_Kernel` instance, with the supplied parameters already set if parameters were valid. Returns :code:`None` if given kernel codename was invalid.
    '''

    dt = dtype if dtype is not None else default_dtype
    kernel = KernelConstructor(name, dtype=dt)
    pvec = None
    if isinstance(pars, array_types):
        pvec = np.array(pars, dtype=dt).flatten()
    #elif isinstance(pars, np.ndarray):
    #    pvec = pars.flatten()
    if isinstance(kernel, _Kernel) and pvec is not None:
        nhyp = kernel.hyperparameters.size
        ncst = kernel.constants.size
        if ncst > 0 and pvec.size >= (nhyp + ncst):
            csts = pvec[nhyp:nhyp+ncst].copy() if pvec.size > (nhyp + ncst) else pvec[nhyp:].copy()
            kernel.constants = csts
        if pvec.size >= nhyp:
            theta = pvec[:nhyp].copy() if pvec.size > nhyp else pvec.copy()
            kernel.hyperparameters = theta
    return kernel


def diagonal(matrix, dtype=None):
    r'''
    Function to compute diagonal of N x [D x ...] x N matrix.

    :returns: array. The desired N x D matrix, which contains the diagonal of the inner matrix in dimension D.
    '''
    dt = dtype if dtype is not None else default_dtype
    mat = None
    diag = None
    if isinstance(matrix, array_types):
        mat = np.array(matrix, dtype=dt)
    if mat is not None and mat.ndim > 1 and mat.shape[0] == mat.shape[-1]:
        if mat.ndim > 2:
            diag = np.zeros((mat.shape[0], mat.shape[1]), dtype=dt)
            for ii in range(mat.shape[0]):
                for jj in range(mat.shape[1]):
                    index = [jj] * (mat.ndim - 2)
                    diag[ii, jj] = mat[ii, *index, ii]
        else:
            diag = np.diag(mat)
    return diag


def diagonalize(diagonal, full=True, dtype=None):
    r'''
    Function to compute diagonal matrix from N x D matrix.

    :returns: array. The desired N x [D x ...] x N matrix, which expands the given diagonal across the 2- or D-dimensional inner matrix.
    '''
    dt = dtype if dtype is not None else default_dtype
    diag = None
    mat = None
    if isinstance(diagonal, array_types):
        diag = np.array(diagonal, dtype=dt)
    if diag is not None and diag.ndim > 0 and diag.ndim <= 2:
        if diag.ndim == 2:
            ndim = diag.shape[-1] if full else 2
            dshape = [diag.shape[-1]] * ndim
            mat = np.zeros((diag.shape[0], *dshape, diag.shape[0]), dtype=dt)
            for ii in range(mat.shape[0]):
                for jj in range(mat.shape[1]):
                    element = [jj] * ndim
                    mat[ii, *element, ii] = diag[ii, jj]
        elif diag.ndim == 1:
            if full:
                dshape = [diag.shape[0]] * diag.shape[0]
                mat = np.zeros(dshape, dtype=dt)
                for ii in range(mat.shape[0]):
                    element = [ii] * mat.shape[0]
                    mat[*element] = diag[ii]
            else:
                mat = np.diag(diag)
    return mat
