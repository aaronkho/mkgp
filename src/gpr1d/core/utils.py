r'''
Some helpful functions and classes for reproducability and user-friendliness

These functions were developed by Aaron Ho [1].

[1] A. Ho, J. Citrin, C. Bourdelle, Y. Camenen, F. Felici, M. Maslov, K.L. Van De Plassche, H. Weisen, and JET Contributors
    IAEA Technical Meeting on Fusion Data Processing, Validation and Analysis, Boston, MA (2017)
    `<https://nucleus.iaea.org/sites/fusionportal/Shared\ Documents/Fusion\ Data\ Processing\ 2nd/31.05/Ho.pdf>`_

'''
# Required imports
import re
import numpy as np

np_itypes = (np.int8, np.int16, np.int32, np.int64)
np_utypes = (np.uint8, np.uint16, np.uint32, np.uint64)
np_ftypes = (np.float16, np.float32, np.float64)
number_types = (float, int, np_itypes, np_utypes, np_ftypes)
array_types = (list, tuple, np.ndarray)
default_dtype = np.float64

__all__ = [
    'KernelConstructor', 'KernelReconstructor',  # Kernel construction functions
]


def KernelConstructor(name):
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
            for jj in np.arange(0, len(links)):
                rname = links[jj] if not bflag else rname + '-' + links[jj]
                if re.search(r'\(', links[jj]):
                    bflag = True
                if re.search(r'\)', links[jj]):
                    bflag = False
                if not bflag:
                    names.append(rname)
            kklist = []
            for ii in np.arange(0, len(names)):
                kklist.append(KernelConstructor(names[ii]))
            if re.search(r'^Sum$', m.group(1)):
                kernel = Sum_Kernel(klist=kklist)
            elif re.search(r'^Prod$', m.group(1)):
                kernel = Product_Kernel(klist=kklist)
            elif re.search(r'^Sym$', m.group(1)):
                kernel = Symmetric_Kernel(klist=kklist)
        else:
            if re.search(r'^C$', name):
                kernel = Constant_Kernel()
            elif re.search(r'^n$', name):
                kernel = Noise_Kernel()
            elif re.search(r'^L$', name):
                kernel = Linear_Kernel()
            elif re.search(r'^P$', name):
                kernel = Poly_Order_Kernel()
            elif re.search(r'^SE$', name):
                kernel = SE_Kernel()
            elif re.search(r'^RQ$', name):
                kernel = RQ_Kernel()
            elif re.search(r'^MH$', name):
                kernel = Matern_HI_Kernel()
            elif re.search(r'^NN$', name):
                kernel = NN_Kernel()
            elif re.search(r'^Gw', name):
                wname = re.search(r'^Gw(.*)$', name).group(1)
                wfunc = None
                if re.search(r'^C$', wname):
                    wfunc = Constant_WarpingFunction()
                elif re.search(r'^IG$', wname):
                    wfunc = IG_WarpingFunction()
                kernel = Gibbs_Kernel(wfunc=wfunc)
    return kernel


def KernelReconstructor(name, pars=None):
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

    kernel = KernelConstructor(name)
    pvec = None
    if isinstance(pars, (list, tuple)):
        pvec = np.array(pars).flatten()
    elif isinstance(pars, np.ndarray):
        pvec = pars.flatten()
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

