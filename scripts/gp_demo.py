#!/usr/local/depot/Python-3.5.1/bin/python

import os
import sys
import re
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from GPR1D import GPR1D

pdir = './GPPlots/'
if not os.path.isdir(pdir):
    os.makedirs(pdir)

# Data generation

xsig = 0.01
ysig = 0.25
oslope = 1.0
xx = np.linspace(0.0,1.0,21)
yy = oslope * xx + 3.0
bnd1 = 0.3
nslope = 16.0
rfilt = (xx >= bnd1)
yy[rfilt] = yy[rfilt] - nslope * (xx[rfilt] - bnd1)
bnd2 = 0.7
rfilt = (xx >= bnd2)
yy[rfilt] = yy[rfilt] + (nslope + oslope) * (xx[rfilt] - bnd2)
nxx = xx + xsig * np.random.randn(xx.size)
yy = yy + ysig * np.random.randn(yy.size)
xe = np.full(xx.shape,xsig)
ye = np.full(yy.shape,ysig)

# Fitting

xn = np.linspace(0.0,1.0,100)
kernel = GPR1D.RQ_Kernel(1.0e0,1.0e0,1.0e0)
kbounds = np.atleast_2d([[1.0e-1,1.0e-2,1.0e0],[1.0e1,1.0e0,1.0e1]])
nkernel = GPR1D.Sum_Kernel(GPR1D.RQ_Kernel(1.0e0,1.0e0,1.0e0),GPR1D.Noise_Kernel(1.0e-2))
nkbounds = np.atleast_2d([[1.0e-1,1.0e-1,1.0e-1,1.0e-3],[1.0e1,1.0e0,1.0e1,1.0e-1]])
GPR = GPR1D.GPR1D()
GPR.set_kernel(kernel=kernel,kbounds=kbounds,regpar=1.0)
GPR.set_error_kernel(kernel=nkernel,kbounds=nkbounds,regpar=10.0)
GPR.set_raw_data(xdata=xx,ydata=yy,yerr=ye,xerr=xe,dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
GPR.set_search_parameters(epsilon=1.0e-3)
GPR.GPRFit(xn,nrestarts=5)
(gpname,gppars) = GPR.get_gp_kernel_details()
(egpname,egppars) = GPR.get_gp_error_kernel_details()
(vgp,egp,vdgp,edgp) = GPR.get_gp_results()

NIGPR = GPR1D.GPR1D()
NIGPR.set_kernel(kernel=kernel,kbounds=kbounds,regpar=1.0)
NIGPR.set_error_kernel(kernel=nkernel,kbounds=nkbounds,regpar=10.0)
NIGPR.set_raw_data(xdata=xx,ydata=yy,yerr=ye,xerr=xe,dxdata=[0.0],dydata=[0.0],dyerr=[0.0])
NIGPR.set_search_parameters(epsilon=1.0e-3)
NIGPR.GPRFit(xn,nigp_flag=True,nrestarts=5)
(nigpname,nigppars) = NIGPR.get_gp_kernel_details()
(enigpname,enigppars) = NIGPR.get_gp_error_kernel_details()
(vnigp,enigp,vdnigp,ednigp) = NIGPR.get_gp_results()

# Plotting

psig = 2.0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(xx,yy,yerr=psig*ysig,ls='',marker='.',color='k')
ax.plot(xn,vgp,color='r')
yl = vgp - psig * egp
yu = vgp + psig * egp
ax.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(pdir+'gp_test.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xn,vdgp,color='r')
yl = vdgp - psig * edgp
yu = vdgp + psig * edgp
ax.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(pdir+'gp_dtest.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(nxx,yy,xerr=psig*xsig,yerr=psig*ysig,ls='',marker='.',color='k')
ax.plot(xn,vgp,color='r')
yl = vgp - psig * egp
yu = vgp + psig * egp
ax.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.2)
ax.plot(xn,vnigp,color='b')
nyl = vnigp - psig * enigp
nyu = vnigp + psig * enigp
ax.fill_between(xn,nyl,nyu,facecolor='b',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(pdir+'nigp_test.png')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xn,vdgp,color='r')
yl = vdgp - psig * edgp
yu = vdgp + psig * edgp
ax.fill_between(xn,yl,yu,facecolor='r',edgecolor='None',alpha=0.2)
ax.plot(xn,vdnigp,color='b')
nyl = vdnigp - psig * ednigp
nyu = vdnigp + psig * ednigp
ax.fill_between(xn,nyl,nyu,facecolor='b',edgecolor='None',alpha=0.2)
ax.set_xlim(0.0,1.0)
fig.savefig(pdir+'nigp_dtest.png')
plt.close(fig)
