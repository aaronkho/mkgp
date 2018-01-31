# Required imports
import os
import sys
import re
import pwd
import time
import copy
import pickle
import numpy as np

import matplotlib
matplotlib.use("Qt4Agg")

from PyQt4 import QtCore, QtGui
from matplotlib.backends import backend_qt4agg as mplqt

#import pyqtgraph as pg
#from pyqtgraph import setConfigOption

from GPR1D import GPR1D

class KernelWidget(QtGui.QWidget):

    def __init__(self,name,fOn=True,fRestart=False):
        super(KernelWidget, self).__init__()
        self.name = name
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.ckeys = []
        self.clabels = dict()
        self.cwidgets = dict()
        self.pkeys = []
        self.plabels = dict()
        self.pwidgets = dict()
        self.lbwidgets = dict()
        self.ubwidgets = dict()

    def addConstant(self,key,widget,label=None):
        if isinstance(widget,QtGui.QLineEdit):
            self.ckeys.append(key)
            self.cwidgets[key] = widget
            if isinstance(label,QtGui.QLabel):
                self.clabels[key] = label
            elif label is None:
                self.clabels[key] = None
            else:
                raise TypeError("Invalid input type for constant label")
        else:
            raise TypeError("Input constant to KernelWidget must be a QtGui.QLineEdit widget")

    def addParameter(self,key,widget,label=None,lbwidget=None,ubwidget=None):
        if isinstance(widget,QtGui.QLineEdit):
            self.pkeys.append(key)
            self.pwidgets[key] = widget
            if isinstance(label,QtGui.QLabel):
                self.plabels[key] = label
            elif label is None:
                self.plabels[key] = None
            else:
                raise TypeError("Invalid input type for parameter label")
            if isinstance(lbwidget,QtGui.QLineEdit) and isinstance(ubwidget,QtGui.QLineEdit):
                self.lbwidgets[key] = lbwidget
                self.ubwidgets[key] = ubwidget
            elif lbwidget is None or ubwidget is None:
                self.lbwidgets[key] = None
                self.ubwidgets[key] = None
            else:
                raise TypeError("Invalid input type for lower / upper bound widget")
        else:
            raise TypeError("Input parameter to KernelWidget must be a QtGui.QLineEdit widget")

    def _removeConstant(self,key):
        if key in self.ckeys:
            del self.cwidgets[key]
            del self.clabels[key]

    def _removeParameter(self,key):
        if key in self.pkeys:
            del self.pwidgets[key]
            del self.plabels[key]
            del self.lbwidgets[key]
            del self.ubwidgets[key]

    def makeLayout(self):
        cbox = None
        pbox = None
        if len(self.ckeys) > 0:
            cbox = QtGui.QFormLayout()
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.clabels[self.ckeys[ii]],QtGui.QLabel):
                    cbox.addRow(self.clabels[self.ckeys[ii]],self.cwidgets[self.ckeys[ii]])
                else:
                    cbox.addRow(self.ckeys[ii],self.cwidgets[self.ckeys[ii]])

        if len(self.pkeys) > 0:
            self.InitGuessLabel = QtGui.QLabel("Initial Guess")
            self.InitGuessLabel.setEnabled(self.aflag)
            self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
            self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
            self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
            self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

            pbox = QtGui.QGridLayout()
            pbox.addWidget(self.InitGuessLabel,0,1)
            pbox.addWidget(self.LowerBoundLabel,0,2)
            pbox.addWidget(self.UpperBoundLabel,0,3)
            for ii in np.arange(0,len(self.pkeys)):
                if self.plabels[self.pkeys[ii]] is not None:
                    pbox.addWidget(self.plabels[self.pkeys[ii]],ii+1,0)
                if self.pwidgets[self.pkeys[ii]] is not None:
                    pbox.addWidget(self.pwidgets[self.pkeys[ii]],ii+1,1)
                if self.lbwidgets[self.pkeys[ii]] is not None:
                    pbox.addWidget(self.lbwidgets[self.pkeys[ii]],ii+1,2)
                if self.ubwidgets[self.pkeys[ii]] is not None:
                    pbox.addWidget(self.ubwidgets[self.pkeys[ii]],ii+1,3)

        layoutBox = None
        if cbox is not None and pbox is not None:
            layoutBox = QtGui.QVBoxLayout()
            layoutBox.addLayout(cbox)
            layoutBox.addLayout(pbox)
        elif cbox is not None:
            layoutBox = cbox
        elif pbox is not None:
            layoutBox = pbox
        else:
            print("No parameters added to KernelWidget, layout cannot be made")

        return layoutBox

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.pkeys)):
            if isinstance(self.lbwidgets[self.pkeys[ii]],QtGui.QWidget):
                self.lbwidgets[self.pkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.pkeys[ii]],QtGui.QWidget):
                self.ubwidgets[self.pkeys[ii]].setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        for ii in np.arange(0,len(self.ckeys)):
            if isinstance(self.cwidgets[self.ckeys[ii]],QtGui.QWidget):
                self.cwidgets[self.ckeys[ii]].setEnabled(self.aflag)
            if isinstance(self.clabels[self.ckeys[ii]],QtGui.QWidget):
                self.clabels[self.ckeys[ii]].setEnabled(self.aflag)
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.pkeys)):
            if isinstance(self.pwidgets[self.pkeys[ii]],QtGui.QWidget):
                self.pwidgets[self.pkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.plabels[self.pkeys[ii]],QtGui.QWidget):
                self.plabels[self.pkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.lbwidgets[self.pkeys[ii]],QtGui.QWidget):
                self.lbwidgets[self.pkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.pkeys[ii]],QtGui.QWidget):
                self.ubwidgets[self.pkeys[ii]].setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            csts = []
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.cwidgets[self.ckeys[ii]],QtGui.LineEdit):
                    csts.append(float(self.cwidgets[self.ckeys[ii]].text()))
            hyps = []
            for ii in np.arange(0,len(self.pkeys)):
                if isinstance(self.pwidgets[self.pkeys[ii]],QtGui.QLineEdit):
                    hyps.append(float(self.pwidgets[self.pkeys[ii]].text()))
            hyps = np.array(hyps).flatten()
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            for ii in np.arange(0,len(self.pkeys)):
                if isinstance(self.lbwidgets[self.pkeys[ii]],QtGui.QLineEdit) and isinstance(self.ubwidgets[self.pkeys[ii]],QtGui.QLineEdit):
                    bounds.append([float(self.lbwidgets[self.pkeys[ii]].text()),float(self.ubwidgets[self.pkeys[ii]].text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class SEKernelWidget(KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(SEKernelWidget, self).__init__("SE",fOn,fRestart)
        self.SEKernelUI()

    def SEKernelUI(self):

        SigmaHypLabel = QtGui.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtGui.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtGui.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtGui.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtGui.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        kbox = super(SEKernelWidget, self).makeLayout()
        self.setLayout(kbox)


class RQKernelWidget(KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(RQKernelWidget, self).__init__("RQ",fOn,fRestart)
        self.RQKernelUI()

    def RQKernelUI(self):

        SigmaHypLabel = QtGui.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtGui.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtGui.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtGui.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtGui.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        AlphaHypLabel = QtGui.QLabel("Exponent:")
        AlphaHypLabel.setEnabled(self.aflag)
        AlphaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        AlphaHypEntry = QtGui.QLineEdit("1.0e0")
        AlphaHypEntry.setEnabled(self.aflag)
        AlphaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        AlphaLBEntry = QtGui.QLineEdit("1.0e0")
        AlphaLBEntry.setEnabled(self.aflag and self.bflag)
        AlphaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        AlphaUBEntry = QtGui.QLineEdit("1.0e0")
        AlphaUBEntry.setEnabled(self.aflag and self.bflag)
        AlphaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('alpha',AlphaHypEntry,label=AlphaHypLabel,lbwidget=AlphaLBEntry,ubwidget=AlphaUBEntry)

        kbox = super(RQKernelWidget, self).makeLayout()
        self.setLayout(kbox)


class MHKernelWidget(KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(MHKernelWidget, self).__init__("MH",fOn,fRestart)
        self.MHKernelUI()

    def MHKernelUI(self):

        NuParLabel = QtGui.QLabel("Integer:")
        NuParLabel.setEnabled(self.aflag)
        NuParLabel.setAlignment(QtCore.Qt.AlignRight)
        NuParEntry = QtGui.QLineEdit("2")
        NuParEntry.setEnabled(self.aflag)
        NuParEntry.setValidator(QtGui.QIntValidator(0,100,None))
        self.addConstant('nu',NuParEntry,label=NuParLabel)

        SigmaHypLabel = QtGui.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtGui.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtGui.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtGui.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtGui.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        kbox = super(MHKernelWidget, self).makeLayout()
        self.setLayout(kbox)


class GibbsKernelWidget(KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(GibbsKernelWidget, self).__init__("Gw",fOn,fRestart)
        self.GibbsKernelUI()

    def GibbsKernelUI(self):

        SigmaHypLabel = QtGui.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.addParameter('gsigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        kbox = super(GibbsKernelWidget, self).makeLayout()

        self.WarpFuncSelectionLabel = QtGui.QLabel("Warping Function:")
        self.WarpFuncSelectionList = QtGui.QComboBox()
        self.WarpFuncSelectionList.addItem("Constant Function")
        self.WarpFuncSelectionList.addItem("Inverse Gaussian")
        self.WarpFuncSelectionList.setCurrentIndex(1)
        self.WarpFuncSelectionList.currentIndexChanged.connect(self.switch_warpfunc_ui)

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.WarpFuncSelectionLabel,self.WarpFuncSelectionList)

        self.CWarpFuncSettings = CWarpFunctionWidget(self.aflag,self.bflag)
        self.IGWarpFuncSettings = IGWarpFunctionWidget(self.aflag,self.bflag)

        self.WarpFuncSettings = QtGui.QStackedLayout()
        self.WarpFuncSettings.addWidget(self.CWarpFuncSettings)
        self.WarpFuncSettings.addWidget(self.IGWarpFuncSettings)
        self.WarpFuncSettings.setCurrentIndex(1)

        tbox = QtGui.QVBoxLayout()
        tbox.addLayout(kbox)
        tbox.addLayout(fbox)
        tbox.addLayout(self.WarpFuncSettings)

        self.setLayout(tbox)

    def switch_warpfunc_ui(self):
        self.WarpFuncSettings.setCurrentIndex(self.WarpFuncSelectionList.currentIndex())

    def toggle_bounds(self,tRestart=None):
        super(GibbsKernelWidget, self).toggle_bounds(tRestart)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_bounds(self.bflag)

    def toggle_all(self,tOn=None):
        super(GibbsKernelWidget, self).toggle_all(tOn)
        self.WarpFuncSelectionLabel.setEnabled(self.aflag)
        self.WarpFuncSelectionList.setEnabled(self.aflag)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_all(self.aflag)

    def get_name(self):
        name = super(GibbsKernelWidget, self).get_name()
        if name is not None:
            idx = self.WarpFuncSettings.currentIndex()
            wname = self.WarpFuncSettings.widget(idx).get_name()
            name = name + wname
        return name

    def get_initial_guess(self):
        (hyps,csts) = super(GibbsKernelWidget, self).get_initial_guess()
        if hyps is not None and csts is not None:
            idx = self.WarpFuncSettings.currentIndex()
            (whyps,wcsts) = self.WarpFuncSettings.widget(idx).get_initial_guess()
            if whyps is not None:
                hyps = np.hstack((hyps,whyps))
            if wcsts is not None:
                csts = np.hstack((csts,wcsts))
        return (hyps,csts)

    def get_bounds(self):
        bounds = super(GibbsKernelWidget, self).get_bounds()
        if bounds is not None:
            idx = self.WarpFuncSettings.currentIndex()
            wbounds = self.WarpFuncSettings.widget(idx).get_bounds()
            if wbounds is not None:
                bounds = np.vstack((bounds,wbounds))
        return bounds


class CWarpFunctionWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(CWarpFunctionWidget,self).__init__()
        self.name = "C"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.CWarpFunctionUI()

    def CWarpFunctionUI(self):

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthHypLabel = QtGui.QLabel("Base Length:")
        self.ConstantLengthHypLabel.setEnabled(self.aflag)
        self.ConstantLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.ConstantLengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.ConstantLengthHypEntry.setEnabled(self.aflag)
        self.ConstantLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ConstantLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.ConstantLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ConstantLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.ConstantLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.ConstantLengthHypLabel,1,0)
        gbox.addWidget(self.ConstantLengthHypEntry,1,1)
        gbox.addWidget(self.ConstantLengthLBEntry,1,2)
        gbox.addWidget(self.ConstantLengthUBEntry,1,3)

        self.setLayout(gbox)

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthUBEntry.setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthHypLabel.setEnabled(self.aflag)
        self.ConstantLengthHypEntry.setEnabled(self.aflag)
        self.ConstantLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthUBEntry.setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            hyps = []
            hyps.append(float(self.ConstantLengthHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            bounds.append([float(self.ConstantLengthLBEntry.text()),float(self.ConstantLengthUBEntry.text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class IGWarpFunctionWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(IGWarpFunctionWidget,self).__init__()
        self.name = "IG"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.IGWarpFunctionUI()

    def IGWarpFunctionUI(self):

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.BaseLengthHypLabel = QtGui.QLabel("Base Length:")
        self.BaseLengthHypLabel.setEnabled(self.aflag)
        self.BaseLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.BaseLengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.BaseLengthHypEntry.setEnabled(self.aflag)
        self.BaseLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.BaseLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.BaseLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.BaseLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.BaseLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakLengthHypLabel = QtGui.QLabel("Gaussian Height:")
        self.PeakLengthHypLabel.setEnabled(self.aflag)
        self.PeakLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.PeakLengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthHypEntry.setEnabled(self.aflag)
        self.PeakLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakWidthHypLabel = QtGui.QLabel("Gaussian Width:")
        self.PeakWidthHypLabel.setEnabled(self.aflag)
        self.PeakWidthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.PeakWidthHypEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthHypEntry.setEnabled(self.aflag)
        self.PeakWidthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakWidthLBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.PeakWidthUBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.BaseLengthHypLabel,1,0)
        gbox.addWidget(self.BaseLengthHypEntry,1,1)
        gbox.addWidget(self.BaseLengthLBEntry,1,2)
        gbox.addWidget(self.BaseLengthUBEntry,1,3)
        gbox.addWidget(self.PeakLengthHypLabel,2,0)
        gbox.addWidget(self.PeakLengthHypEntry,2,1)
        gbox.addWidget(self.PeakLengthLBEntry,2,2)
        gbox.addWidget(self.PeakLengthUBEntry,2,3)
        gbox.addWidget(self.PeakWidthHypLabel,3,0)
        gbox.addWidget(self.PeakWidthHypEntry,3,1)
        gbox.addWidget(self.PeakWidthLBEntry,3,2)
        gbox.addWidget(self.PeakWidthUBEntry,3,3)

        self.MuCstLabel = QtGui.QLabel("Gaussian Peak Location:")
        self.MuCstLabel.setEnabled(self.aflag)
        self.MuCstEntry = QtGui.QLineEdit("1.0e0")
        self.MuCstEntry.setEnabled(self.aflag)
        self.MuCstEntry.setValidator(QtGui.QDoubleValidator(None))
        self.MaxFracCstLabel = QtGui.QLabel("Maximum Peak-to-Base Ratio:")
        self.MaxFracCstLabel.setEnabled(self.aflag)
        self.MaxFracCstEntry = QtGui.QLineEdit("0.5")
        self.MaxFracCstEntry.setEnabled(self.aflag)
        self.MaxFracCstEntry.setValidator(QtGui.QDoubleValidator(0.0,1.0,100,None))

        cbox = QtGui.QFormLayout()
        cbox.addRow(self.MuCstLabel,self.MuCstEntry)
        cbox.addRow(self.MaxFracCstLabel,self.MaxFracCstEntry)

        tbox = QtGui.QVBoxLayout()
        tbox.addLayout(gbox)
        tbox.addLayout(cbox)

        self.setLayout(tbox)

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.BaseLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthUBEntry.setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.BaseLengthHypLabel.setEnabled(self.aflag)
        self.BaseLengthHypEntry.setEnabled(self.aflag)
        self.BaseLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthHypLabel.setEnabled(self.aflag)
        self.PeakLengthHypEntry.setEnabled(self.aflag)
        self.PeakLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthHypLabel.setEnabled(self.aflag)
        self.PeakWidthHypEntry.setEnabled(self.aflag)
        self.PeakWidthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthUBEntry.setEnabled(self.aflag and self.bflag)
        self.MuCstLabel.setEnabled(self.aflag)
        self.MuCstEntry.setEnabled(self.aflag)
        self.MaxFracCstLabel.setEnabled(self.aflag)
        self.MaxFracCstEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            hyps = []
            hyps.append(float(self.BaseLengthHypEntry.text()))
            hyps.append(float(self.PeakLengthHypEntry.text()))
            hyps.append(float(self.PeakWidthHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts.append(float(self.MuCstEntry.text()))
            csts.append(float(self.MaxFracCstEntry.text()))
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            bounds.append([float(self.BaseLengthLBEntry.text()),float(self.BaseLengthUBEntry.text())])
            bounds.append([float(self.PeakLengthLBEntry.text()),float(self.PeakLengthUBEntry.text())])
            bounds.append([float(self.PeakWidthLBEntry.text()),float(self.PeakWidthUBEntry.text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class GradAscentOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(GradAscentOptimizerWidget, self).__init__()
        self.name = 'grad'
        self.aflag = True if fOn else False
        self.GradOptUI()

    def GradOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-5")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars = np.array(pars)
        return pars


class MomentumOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(MomentumOptimizerWidget, self).__init__()
        self.name = 'mom'
        self.aflag = True if fOn else False
        self.MomOptUI()

    def MomOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-5")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.MomentumLabel = QtGui.QLabel("Momentum Factor:")
        self.MomentumLabel.setEnabled(self.aflag)
        self.MomentumEntry = QtGui.QLineEdit("0.9")
        self.MomentumEntry.setEnabled(self.aflag)
        self.MomentumEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.MomentumLabel,self.MomentumEntry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.MomentumLabel.setEnabled(self.aflag)
        self.MomentumEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.MomentumEntry.text()))
            pars = np.array(pars)
        return pars


class NesterovOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(NesterovOptimizerWidget, self).__init__()
        self.name = 'nag'
        self.aflag = True if fOn else False
        self.NagOptUI()

    def NagOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-5")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.MomentumLabel = QtGui.QLabel("Momentum Factor:")
        self.MomentumLabel.setEnabled(self.aflag)
        self.MomentumEntry = QtGui.QLineEdit("0.9")
        self.MomentumEntry.setEnabled(self.aflag)
        self.MomentumEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.MomentumLabel,self.MomentumEntry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.MomentumLabel.setEnabled(self.aflag)
        self.MomentumEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.MomentumEntry.text()))
            pars = np.array(pars)
        return pars


class AdagradOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(AdagradOptimizerWidget, self).__init__()
        self.name = 'adagrad'
        self.aflag = True if fOn else False
        self.AdagradOptUI()

    def AdagradOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-2")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars = np.array(pars)
        return pars


class AdadeltaOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(AdadeltaOptimizerWidget, self).__init__()
        self.name = 'adadelta'
        self.aflag = True if fOn else False
        self.AdadeltaOptUI()

    def AdadeltaOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-2")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ForgetLabel = QtGui.QLabel("Forgetting Factor:")
        self.ForgetLabel.setEnabled(self.aflag)
        self.ForgetEntry = QtGui.QLineEdit("0.9")
        self.ForgetEntry.setEnabled(self.aflag)
        self.ForgetEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.ForgetLabel,self.ForgetEntry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.ForgetLabel.setEnabled(self.aflag)
        self.ForgetEntry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.ForgetEntry.text()))
            pars = np.array(pars)
        return pars


class AdamOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(AdamOptimizerWidget, self).__init__()
        self.name = 'adam'
        self.aflag = True if fOn else False
        self.AdamOptUI()

    def AdamOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-3")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta1Label = QtGui.QLabel("Gradient Factor:")
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry = QtGui.QLineEdit("0.9")
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta2Label = QtGui.QLabel("Sq. Gradient Factor:")
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry = QtGui.QLineEdit("0.999")
        self.Beta2Entry.setEnabled(self.aflag)
        self.Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.Beta1Label,self.Beta1Entry)
        fbox.addRow(self.Beta2Label,self.Beta2Entry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.Beta1Entry.text()))
            pars.append(float(self.Beta2Entry.text()))
            pars = np.array(pars)
        return pars


class AdamaxOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(AdamaxOptimizerWidget, self).__init__()
        self.name = 'adamax'
        self.aflag = True if fOn else False
        self.AdamaxOptUI()

    def AdamaxOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("2.0e-3")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta1Label = QtGui.QLabel("Gradient Factor:")
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry = QtGui.QLineEdit("0.9")
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta2Label = QtGui.QLabel("Sq. Gradient Factor:")
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry = QtGui.QLineEdit("0.999")
        self.Beta2Entry.setEnabled(self.aflag)
        self.Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.Beta1Label,self.Beta1Entry)
        fbox.addRow(self.Beta2Label,self.Beta2Entry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.Beta1Entry.text()))
            pars.append(float(self.Beta2Entry.text()))
            pars = np.array(pars)
        return pars


class NadamOptimizerWidget(QtGui.QWidget):

    def __init__(self,fOn=True):
        super(NadamOptimizerWidget, self).__init__()
        self.name = 'nadam'
        self.aflag = True if fOn else False
        self.NadamOptUI()

    def NadamOptUI(self):

        self.GainLabel = QtGui.QLabel("Gain Factor:")
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry = QtGui.QLineEdit("1.0e-3")
        self.GainEntry.setEnabled(self.aflag)
        self.GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta1Label = QtGui.QLabel("Gradient Factor:")
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry = QtGui.QLineEdit("0.9")
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.Beta2Label = QtGui.QLabel("Sq. Gradient Factor:")
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry = QtGui.QLineEdit("0.999")
        self.Beta2Entry.setEnabled(self.aflag)
        self.Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        fbox = QtGui.QFormLayout()
        fbox.addRow(self.GainLabel,self.GainEntry)
        fbox.addRow(self.Beta1Label,self.Beta1Entry)
        fbox.addRow(self.Beta2Label,self.Beta2Entry)

        self.setLayout(fbox)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.GainLabel.setEnabled(self.aflag)
        self.GainEntry.setEnabled(self.aflag)
        self.Beta1Label.setEnabled(self.aflag)
        self.Beta1Entry.setEnabled(self.aflag)
        self.Beta2Label.setEnabled(self.aflag)
        self.Beta2Entry.setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            pars.append(float(self.GainEntry.text()))
            pars.append(float(self.Beta1Entry.text()))
            pars.append(float(self.Beta2Entry.text()))
            pars = np.array(pars)
        return pars


class QCustomTableWidgetItem(QtGui.QTableWidgetItem):

    def __init__(self, value):
        super(QCustomTableWidgetItem, self).__init__('%s' % value)

    def __lt__(self, other):
        if (isinstance(other, QCustomTableWidgetItem)):
            selfDataValue = float(self.data(QtCore.Qt.EditRole))
            otherDataValue = float(other.data(QtCore.Qt.EditRole))
            return selfDataValue < otherDataValue
        else:
            return QtGui.QTableWidgetItem.__lt__(self, other)


class GPR1D_GUI(QtGui.QWidget):

    def __init__(self):
        super(GPR1D_GUI, self).__init__()
        self.fNewData = False
        self.gpr = GPR1D.GPR1D()
        self.initUI()

    def initUI(self):

#        pg.setConfigOption('background', 'w')
#        pg.setConfigOption('foreground', 'k')

        self.TabPanel = QtGui.QTabWidget()

        self.DataEntryTab = QtGui.QWidget()
        self.DataEntryUI()

        self.YKernelSelectionTab = QtGui.QWidget()
        self.YKernelSelectionUI()

        self.EKernelSelectionTab = QtGui.QWidget()
        self.EKernelSelectionUI()

        self.TabPanel.addTab(self.DataEntryTab,"Data Entry")
        self.TabPanel.addTab(self.YKernelSelectionTab,"Fit Kernel")
        self.TabPanel.addTab(self.EKernelSelectionTab,"Error Kernel")

        self.PredictStartLabel = QtGui.QLabel("Start:")
        self.PredictStartEntry = QtGui.QLineEdit("0.0")
        self.PredictStartEntry.setValidator(QtGui.QDoubleValidator(None))
        self.PredictEndLabel = QtGui.QLabel("End:")
        self.PredictEndEntry = QtGui.QLineEdit("1.0")
        self.PredictEndEntry.setValidator(QtGui.QDoubleValidator(None))
        self.PredictNPointsLabel = QtGui.QLabel("Points:")
        self.PredictNPointsEntry = QtGui.QLineEdit("100")
        self.PredictNPointsEntry.setValidator(QtGui.QIntValidator(1,100000,None))

        xnbox = QtGui.QHBoxLayout()
        xnbox.addWidget(self.PredictStartLabel)
        xnbox.addWidget(self.PredictStartEntry)
        xnbox.addWidget(self.PredictEndLabel)
        xnbox.addWidget(self.PredictEndEntry)
        xnbox.addWidget(self.PredictNPointsLabel)
        xnbox.addWidget(self.PredictNPointsEntry)

        self.FitDataButton = QtGui.QPushButton("Fit Data")
        self.FitDataButton.clicked.connect(self.fit_data)
        self.PlotDataButton = QtGui.QPushButton("Plot Data")
        self.PlotDataButton.clicked.connect(self.plot_data)
        self.SaveRawDataButton = QtGui.QPushButton("Save Raw Data")
        self.SaveRawDataButton.clicked.connect(self.save_raw_data)
        self.SaveFitDataButton = QtGui.QPushButton("Save Fit Data")
        self.SaveFitDataButton.clicked.connect(self.save_fit_data)

        dobox = QtGui.QHBoxLayout()
        dobox.addWidget(self.FitDataButton)
        dobox.addWidget(self.PlotDataButton)
        dobox.addWidget(self.SaveRawDataButton)
        dobox.addWidget(self.SaveFitDataButton)

#        self.p1 = pg.PlotWidget()
        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
        self.p1 = mplqt.FigureCanvasQTAgg(fig)
        self.p1.figure.patch.set_facecolor('w')
        self.p1.figure.tight_layout()

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(xnbox)
        vbox.addLayout(dobox)
        vbox.addWidget(self.p1)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.TabPanel)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

        self.setGeometry(20, 20, 1500, 700)
        self.setWindowTitle("GPR1D GUI")
        self.show()

    def DataEntryUI(self):

        self.AddYDataButton = QtGui.QPushButton("Add Y Data")
        self.AddYDataButton.clicked.connect(self.add_data)
        self.AddDDataButton = QtGui.QPushButton("Add dY Data")
        self.AddDDataButton.clicked.connect(self.add_derivative_data)
        self.LoadDataButton = QtGui.QPushButton("Load Data")
        self.LoadDataButton.clicked.connect(self.load_data)

        dabox = QtGui.QHBoxLayout()
        dabox.addWidget(self.AddYDataButton)
        dabox.addWidget(self.AddDDataButton)
        dabox.addWidget(self.LoadDataButton)

        self.SortDataButton = QtGui.QPushButton("Sort Data")
        self.SortDataButton.clicked.connect(self.sort_data)
        self.CleanDataButton = QtGui.QPushButton("Clean Data")
        self.CleanDataButton.clicked.connect(self.clean_data)
        self.ClearDataButton = QtGui.QPushButton("Clear Data")
        self.ClearDataButton.clicked.connect(self.clear_data)

        dbbox = QtGui.QHBoxLayout()
        dbbox.addWidget(self.SortDataButton)
        dbbox.addWidget(self.CleanDataButton)
        dbbox.addWidget(self.ClearDataButton)

        self.DataTable = QtGui.QTableWidget()
        self.DataTable.setColumnCount(4)
        self.DataTable.setHorizontalHeaderLabels("X;Y;Y Err.;X Err.".split(";"))
        self.DataTable.horizontalHeader().hideSection(3)
        self.DataTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.DataTable.cellChanged.connect(self.flag_new_data)
        self.DataTable.itemChanged.connect(self.flag_new_data)
        self.DerivativeBox = QtGui.QCheckBox("Use derivative constraints")
        self.DerivativeBox.toggled.connect(self.toggle_derivatives)
        self.DerivativeTable = QtGui.QTableWidget()
        self.DerivativeTable.setEnabled(False)
        self.DerivativeTable.setColumnCount(4)
        self.DerivativeTable.setHorizontalHeaderLabels("X;dY;dY Err.;X Err.".split(";"))
        self.DerivativeTable.horizontalHeader().hideSection(3)
        self.DerivativeTable.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.DerivativeTable.cellChanged.connect(self.flag_new_data)
        self.DerivativeTable.itemChanged.connect(self.flag_new_data)
        self.UseXErrorsBox = QtGui.QCheckBox("Use x-errors")
        self.UseXErrorsBox.toggled.connect(self.toggle_xerror_display)

        debox = QtGui.QVBoxLayout()
        debox.addLayout(dabox)
        debox.addLayout(dbbox)
        debox.addWidget(self.DataTable)
        debox.addWidget(self.DerivativeBox)
        debox.addWidget(self.DerivativeTable)
        debox.addWidget(self.UseXErrorsBox)

        self.DataEntryTab.setLayout(debox)

    def YKernelSelectionUI(self):

        self.YKernelSelectionLabel = QtGui.QLabel("Kernel:")
        self.YKernelSelectionList = QtGui.QComboBox()
        self.YKernelSelectionList.addItem("Squared Exponential")
        self.YKernelSelectionList.addItem("Rational Quadratic")
        self.YKernelSelectionList.addItem("Matern Half-Integer")
        self.YKernelSelectionList.addItem("Gibbs Kernel")
        self.YKernelSelectionList.setCurrentIndex(0)
        self.YKernelSelectionList.currentIndexChanged.connect(self.switch_kernel_ui_y)

        self.YOptimizeBox = QtGui.QCheckBox("Optimize")
        self.YOptimizeBox.toggled.connect(self.toggle_optimize_y)
        self.YAddNoiseBox = QtGui.QCheckBox("Add Noise Kernel")
        self.YAddNoiseBox.toggled.connect(self.toggle_noise_kernel_y)
        self.YKernelRestartBox = QtGui.QCheckBox("Use Kernel Restarts")
        self.YKernelRestartBox.toggled.connect(self.toggle_kernel_restarts_y)

        self.YRegularizationLabel = QtGui.QLabel("Reg. Parameter:")
        self.YRegularizationEntry = QtGui.QLineEdit("1.0")
        self.YRegularizationEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YEpsilonLabel = QtGui.QLabel("Convergence Criteria:")
        self.YEpsilonLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry = QtGui.QLineEdit("1.0e-3")
        self.YEpsilonEntry.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        self.YOptimizerSelectionLabel = QtGui.QLabel("Optimizer:")
        self.YOptimizerSelectionLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionList = QtGui.QComboBox()
        self.YOptimizerSelectionList.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionList.addItem("Gradient Ascent")
        self.YOptimizerSelectionList.addItem("Momentum Gradient Ascent")
        self.YOptimizerSelectionList.addItem("Nesterov-Accelerated Momentum Gradient Ascent")
        self.YOptimizerSelectionList.addItem("Adaptive Gradient Ascent")
        self.YOptimizerSelectionList.addItem("Decaying Adaptive Gradient Ascent")
        self.YOptimizerSelectionList.addItem("Adaptive Moment Estimation")
        self.YOptimizerSelectionList.addItem("Adaptive Moment Estimation with L-Infinity")
        self.YOptimizerSelectionList.addItem("Nesterov-Accelerated Adaptive Moment Estimation")
        self.YOptimizerSelectionList.setCurrentIndex(0)
        self.YOptimizerSelectionList.currentIndexChanged.connect(self.switch_optimizer_ui_y)

        self.YNoiseInitGuessLabel = QtGui.QLabel("Initial Guess")
        self.YNoiseInitGuessLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseHypLabel = QtGui.QLabel("Noise Hyperparameter:")
        self.YNoiseHypLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry = QtGui.QLineEdit("1.0e-2")
        self.YNoiseHypEntry.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNoiseLBEntry = QtGui.QLineEdit("1.0e-3")
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNoiseUBEntry = QtGui.QLineEdit("1.0e-2")
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNRestartsLabel = QtGui.QLabel("Number of Restarts:")
        self.YNRestartsLabel.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry = QtGui.QLineEdit("5")
        self.YNRestartsEntry.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry.setValidator(QtGui.QIntValidator(1,1000,None))

        self.YSEKernelSettings = SEKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YRQKernelSettings = RQKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YMHKernelSettings = MHKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YGGKernelSettings = GibbsKernelWidget(True,self.YKernelRestartBox.isChecked())

        self.YKernelSettings = QtGui.QStackedLayout()
        self.YKernelSettings.addWidget(self.YSEKernelSettings)
        self.YKernelSettings.addWidget(self.YRQKernelSettings)
        self.YKernelSettings.addWidget(self.YMHKernelSettings)
        self.YKernelSettings.addWidget(self.YGGKernelSettings)
        self.YKernelSettings.setCurrentIndex(self.YKernelSelectionList.currentIndex())

        self.YGradOptSettings = GradAscentOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YMomOptSettings = MomentumOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YNagOptSettings = NesterovOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YAdagradOptSettings = AdagradOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YAdadeltaOptSettings = AdadeltaOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YAdamOptSettings = AdamOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YAdamaxOptSettings = AdamaxOptimizerWidget(self.YOptimizeBox.isChecked())
        self.YNadamOptSettings = NadamOptimizerWidget(self.YOptimizeBox.isChecked())

        self.YOptimizerSettings = QtGui.QStackedLayout()
        self.YOptimizerSettings.addWidget(self.YGradOptSettings)
        self.YOptimizerSettings.addWidget(self.YMomOptSettings)
        self.YOptimizerSettings.addWidget(self.YNagOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdagradOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdadeltaOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdamOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdamaxOptSettings)
        self.YOptimizerSettings.addWidget(self.YNadamOptSettings)
        self.YOptimizerSettings.setCurrentIndex(self.YOptimizerSelectionList.currentIndex())

        ynlbox = QtGui.QHBoxLayout()
        ynlbox.addWidget(self.YNoiseInitGuessLabel)
        ynlbox.addWidget(self.YNoiseLowerBoundLabel)
        ynlbox.addWidget(self.YNoiseUpperBoundLabel)

        ynebox = QtGui.QHBoxLayout()
        ynebox.addWidget(self.YNoiseHypEntry)
        ynebox.addWidget(self.YNoiseLBEntry)
        ynebox.addWidget(self.YNoiseUBEntry)

        ykbox = QtGui.QFormLayout()
        ykbox.addRow(self.YKernelSelectionLabel,self.YKernelSelectionList)
        ykbox.addRow(self.YKernelSettings)
        ykbox.addRow(self.YRegularizationLabel,self.YRegularizationEntry)
        ykbox.addRow(self.YOptimizeBox)
        ykbox.addRow(self.YEpsilonLabel,self.YEpsilonEntry)
        ykbox.addRow(self.YOptimizerSelectionLabel,self.YOptimizerSelectionList)
        ykbox.addRow(self.YOptimizerSettings)
        ykbox.addRow(self.YAddNoiseBox)
        ykbox.addRow("",ynlbox)
        ykbox.addRow(self.YNoiseHypLabel,ynebox)
        ykbox.addRow(self.YKernelRestartBox)
        ykbox.addRow(self.YNRestartsLabel,self.YNRestartsEntry)
        ykbox.setLabelAlignment(QtCore.Qt.AlignBottom)

        self.YKernelSelectionTab.setLayout(ykbox)

    def EKernelSelectionUI(self):

        self.HeteroscedasticBox = QtGui.QCheckBox("Enable Error Kernel")
        self.HeteroscedasticBox.toggled.connect(self.toggle_error_kernel)

        self.EKernelSelectionLabel = QtGui.QLabel("Kernel:")
        self.EKernelSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSelectionList = QtGui.QComboBox()
        self.EKernelSelectionList.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSelectionList.addItem("Squared Exponential")
        self.EKernelSelectionList.addItem("Rational Quadratic")
        self.EKernelSelectionList.addItem("Matern Half-Integer")
        self.EKernelSelectionList.addItem("Gibbs Kernel")
        self.EKernelSelectionList.setCurrentIndex(1)
        self.EKernelSelectionList.currentIndexChanged.connect(self.switch_kernel_ui_e)

        self.EOptimizeBox = QtGui.QCheckBox("Optimize")
        self.EOptimizeBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EOptimizeBox.toggled.connect(self.toggle_optimize_e)
        self.EAddNoiseBox = QtGui.QCheckBox("Add Noise Kernel")
        self.EAddNoiseBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EAddNoiseBox.toggled.connect(self.toggle_noise_kernel_e)
        self.EKernelRestartBox = QtGui.QCheckBox("Use Kernel Restarts")
        self.EKernelRestartBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelRestartBox.toggled.connect(self.toggle_kernel_restarts_e)

        self.ERegularizationLabel = QtGui.QLabel("Reg. Parameter:")
        self.ERegularizationLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ERegularizationEntry = QtGui.QLineEdit("6.0")
        self.ERegularizationEntry.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ERegularizationEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.EEpsilonLabel = QtGui.QLabel("Convergence Criteria:")
        self.EEpsilonLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EEpsilonEntry = QtGui.QLineEdit("1.0e-1")
        self.EEpsilonEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EEpsilonEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        self.EOptimizerSelectionLabel = QtGui.QLabel("Optimizer:")
        self.EOptimizerSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList = QtGui.QComboBox()
        self.EOptimizerSelectionList.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList.addItem("Gradient Ascent")
        self.EOptimizerSelectionList.addItem("Momentum Gradient Ascent")
        self.EOptimizerSelectionList.addItem("Nesterov-Accelerated Momentum Gradient Ascent")
        self.EOptimizerSelectionList.addItem("Adaptive Gradient Ascent")
        self.EOptimizerSelectionList.addItem("Decaying Adaptive Gradient Ascent")
        self.EOptimizerSelectionList.addItem("Adaptive Moment Estimation")
        self.EOptimizerSelectionList.addItem("Adaptive Moment Estimation with L-Infinity")
        self.EOptimizerSelectionList.addItem("Nesterov-Accelerated Adaptive Moment Estimation")
        self.EOptimizerSelectionList.setCurrentIndex(0)
        self.EOptimizerSelectionList.currentIndexChanged.connect(self.switch_optimizer_ui_e)

        self.ENoiseInitGuessLabel = QtGui.QLabel("Initial Guess")
        self.ENoiseInitGuessLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseLowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.ENoiseLowerBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.ENoiseUpperBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseHypLabel = QtGui.QLabel("Noise Hyperparameter:")
        self.ENoiseHypLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry = QtGui.QLineEdit("1.0e-3")
        self.ENoiseHypEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENoiseLBEntry = QtGui.QLineEdit("1.0e-3")
        self.ENoiseLBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENoiseUBEntry = QtGui.QLineEdit("1.0e-2")
        self.ENoiseUBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENRestartsLabel = QtGui.QLabel("Number of Restarts:")
        self.ENRestartsLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry = QtGui.QLineEdit("5")
        self.ENRestartsEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry.setValidator(QtGui.QIntValidator(1,1000,None))

        self.ESEKernelSettings = SEKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.ERQKernelSettings = RQKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.EMHKernelSettings = MHKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.EGGKernelSettings = GibbsKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())

        self.EKernelSettings = QtGui.QStackedLayout()
        self.EKernelSettings.addWidget(self.ESEKernelSettings)
        self.EKernelSettings.addWidget(self.ERQKernelSettings)
        self.EKernelSettings.addWidget(self.EMHKernelSettings)
        self.EKernelSettings.addWidget(self.EGGKernelSettings)
        self.EKernelSettings.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSettings.setCurrentIndex(self.EKernelSelectionList.currentIndex())

        self.EGradOptSettings = GradAscentOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EMomOptSettings = MomentumOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.ENagOptSettings = NesterovOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EAdagradOptSettings = AdagradOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EAdadeltaOptSettings = AdadeltaOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EAdamOptSettings = AdamOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EAdamaxOptSettings = AdamaxOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.ENadamOptSettings = NadamOptimizerWidget(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())

        self.EOptimizerSettings = QtGui.QStackedLayout()
        self.EOptimizerSettings.addWidget(self.EGradOptSettings)
        self.EOptimizerSettings.addWidget(self.EMomOptSettings)
        self.EOptimizerSettings.addWidget(self.ENagOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdagradOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdadeltaOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdamOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdamaxOptSettings)
        self.EOptimizerSettings.addWidget(self.ENadamOptSettings)
        self.EOptimizerSettings.setCurrentIndex(self.EOptimizerSelectionList.currentIndex())

        enlbox = QtGui.QHBoxLayout()
        enlbox.addWidget(self.ENoiseInitGuessLabel)
        enlbox.addWidget(self.ENoiseLowerBoundLabel)
        enlbox.addWidget(self.ENoiseUpperBoundLabel)

        enebox = QtGui.QHBoxLayout()
        enebox.addWidget(self.ENoiseHypEntry)
        enebox.addWidget(self.ENoiseLBEntry)
        enebox.addWidget(self.ENoiseUBEntry)

        ekbox = QtGui.QFormLayout()
        ekbox.addRow(self.EKernelSelectionLabel,self.EKernelSelectionList)
        ekbox.addRow(self.EKernelSettings)
        ekbox.addRow(self.ERegularizationLabel,self.ERegularizationEntry)
        ekbox.addRow(self.EOptimizeBox)
        ekbox.addRow(self.EEpsilonLabel,self.EEpsilonEntry)
        ekbox.addRow(self.EOptimizerSelectionLabel,self.EOptimizerSelectionList)
        ekbox.addRow(self.EOptimizerSettings)
        ekbox.addRow(self.EAddNoiseBox)
        ekbox.addRow("",enlbox)
        ekbox.addRow(self.ENoiseHypLabel,enebox)
        ekbox.addRow(self.EKernelRestartBox)
        ekbox.addRow(self.ENRestartsLabel,self.ENRestartsEntry)
        ekbox.setLabelAlignment(QtCore.Qt.AlignBottom)

        eebox = QtGui.QVBoxLayout()
        eebox.addWidget(self.HeteroscedasticBox)
        eebox.addLayout(ekbox)

        self.EKernelSelectionTab.setLayout(eebox)


    def flag_new_data(self):
        self.fNewData = True

    def add_data(self):
        idx = self.DataTable.rowCount()
        self.DataTable.insertRow(idx)
        self.DataTable.resizeRowsToContents()

    def add_derivative_data(self):
        idx = self.DerivativeTable.rowCount()
        self.DerivativeTable.insertRow(idx)
        self.DerivativeTable.resizeRowsToContents()

    def load_data(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '', 'Text files (*.txt);;All files (*)')
        if filename:
            with open(filename,'r') as ff:
                for tline in ff:
                    tline = tline.strip()
                    if re.search(r'^[0-9+\-]',tline):
                        dline = tline.split()
                        data = []
                        if len(dline) >= 2:
                            data.append(dline[0])
                            data.append(dline[1])
                            if len(dline) >= 3:
                                data.append(dline[2].strip('-'))
                            else:
                                data.append("0.0")
                            if len(dline) >= 4:
                                data.append(dline[3].strip('-'))
                            else:
                                data.append("0.0")
                        if len(data) > 0:
                            idx = self.DataTable.rowCount()
                            self.DataTable.insertRow(idx)
                            self.DataTable.setItem(idx,0,QCustomTableWidgetItem(data[0]))
                            self.DataTable.setItem(idx,1,QCustomTableWidgetItem(data[1]))
                            self.DataTable.setItem(idx,2,QCustomTableWidgetItem(data[2]))
                            self.DataTable.setItem(idx,3,QCustomTableWidgetItem(data[3]))
                            self.flag_new_data()
                    elif re.search(r'^!+\s+',tline):
                        dline = tline.split()
                        data = []
                        if len(dline) >= 3:
                            data.append(dline[1])
                            data.append(dline[2])
                            if len(dline) >= 4:
                                data.append(dline[3].strip('-'))
                            else:
                                data.append("0.0")
                            if len(dline) >= 5:
                                data.append(dline[4].strip('-'))
                            else:
                                data.append("0.0")
                        if len(data) > 0:
                            idx = self.DerivativeTable.rowCount()
                            self.DerivativeTable.insertRow(idx)
                            self.DerivativeTable.setItem(idx,0,QCustomTableWidgetItem(data[0]))
                            self.DerivativeTable.setItem(idx,1,QCustomTableWidgetItem(data[1]))
                            self.DerivativeTable.setItem(idx,2,QCustomTableWidgetItem(data[2]))
                            self.DerivativeTable.setItem(idx,3,QCustomTableWidgetItem(data[3]))
                            self.flag_new_data()
            self.DataTable.resizeRowsToContents()
            self.DerivativeTable.resizeRowsToContents()

    def sort_data(self):
        self.DataTable.sortItems(0,QtCore.Qt.AscendingOrder)
        self.DerivativeTable.sortItems(0,QtCore.Qt.AscendingOrder)

    def clean_data(self):
        for ii in np.arange(self.DataTable.rowCount()-1,-1,-1):
            gflag = True
            if self.DataTable.item(ii,0):
                try:
                    test = float(self.DataTable.item(ii,0).text())
                except:
                    gflag = False
            else:
                gflag = False
            if self.DataTable.item(ii,1):
                try:
                    test = float(self.DataTable.item(ii,1).text())
                except:
                    gflag = False
            else:
                gflag = False
            if self.DataTable.item(ii,2):
                try:
                    test = float(self.DataTable.item(ii,2).text())
                    if test < 0.0:
                        self.DataTable.item(ii,2).setText(self.DataTable.item(ii,2).text().strip('-'))
                except:
                    gflag = False
            else:
                gflag = False
            if self.DataTable.item(ii,3):
                try:
                    test = float(self.DataTable.item(ii,3).text())
                    if test < 0.0:
                        self.DataTable.item(ii,3).setText(self.DataTable.item(ii,3).text().strip('-'))
                except:
                    if not self.UseXErrorsBox.isChecked():
                        self.DataTable.item(ii,3).setText("0.0")
                        self.flag_new_data()
                    else:
                        gflag = False
            elif not self.UseXErrorsBox.isChecked():
                self.DataTable.setItem(ii,3,QtGui.QTableWidgetItem("0.0"))
                self.flag_new_data()
            else:
                gflag = False
            if not gflag:
                self.DataTable.removeRow(ii)
                self.flag_new_data()
        for ii in np.arange(self.DerivativeTable.rowCount()-1,-1,-1):
            gflag = True
            if self.DerivativeTable.item(ii,0):
                try:
                    test = float(self.DerivativeTable.item(ii,0).text())
                except:
                    gflag = False
            else:
                gflag = False
            if self.DerivativeTable.item(ii,1):
                try:
                    test = float(self.DerivativeTable.item(ii,1).text())
                except:
                    gflag = False
            else:
                gflag = False
            if self.DerivativeTable.item(ii,2):
                try:
                    test = float(self.DerivativeTable.item(ii,2).text())
                    if test < 0.0:
                        self.DerivativeTable.item(ii,2).setText(self.DerivativeTable.item(ii,2).text().strip('-'))
                except:
                    gflag = False
            else:
                gflag = False
            if self.DerivativeTable.item(ii,3):
                try:
                    test = float(self.DerivativeTable.item(ii,3).text())
                    if test < 0.0:
                        self.DerivativeTable.item(ii,3).setText(self.DerivativeTable.item(ii,3).text().strip('-'))
                except:
                    if not self.UseXErrorsBox.isChecked():
                        self.DerivativeTable.item(ii,3).setText("0.0")
                        self.flag_new_data()
                    else:
                        gflag = False
            elif not self.UseXErrorsBox.isChecked():
                self.DerivativeTable.setItem(ii,3,QtGui.QTableWidgetItem("0.0"))
                self.flag_new_data()
            else:
                gflag = False
            if not gflag:
                self.DerivativeTable.removeRow(ii)
                self.flag_new_data()

    def clear_data(self):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Question)
        msg.setWindowTitle("Clear Data")
        msg.setText("Are you sure you want to clear all stored raw data?")
        msg.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        retval = msg.exec_()
        if retval == QtGui.QMessageBox.Yes:
            self.DataTable.clearContents()
            self.DerivativeTable.clearContents()
            self.clean_data()
            self.flag_new_data()

    def toggle_derivatives(self):
        self.DerivativeTable.setEnabled(self.DerivativeBox.isChecked())

    def toggle_xerror_display(self):
        if self.UseXErrorsBox.isChecked():
            self.DataTable.horizontalHeader().showSection(3)
            self.DerivativeTable.horizontalHeader().showSection(3)
        else:
            self.DataTable.horizontalHeader().hideSection(3)
            self.DerivativeTable.horizontalHeader().hideSection(3)

    def switch_kernel_ui_y(self):
        self.YKernelSettings.setCurrentIndex(self.YKernelSelectionList.currentIndex())

    def toggle_optimize_y(self):
        self.YEpsilonLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionList.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSettings.setEnabled(self.YOptimizeBox.isChecked())
        for ii in np.arange(0,self.YOptimizerSettings.count()):
            self.YOptimizerSettings.widget(ii).toggle_all(self.YOptimizeBox.isChecked())

    def switch_optimizer_ui_y(self):
        self.YOptimizerSettings.setCurrentIndex(self.YOptimizerSelectionList.currentIndex())

    def toggle_noise_kernel_y(self):
        self.YNoiseInitGuessLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseHypLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())

    def toggle_kernel_restarts_y(self):
        self.YNRestartsLabel.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry.setEnabled(self.YKernelRestartBox.isChecked())
        for ii in np.arange(0,self.YKernelSettings.count()):
            self.YKernelSettings.widget(ii).toggle_bounds(self.YKernelRestartBox.isChecked())
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())

    def toggle_error_kernel(self):
        self.EKernelSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSelectionList.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSettings.setEnabled(self.HeteroscedasticBox.isChecked())
        for ii in np.arange(0,self.EKernelSettings.count()):
            self.EKernelSettings.widget(ii).toggle_all(self.HeteroscedasticBox.isChecked())
        self.ERegularizationLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ERegularizationEntry.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EOptimizeBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EEpsilonLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EEpsilonEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSettings.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        for ii in np.arange(0,self.EOptimizerSettings.count()):
            self.EOptimizerSettings.widget(ii).toggle_all(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EAddNoiseBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ENoiseInitGuessLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseLowerBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseHypLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseLBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.EKernelRestartBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ENRestartsLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())

    def switch_kernel_ui_e(self):
        self.EKernelSettings.setCurrentIndex(self.EKernelSelectionList.currentIndex())

    def toggle_optimize_e(self):
        self.EEpsilonLabel.setEnabled(self.EOptimizeBox.isChecked())
        self.EEpsilonEntry.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionLabel.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSettings.setEnabled(self.EOptimizeBox.isChecked())
        for ii in np.arange(0,self.EOptimizerSettings.count()):
            self.EOptimizerSettings.widget(ii).toggle_all(self.EOptimizeBox.isChecked())

    def switch_optimizer_ui_e(self):
        self.EOptimizerSettings.setCurrentIndex(self.EOptimizerSelectionList.currentIndex())

    def toggle_noise_kernel_e(self):
        self.ENoiseInitGuessLabel.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseLowerBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseHypLabel.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseLBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())

    def toggle_kernel_restarts_e(self):
        self.ENRestartsLabel.setEnabled(self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry.setEnabled(self.EKernelRestartBox.isChecked())
        for ii in np.arange(0,self.EKernelSettings.count()):
            self.EKernelSettings.widget(ii).toggle_bounds(self.EKernelRestartBox.isChecked())
        self.ENoiseLowerBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseLBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())

    def fit_data(self):
        self.clean_data()
        npts = self.DataTable.rowCount()
        if npts > 0:
            xx = np.array([])
            yy = np.array([])
            ye = np.array([])
            xe = np.array([])
            for ii in np.arange(0,npts):
                xx = np.hstack((xx,float(self.DataTable.item(ii,0).text())))
                yy = np.hstack((yy,float(self.DataTable.item(ii,1).text())))
                ye = np.hstack((ye,float(self.DataTable.item(ii,2).text())))
                xe = np.hstack((xe,float(self.DataTable.item(ii,3).text())))
            dxx = 'None'
            dyy = 'None'
            dye = 'None'
            dxe = 'None'
            ndpts = self.DerivativeTable.rowCount()
            if self.DerivativeBox.isChecked() and ndpts > 0:
                dxx = np.array([])
                dyy = np.array([])
                dye = np.array([])
                dxe = np.array([])
                for ii in np.arange(0,ndpts):
                    dxx = np.hstack((dxx,float(self.DerivativeTable.item(ii,0).text())))
                    dyy = np.hstack((dyy,float(self.DerivativeTable.item(ii,1).text())))
                    dye = np.hstack((dye,float(self.DerivativeTable.item(ii,2).text())))
                    dxe = np.hstack((dxe,float(self.DerivativeTable.item(ii,3).text())))
            use_xerrs = self.UseXErrorsBox.isChecked()

            ykname = self.YKernelSettings.currentWidget().get_name()
            (ykhyps,ykcsts) = self.YKernelSettings.currentWidget().get_initial_guess()
            ykbounds = self.YKernelSettings.currentWidget().get_bounds()
            yregpar = float(self.YRegularizationEntry.text()) if self.YRegularizationEntry.text() else None
            yeps = float(self.YEpsilonEntry.text()) if self.YOptimizeBox.isChecked() and self.YEpsilonEntry.text() else 'None'
            yopm = self.YOptimizerSettings.widget(self.YOptimizerSelectionList.currentIndex()).get_name()
            yopp = self.YOptimizerSettings.widget(self.YOptimizerSelectionList.currentIndex()).get_parameters()
            if self.YAddNoiseBox.isChecked():
                ykname = 'Sum_' + ykname + '-n'
                ykhyps = np.hstack((ykhyps,float(self.YNoiseHypEntry.text()))) if ykhyps is not None else None
                ykbounds = np.vstack((ykbounds,np.atleast_2d([float(self.YNoiseLBEntry.text()),float(self.YNoiseUBEntry.text())]))) if ykbounds is not None else None
            ynres = int(float(self.YNRestartsEntry.text())) if self.YKernelRestartBox.isChecked() else None
            ykernel = GPR1D.KernelReconstructor(ykname,pars=np.hstack((ykhyps,ykcsts)))

            ekernel = None
            ekname = self.EKernelSettings.currentWidget().get_name()
            (ekhyps,ekcsts) = self.EKernelSettings.currentWidget().get_initial_guess()
            ekbounds = self.EKernelSettings.currentWidget().get_bounds()
            eregpar = None
            eeps = 'None'
            eopm = None
            eopp = None
            enres = None
            if self.HeteroscedasticBox.isChecked():
                eregpar = float(self.ERegularizationEntry.text()) if self.ERegularizationEntry.text() else None
                eeps = float(self.EEpsilonEntry.text()) if self.EOptimizeBox.isChecked() and self.EEpsilonEntry.text() else 'None'
                eopm = self.EOptimizerSettings.widget(self.EOptimizerSelectionList.currentIndex()).get_name()
                eopp = self.EOptimizerSettings.widget(self.EOptimizerSelectionList.currentIndex()).get_parameters()
                if self.EAddNoiseBox.isChecked():
                    ekname = 'Sum_' + ekname + '-n'
                    ekhyps = np.hstack((ekhyps,float(self.ENoiseHypEntry.text()))) if ekhyps is not None else None
                    ekbounds = np.vstack((ekbounds,np.atleast_2d([float(self.ENoiseLBEntry.text()),float(self.ENoiseUBEntry.text())]))) if ekbounds is not None else None
                enres = int(float(self.ENRestartsEntry.text())) if self.EKernelRestartBox.isChecked() else None
                ekernel = GPR1D.KernelReconstructor(ekname,pars=np.hstack((ekhyps,ekcsts)))

            try:
                tic = time.perf_counter()
                xnew = np.linspace(float(self.PredictStartEntry.text()),float(self.PredictEndEntry.text()),int(float(self.PredictNPointsEntry.text())))
                self.gpr.set_raw_data(xdata=xx,ydata=yy,yerr=ye,xerr=xe,dxdata=dxx,dydata=dyy,dyerr=dye)
                self.gpr.set_kernel(kernel=ykernel,kbounds=ykbounds,regpar=yregpar)
                self.gpr.set_error_kernel(kernel=ekernel,kbounds=ekbounds,regpar=eregpar,nrestarts=enres)
                self.gpr.set_search_parameters(epsilon=yeps,method=yopm,spars=yopp)
                self.gpr.set_error_search_parameters(epsilon=eeps,method=eopm,spars=eopp)
                self.gpr.GPRFit(xnew,nigp_flag=use_xerrs,nrestarts=ynres)
                self.fNewData = False
                toc = time.perf_counter()
                print("Fitting routine completed. Elapsed time: %.3f s" % (toc - tic))
                ylml = self.gpr.get_gp_lml()
                print("Final log-marginal-likelihood: %15.8f" % (ylml))
                if (isinstance(eeps,(float,int)) and eeps > 0.0) or (isinstance(enres,(float,int)) and enres > 0):
                    ehyps = self.gpr.get_error_kernel().get_hyperparameters()
                    print("   --- Optimized error kernel hyperparameters: ---")
                    print(ehyps)
                if (isinstance(yeps,(float,int)) and yeps > 0.0) or (isinstance(ynres,(float,int)) and ynres > 0):
                    yhyps = self.gpr.get_gp_kernel().get_hyperparameters()
                    print("   *** Optimized kernel hyperparameters ***")
                    print(yhyps)
            except Exception as e:
                print(repr(e))
                msg = QtGui.QMessageBox()
                msg.setIcon(QtGui.QMessageBox.Critical)
                msg.setWindowTitle("Fitting Routine Failed")
                msg.setText("Fitting routine failure: Please see console messages for more details.")
                msg.exec_()
        else:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Raw data table is empty or was improperly filled.")
            msg.exec_()

    def plot_data(self):
        self.clean_data()
        retval = QtGui.QMessageBox.Yes
        if self.fNewData and self.gpr.get_gp_x() is not None:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Question)
            msg.setWindowTitle("Raw Data and Fit Mismatched")
            msg.setText("Changes to stored raw data have been detected and stored fit may no longer correspond with it. Plot anyway?")
            msg.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            retval = msg.exec_()
        if retval == QtGui.QMessageBox.Yes:
            sigma = 1.0

            fig = self.p1.figure
            fig.clear()
            ax = fig.add_subplot(111)
            xraw = np.array([])
            yraw = np.array([])
            yeraw = np.array([])
            xeraw = np.array([])
            for ii in np.arange(0,self.DataTable.rowCount()):
                xraw = np.hstack((xraw,float(self.DataTable.item(ii,0).text())))
                yraw = np.hstack((yraw,float(self.DataTable.item(ii,1).text())))
                yeraw = np.hstack((yeraw,float(self.DataTable.item(ii,2).text())))
                xeraw = np.hstack((xeraw,float(self.DataTable.item(ii,3).text())))
            if self.UseXErrorsBox.isChecked():
                ax.errorbar(xraw,yraw,xerr=xeraw,yerr=yeraw,ls='',color='k')
            else:
                ax.errorbar(xraw,yraw,yerr=yeraw,ls='',color='k')
            xfit = self.gpr.get_gp_x()
            yfit = None
            yefit = None
            if xfit is not None:
                yfit = self.gpr.get_gp_mean()
                yefit = self.gpr.get_gp_std(noise_flag=True)
                ax.plot(xfit,yfit,ls='-',color='r')
                ylower = yfit - sigma * yefit
                yupper = yfit + sigma * yefit
                ax.fill_between(xfit,ylower,yupper,facecolor='r',edgecolor='None',alpha=0.2)
            ymin = np.amin([np.amin(yraw-sigma*yeraw),np.amin(yfit-sigma*yefit)]) if yfit is not None else np.amin(yraw-sigma*yeraw)
            ymax = np.amax([np.amax(yraw+sigma*yeraw),np.amax(yfit+sigma*yefit)]) if yfit is not None else np.amax(yraw+sigma*yeraw)
            ybuf = 0.025 * (ymax - ymin)
            ax.set_ylim([ymin-ybuf,ymax+ybuf])
            ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
            self.p1.draw()

## Procedure to use pyqtgraph
#            self.p1.clear()
#            rawPen = pg.mkPen(color='k',width=1)
#            for ii in np.arange(0,self.DataTable.rowCount()):
#                xraw = float(self.DataTable.item(ii,0).text())
#                yraw = float(self.DataTable.item(ii,1).text())
#                yeraw = float(self.DataTable.item(ii,2).text())
#                xeraw = float(self.DataTable.item(ii,3).text())
#                self.p1.plot([xraw,xraw],[yraw-sigma*yeraw,yraw+sigma*yeraw],pen=rawPen)
#                if self.UseXErrorsBox.isChecked():
#                    self.p1.plot([xraw-sigma*xeraw,xraw+sigma*xeraw],[yraw,yraw],pen=rawPen)
#            xfit = self.gpr.get_gp_x()
#            fitPen = pg.mkPen(color='r',width=1)
#            errPen = pg.mkPen(color='r',width=1,style=QtCore.Qt.DashLine)
#            if xfit is not None:
#                yfit = self.gpr.get_gp_mean()
#                yefit = self.gpr.get_gp_std(noise_flag=True)
#                self.p1.plot(xfit,yfit,pen=fitPen)
#                self.p1.plot(xfit,yfit-sigma*yefit,pen=errPen)
#                self.p1.plot(xfit,yfit+sigma*yefit,pen=errPen)


    def save_raw_data(self,sortflag=False):
        self.clean_data()
        if self.DataTable.rowCount() > 0:
            filename = QtGui.QFileDialog.getSaveFileName(self, 'Save As...', '', 'Text files (*.txt);;All files (*)')
            if filename:
                ff = open(filename,'w')
                ff.write("%15s%15s%15s%15s\n" % ("X","Y","Y Err.","X Err."))
                for ii in np.arange(0,self.DataTable.rowCount()):
                    xraw = float(self.DataTable.item(ii,0).text())
                    yraw = float(self.DataTable.item(ii,1).text())
                    yeraw = float(self.DataTable.item(ii,2).text())
                    xeraw = float(self.DataTable.item(ii,3).text())
                    ff.write("%15.6e%15.6e%15.6e%15.6e\n" % (xraw,yraw,yeraw,xeraw))
                if self.DerivativeTable.rowCount() > 0:
                    ff.write("\n")
                    ff.write("  %15s%15s%15s%15s\n" % ("X","dY","dY Err.","X Err."))
                    for ii in np.arange(0,self.DerivativeTable.rowCount()):
                        xraw = float(self.DerivativeTable.item(ii,0).text())
                        yraw = float(self.DerivativeTable.item(ii,1).text())
                        yeraw = float(self.DerivativeTable.item(ii,2).text())
                        xeraw = float(self.DerivativeTable.item(ii,3).text())
                        ff.write("! %15.6e%15.6e%15.6e%15.6e\n" % (xraw,yraw,yeraw,xeraw))
                ff.close()
                print("Raw data written into %s." % (filename))
        else:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Raw data table is empty or was improperly filled.")
            msg.exec_()

    def save_fit_data(self):
        if self.gpr.get_gp_x() is not None:
            filename = QtGui.QFileDialog.getSaveFileName(self, 'Save As...', '', 'Text files (*.txt);;All files (*)')
            if filename:
                xfit = self.gpr.get_gp_x()
                yfit = self.gpr.get_gp_mean()
                yefit = self.gpr.get_gp_std(noise_flag=True)
                dyfit = self.gpr.get_gp_drv_mean()
                dyefit = self.gpr.get_gp_drv_std(noise_flag=False)
                ff = open(filename,'w')
                ff.write("%15s%15s%15s%15s%15s\n" % ("X","Y","Y Err.","dY","dY Err."))
                for ii in np.arange(0,xfit.size):
                    ff.write("%15.6e%15.6e%15.6e%15.6e%15.6e\n" % (xfit[ii],yfit[ii],yefit[ii],dyfit[ii],dyefit[ii]))
                ff.close()
                print("Fit data written into %s." % (filename))
        else:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Fit data not yet populated.")
            msg.exec_()

def main():

    app = QtGui.QApplication(sys.argv)
    app.setApplicationName('GPR1D')
    ex = GPR1D_GUI()

    sys.exit(app.exec_())

if __name__ == '__main__':

    main()
