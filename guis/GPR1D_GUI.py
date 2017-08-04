#!/usr/local/depot/Python-3.5.1/bin/python

# Required imports
import os
import sys
import re
import pwd
import time
import copy
import pickle
import numpy as np

from PyQt4 import QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import setConfigOption

from GPR1D import GPR1D

class SEKernelWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(SEKernelWidget, self).__init__()
        self.name = "SE"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.SEKernelUI()

    def SEKernelUI(self):

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

        self.SigmaHypLabel = QtGui.QLabel("Amplitude:")
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaHypEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthHypLabel = QtGui.QLabel("Length:")
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.LengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setValidator(QtGui.QDoubleValidator())

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.SigmaHypLabel,1,0)
        gbox.addWidget(self.SigmaHypEntry,1,1)
        gbox.addWidget(self.SigmaLBEntry,1,2)
        gbox.addWidget(self.SigmaUBEntry,1,3)
        gbox.addWidget(self.LengthHypLabel,2,0)
        gbox.addWidget(self.LengthHypEntry,2,1)
        gbox.addWidget(self.LengthLBEntry,2,2)
        gbox.addWidget(self.LengthUBEntry,2,3)

        self.setLayout(gbox)

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            hyps = []
            hyps.append(float(self.SigmaHypEntry.text()))
            hyps.append(float(self.LengthHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            bounds.append([float(self.SigmaLBEntry.text()),float(self.SigmaUBEntry.text())])
            bounds.append([float(self.LengthLBEntry.text()),float(self.LengthUBEntry.text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class RQKernelWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(RQKernelWidget, self).__init__()
        self.name = "RQ"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.RQKernelUI()

    def RQKernelUI(self):

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

        self.SigmaHypLabel = QtGui.QLabel("Amplitude:")
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaHypEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthHypLabel = QtGui.QLabel("Length:")
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.LengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setValidator(QtGui.QDoubleValidator())
        self.AlphaHypLabel = QtGui.QLabel("Exponent:")
        self.AlphaHypLabel.setEnabled(self.aflag)
        self.AlphaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.AlphaHypEntry = QtGui.QLineEdit("1.0e0")
        self.AlphaHypEntry.setEnabled(self.aflag)
        self.AlphaHypEntry.setValidator(QtGui.QDoubleValidator())
        self.AlphaLBEntry = QtGui.QLineEdit("1.0e0")
        self.AlphaLBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaLBEntry.setValidator(QtGui.QDoubleValidator())
        self.AlphaUBEntry = QtGui.QLineEdit("1.0e0")
        self.AlphaUBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaUBEntry.setValidator(QtGui.QDoubleValidator())

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.SigmaHypLabel,1,0)
        gbox.addWidget(self.SigmaHypEntry,1,1)
        gbox.addWidget(self.SigmaLBEntry,1,2)
        gbox.addWidget(self.SigmaUBEntry,1,3)
        gbox.addWidget(self.LengthHypLabel,2,0)
        gbox.addWidget(self.LengthHypEntry,2,1)
        gbox.addWidget(self.LengthLBEntry,2,2)
        gbox.addWidget(self.LengthUBEntry,2,3)
        gbox.addWidget(self.AlphaHypLabel,3,0)
        gbox.addWidget(self.AlphaHypEntry,3,1)
        gbox.addWidget(self.AlphaLBEntry,3,2)
        gbox.addWidget(self.AlphaUBEntry,3,3)

        self.setLayout(gbox)

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaLBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaUBEntry.setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaHypLabel.setEnabled(self.aflag)
        self.AlphaHypEntry.setEnabled(self.aflag)
        self.AlphaLBEntry.setEnabled(self.aflag and self.bflag)
        self.AlphaUBEntry.setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            hyps = []
            hyps.append(float(self.SigmaHypEntry.text()))
            hyps.append(float(self.LengthHypEntry.text()))
            hyps.append(float(self.AlphaHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            bounds.append([float(self.SigmaLBEntry.text()),float(self.SigmaUBEntry.text())])
            bounds.append([float(self.LengthLBEntry.text()),float(self.LengthUBEntry.text())])
            bounds.append([float(self.AlphaLBEntry.text()),float(self.AlphaUBEntry.text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class MHKernelWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(MHKernelWidget, self).__init__()
        self.name = "MH"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.MHKernelUI()

    def MHKernelUI(self):

        self.NuParLabel = QtGui.QLabel("Integer:")
        self.NuParLabel.setEnabled(self.aflag)
        self.NuParLabel.setAlignment(QtCore.Qt.AlignRight)
        self.NuParEntry = QtGui.QLineEdit("2")
        self.NuParEntry.setEnabled(self.aflag)
        self.NuParEntry.setValidator(QtGui.QIntValidator())

        pbox = QtGui.QHBoxLayout()
        pbox.addWidget(self.NuParLabel)
        pbox.addWidget(self.NuParEntry)

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

        self.SigmaHypLabel = QtGui.QLabel("Amplitude:")
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaHypEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthHypLabel = QtGui.QLabel("Length:")
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.LengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.LengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setValidator(QtGui.QDoubleValidator())

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.SigmaHypLabel,1,0)
        gbox.addWidget(self.SigmaHypEntry,1,1)
        gbox.addWidget(self.SigmaLBEntry,1,2)
        gbox.addWidget(self.SigmaUBEntry,1,3)
        gbox.addWidget(self.LengthHypLabel,2,0)
        gbox.addWidget(self.LengthHypEntry,2,1)
        gbox.addWidget(self.LengthLBEntry,2,2)
        gbox.addWidget(self.LengthUBEntry,2,3)

        tbox = QtGui.QVBoxLayout()
        tbox.addLayout(pbox)
        tbox.addLayout(gbox)

        self.setLayout(tbox)

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.NuParLabel.setEnabled(self.aflag)
        self.NuParEntry.setEnabled(self.aflag)
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthHypLabel.setEnabled(self.aflag)
        self.LengthHypEntry.setEnabled(self.aflag)
        self.LengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.LengthUBEntry.setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            hyps = []
            hyps.append(float(self.SigmaHypEntry.text()))
            hyps.append(float(self.LengthHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts.append(float(self.NuParEntry.text()))
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            bounds.append([float(self.SigmaLBEntry.text()),float(self.SigmaUBEntry.text())])
            bounds.append([float(self.LengthLBEntry.text()),float(self.LengthUBEntry.text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class GibbsKernelWidget(QtGui.QWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(GibbsKernelWidget, self).__init__()
        self.name = "Gw"
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.GibbsKernelUI()

    def GibbsKernelUI(self):

        self.InitGuessLabel = QtGui.QLabel("Initial Guess")
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaHypLabel = QtGui.QLabel("Amplitude:")
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.SigmaHypEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaHypEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaLBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setValidator(QtGui.QDoubleValidator())
        self.SigmaUBEntry = QtGui.QLineEdit("1.0e0")
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setValidator(QtGui.QDoubleValidator())

        gbox = QtGui.QGridLayout()
        gbox.addWidget(self.InitGuessLabel,0,1)
        gbox.addWidget(self.LowerBoundLabel,0,2)
        gbox.addWidget(self.UpperBoundLabel,0,3)
        gbox.addWidget(self.SigmaHypLabel,1,0)
        gbox.addWidget(self.SigmaHypEntry,1,1)
        gbox.addWidget(self.SigmaLBEntry,1,2)
        gbox.addWidget(self.SigmaUBEntry,1,3)

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
        tbox.addLayout(gbox)
        tbox.addLayout(fbox)
        tbox.addLayout(self.WarpFuncSettings)

        self.setLayout(tbox)

    def switch_warpfunc_ui(self):
        self.WarpFuncSettings.setCurrentIndex(self.WarpFuncSelectionList.currentIndex())

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_bounds(self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        self.SigmaHypLabel.setEnabled(self.aflag)
        self.SigmaHypEntry.setEnabled(self.aflag)
        self.SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        self.SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        self.WarpFuncSelectionLabel.setEnabled(self.aflag)
        self.WarpFuncSelectionList.setEnabled(self.aflag)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_all(self.aflag)

    def get_name(self):
        name = None
        if self.aflag:
            idx = self.WarpFuncSettings.currentIndex()
            wname = self.WarpFuncSettings.widget(idx).get_name()
            name = self.name + wname
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            idx = self.WarpFuncSettings.currentIndex()
            (whyps,wcsts) = self.WarpFuncSettings.widget(idx).get_initial_guess()
            hyps = []
            hyps.append(float(self.SigmaHypEntry.text()))
            hyps = np.array(hyps).flatten()
            csts = []
            csts = np.array(csts).flatten()
            if whyps is not None:
                hyps = np.hstack((hyps,whyps))
            if wcsts is not None:
                csts = np.hstack((csts,wcsts))
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            idx = self.WarpFuncSettings.currentIndex()
            wbounds = self.WarpFuncSettings.widget(idx).get_bounds()
            bounds = []
            bounds.append([float(self.SigmaLBEntry.text()),float(self.SigmaUBEntry.text())])
            bounds = np.atleast_2d(bounds)
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
        self.ConstantLengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.ConstantLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.ConstantLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.ConstantLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.ConstantLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.ConstantLengthUBEntry.setValidator(QtGui.QDoubleValidator())

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
        self.BaseLengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.BaseLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.BaseLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.BaseLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.BaseLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.BaseLengthUBEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakLengthHypLabel = QtGui.QLabel("Gaussian Height:")
        self.PeakLengthHypLabel.setEnabled(self.aflag)
        self.PeakLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.PeakLengthHypEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthHypEntry.setEnabled(self.aflag)
        self.PeakLengthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakLengthLBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakLengthUBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakLengthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakLengthUBEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakWidthHypLabel = QtGui.QLabel("Gaussian Width:")
        self.PeakWidthHypLabel.setEnabled(self.aflag)
        self.PeakWidthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        self.PeakWidthHypEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthHypEntry.setEnabled(self.aflag)
        self.PeakWidthHypEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakWidthLBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthLBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthLBEntry.setValidator(QtGui.QDoubleValidator())
        self.PeakWidthUBEntry = QtGui.QLineEdit("1.0e0")
        self.PeakWidthUBEntry.setEnabled(self.aflag and self.bflag)
        self.PeakWidthUBEntry.setValidator(QtGui.QDoubleValidator())

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
        self.MuCstEntry.setValidator(QtGui.QDoubleValidator())
        self.MaxFracCstLabel = QtGui.QLabel("Maximum Peak-to-Base Ratio:")
        self.MaxFracCstLabel.setEnabled(self.aflag)
        self.MaxFracCstEntry = QtGui.QLineEdit("5.0e-1")
        self.MaxFracCstEntry.setEnabled(self.aflag)
        self.MaxFracCstEntry.setValidator(QtGui.QDoubleValidator())

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

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

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
        self.PredictStartEntry.setValidator(QtGui.QDoubleValidator())
        self.PredictEndLabel = QtGui.QLabel("End:")
        self.PredictEndEntry = QtGui.QLineEdit("1.0")
        self.PredictEndEntry.setValidator(QtGui.QDoubleValidator())
        self.PredictNPointsLabel = QtGui.QLabel("Points:")
        self.PredictNPointsEntry = QtGui.QLineEdit("100")
        self.PredictNPointsEntry.setValidator(QtGui.QIntValidator())

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

        self.p1 = pg.PlotWidget()

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

        self.YRegularizationLabel = QtGui.QLabel("Reg. Parameter:")
        self.YRegularizationEntry = QtGui.QLineEdit("1.0")
        self.YRegularizationEntry.setValidator(QtGui.QDoubleValidator())
        self.YOptimizeBox = QtGui.QCheckBox("Optimize")
        self.YOptimizeBox.toggled.connect(self.toggle_optimize_y)
        self.YEpsilonLabel = QtGui.QLabel("Convergence Criteria:")
        self.YEpsilonLabel.setEnabled(False)
        self.YEpsilonEntry = QtGui.QLineEdit("1.0e-3")
        self.YEpsilonEntry.setEnabled(False)
        self.YEpsilonEntry.setValidator(QtGui.QDoubleValidator())
        self.YAddNoiseBox = QtGui.QCheckBox("Add Noise Kernel")
        self.YAddNoiseBox.toggled.connect(self.toggle_noise_kernel_y)
        self.YNoiseInitGuessLabel = QtGui.QLabel("Initial Guess")
        self.YNoiseInitGuessLabel.setEnabled(False)
        self.YNoiseLowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.YNoiseLowerBoundLabel.setEnabled(False)
        self.YNoiseUpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.YNoiseUpperBoundLabel.setEnabled(False)
        self.YNoiseHypLabel = QtGui.QLabel("Noise Hyperparameter:")
        self.YNoiseHypLabel.setEnabled(False)
        self.YNoiseHypEntry = QtGui.QLineEdit("1.0e-2")
        self.YNoiseHypEntry.setEnabled(False)
        self.YNoiseHypEntry.setValidator(QtGui.QDoubleValidator())
        self.YNoiseLBEntry = QtGui.QLineEdit("1.0e-3")
        self.YNoiseLBEntry.setEnabled(False)
        self.YNoiseLBEntry.setValidator(QtGui.QDoubleValidator())
        self.YNoiseUBEntry = QtGui.QLineEdit("1.0e-2")
        self.YNoiseUBEntry.setEnabled(False)
        self.YNoiseUBEntry.setValidator(QtGui.QDoubleValidator())
        self.YKernelRestartBox = QtGui.QCheckBox("Use Kernel Restarts")
        self.YKernelRestartBox.toggled.connect(self.toggle_kernel_restarts_y)
        self.YNRestartsLabel = QtGui.QLabel("Number of Restarts:")
        self.YNRestartsLabel.setEnabled(False)
        self.YNRestartsEntry = QtGui.QLineEdit("5")
        self.YNRestartsEntry.setEnabled(False)
        self.YNRestartsEntry.setValidator(QtGui.QIntValidator())

        self.YSEKernelSettings = SEKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YRQKernelSettings = RQKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YMHKernelSettings = MHKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YGGKernelSettings = GibbsKernelWidget(True,self.YKernelRestartBox.isChecked())

        self.YKernelSettings = QtGui.QStackedLayout()
        self.YKernelSettings.addWidget(self.YSEKernelSettings)
        self.YKernelSettings.addWidget(self.YRQKernelSettings)
        self.YKernelSettings.addWidget(self.YMHKernelSettings)
        self.YKernelSettings.addWidget(self.YGGKernelSettings)
        self.YKernelSettings.setCurrentIndex(0)

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
        ykbox.addRow(self.YAddNoiseBox)
        ykbox.addRow("",ynlbox)
        ykbox.addRow(self.YNoiseHypLabel,ynebox)
        ykbox.addRow(self.YKernelRestartBox)
        ykbox.addRow(self.YNRestartsLabel,self.YNRestartsEntry)

        self.YKernelSelectionTab.setLayout(ykbox)

    def EKernelSelectionUI(self):

        self.HeteroscedasticBox = QtGui.QCheckBox("Enable Error Kernel")
        self.HeteroscedasticBox.toggled.connect(self.toggle_error_kernel)

        self.EKernelSelectionLabel = QtGui.QLabel("Kernel:")
        self.EKernelSelectionLabel.setEnabled(False)
        self.EKernelSelectionList = QtGui.QComboBox()
        self.EKernelSelectionList.setEnabled(False)
        self.EKernelSelectionList.addItem("Squared Exponential")
        self.EKernelSelectionList.addItem("Rational Quadratic")
        self.EKernelSelectionList.addItem("Matern Half-Integer")
        self.EKernelSelectionList.addItem("Gibbs Kernel")
        self.EKernelSelectionList.setCurrentIndex(1)
        self.EKernelSelectionList.currentIndexChanged.connect(self.switch_kernel_ui_e)

        self.ERegularizationLabel = QtGui.QLabel("Reg. Parameter:")
        self.ERegularizationLabel.setEnabled(False)
        self.ERegularizationEntry = QtGui.QLineEdit("6.0")
        self.ERegularizationEntry.setEnabled(False)
        self.ERegularizationEntry.setValidator(QtGui.QDoubleValidator())
        self.EOptimizeBox = QtGui.QCheckBox("Optimize")
        self.EOptimizeBox.setEnabled(False)
        self.EOptimizeBox.toggled.connect(self.toggle_optimize_e)
        self.EEpsilonLabel = QtGui.QLabel("Convergence Criteria:")
        self.EEpsilonLabel.setEnabled(False)
        self.EEpsilonEntry = QtGui.QLineEdit("1.0e-1")
        self.EEpsilonEntry.setEnabled(False)
        self.EEpsilonEntry.setValidator(QtGui.QDoubleValidator())
        self.EAddNoiseBox = QtGui.QCheckBox("Add Noise Kernel")
        self.EAddNoiseBox.setEnabled(False)
        self.EAddNoiseBox.toggled.connect(self.toggle_noise_kernel_e)
        self.ENoiseInitGuessLabel = QtGui.QLabel("Initial Guess")
        self.ENoiseInitGuessLabel.setEnabled(False)
        self.ENoiseLowerBoundLabel = QtGui.QLabel("Lower Bound")
        self.ENoiseLowerBoundLabel.setEnabled(False)
        self.ENoiseUpperBoundLabel = QtGui.QLabel("Upper Bound")
        self.ENoiseUpperBoundLabel.setEnabled(False)
        self.ENoiseHypLabel = QtGui.QLabel("Noise Hyperparameter:")
        self.ENoiseHypLabel.setEnabled(False)
        self.ENoiseHypEntry = QtGui.QLineEdit("1.0e-3")
        self.ENoiseHypEntry.setEnabled(False)
        self.ENoiseHypEntry.setValidator(QtGui.QDoubleValidator())
        self.ENoiseLBEntry = QtGui.QLineEdit("1.0e-3")
        self.ENoiseLBEntry.setEnabled(False)
        self.ENoiseLBEntry.setValidator(QtGui.QDoubleValidator())
        self.ENoiseUBEntry = QtGui.QLineEdit("1.0e-2")
        self.ENoiseUBEntry.setEnabled(False)
        self.ENoiseUBEntry.setValidator(QtGui.QDoubleValidator())
        self.EKernelRestartBox = QtGui.QCheckBox("Use Kernel Restarts")
        self.EKernelRestartBox.setEnabled(False)
        self.EKernelRestartBox.toggled.connect(self.toggle_kernel_restarts_e)
        self.ENRestartsLabel = QtGui.QLabel("Number of Restarts:")
        self.ENRestartsLabel.setEnabled(False)
        self.ENRestartsEntry = QtGui.QLineEdit("5")
        self.ENRestartsEntry.setEnabled(False)
        self.ENRestartsEntry.setValidator(QtGui.QIntValidator())

        self.ESEKernelSettings = SEKernelWidget(self.HeteroscedasticBox.isChecked(),self.YKernelRestartBox.isChecked())
        self.ERQKernelSettings = RQKernelWidget(self.HeteroscedasticBox.isChecked(),self.YKernelRestartBox.isChecked())
        self.EMHKernelSettings = MHKernelWidget(self.HeteroscedasticBox.isChecked(),self.YKernelRestartBox.isChecked())
        self.EGGKernelSettings = GibbsKernelWidget(self.HeteroscedasticBox.isChecked(),self.YKernelRestartBox.isChecked())

        self.EKernelSettings = QtGui.QStackedLayout()
        self.EKernelSettings.addWidget(self.ESEKernelSettings)
        self.EKernelSettings.addWidget(self.ERQKernelSettings)
        self.EKernelSettings.addWidget(self.EMHKernelSettings)
        self.EKernelSettings.addWidget(self.EGGKernelSettings)
        self.EKernelSettings.setEnabled(False)
        self.EKernelSettings.setCurrentIndex(1)

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
        ekbox.addRow(self.EAddNoiseBox)
        ekbox.addRow("",enlbox)
        ekbox.addRow(self.ENoiseHypLabel,enebox)
        ekbox.addRow(self.EKernelRestartBox)
        ekbox.addRow(self.ENRestartsLabel,self.ENRestartsEntry)

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
            dxx = None
            dyy = None
            dye = None
            dxe = None
            ndpts = self.DerivativeTable.rowCount()
            if ndpts > 0:
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
            enres = None
            if self.HeteroscedasticBox.isChecked():
                eregpar = float(self.ERegularizationEntry.text()) if self.ERegularizationEntry.text() else None
                eeps = float(self.EEpsilonEntry.text()) if self.EOptimizeBox.isChecked() and self.EEpsilonEntry.text() else 'None'
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
                self.gpr.set_error_kernel(kernel=ekernel,kbounds=ekbounds,regpar=eregpar,nrestarts=enres,epsilon=eeps)
                self.gpr.set_search_parameters(epsilon=yeps)
                self.gpr.GPRFit(xnew,nigp_flag=use_xerrs,nrestarts=ynres)
                self.fNewData = False
                toc = time.perf_counter()
                print("Fitting routine completed. Elapsed time: %.5f" % (toc - tic))
                if (yeps is not None and yeps > 0.0) or (ynres is not None and ynres > 0):
                    fhyps = self.gpr.get_gp_kernel().get_hyperparameters()
                    print("   Optimized hyp.s: ",fhyps)
            except Exception as e:
                print(repr(e))
                msg = QtGui.QMessageBox()
                msg.setIcon(QtGui.QMessageBox.Critical)
                msg.setWindowTitle("Fitting Routine Failed")
                msg.setText("Fitting routine failure: Please check inputs and try again.")
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
            self.p1.clear()
            sigma = 1.0
            rawPen = pg.mkPen(color='k',width=1)
            for ii in np.arange(0,self.DataTable.rowCount()):
                xraw = float(self.DataTable.item(ii,0).text())
                yraw = float(self.DataTable.item(ii,1).text())
                yeraw = float(self.DataTable.item(ii,2).text())
                xeraw = float(self.DataTable.item(ii,3).text())
                self.p1.plot([xraw,xraw],[yraw-sigma*yeraw,yraw+sigma*yeraw],pen=rawPen)
                if self.UseXErrorsBox.isChecked():
                    self.p1.plot([xraw-sigma*xeraw,xraw+sigma*xeraw],[yraw,yraw],pen=rawPen)
            xfit = self.gpr.get_gp_x()
            fitPen = pg.mkPen(color='r',width=1)
            errPen = pg.mkPen(color='r',width=1,style=QtCore.Qt.DashLine)
            if xfit is not None:
                yfit = self.gpr.get_gp_mean()
                yefit = self.gpr.get_gp_std(noise_flag=True)
                self.p1.plot(xfit,yfit,pen=fitPen)
                self.p1.plot(xfit,yfit-sigma*yefit,pen=errPen)
                self.p1.plot(xfit,yfit+sigma*yefit,pen=errPen)

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
