#!/usr/bin/env python

# Required imports
import os
import sys
import re
import pwd
import time
import copy
import pickle
import numpy as np
import distutils
import inspect

import matplotlib
old_mpl = distutils.version.LooseVersion(matplotlib.__version__) <= distutils.version.LooseVersion("2.0.0")

### The following is done simply for systems which do not have matplotlib-2.0.1 or greater

QtCore = None
QtGui = None
QtWidgets = None
pyqtversion = None
if sys.version_info[0] >= 3:
    try:
        if old_mpl:
            raise ImportError("Package matplotlib-2.0.0 and earlier do not support PySide2")
        from PySide2 import QtCore, QtGui, QtWidgets
        pyqtversion = QtCore.__version__
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
            pyqtversion = QtCore.PYQT_VERSION_STR
        except ImportError:
            QtGui = None
if QtGui is None:
    try:
        from PySide import QtCore, QtGui
        pyqtversion = QtCore.__version__
    except:
        try:
            from PyQt4 import QtCore, QtGui
            pyqtversion = QtCore.PYQT_VERSION_STR
        except ImportError:
            raise ImportError("No PyQt modules found.")
if QtWidgets is None:
    QtWidgets = QtGui

fqt5 = distutils.version.LooseVersion(pyqtversion) >= distutils.version.LooseVersion("5.0.0")
if fqt5:
    matplotlib.use("Qt5Agg")
    from matplotlib.backends import backend_qt5agg as mplqt
else:
    matplotlib.use("Qt4Agg")
    from matplotlib.backends import backend_qt4agg as mplqt

import GPR1D


##### Base classes to be inherited by custom implementations - do not modify! #####


class _KernelWidget(QtWidgets.QWidget):

    def __init__(self,name,fOn=True,fRestart=False):
        super(_KernelWidget, self).__init__()
        self.name = name
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.ckeys = []
        self.clabels = dict()
        self.cwidgets = dict()
        self.hkeys = []
        self.hlabels = dict()
        self.hwidgets = dict()
        self.lbwidgets = dict()
        self.ubwidgets = dict()

    def add_constant(self,key,widget,label=None):
        if isinstance(widget,QtWidgets.QLineEdit):
            self.ckeys.append(key)
            self.cwidgets[key] = widget
            if isinstance(label,QtWidgets.QLabel):
                self.clabels[key] = label
            elif label is None:
                self.clabels[key] = None
            else:
                raise TypeError("Invalid input type for constant label")
        else:
            raise TypeError("Input constant to KernelWidget must be a QLineEdit widget")

    def add_hyperparameter(self,key,widget,label=None,lbwidget=None,ubwidget=None):
        if isinstance(widget,QtWidgets.QLineEdit):
            self.hkeys.append(key)
            self.hwidgets[key] = widget
            if isinstance(label,QtWidgets.QLabel):
                self.hlabels[key] = label
            elif label is None:
                self.hlabels[key] = None
            else:
                raise TypeError("Invalid input type for hyperparameter label")
            if isinstance(lbwidget,QtWidgets.QLineEdit) and isinstance(ubwidget,QtWidgets.QLineEdit):
                self.lbwidgets[key] = lbwidget
                self.ubwidgets[key] = ubwidget
            elif lbwidget is None or ubwidget is None:
                self.lbwidgets[key] = None
                self.ubwidgets[key] = None
            else:
                raise TypeError("Invalid input type for lower / upper bound widget")
        else:
            raise TypeError("Input hyperparameter to KernelWidget must be a QLineEdit widget")

    def _remove_constant(self,key):
        try:
            self.ckeys.remove(key)
            del self.cwidgets[key]
            del self.clabels[key]
        except ValueError:
            print("Constant key %s not found in KernelWidget" % (key))

    def _remove_hyperparameter(self,key):
        try:
            self.hkeys.remove(key)
            del self.hwidgets[key]
            del self.hlabels[key]
            del self.lbwidgets[key]
            del self.ubwidgets[key]
        except ValueError:
            print("Hyperparameter key %s not found in KernelWidget" % (key))

    def make_layout(self):
        cbox = None
        hbox = None
        if len(self.ckeys) > 0:
            cbox = QtWidgets.QFormLayout()
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.clabels[self.ckeys[ii]],QtWidgets.QLabel):
                    cbox.addRow(self.clabels[self.ckeys[ii]],self.cwidgets[self.ckeys[ii]])
                else:
                    cbox.addRow(self.ckeys[ii],self.cwidgets[self.ckeys[ii]])

        if len(self.hkeys) > 0:
            self.InitGuessLabel = QtWidgets.QLabel("Initial Guess")
            self.InitGuessLabel.setEnabled(self.aflag)
            self.LowerBoundLabel = QtWidgets.QLabel("Lower Bound")
            self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
            self.UpperBoundLabel = QtWidgets.QLabel("Upper Bound")
            self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

            hbox = QtWidgets.QGridLayout()
            hbox.addWidget(self.InitGuessLabel,0,1)
            hbox.addWidget(self.LowerBoundLabel,0,2)
            hbox.addWidget(self.UpperBoundLabel,0,3)
            for ii in np.arange(0,len(self.hkeys)):
                if self.hlabels[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.hlabels[self.hkeys[ii]],ii+1,0)
                if self.hwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.hwidgets[self.hkeys[ii]],ii+1,1)
                if self.lbwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.lbwidgets[self.hkeys[ii]],ii+1,2)
                if self.ubwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.ubwidgets[self.hkeys[ii]],ii+1,3)

        layoutBox = None
        if isinstance(cbox,QtWidgets.QLayout) and isinstance(hbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(cbox)
            layoutBox.addLayout(hbox)
        elif isinstance(cbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(cbox)
        elif isinstance(hbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(hbox)
        else:
            print("No parameters added to KernelWidget, layout cannot be made")
        if isinstance(layoutBox,QtWidgets.QVBoxLayout):
            layoutBox.addStretch(1)

        return layoutBox

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.hkeys)):
            if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.lbwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.ubwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        for ii in np.arange(0,len(self.ckeys)):
            if isinstance(self.cwidgets[self.ckeys[ii]],QtWidgets.QWidget):
                self.cwidgets[self.ckeys[ii]].setEnabled(self.aflag)
            if isinstance(self.clabels[self.ckeys[ii]],QtWidgets.QWidget):
                self.clabels[self.ckeys[ii]].setEnabled(self.aflag)
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.hkeys)):
            if isinstance(self.hwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.hwidgets[self.hkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.hlabels[self.hkeys[ii]],QtWidgets.QWidget):
                self.hlabels[self.hkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.lbwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.ubwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            csts = []
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.cwidgets[self.ckeys[ii]],QtWidgets.QLineEdit):
                    csts.append(float(self.cwidgets[self.ckeys[ii]].text()))
            hyps = []
            for ii in np.arange(0,len(self.hkeys)):
                if isinstance(self.hwidgets[self.hkeys[ii]],QtWidgets.QLineEdit):
                    hyps.append(float(self.hwidgets[self.hkeys[ii]].text()))
            hyps = np.array(hyps).flatten()
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            for ii in np.arange(0,len(self.hkeys)):
                if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QLineEdit) and isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QLineEdit):
                    bounds.append([float(self.lbwidgets[self.hkeys[ii]].text()),float(self.ubwidgets[self.hkeys[ii]].text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class _WarpFunctionWidget(QtWidgets.QWidget):

    def __init__(self,name,fOn=True,fRestart=False):
        super(_WarpFunctionWidget,self).__init__()
        self.name = name
        self.aflag = True if fOn else False
        self.bflag = True if fRestart else False
        self.ckeys = []
        self.clabels = dict()
        self.cwidgets = dict()
        self.hkeys = []
        self.hlabels = dict()
        self.hwidgets = dict()
        self.lbwidgets = dict()
        self.ubwidgets = dict()

    def add_constant(self,key,widget,label=None):
        if isinstance(widget,QtWidgets.QLineEdit):
            self.ckeys.append(key)
            self.cwidgets[key] = widget
            if isinstance(label,QtWidgets.QLabel):
                self.clabels[key] = label
            elif label is None:
                self.clabels[key] = None
            else:
                raise TypeError("Invalid input type for constant label")
        else:
            raise TypeError("Input constant to WarpFunctionWidget must be a QLineEdit widget")

    def add_hyperparameter(self,key,widget,label=None,lbwidget=None,ubwidget=None):
        if isinstance(widget,QtWidgets.QLineEdit):
            self.hkeys.append(key)
            self.hwidgets[key] = widget
            if isinstance(label,QtWidgets.QLabel):
                self.hlabels[key] = label
            elif label is None:
                self.hlabels[key] = None
            else:
                raise TypeError("Invalid input type for hyperparameter label")
            if isinstance(lbwidget,QtWidgets.QLineEdit) and isinstance(ubwidget,QtWidgets.QLineEdit):
                self.lbwidgets[key] = lbwidget
                self.ubwidgets[key] = ubwidget
            elif lbwidget is None or ubwidget is None:
                self.lbwidgets[key] = None
                self.ubwidgets[key] = None
            else:
                raise TypeError("Invalid input type for lower / upper bound widget")
        else:
            raise TypeError("Input hyperparameter to WarpFunctionWidget must be a QLineEdit widget")

    def _remove_constant(self,key):
        try:
            self.ckeys.remove(key)
            del self.cwidgets[key]
            del self.clabels[key]
        except ValueError:
            print("Constant key %s not found in WarpFunctionWidget" % (key))

    def _remove_hyperparameter(self,key):
        try:
            self.hkeys.remove(key)
            del self.hwidgets[key]
            del self.hlabels[key]
            del self.lbwidgets[key]
            del self.ubwidgets[key]
        except ValueError:
            print("Hyperparameter key %s not found in WarpFunctionWidget" % (key))

    def make_layout(self):
        cbox = None
        hbox = None

        if len(self.hkeys) > 0:
            self.InitGuessLabel = QtWidgets.QLabel("Initial Guess")
            self.InitGuessLabel.setEnabled(self.aflag)
            self.LowerBoundLabel = QtWidgets.QLabel("Lower Bound")
            self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
            self.UpperBoundLabel = QtWidgets.QLabel("Upper Bound")
            self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)

            hbox = QtWidgets.QGridLayout()
            hbox.addWidget(self.InitGuessLabel,0,1)
            hbox.addWidget(self.LowerBoundLabel,0,2)
            hbox.addWidget(self.UpperBoundLabel,0,3)
            for ii in np.arange(0,len(self.hkeys)):
                if self.hlabels[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.hlabels[self.hkeys[ii]],ii+1,0)
                if self.hwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.hwidgets[self.hkeys[ii]],ii+1,1)
                if self.lbwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.lbwidgets[self.hkeys[ii]],ii+1,2)
                if self.ubwidgets[self.hkeys[ii]] is not None:
                    hbox.addWidget(self.ubwidgets[self.hkeys[ii]],ii+1,3)

        if len(self.ckeys) > 0:
            cbox = QtWidgets.QFormLayout()
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.clabels[self.ckeys[ii]],QtWidgets.QLabel):
                    cbox.addRow(self.clabels[self.ckeys[ii]],self.cwidgets[self.ckeys[ii]])
                else:
                    cbox.addRow(self.ckeys[ii],self.cwidgets[self.ckeys[ii]])

        layoutBox = None
        if isinstance(hbox,QtWidgets.QLayout) and isinstance(cbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(hbox)
            layoutBox.addLayout(cbox)
        elif isinstance(hbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(hbox)
        elif isinstance(cbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(cbox)
        else:
            print("No parameters added to WarpFunctionWidget, layout cannot be made")
        if isinstance(layoutBox,QtWidgets.QVBoxLayout):
            layoutBox.addStretch(1)

        return layoutBox

    def toggle_bounds(self,tRestart=None):
        if tRestart is None:
            self.bflag = (not self.bflag)
        else:
            self.bflag = True if tRestart else False
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.hkeys)):
            if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.lbwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.ubwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        for ii in np.arange(0,len(self.ckeys)):
            if isinstance(self.cwidgets[self.ckeys[ii]],QtWidgets.QWidget):
                self.cwidgets[self.ckeys[ii]].setEnabled(self.aflag)
            if isinstance(self.clabels[self.ckeys[ii]],QtWidgets.QWidget):
                self.clabels[self.ckeys[ii]].setEnabled(self.aflag)
        self.InitGuessLabel.setEnabled(self.aflag)
        self.LowerBoundLabel.setEnabled(self.aflag and self.bflag)
        self.UpperBoundLabel.setEnabled(self.aflag and self.bflag)
        for ii in np.arange(0,len(self.hkeys)):
            if isinstance(self.hwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.hwidgets[self.hkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.hlabels[self.hkeys[ii]],QtWidgets.QWidget):
                self.hlabels[self.hkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.lbwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)
            if isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QWidget):
                self.ubwidgets[self.hkeys[ii]].setEnabled(self.aflag and self.bflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_initial_guess(self):
        hyps = None
        csts = None
        if self.aflag:
            csts = []
            for ii in np.arange(0,len(self.ckeys)):
                if isinstance(self.cwidgets[self.ckeys[ii]],QtWidgets.QLineEdit):
                    csts.append(float(self.cwidgets[self.ckeys[ii]].text()))
            hyps = []
            for ii in np.arange(0,len(self.hkeys)):
                if isinstance(self.hwidgets[self.hkeys[ii]],QtWidgets.QLineEdit):
                    hyps.append(float(self.hwidgets[self.hkeys[ii]].text()))
            hyps = np.array(hyps).flatten()
            csts = np.array(csts).flatten()
        return (hyps,csts)

    def get_bounds(self):
        bounds = None
        if self.aflag and self.bflag:
            bounds = []
            for ii in np.arange(0,len(self.hkeys)):
                if isinstance(self.lbwidgets[self.hkeys[ii]],QtWidgets.QLineEdit) and isinstance(self.ubwidgets[self.hkeys[ii]],QtWidgets.QLineEdit):
                    bounds.append([float(self.lbwidgets[self.hkeys[ii]].text()),float(self.ubwidgets[self.hkeys[ii]].text())])
            bounds = np.atleast_2d(bounds)
        return bounds


class _OptimizerWidget(QtWidgets.QWidget):

    def __init__(self,name,fOn=True):
        super(_OptimizerWidget,self).__init__()
        self.name = name
        self.aflag = True if fOn else False
        self.pkeys = []
        self.plabels = dict()
        self.pwidgets = dict()

    def add_parameter(self,key,widget,label=None):
        if isinstance(widget,QtWidgets.QLineEdit):
            self.pkeys.append(key)
            self.pwidgets[key] = widget
            if isinstance(label,QtWidgets.QLabel):
                self.plabels[key] = label
            elif label is None:
                self.plabels[key] = None
            else:
                raise TypeError("Invalid input type for parameter label")
        else:
            raise TypeError("Input parameter to OptimizerWidget must be a QLineEdit widget")

    def _remove_parameter(self,key):
        try:
            self.pkeys.remove(key)
            del self.pwidgets[key]
            del self.plabels[key]
        except ValueError:
            print("Parameter key %s not found in OptimizerWidget" % (key))

    def make_layout(self):
        pbox = None

        if len(self.pkeys) > 0:
            pbox = QtWidgets.QFormLayout()
            for ii in np.arange(0,len(self.pkeys)):
                if isinstance(self.plabels[self.pkeys[ii]],QtWidgets.QLabel):
                    pbox.addRow(self.plabels[self.pkeys[ii]],self.pwidgets[self.pkeys[ii]])
                else:
                    pbox.addRow(self.pkeys[ii],self.pwidgets[self.pkeys[ii]])

        layoutBox = None
        if isinstance(pbox,QtWidgets.QLayout):
            layoutBox = QtWidgets.QVBoxLayout()
            layoutBox.addLayout(pbox)
        else:
            print("No parameters added to OptimizerWidget, layout cannot be made")
        if isinstance(layoutBox,QtWidgets.QVBoxLayout):
            layoutBox.addStretch(1)

        return layoutBox

    def toggle_all(self,tOn=None):
        if tOn is None:
            self.aflag = (not self.aflag)
        else:
            self.aflag = True if tOn else False
        for ii in np.arange(0,len(self.pkeys)):
            if isinstance(self.pwidgets[self.pkeys[ii]],QtWidgets.QWidget):
                self.pwidgets[self.pkeys[ii]].setEnabled(self.aflag)
            if isinstance(self.plabels[self.pkeys[ii]],QtWidgets.QWidget):
                self.plabels[self.pkeys[ii]].setEnabled(self.aflag)

    def get_name(self):
        name = self.name if self.aflag else None
        return name

    def get_parameters(self):
        pars = None
        if self.aflag:
            pars = []
            for ii in np.arange(0,len(self.pkeys)):
                if isinstance(self.pwidgets[self.pkeys[ii]],QtWidgets.QLineEdit):
                    pars.append(float(self.pwidgets[self.pkeys[ii]].text()))
            pars = np.array(pars).flatten()
        return pars


##### Custom implementations to be placed below #####


class SEKernelWidget(_KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(SEKernelWidget,self).__init__("SE",fOn,fRestart)
        self.SEKernelUI()

    def SEKernelUI(self):

        SigmaHypLabel = QtWidgets.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtWidgets.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        kbox = self.make_layout()
        self.setLayout(kbox)


class RQKernelWidget(_KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(RQKernelWidget,self).__init__("RQ",fOn,fRestart)
        self.RQKernelUI()

    def RQKernelUI(self):

        SigmaHypLabel = QtWidgets.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtWidgets.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        AlphaHypLabel = QtWidgets.QLabel("Exponent:")
        AlphaHypLabel.setEnabled(self.aflag)
        AlphaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        AlphaHypEntry = QtWidgets.QLineEdit("1.0e0")
        AlphaHypEntry.setEnabled(self.aflag)
        AlphaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        AlphaLBEntry = QtWidgets.QLineEdit("1.0e0")
        AlphaLBEntry.setEnabled(self.aflag and self.bflag)
        AlphaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        AlphaUBEntry = QtWidgets.QLineEdit("1.0e0")
        AlphaUBEntry.setEnabled(self.aflag and self.bflag)
        AlphaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('alpha',AlphaHypEntry,label=AlphaHypLabel,lbwidget=AlphaLBEntry,ubwidget=AlphaUBEntry)

        kbox = self.make_layout()
        self.setLayout(kbox)


class MHKernelWidget(_KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(MHKernelWidget,self).__init__("MH",fOn,fRestart)
        self.MHKernelUI()

    def MHKernelUI(self):

        NuCstLabel = QtWidgets.QLabel("Integer:")
        NuCstLabel.setEnabled(self.aflag)
        NuCstLabel.setAlignment(QtCore.Qt.AlignRight)
        NuCstEntry = QtWidgets.QLineEdit("2")
        NuCstEntry.setEnabled(self.aflag)
        NuCstEntry.setValidator(QtGui.QIntValidator(0,100,None))
        self.add_constant('nu',NuCstEntry,label=NuCstLabel)

        SigmaHypLabel = QtWidgets.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('sigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        LengthHypLabel = QtWidgets.QLabel("Length:")
        LengthHypLabel.setEnabled(self.aflag)
        LengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        LengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        LengthHypEntry.setEnabled(self.aflag)
        LengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthLBEntry.setEnabled(self.aflag and self.bflag)
        LengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        LengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        LengthUBEntry.setEnabled(self.aflag and self.bflag)
        LengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('length',LengthHypEntry,label=LengthHypLabel,lbwidget=LengthLBEntry,ubwidget=LengthUBEntry)

        kbox = self.make_layout()
        self.setLayout(kbox)


class GibbsKernelWidget(_KernelWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(GibbsKernelWidget,self).__init__("Gw",fOn,fRestart)
        self.GibbsKernelUI()

    def GibbsKernelUI(self):

        SigmaHypLabel = QtWidgets.QLabel("Amplitude:")
        SigmaHypLabel.setEnabled(self.aflag)
        SigmaHypLabel.setAlignment(QtCore.Qt.AlignRight)
        SigmaHypEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaHypEntry.setEnabled(self.aflag)
        SigmaHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaLBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaLBEntry.setEnabled(self.aflag and self.bflag)
        SigmaLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        SigmaUBEntry = QtWidgets.QLineEdit("1.0e0")
        SigmaUBEntry.setEnabled(self.aflag and self.bflag)
        SigmaUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('gsigma',SigmaHypEntry,label=SigmaHypLabel,lbwidget=SigmaLBEntry,ubwidget=SigmaUBEntry)

        kbox = self.make_layout()

        self.WarpFuncSelectionLabel = QtWidgets.QLabel("Warping Function:")
        self.WarpFuncSelectionList = QtWidgets.QComboBox()
        self.WarpFuncSelectionList.addItem("Constant Function")
        self.WarpFuncSelectionList.addItem("Inverse Gaussian")
        self.WarpFuncSelectionList.setCurrentIndex(1)
        self.WarpFuncSelectionList.currentIndexChanged.connect(self.switch_warpfunc_ui)

        wbox = QtWidgets.QFormLayout()
        wbox.addRow(self.WarpFuncSelectionLabel,self.WarpFuncSelectionList)

        self.CWarpFuncSettings = CWarpFunctionWidget(self.aflag,self.bflag)
        self.IGWarpFuncSettings = IGWarpFunctionWidget(self.aflag,self.bflag)

        self.WarpFuncSettings = QtWidgets.QStackedLayout()
        self.WarpFuncSettings.addWidget(self.CWarpFuncSettings)
        self.WarpFuncSettings.addWidget(self.IGWarpFuncSettings)
        self.WarpFuncSettings.setCurrentIndex(self.WarpFuncSelectionList.currentIndex())

        tbox = QtWidgets.QVBoxLayout()
        tbox.addLayout(kbox)
        tbox.addLayout(wbox)
        tbox.addLayout(self.WarpFuncSettings)

        self.setLayout(tbox)

    def switch_warpfunc_ui(self):
        self.WarpFuncSettings.setCurrentIndex(self.WarpFuncSelectionList.currentIndex())

    def toggle_bounds(self,tRestart=None):
        super(GibbsKernelWidget,self).toggle_bounds(tRestart)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_bounds(self.bflag)

    def toggle_all(self,tOn=None):
        super(GibbsKernelWidget,self).toggle_all(tOn)
        self.WarpFuncSelectionLabel.setEnabled(self.aflag)
        self.WarpFuncSelectionList.setEnabled(self.aflag)
        for ii in np.arange(0,self.WarpFuncSettings.count()):
            self.WarpFuncSettings.widget(ii).toggle_all(self.aflag)

    def get_name(self):
        name = super(GibbsKernelWidget,self).get_name()
        if name is not None:
            idx = self.WarpFuncSettings.currentIndex()
            wname = self.WarpFuncSettings.widget(idx).get_name()
            name = name + wname
        return name

    def get_initial_guess(self):
        (hyps,csts) = super(GibbsKernelWidget,self).get_initial_guess()
        if hyps is not None and csts is not None:
            idx = self.WarpFuncSettings.currentIndex()
            (whyps,wcsts) = self.WarpFuncSettings.widget(idx).get_initial_guess()
            if whyps is not None:
                hyps = np.hstack((hyps,whyps))
            if wcsts is not None:
                csts = np.hstack((csts,wcsts))
        return (hyps,csts)

    def get_bounds(self):
        bounds = super(GibbsKernelWidget,self).get_bounds()
        if bounds is not None:
            idx = self.WarpFuncSettings.currentIndex()
            wbounds = self.WarpFuncSettings.widget(idx).get_bounds()
            if wbounds is not None:
                bounds = np.vstack((bounds,wbounds))
        return bounds


class CWarpFunctionWidget(_WarpFunctionWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(CWarpFunctionWidget,self).__init__("C",fOn,fRestart)
        self.CWarpFunctionUI()

    def CWarpFunctionUI(self):

        ConstantLengthHypLabel = QtWidgets.QLabel("Base Length:")
        ConstantLengthHypLabel.setEnabled(self.aflag)
        ConstantLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        ConstantLengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        ConstantLengthHypEntry.setEnabled(self.aflag)
        ConstantLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        ConstantLengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        ConstantLengthLBEntry.setEnabled(self.aflag and self.bflag)
        ConstantLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        ConstantLengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        ConstantLengthUBEntry.setEnabled(self.aflag and self.bflag)
        ConstantLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('constl',ConstantLengthHypEntry,label=ConstantLengthHypLabel,lbwidget=ConstantLengthLBEntry,ubwidget=ConstantLengthUBEntry)

        wbox = self.make_layout()
        self.setLayout(wbox)


class IGWarpFunctionWidget(_WarpFunctionWidget):

    def __init__(self,fOn=True,fRestart=False):
        super(IGWarpFunctionWidget,self).__init__("IG",fOn,fRestart)
        self.IGWarpFunctionUI()

    def IGWarpFunctionUI(self):

        BaseLengthHypLabel = QtWidgets.QLabel("Base Length:")
        BaseLengthHypLabel.setEnabled(self.aflag)
        BaseLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        BaseLengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        BaseLengthHypEntry.setEnabled(self.aflag)
        BaseLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        BaseLengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        BaseLengthLBEntry.setEnabled(self.aflag and self.bflag)
        BaseLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        BaseLengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        BaseLengthUBEntry.setEnabled(self.aflag and self.bflag)
        BaseLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('basel',BaseLengthHypEntry,label=BaseLengthHypLabel,lbwidget=BaseLengthLBEntry,ubwidget=BaseLengthUBEntry)

        PeakLengthHypLabel = QtWidgets.QLabel("Gaussian Height:")
        PeakLengthHypLabel.setEnabled(self.aflag)
        PeakLengthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        PeakLengthHypEntry = QtWidgets.QLineEdit("1.0e0")
        PeakLengthHypEntry.setEnabled(self.aflag)
        PeakLengthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        PeakLengthLBEntry = QtWidgets.QLineEdit("1.0e0")
        PeakLengthLBEntry.setEnabled(self.aflag and self.bflag)
        PeakLengthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        PeakLengthUBEntry = QtWidgets.QLineEdit("1.0e0")
        PeakLengthUBEntry.setEnabled(self.aflag and self.bflag)
        PeakLengthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('peakl',PeakLengthHypEntry,label=PeakLengthHypLabel,lbwidget=PeakLengthLBEntry,ubwidget=PeakLengthUBEntry)

        PeakWidthHypLabel = QtWidgets.QLabel("Gaussian Width:")
        PeakWidthHypLabel.setEnabled(self.aflag)
        PeakWidthHypLabel.setAlignment(QtCore.Qt.AlignRight)
        PeakWidthHypEntry = QtWidgets.QLineEdit("1.0e0")
        PeakWidthHypEntry.setEnabled(self.aflag)
        PeakWidthHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        PeakWidthLBEntry = QtWidgets.QLineEdit("1.0e0")
        PeakWidthLBEntry.setEnabled(self.aflag and self.bflag)
        PeakWidthLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        PeakWidthUBEntry = QtWidgets.QLineEdit("1.0e0")
        PeakWidthUBEntry.setEnabled(self.aflag and self.bflag)
        PeakWidthUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_hyperparameter('peakw',PeakWidthHypEntry,label=PeakWidthHypLabel,lbwidget=PeakWidthLBEntry,ubwidget=PeakWidthUBEntry)

        MuCstLabel = QtWidgets.QLabel("Gaussian Peak Location:")
        MuCstLabel.setEnabled(self.aflag)
        MuCstEntry = QtWidgets.QLineEdit("1.0e0")
        MuCstEntry.setEnabled(self.aflag)
        MuCstEntry.setValidator(QtGui.QDoubleValidator(None))
        self.add_constant('mu',MuCstEntry,label=MuCstLabel)

        MaxFracCstLabel = QtWidgets.QLabel("Maximum Peak-to-Base Ratio:")
        MaxFracCstLabel.setEnabled(self.aflag)
        MaxFracCstEntry = QtWidgets.QLineEdit("0.5")
        MaxFracCstEntry.setEnabled(self.aflag)
        MaxFracCstEntry.setValidator(QtGui.QDoubleValidator(0.0,1.0,100,None))
        self.add_constant('maxfrac',MaxFracCstEntry,label=MaxFracCstLabel)

        wbox = self.make_layout()
        self.setLayout(wbox)


class GradAscentOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(GradAscentOptimizerWidget,self).__init__('grad',fOn)
        self.GradOptUI()

    def GradOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-4")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        obox = self.make_layout()
        self.setLayout(obox)


class MomentumOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(MomentumOptimizerWidget,self).__init__('mom',fOn)
        self.MomOptUI()

    def MomOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-4")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        MomentumLabel = QtWidgets.QLabel("Momentum Factor:")
        MomentumLabel.setEnabled(self.aflag)
        MomentumEntry = QtWidgets.QLineEdit("0.9")
        MomentumEntry.setEnabled(self.aflag)
        MomentumEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('momentum',MomentumEntry,label=MomentumLabel)

        obox = self.make_layout()
        self.setLayout(obox)


class NesterovOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(NesterovOptimizerWidget,self).__init__('nag',fOn)
        self.NagOptUI()

    def NagOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-4")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        MomentumLabel = QtWidgets.QLabel("Momentum Factor:")
        MomentumLabel.setEnabled(self.aflag)
        MomentumEntry = QtWidgets.QLineEdit("0.9")
        MomentumEntry.setEnabled(self.aflag)
        MomentumEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('momentum',MomentumEntry,label=MomentumLabel)

        obox = self.make_layout()
        self.setLayout(obox)


class AdagradOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(AdagradOptimizerWidget,self).__init__('adagrad',fOn)
        self.AdagradOptUI()

    def AdagradOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-2")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        obox = self.make_layout()
        self.setLayout(obox)


class AdadeltaOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(AdadeltaOptimizerWidget,self).__init__('adadelta',fOn)
        self.AdadeltaOptUI()

    def AdadeltaOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-2")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        ForgetLabel = QtWidgets.QLabel("Forgetting Factor:")
        ForgetLabel.setEnabled(self.aflag)
        ForgetEntry = QtWidgets.QLineEdit("0.9")
        ForgetEntry.setEnabled(self.aflag)
        ForgetEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta',ForgetEntry,label=ForgetLabel)

        obox = self.make_layout()
        self.setLayout(obox)


class AdamOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(AdamOptimizerWidget, self).__init__('adam',fOn)
        self.AdamOptUI()

    def AdamOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-3")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        Beta1Label = QtWidgets.QLabel("Gradient Factor:")
        Beta1Label.setEnabled(self.aflag)
        Beta1Entry = QtWidgets.QLineEdit("0.9")
        Beta1Entry.setEnabled(self.aflag)
        Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta1',Beta1Entry,label=Beta1Label)

        Beta2Label = QtWidgets.QLabel("Sq. Gradient Factor:")
        Beta2Label.setEnabled(self.aflag)
        Beta2Entry = QtWidgets.QLineEdit("0.999")
        Beta2Entry.setEnabled(self.aflag)
        Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta2',Beta2Entry,label=Beta2Label)

        obox = self.make_layout()
        self.setLayout(obox)


class AdamaxOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(AdamaxOptimizerWidget,self).__init__('adamax',fOn)
        self.AdamaxOptUI()

    def AdamaxOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("2.0e-3")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        Beta1Label = QtWidgets.QLabel("Gradient Factor:")
        Beta1Label.setEnabled(self.aflag)
        Beta1Entry = QtWidgets.QLineEdit("0.9")
        Beta1Entry.setEnabled(self.aflag)
        Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta1',Beta1Entry,label=Beta1Label)

        Beta2Label = QtWidgets.QLabel("Sq. Gradient Factor:")
        Beta2Label.setEnabled(self.aflag)
        Beta2Entry = QtWidgets.QLineEdit("0.999")
        Beta2Entry.setEnabled(self.aflag)
        Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta2',Beta2Entry,label=Beta2Label)

        obox = self.make_layout()
        self.setLayout(obox)


class NadamOptimizerWidget(_OptimizerWidget):

    def __init__(self,fOn=True):
        super(NadamOptimizerWidget,self).__init__('nadam',fOn)
        self.NadamOptUI()

    def NadamOptUI(self):

        GainLabel = QtWidgets.QLabel("Gain Factor:")
        GainLabel.setEnabled(self.aflag)
        GainEntry = QtWidgets.QLineEdit("1.0e-3")
        GainEntry.setEnabled(self.aflag)
        GainEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('gain',GainEntry,label=GainLabel)

        Beta1Label = QtWidgets.QLabel("Gradient Factor:")
        Beta1Label.setEnabled(self.aflag)
        Beta1Entry = QtWidgets.QLineEdit("0.9")
        Beta1Entry.setEnabled(self.aflag)
        Beta1Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta1',Beta1Entry,label=Beta1Label)

        Beta2Label = QtWidgets.QLabel("Sq. Gradient Factor:")
        Beta2Label.setEnabled(self.aflag)
        Beta2Entry = QtWidgets.QLineEdit("0.999")
        Beta2Entry.setEnabled(self.aflag)
        Beta2Entry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.add_parameter('beta2',Beta2Entry,label=Beta2Label)

        obox = self.make_layout()
        self.setLayout(obox)


class QCustomTableWidgetItem(QtWidgets.QTableWidgetItem):

    def __init__(self, value):
        super(QCustomTableWidgetItem, self).__init__('%s' % value)

    def __lt__(self, other):
        if (isinstance(other, QCustomTableWidgetItem)):
            selfDataValue = float(self.data(QtCore.Qt.EditRole))
            otherDataValue = float(other.data(QtCore.Qt.EditRole))
            return selfDataValue < otherDataValue
        else:
            return QtWidgets.QTableWidgetItem.__lt__(self, other)


class GPR1D_GUI(QtWidgets.QWidget):

    def __init__(self):
        super(GPR1D_GUI, self).__init__()
        self.fNewData = False
        self.gpr = GPR1D.GaussianProcessRegression1D()
        location = inspect.getsourcefile(type(self.gpr))
        self.srcdir = os.path.dirname(location) + '/'
        print("Using GPR1D definition from: %s" % (self.srcdir))
        self.initUI()

    def initUI(self):

        self.TabPanel = QtWidgets.QTabWidget()

        self.DataEntryTab = QtWidgets.QWidget()
        self.DataEntryUI()

        self.YKernelSelectionTab = QtWidgets.QWidget()
        self.YKernelSelectionUI()

        self.EKernelSelectionTab = QtWidgets.QWidget()
        self.EKernelSelectionUI()

        self.TabPanel.addTab(self.DataEntryTab,"Data Entry")
        self.TabPanel.addTab(self.YKernelSelectionTab,"Fit Kernel")
        self.TabPanel.addTab(self.EKernelSelectionTab,"Error Kernel")

        self.PredictStartLabel = QtWidgets.QLabel("Start:")
        self.PredictStartEntry = QtWidgets.QLineEdit("0.0")
        self.PredictStartEntry.setValidator(QtGui.QDoubleValidator(None))
        self.PredictEndLabel = QtWidgets.QLabel("End:")
        self.PredictEndEntry = QtWidgets.QLineEdit("1.0")
        self.PredictEndEntry.setValidator(QtGui.QDoubleValidator(None))
        self.PredictNPointsLabel = QtWidgets.QLabel("Points:")
        self.PredictNPointsEntry = QtWidgets.QLineEdit("100")
        self.PredictNPointsEntry.setValidator(QtGui.QIntValidator(1,100000,None))

        xnbox = QtWidgets.QHBoxLayout()
        xnbox.addWidget(self.PredictStartLabel)
        xnbox.addWidget(self.PredictStartEntry)
        xnbox.addWidget(self.PredictEndLabel)
        xnbox.addWidget(self.PredictEndEntry)
        xnbox.addWidget(self.PredictNPointsLabel)
        xnbox.addWidget(self.PredictNPointsEntry)

        self.FitDataButton = QtWidgets.QPushButton("Fit Data")
        self.FitDataButton.clicked.connect(self._fit_data)
        self.PlotDataButton = QtWidgets.QPushButton("Plot Data")
        self.PlotDataButton.clicked.connect(self._plot_data)
        self.SaveRawDataButton = QtWidgets.QPushButton("Save Raw Data")
        self.SaveRawDataButton.clicked.connect(self._save_raw_data)
        self.SaveFitDataButton = QtWidgets.QPushButton("Save Fit Data")
        self.SaveFitDataButton.clicked.connect(self._save_fit_data)

        dobox = QtWidgets.QHBoxLayout()
        dobox.addWidget(self.FitDataButton)
        dobox.addWidget(self.PlotDataButton)
        dobox.addWidget(self.SaveRawDataButton)
        dobox.addWidget(self.SaveFitDataButton)

        fig = matplotlib.figure.Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
        self.p1 = mplqt.FigureCanvasQTAgg(fig)
        self.p1.figure.patch.set_facecolor('w')
        self.p1.figure.tight_layout()

        vbox = QtWidgets.QVBoxLayout()
        vbox.addLayout(xnbox)
        vbox.addLayout(dobox)
        vbox.addWidget(self.p1)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.TabPanel)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

        self.setGeometry(20, 20, 1500, 700)
        self.setWindowTitle("GPR1D GUI")
        self.show()

    def DataEntryUI(self):

        self.AddYDataButton = QtWidgets.QPushButton("Add Y Data")
        self.AddYDataButton.clicked.connect(self._add_data)
        self.AddDDataButton = QtWidgets.QPushButton("Add dY Data")
        self.AddDDataButton.clicked.connect(self._add_derivative_data)
        self.LoadDataButton = QtWidgets.QPushButton("Load Data")
        self.LoadDataButton.clicked.connect(self._load_data)

        dabox = QtWidgets.QHBoxLayout()
        dabox.addWidget(self.AddYDataButton)
        dabox.addWidget(self.AddDDataButton)
        dabox.addWidget(self.LoadDataButton)

        self.SortDataButton = QtWidgets.QPushButton("Sort Data")
        self.SortDataButton.clicked.connect(self._sort_data)
        self.CleanDataButton = QtWidgets.QPushButton("Clean Data")
        self.CleanDataButton.clicked.connect(self._clean_data)
        self.ClearDataButton = QtWidgets.QPushButton("Clear Data")
        self.ClearDataButton.clicked.connect(self._clear_data)

        dbbox = QtWidgets.QHBoxLayout()
        dbbox.addWidget(self.SortDataButton)
        dbbox.addWidget(self.CleanDataButton)
        dbbox.addWidget(self.ClearDataButton)

        self.DataTable = QtWidgets.QTableWidget()
        self.DataTable.setColumnCount(4)
        self.DataTable.setHorizontalHeaderLabels("X;Y;Y Err.;X Err.".split(";"))
        self.DataTable.horizontalHeader().hideSection(3)
        if fqt5:
            self.DataTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        else:
            self.DataTable.horizontalHeader().setResizeMode(QtWidgets.QHeaderView.Stretch)
        self.DataTable.cellChanged.connect(self._flag_new_data)
        self.DataTable.itemChanged.connect(self._flag_new_data)
        self.DerivativeBox = QtWidgets.QCheckBox("Use derivative constraints")
        self.DerivativeBox.toggled.connect(self._toggle_derivatives)
        self.DerivativeTable = QtWidgets.QTableWidget()
        self.DerivativeTable.setEnabled(False)
        self.DerivativeTable.setColumnCount(4)
        self.DerivativeTable.setHorizontalHeaderLabels("X;dY;dY Err.;X Err.".split(";"))
        self.DerivativeTable.horizontalHeader().hideSection(3)
        if fqt5:
            self.DerivativeTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        else:
            self.DerivativeTable.horizontalHeader().setResizeMode(QtWidgets.QHeaderView.Stretch)
        self.DerivativeTable.cellChanged.connect(self._flag_new_data)
        self.DerivativeTable.itemChanged.connect(self._flag_new_data)
        self.UseXErrorsBox = QtWidgets.QCheckBox("Use x-errors")
        self.UseXErrorsBox.toggled.connect(self._toggle_xerror_display)

        debox = QtWidgets.QVBoxLayout()
        debox.addLayout(dabox)
        debox.addLayout(dbbox)
        debox.addWidget(self.DataTable)
        debox.addWidget(self.DerivativeBox)
        debox.addWidget(self.DerivativeTable)
        debox.addWidget(self.UseXErrorsBox)

        self.DataEntryTab.setLayout(debox)

    def YKernelSelectionUI(self):

        self.YKernelSelectionLabel = QtWidgets.QLabel("Kernel:")
        self.YKernelSelectionList = QtWidgets.QComboBox()
        self.YKernelSelectionList.addItem("Squared Exponential")
        self.YKernelSelectionList.addItem("Rational Quadratic")
        self.YKernelSelectionList.addItem("Matern Half-Integer")
        self.YKernelSelectionList.addItem("Gibbs Kernel")
        self.YKernelSelectionList.setCurrentIndex(0)
        self.YKernelSelectionList.currentIndexChanged.connect(self._switch_kernel_ui_y)

        self.YOptimizeBox = QtWidgets.QCheckBox("Optimize")
        self.YOptimizeBox.toggled.connect(self._toggle_optimize_y)
        self.YAddNoiseBox = QtWidgets.QCheckBox("Add Noise Kernel")
        self.YAddNoiseBox.toggled.connect(self._toggle_noise_kernel_y)
        self.YKernelRestartBox = QtWidgets.QCheckBox("Use Kernel Restarts")
        self.YKernelRestartBox.toggled.connect(self._toggle_kernel_restarts_y)

        self.YRegularizationLabel = QtWidgets.QLabel("Reg. Parameter:")
        self.YRegularizationEntry = QtWidgets.QLineEdit("1.0")
        self.YRegularizationEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YEpsilonLabel = QtWidgets.QLabel("Convergence Criteria:")
        self.YEpsilonLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry = QtWidgets.QLineEdit("1.0e-3")
        self.YEpsilonEntry.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        self.YOptimizerSelectionLabel = QtWidgets.QLabel("Optimizer:")
        self.YOptimizerSelectionLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionList = QtWidgets.QComboBox()
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
        self.YOptimizerSelectionList.currentIndexChanged.connect(self._switch_optimizer_ui_y)

        self.YNoiseInitGuessLabel = QtWidgets.QLabel("Initial Guess")
        self.YNoiseInitGuessLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLowerBoundLabel = QtWidgets.QLabel("Lower Bound")
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel = QtWidgets.QLabel("Upper Bound")
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseHypLabel = QtWidgets.QLabel("Noise Hyperparameter:")
        self.YNoiseHypLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry = QtWidgets.QLineEdit("1.0e-2")
        self.YNoiseHypEntry.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNoiseLBEntry = QtWidgets.QLineEdit("1.0e-3")
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNoiseUBEntry = QtWidgets.QLineEdit("1.0e-2")
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.YNRestartsLabel = QtWidgets.QLabel("Number of Restarts:")
        self.YNRestartsLabel.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry = QtWidgets.QLineEdit("5")
        self.YNRestartsEntry.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry.setValidator(QtGui.QIntValidator(1,1000,None))

        self.YSEKernelSettings = SEKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YRQKernelSettings = RQKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YMHKernelSettings = MHKernelWidget(True,self.YKernelRestartBox.isChecked())
        self.YGGKernelSettings = GibbsKernelWidget(True,self.YKernelRestartBox.isChecked())

        self.YKernelSettings = QtWidgets.QStackedLayout()
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

        self.YOptimizerSettings = QtWidgets.QStackedLayout()
        self.YOptimizerSettings.addWidget(self.YGradOptSettings)
        self.YOptimizerSettings.addWidget(self.YMomOptSettings)
        self.YOptimizerSettings.addWidget(self.YNagOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdagradOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdadeltaOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdamOptSettings)
        self.YOptimizerSettings.addWidget(self.YAdamaxOptSettings)
        self.YOptimizerSettings.addWidget(self.YNadamOptSettings)
        self.YOptimizerSettings.setCurrentIndex(self.YOptimizerSelectionList.currentIndex())

        ynlbox = QtWidgets.QHBoxLayout()
        ynlbox.addWidget(self.YNoiseInitGuessLabel)
        ynlbox.addWidget(self.YNoiseLowerBoundLabel)
        ynlbox.addWidget(self.YNoiseUpperBoundLabel)

        ynebox = QtWidgets.QHBoxLayout()
        ynebox.addWidget(self.YNoiseHypEntry)
        ynebox.addWidget(self.YNoiseLBEntry)
        ynebox.addWidget(self.YNoiseUBEntry)

        ykbox = QtWidgets.QFormLayout()
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

        self.HeteroscedasticBox = QtWidgets.QCheckBox("Enable Error Kernel")
        self.HeteroscedasticBox.toggled.connect(self._toggle_error_kernel)

        self.EKernelSelectionLabel = QtWidgets.QLabel("Kernel:")
        self.EKernelSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSelectionList = QtWidgets.QComboBox()
        self.EKernelSelectionList.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelSelectionList.addItem("Squared Exponential")
        self.EKernelSelectionList.addItem("Rational Quadratic")
        self.EKernelSelectionList.addItem("Matern Half-Integer")
        self.EKernelSelectionList.addItem("Gibbs Kernel")
        self.EKernelSelectionList.setCurrentIndex(1)
        self.EKernelSelectionList.currentIndexChanged.connect(self._switch_kernel_ui_e)

        self.EOptimizeBox = QtWidgets.QCheckBox("Optimize")
        self.EOptimizeBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EOptimizeBox.toggled.connect(self._toggle_optimize_e)
        self.EAddNoiseBox = QtWidgets.QCheckBox("Add Noise Kernel")
        self.EAddNoiseBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EAddNoiseBox.toggled.connect(self._toggle_noise_kernel_e)
        self.EKernelRestartBox = QtWidgets.QCheckBox("Use Kernel Restarts")
        self.EKernelRestartBox.setEnabled(self.HeteroscedasticBox.isChecked())
        self.EKernelRestartBox.toggled.connect(self._toggle_kernel_restarts_e)

        self.ERegularizationLabel = QtWidgets.QLabel("Reg. Parameter:")
        self.ERegularizationLabel.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ERegularizationEntry = QtWidgets.QLineEdit("6.0")
        self.ERegularizationEntry.setEnabled(self.HeteroscedasticBox.isChecked())
        self.ERegularizationEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.EEpsilonLabel = QtWidgets.QLabel("Convergence Criteria:")
        self.EEpsilonLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EEpsilonEntry = QtWidgets.QLineEdit("1.0e-1")
        self.EEpsilonEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EEpsilonEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))

        self.EOptimizerSelectionLabel = QtWidgets.QLabel("Optimizer:")
        self.EOptimizerSelectionLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList = QtWidgets.QComboBox()
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
        self.EOptimizerSelectionList.currentIndexChanged.connect(self._switch_optimizer_ui_e)

        self.ENoiseInitGuessLabel = QtWidgets.QLabel("Initial Guess")
        self.ENoiseInitGuessLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseLowerBoundLabel = QtWidgets.QLabel("Lower Bound")
        self.ENoiseLowerBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel = QtWidgets.QLabel("Upper Bound")
        self.ENoiseUpperBoundLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseHypLabel = QtWidgets.QLabel("Noise Hyperparameter:")
        self.ENoiseHypLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry = QtWidgets.QLineEdit("1.0e-3")
        self.ENoiseHypEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENoiseLBEntry = QtWidgets.QLineEdit("1.0e-3")
        self.ENoiseLBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseLBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENoiseUBEntry = QtWidgets.QLineEdit("1.0e-2")
        self.ENoiseUBEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setValidator(QtGui.QDoubleValidator(0.0,np.Inf,100,None))
        self.ENRestartsLabel = QtWidgets.QLabel("Number of Restarts:")
        self.ENRestartsLabel.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry = QtWidgets.QLineEdit("5")
        self.ENRestartsEntry.setEnabled(self.HeteroscedasticBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry.setValidator(QtGui.QIntValidator(1,1000,None))

        self.ESEKernelSettings = SEKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.ERQKernelSettings = RQKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.EMHKernelSettings = MHKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())
        self.EGGKernelSettings = GibbsKernelWidget(self.HeteroscedasticBox.isChecked(),self.EKernelRestartBox.isChecked())

        self.EKernelSettings = QtWidgets.QStackedLayout()
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

        self.EOptimizerSettings = QtWidgets.QStackedLayout()
        self.EOptimizerSettings.addWidget(self.EGradOptSettings)
        self.EOptimizerSettings.addWidget(self.EMomOptSettings)
        self.EOptimizerSettings.addWidget(self.ENagOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdagradOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdadeltaOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdamOptSettings)
        self.EOptimizerSettings.addWidget(self.EAdamaxOptSettings)
        self.EOptimizerSettings.addWidget(self.ENadamOptSettings)
        self.EOptimizerSettings.setCurrentIndex(self.EOptimizerSelectionList.currentIndex())

        enlbox = QtWidgets.QHBoxLayout()
        enlbox.addWidget(self.ENoiseInitGuessLabel)
        enlbox.addWidget(self.ENoiseLowerBoundLabel)
        enlbox.addWidget(self.ENoiseUpperBoundLabel)

        enebox = QtWidgets.QHBoxLayout()
        enebox.addWidget(self.ENoiseHypEntry)
        enebox.addWidget(self.ENoiseLBEntry)
        enebox.addWidget(self.ENoiseUBEntry)

        ekbox = QtWidgets.QFormLayout()
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

        eebox = QtWidgets.QVBoxLayout()
        eebox.addWidget(self.HeteroscedasticBox)
        eebox.addLayout(ekbox)

        self.EKernelSelectionTab.setLayout(eebox)


    def _flag_new_data(self):
        self.fNewData = True

    def _add_data(self):
        idx = self.DataTable.rowCount()
        self.DataTable.insertRow(idx)
        self.DataTable.resizeRowsToContents()

    def _add_derivative_data(self):
        idx = self.DerivativeTable.rowCount()
        self.DerivativeTable.insertRow(idx)
        self.DerivativeTable.resizeRowsToContents()

    def _load_data(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', 'Text files (*.txt);;All files (*)')
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
                            self._flag_new_data()
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
                            self._flag_new_data()
            self.DataTable.resizeRowsToContents()
            self.DerivativeTable.resizeRowsToContents()

    def _sort_data(self):
        self.DataTable.sortItems(0,QtCore.Qt.AscendingOrder)
        self.DerivativeTable.sortItems(0,QtCore.Qt.AscendingOrder)

    def _clean_data(self):
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
                        self._flag_new_data()
                    else:
                        gflag = False
            elif not self.UseXErrorsBox.isChecked():
                self.DataTable.setItem(ii,3,QtWidgets.QTableWidgetItem("0.0"))
                self._flag_new_data()
            else:
                gflag = False
            if not gflag:
                self.DataTable.removeRow(ii)
                self._flag_new_data()
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
                        self._flag_new_data()
                    else:
                        gflag = False
            elif not self.UseXErrorsBox.isChecked():
                self.DerivativeTable.setItem(ii,3,QtWidgets.QTableWidgetItem("0.0"))
                self._flag_new_data()
            else:
                gflag = False
            if not gflag:
                self.DerivativeTable.removeRow(ii)
                self._flag_new_data()

    def _clear_data(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setWindowTitle("Clear Data")
        msg.setText("Are you sure you want to clear all stored raw data?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        retval = msg.exec_()
        if retval == QtWidgets.QMessageBox.Yes:
            self.DataTable.clearContents()
            self.DerivativeTable.clearContents()
            self._clean_data()
            self._flag_new_data()

    def _toggle_derivatives(self):
        self.DerivativeTable.setEnabled(self.DerivativeBox.isChecked())

    def _toggle_xerror_display(self):
        if self.UseXErrorsBox.isChecked():
            self.DataTable.horizontalHeader().showSection(3)
            self.DerivativeTable.horizontalHeader().showSection(3)
        else:
            self.DataTable.horizontalHeader().hideSection(3)
            self.DerivativeTable.horizontalHeader().hideSection(3)

    def _switch_kernel_ui_y(self):
        self.YKernelSettings.setCurrentIndex(self.YKernelSelectionList.currentIndex())

    def _toggle_optimize_y(self):
        self.YEpsilonLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YEpsilonEntry.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionLabel.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSelectionList.setEnabled(self.YOptimizeBox.isChecked())
        self.YOptimizerSettings.setEnabled(self.YOptimizeBox.isChecked())
        for ii in np.arange(0,self.YOptimizerSettings.count()):
            self.YOptimizerSettings.widget(ii).toggle_all(self.YOptimizeBox.isChecked())

    def _switch_optimizer_ui_y(self):
        self.YOptimizerSettings.setCurrentIndex(self.YOptimizerSelectionList.currentIndex())

    def _toggle_noise_kernel_y(self):
        self.YNoiseInitGuessLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseHypLabel.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseHypEntry.setEnabled(self.YAddNoiseBox.isChecked())
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())

    def _toggle_kernel_restarts_y(self):
        self.YNRestartsLabel.setEnabled(self.YKernelRestartBox.isChecked())
        self.YNRestartsEntry.setEnabled(self.YKernelRestartBox.isChecked())
        for ii in np.arange(0,self.YKernelSettings.count()):
            self.YKernelSettings.widget(ii).toggle_bounds(self.YKernelRestartBox.isChecked())
        self.YNoiseLowerBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUpperBoundLabel.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseLBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())
        self.YNoiseUBEntry.setEnabled(self.YAddNoiseBox.isChecked() and self.YKernelRestartBox.isChecked())

    def _toggle_error_kernel(self):
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

    def _switch_kernel_ui_e(self):
        self.EKernelSettings.setCurrentIndex(self.EKernelSelectionList.currentIndex())

    def _toggle_optimize_e(self):
        self.EEpsilonLabel.setEnabled(self.EOptimizeBox.isChecked())
        self.EEpsilonEntry.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionLabel.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSelectionList.setEnabled(self.EOptimizeBox.isChecked())
        self.EOptimizerSettings.setEnabled(self.EOptimizeBox.isChecked())
        for ii in np.arange(0,self.EOptimizerSettings.count()):
            self.EOptimizerSettings.widget(ii).toggle_all(self.EOptimizeBox.isChecked())

    def _switch_optimizer_ui_e(self):
        self.EOptimizerSettings.setCurrentIndex(self.EOptimizerSelectionList.currentIndex())

    def _toggle_noise_kernel_e(self):
        self.ENoiseInitGuessLabel.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseLowerBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseHypLabel.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseHypEntry.setEnabled(self.EAddNoiseBox.isChecked())
        self.ENoiseLBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())

    def _toggle_kernel_restarts_e(self):
        self.ENRestartsLabel.setEnabled(self.EKernelRestartBox.isChecked())
        self.ENRestartsEntry.setEnabled(self.EKernelRestartBox.isChecked())
        for ii in np.arange(0,self.EKernelSettings.count()):
            self.EKernelSettings.widget(ii).toggle_bounds(self.EKernelRestartBox.isChecked())
        self.ENoiseLowerBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUpperBoundLabel.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseLBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())
        self.ENoiseUBEntry.setEnabled(self.EAddNoiseBox.isChecked() and self.EKernelRestartBox.isChecked())

    def _fit_data(self):
        self._clean_data()
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
                ykname = 'Sum(' + ykname + '-n)'
                ykhyps = np.hstack((ykhyps,float(self.YNoiseHypEntry.text()))) if ykhyps is not None else None
                ykbounds = np.vstack((ykbounds,np.atleast_2d([float(self.YNoiseLBEntry.text()),float(self.YNoiseUBEntry.text())]))) if ykbounds is not None else None
            ynres = int(float(self.YNRestartsEntry.text())) if self.YKernelRestartBox.isChecked() else None
            ykernel = GPR1D.KernelReconstructor(ykname,pars=np.hstack((ykhyps,ykcsts)))
            if ykbounds is not None:
                ykbounds = np.transpose(ykbounds)

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
                    ekname = 'Sum(' + ekname + '-n)'
                    ekhyps = np.hstack((ekhyps,float(self.ENoiseHypEntry.text()))) if ekhyps is not None else None
                    ekbounds = np.vstack((ekbounds,np.atleast_2d([float(self.ENoiseLBEntry.text()),float(self.ENoiseUBEntry.text())]))) if ekbounds is not None else None
                enres = int(float(self.ENRestartsEntry.text())) if self.EKernelRestartBox.isChecked() else None
                ekernel = GPR1D.KernelReconstructor(ekname,pars=np.hstack((ekhyps,ekcsts)))
            if ekbounds is not None:
                ekbounds = np.transpose(ekbounds)
            vary_yerrs = self.HeteroscedasticBox.isChecked() if ekernel is not None else False

            try:
                tic = time.perf_counter()
                xnew = np.linspace(float(self.PredictStartEntry.text()),float(self.PredictEndEntry.text()),int(float(self.PredictNPointsEntry.text())))
                self.gpr.set_raw_data(xdata=xx,ydata=yy,yerr=ye,xerr=xe,dxdata=dxx,dydata=dyy,dyerr=dye)
                self.gpr.set_kernel(kernel=ykernel,kbounds=ykbounds,regpar=yregpar)
                self.gpr.set_error_kernel(kernel=ekernel,kbounds=ekbounds,regpar=eregpar,nrestarts=enres)
                self.gpr.set_search_parameters(epsilon=yeps,method=yopm,spars=yopp)
                self.gpr.set_error_search_parameters(epsilon=eeps,method=eopm,spars=eopp)
                self.gpr.GPRFit(xnew,hsgp_flag=vary_yerrs,nigp_flag=use_xerrs,nrestarts=ynres)
                self.fNewData = False
                toc = time.perf_counter()
                print("Fitting routine completed. Elapsed time: %.3f s" % (toc - tic))
                ylml = self.gpr.get_gp_lml()
                rsad = self.gpr.get_gp_adjusted_r2()
                rscs = self.gpr.get_gp_generalized_r2()
                print("Final log-marginal-likelihood: %15.8f" % (ylml))
                print("Final adjusted R-squared:      %15.8f" % (rsad))
                print("Final pseudo R-squared (CS):   %15.8f" % (rscs))
                if (isinstance(eeps,(float,int)) and eeps > 0.0) or (isinstance(enres,(float,int)) and enres > 0):
                    ehyps = self.gpr.get_error_kernel().hyperparameters
                    print("   --- Optimized error kernel hyperparameters: ---")
                    print(ehyps)
                if (isinstance(yeps,(float,int)) and yeps > 0.0) or (isinstance(ynres,(float,int)) and ynres > 0):
                    yhyps = self.gpr.get_gp_kernel().hyperparameters
                    print("   *** Optimized kernel hyperparameters: ***")
                    print(yhyps)
            except Exception as e:
                print(repr(e))
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Critical)
                msg.setWindowTitle("Fitting Routine Failed")
                msg.setText("Fitting routine failure: Please see console messages for more details.")
                msg.exec_()
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Raw data table is empty or was improperly filled.")
            msg.exec_()

    def _plot_data(self):
        self._clean_data()
        retval = QtWidgets.QMessageBox.Yes
        if self.fNewData and self.gpr.get_gp_x() is not None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setWindowTitle("Raw Data and Fit Mismatched")
            msg.setText("Changes to stored raw data have been detected and stored fit may no longer correspond with it. Plot anyway?")
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            retval = msg.exec_()
        if retval == QtWidgets.QMessageBox.Yes:
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
                ax.errorbar(xraw,yraw,xerr=xeraw,yerr=yeraw,ls='',marker='x',color='k')
            else:
                ax.errorbar(xraw,yraw,yerr=yeraw,ls='',marker='x',color='k')
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

    def _save_raw_data(self,sortflag=False):
        self._clean_data()
        if self.DataTable.rowCount() > 0:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save As...', '', 'Text files (*.txt);;All files (*)')
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
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Raw data table is empty or was improperly filled.")
            msg.exec_()

    def _save_fit_data(self):
        if self.gpr.get_gp_x() is not None:
            filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save As...', '', 'Text files (*.txt);;All files (*)')
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
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Data Not Found")
            msg.setText("Fit data not yet populated.")
            msg.exec_()

def main():

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('GPR1D')
    ex = GPR1D_GUI()

    sys.exit(app.exec_())

if __name__ == '__main__':

    main()
