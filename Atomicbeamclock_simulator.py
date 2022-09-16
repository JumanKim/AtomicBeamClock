import numpy as np
from matplotlib import pyplot as plt
import tqdm
import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread, QThreadPool, QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import cm
from matplotlib.text import Annotation
import shutil
from numba import njit, prange, jit
import matplotlib
import time
import random
import os
from scipy import fftpack
from scipy.optimize import curve_fit  # https://smlee729.github.io/python/simulation/2015/03/25/2-curve_fitting.html
from multiprocessing import Pool, Process, Queue
import multiprocessing as mp
import warnings
from atomicbeamclock import Atomicbeamclock, WorkerSignals, folder_reset, G1calc, ContourPlot
import threading
import logging
import subprocess

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
## plot colors(defalut cycle) #https://matplotlib.org/3.1.0/gallery/color/color_cycle_default.html
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = ['black', 'red', 'blue']
fmt_list = ['o', 'v', '^', '<', '>', 'o', 'v', '^', '<', '>']
linestyle_list = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
numofproc = mp.cpu_count() - 2  # num of cpu core
logging.basicConfig(format="%(message)s", level=logging.INFO)


# numofproc=10

def normlrtz(x, amp1, cen1, wid1):  # wid1: FWHM
    return amp1 / np.pi * (wid1 / 2.0) / ((x - cen1) ** 2.0 + (wid1 / 2.0) ** 2.0)


def exp_decay(t, sigma,period):
    return np.exp(-t*sigma)*np.cos(2.0*np.pi*t/period)


def normlrtz(x, amp1, cen1, wid1):#wid1: FWHM
    return amp1/np.pi*(wid1/2.0)/((x-cen1)**2.0+(wid1/2.0)**2.0)
# class MyThread(QThread):


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.status = 0
        self.trajcompute = Atomicbeamclock()
        self.g1_calc = None
        self.result = None
        self.thread = QThreadPool()

    def initUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("atomic beam clock")
        self.setWindowIcon(QIcon('icon.png'))

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.navi_toolbar = NavigationToolbar(self.canvas, self)
        self.pbar_display = QProgressBar(self)
        self.pbar_display.setValue(0)

        # Status bar
        self.statusBar().showMessage('Ready')

        # Main buttons
        self.atom_button_tips = QLabel("Atom")
        self.atom_button = QComboBox(self)
        self.atom_button.addItem("Ba-138")
        self.atom_button.addItem("Sr-87")
        self.atom_button.addItem("Ca-40")
        self.atom_button.addItem("Yb-171")
        self.manipulated_button_tips = QLabel("Manipulated variables")
        self.manipulated_button = QComboBox(self)
        self.manipulated_button.addItem("Cluster number")
        self.manipulated_button.addItem("Cavity-atom detuning")
        self.manipulated_button.addItem("Pump-atom detuning")
        self.manipulated_button.addItem("None")
        self.dependent_button_tips = QLabel("Dependent variables")
        self.dependent_button = QComboBox(self)
        self.dependent_button.addItem("output power")
        self.dependent_button.addItem("g1 function")
        self.dependent_button.addItem("time evolution")
        self.randomphase_button_tips = QLabel("random phase")
        self.randomphase_button_group = QButtonGroup(self)
        self.randomphase_on = QRadioButton('on', self)
        self.randomphase_on.setChecked(True)
        self.randomphase_off = QRadioButton('off', self)
        self.randomphase_button_group.addButton(self.randomphase_on)
        self.randomphase_button_group.addButton(self.randomphase_off)
        self.fftcalc_button_tips = QLabel("FFT calculation")
        self.fftcalc_button_group = QButtonGroup(self)
        self.fftcalc_on = QRadioButton('on', self)
        self.fftcalc_on.setChecked(True)
        self.fftcalc_off = QRadioButton('off', self)
        self.fftcalc_button_group.addButton(self.fftcalc_on)
        self.fftcalc_button_group.addButton(self.fftcalc_off)
        self.fftplot_button_tips = QLabel("FFT plot")
        self.fftplot_button_group = QButtonGroup(self)
        self.fftplot_on = QRadioButton('on', self)
        self.fftplot_on.setChecked(True)
        self.fftplot_off = QRadioButton('off', self)
        self.fftplot_button_group.addButton(self.fftplot_on)
        self.fftplot_button_group.addButton(self.fftplot_off)
        self.fftfit_button_tips = QLabel("FFT fit")
        self.fftfit_button_group = QButtonGroup(self)
        self.fftfit_on = QRadioButton('on', self)
        self.fftfit_on.setChecked(True)
        self.fftfit_off = QRadioButton('off', self)
        self.fftfit_button_group.addButton(self.fftfit_on)
        self.fftfit_button_group.addButton(self.fftfit_off)
        self.ntraj_button_tips = QLabel("# of trajectories (for time evolution)")
        self.ntraj_button = QLineEdit()
        self.ntraj_button.setText("1000")
        self.compute_button = QPushButton("Compute")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.atom_sheet = QPushButton("Atom information")
        self.atom_sheet_dialog = QDialog()
        self.parameter_list = QPushButton("Parameter list")
        self.parameter_list_dialog = QDialog()
        self.xyplot_button = QPushButton("Draw XY plot")
        self.contourplot_button = QPushButton("Draw contour plot")
        self.status_display = QPlainTextEdit("Status")
        self.status_display.setReadOnly(True)
        # self.status_display.setAlignment(Qt.AlignTop)
        # self.status_display.setEnabled(False)
        self.status_display.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.tips = QLabel("designed by Juman @ QFLL 2022")

        # Atom information buttons
        self.kappa_label = QLabel("kappa / 2pi (MHz)", self.atom_sheet_dialog)
        self.kappa_display = QLineEdit(self.atom_sheet_dialog)
        self.g_label = QLabel("g / 2pi (MHz)", self.atom_sheet_dialog)
        self.g_display = QLineEdit(self.atom_sheet_dialog)
        self.tau_label = QLabel("tau (us)", self.atom_sheet_dialog)
        self.tau_display = QLineEdit(self.atom_sheet_dialog)
        self.atomtype_label = QLabel("Atom type", self.atom_sheet_dialog)
        self.atomtype_display = QLineEdit("", self.atom_sheet_dialog)
        self.atom_apply_button = QPushButton("Apply", self.atom_sheet_dialog)

        # Parameter list buttons
        self.cs_label = QLabel("1) # of atoms in a cluster", self.parameter_list_dialog)
        self.cs_display = QLineEdit(self.parameter_list_dialog)
        self.clusnum_label = QLabel("2) # of cluster in the cavity", self.parameter_list_dialog)
        self.clusnum_display = QLineEdit(self.parameter_list_dialog)
        self.ctlth_label = QLabel("3) number of g1 time element (g1 time range = 0 ~ 3) * 4) * 5) * 6))",
                                  self.parameter_list_dialog)
        self.ctlth_display = QLineEdit(self.parameter_list_dialog)
        self.stpsize_label = QLabel("4) unit time for recording", self.parameter_list_dialog)
        self.stpsize_display = QLineEdit(self.parameter_list_dialog)
        self.cstpsiz_label = QLabel("5) cstpsiz", self.parameter_list_dialog)
        self.cstpsiz_display = QLineEdit(self.parameter_list_dialog)
        self.dt_label = QLabel("6) dt (tau / 200) (us)", self.parameter_list_dialog)
        self.dt_display = QLineEdit(self.parameter_list_dialog)
        self.delDtau_label = QLabel("7) Doppler width", self.parameter_list_dialog)
        self.delDtau_display = QLineEdit(self.parameter_list_dialog)
        self.delD_label = QLabel("8) Doppler width / tau", self.parameter_list_dialog)
        self.delD_display = QLineEdit(self.parameter_list_dialog)
        self.delT_label = QLabel("9) Doppler shift by tilting atomic beam / 2pi (MHz)", self.parameter_list_dialog)
        self.delT_display = QLineEdit(self.parameter_list_dialog)
        self.delca_label = QLabel("10) cavity-atom detuning / 2pi (MHz)", self.parameter_list_dialog)
        self.delca_display = QLineEdit(self.parameter_list_dialog)
        self.delpa_label = QLabel("11) pump-atom detuning / 2pi (MHz)", self.parameter_list_dialog)
        self.delpa_display = QLineEdit(self.parameter_list_dialog)
        self.rhoee_label = QLabel("12) rhoee", self.parameter_list_dialog)
        self.rhoee_display = QLineEdit(self.parameter_list_dialog)
        self.pumplinewidth_label = QLabel("13) pump linewidth / 2pi (MHz)", self.parameter_list_dialog)
        self.pumplinewidth_display = QLineEdit(self.parameter_list_dialog)
        self.sqrtFWHM_label = QLabel("14) sqrt of pump linewidth", self.parameter_list_dialog)
        self.sqrtFWHM_display = QLineEdit(self.parameter_list_dialog)
        self.parameter_apply_button = QPushButton("Apply", self.parameter_list_dialog)

        # Interaction
        self.atom_button.activated[str].connect(self.atom_selected)
        self.manipulated_button.activated[str].connect(self.manipulated_selected)
        self.dependent_button.activated[str].connect(self.dependent_selected)
        self.randomphase_on.clicked.connect(self.randomphase_selected)
        self.randomphase_off.clicked.connect(self.randomphase_selected)
        self.fftcalc_on.clicked.connect(self.fftcalc_selected)
        self.fftcalc_off.clicked.connect(self.fftcalc_selected)
        self.fftplot_on.clicked.connect(self.fftplot_selected)
        self.fftplot_off.clicked.connect(self.fftplot_selected)
        self.fftfit_on.clicked.connect(self.fftfit_selected)
        self.fftfit_off.clicked.connect(self.fftfit_selected)
        self.ntraj_button.textChanged.connect(self.ntraj_changed)
        self.atom_sheet.clicked.connect(self.atom_sheet_clicked)
        self.atom_apply_button.clicked.connect(self.atom_sheet_close)
        self.parameter_list.clicked.connect(self.parameter_list_clicked)
        self.parameter_apply_button.clicked.connect(self.parameter_list_close)
        self.compute_button.clicked.connect(self.compute_clicked)
        self.stop_button.clicked.connect(self.stop_clicked)
        self.xyplot_button.clicked.connect(self.draw_xyplot)
        self.contourplot_button.clicked.connect(self.draw_contourplot)

        # Figure Layout
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        leftLayout.addWidget(self.navi_toolbar)
        leftLayout.addWidget(self.pbar_display)

        # Main layout
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.atom_button_tips)
        rightLayout.addWidget(self.atom_button)
        rightLayout.addWidget(self.manipulated_button_tips)
        rightLayout.addWidget(self.manipulated_button)
        rightLayout.addWidget(self.dependent_button_tips)
        rightLayout.addWidget(self.dependent_button)
        rightLayout.addWidget(self.randomphase_button_tips)
        randomphaseLayout = QHBoxLayout()
        randomphaseLayout.addWidget(self.randomphase_on)
        randomphaseLayout.addWidget(self.randomphase_off)
        rightLayout.addLayout(randomphaseLayout)
        rightLayout.addWidget(self.fftcalc_button_tips)
        fftcalcLayout = QHBoxLayout()
        fftcalcLayout.addWidget(self.fftcalc_on)
        fftcalcLayout.addWidget(self.fftcalc_off)
        rightLayout.addLayout(fftcalcLayout)
        rightLayout.addWidget(self.fftplot_button_tips)
        fftplotLayout = QHBoxLayout()
        fftplotLayout.addWidget(self.fftplot_on)
        fftplotLayout.addWidget(self.fftplot_off)
        rightLayout.addLayout(fftplotLayout)
        rightLayout.addWidget(self.fftfit_button_tips)
        fftfitLayout = QHBoxLayout()
        fftfitLayout.addWidget(self.fftfit_on)
        fftfitLayout.addWidget(self.fftfit_off)
        rightLayout.addLayout(fftfitLayout)
        rightLayout.addWidget(self.ntraj_button_tips)
        rightLayout.addWidget(self.ntraj_button)
        computeLayout = QVBoxLayout()
        computeLayout.addStretch(0)
        computeLayout.addWidget(self.compute_button)
        computeLayout.addWidget(self.stop_button)
        rightLayout.addLayout(computeLayout)
        rightLayout.addStretch(1)
        # Atom information Layout
        atomLayout = QVBoxLayout(self.atom_sheet_dialog)
        atomLayout.addWidget(self.atomtype_label)
        atomLayout.addWidget(self.atomtype_display)
        atomLayout.addWidget(self.kappa_label)
        atomLayout.addWidget(self.kappa_display)
        atomLayout.addWidget(self.g_label)
        atomLayout.addWidget(self.g_display)
        atomLayout.addWidget(self.tau_label)
        atomLayout.addWidget(self.tau_display)
        atomLayout.addWidget(self.atom_apply_button)

        # Parameter list Layout
        paramLayout = QVBoxLayout(self.parameter_list_dialog)
        paramLayout.addWidget(self.cs_label)
        paramLayout.addWidget(self.cs_display)
        paramLayout.addWidget(self.clusnum_label)
        paramLayout.addWidget(self.clusnum_display)
        paramLayout.addWidget(self.ctlth_label)
        paramLayout.addWidget(self.ctlth_display)
        paramLayout.addWidget(self.stpsize_label)
        paramLayout.addWidget(self.stpsize_display)
        paramLayout.addWidget(self.cstpsiz_label)
        paramLayout.addWidget(self.cstpsiz_display)
        paramLayout.addWidget(self.dt_label)
        paramLayout.addWidget(self.dt_display)
        paramLayout.addWidget(self.delDtau_label)
        paramLayout.addWidget(self.delDtau_display)
        paramLayout.addWidget(self.delD_label)
        paramLayout.addWidget(self.delD_display)
        paramLayout.addWidget(self.delT_label)
        paramLayout.addWidget(self.delT_display)
        paramLayout.addWidget(self.delca_label)
        paramLayout.addWidget(self.delca_display)
        paramLayout.addWidget(self.delpa_label)
        paramLayout.addWidget(self.delpa_display)
        paramLayout.addWidget(self.rhoee_label)
        paramLayout.addWidget(self.rhoee_display)
        paramLayout.addWidget(self.pumplinewidth_label)
        paramLayout.addWidget(self.pumplinewidth_display)
        paramLayout.addWidget(self.sqrtFWHM_label)
        paramLayout.addWidget(self.sqrtFWHM_display)
        paramLayout.addWidget(self.parameter_apply_button)

        # Specsheet Layout
        thirdLayout = QVBoxLayout()
        thirdLayout.addWidget(self.atom_sheet)
        thirdLayout.addWidget(self.parameter_list)
        thirdLayout.addWidget(self.contourplot_button)
        thirdLayout.addWidget(self.xyplot_button)
        thirdLayout.addWidget(self.status_display)
        # thirdLayout.addStretch(1)
        thirdLayout.addWidget(self.tips)
        # thirdLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.addLayout(thirdLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        layout.setStretchFactor(thirdLayout, 0)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def atom_selected(self, text):
        self.trajcompute.changeVar()
        if text == "Ba-138":
            self.trajcompute.atomtype = "Ba-138"
            self.trajcompute.kappa = 2 * np.pi * 230  # MHz cavity dissipation rate
            self.trajcompute.g = 2 * np.pi * 0.22  # MHz coupling strength
            self.trajcompute.tau = 0.14  # microsecond interaction time
        if text == "Sr-87":
            self.trajcompute.atomtype = "Sr-87"
            self.trajcompute.kappa = 2 * np.pi * 240  # MHz cavity dissipation rate
            self.trajcompute.g = 2 * np.pi * 0.078  # MHz coupling strength
            self.trajcompute.tau = 0.13  # microsecond interaction time
        if text == "Ca-40":
            self.trajcompute.atomtype = "Ca-40"
            self.trajcompute.kappa = 2 * np.pi * 250  # MHz cavity dissipation rate
            self.trajcompute.g = 2 * np.pi * 0.0174  # MHz coupling strength
            self.trajcompute.tau = 0.13  # microsecond interaction time
        if text == "Yb-171":
            self.trajcompute.atomtype = "Yb-171"
            self.trajcompute.kappa = 2 * np.pi * 0.202  # MHz cavity dissipation rate
            self.trajcompute.g = 2 * np.pi * 48E-6  # MHz coupling strength
            self.trajcompute.tau = 140  # microsecond interaction time

    def manipulated_selected(self, text):
        self.trajcompute.changeVar()
        if text == "Cluster number":
            self.trajcompute.manipulated_var = "clusnum"

        if text == "Cavity-atom detuning":
            self.trajcompute.manipulated_var = "dca"

        if text == "Pump-atom detuning":
            self.trajcompute.manipulated_var = "dpa"

        if text == "None":
            self.trajcompute.manipulated_var = "none"
            self.dependent_button.setCurrentText("evolve")
            self.trajcompute.dependent_var = 'evolve'

    def dependent_selected(self, text):
        self.trajcompute.changeVar()
        if text == "output power":
            self.trajcompute.dependent_var = "outputpower"

        if text == "g1 function":
            self.trajcompute.dependent_var = "g1calc"

        if text == "time evolution":
            self.trajcompute.dependent_var = "evolve"
            self.manipulated_button.setCurrentText("None")
            self.trajcompute.manipulated_var = 'none'

    def randomphase_selected(self):
        if self.randomphase_on.isChecked():
            self.trajcompute.randomphase = 'on'
        if self.randomphase_off.isChecked():
            self.trajcompute.randomphase = 'off'

    def fftcalc_selected(self):
        if self.fftcalc_on.isChecked():
            self.trajcompute.fftcalc = 'on'
        if self.fftcalc_off.isChecked():
            self.trajcompute.fftcalc = 'off'

    def fftplot_selected(self):
        if self.fftplot_on.isChecked():
            self.trajcompute.fftplot = 'on'
        if self.fftplot_off.isChecked():
            self.trajcompute.fftplot = 'off'

    def fftfit_selected(self):
        if self.fftfit_on.isChecked():
            self.trajcompute.fftfit = 'on'
        if self.fftfit_off.isChecked():
            self.trajcompute.fftfit = 'off'

    def ntraj_changed(self):
        self.trajcompute.ntraj = self.ntraj_button.text()

    def atom_sheet_clicked(self):
        self.atomtype_display.setText(self.trajcompute.atomtype)
        self.kappa_display.setText(str(self.trajcompute.kappa / 2 / np.pi))
        self.g_display.setText(str(self.trajcompute.g / 2 / np.pi))
        self.tau_display.setText(str(self.trajcompute.tau))
        self.atom_sheet_dialog.setWindowTitle("Atom sheet")
        self.atom_sheet_dialog.setWindowModality(2)
        self.atom_sheet_dialog.resize(200, 300)
        self.atom_sheet_dialog.show()

    def atom_sheet_close(self):
        self.trajcompute.kappa = float(self.kappa_display.text()) * 2 * np.pi
        self.trajcompute.g = float(self.g_display.text()) * 2 * np.pi
        self.trajcompute.tau = float(self.tau_display.text())
        self.atom_sheet_dialog.close()

    def parameter_list_clicked(self):
        self.trajcompute.changeVar()
        self.cs_display.setText(str(self.trajcompute.cs))
        self.clusnum_display.setText(str(self.trajcompute.clusnum))
        self.ctlth_display.setText(str(self.trajcompute.ctlth))
        self.stpsize_display.setText(str(self.trajcompute.stpsize))
        self.cstpsiz_display.setText(str(self.trajcompute.cstpsiz))
        self.dt_display.setText(str(self.trajcompute.dt))
        self.delDtau_display.setText(str(self.trajcompute.delDtau))
        self.delD_display.setText(str(self.trajcompute.delD))
        self.delT_display.setText(str(self.trajcompute.delT))
        self.delca_display.setText(str(self.trajcompute.delca / 2 / np.pi))
        self.delpa_display.setText(str(self.trajcompute.delpa / 2 / np.pi))
        self.rhoee_display.setText(str(self.trajcompute.rhoee))
        self.pumplinewidth_display.setText(str(self.trajcompute.pumplinewidth / 2 / np.pi))
        self.sqrtFWHM_display.setText(str(self.trajcompute.sqrtFWHM))
        self.parameter_list_dialog.setWindowTitle("Parameter list")
        self.parameter_list_dialog.setWindowModality(2)
        self.parameter_list_dialog.resize(200, 300)
        self.parameter_list_dialog.show()

    def parameter_list_close(self):
        self.trajcompute.cs = int(str(self.cs_display.text()))
        self.trajcompute.clusnum = int(str(self.clusnum_display.text()))
        self.trajcompute.ctlth = int(str(self.ctlth_display.text()))
        self.trajcompute.stpsize = int(str(self.stpsize_display.text()))
        self.trajcompute.cstpsiz = int(str(self.cstpsiz_display.text()))
        self.trajcompute.dt = float(str(self.dt_display.text()))
        self.trajcompute.delDtau = float(str(self.delDtau_display.text()))
        self.trajcompute.delD = float(str(self.delD_display.text()))
        self.trajcompute.delT = float(str(self.delT_display.text()))
        self.trajcompute.delca = float(str(float(self.delca_display.text()) * 2 * np.pi))
        self.trajcompute.delpa = float(str(float(self.delpa_display.text()) * 2 * np.pi))
        self.trajcompute.rhoee = float(str(self.rhoee_display.text()))
        self.trajcompute.pumplinewidth = float(str(float(self.pumplinewidth_display.text()) * 2 * np.pi))
        self.trajcompute.sqrtFWHM = float(str(self.sqrtFWHM_display.text()))
        self.parameter_list_dialog.close()

    # https: // realpython.com / python - pyqt - qthread /
    def compute_clicked(self):
        self.trajcompute.changeVar()
        self.status_display.clear()
        self.statusBar().showMessage('Computing')
        self.pbar_display.setValue(0)
        self.trajcompute.show_params()
        self.compute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.compute_function()

    @pyqtSlot(list, np.ndarray)
    def compute_function(self):
        self.status_display.appendPlainText("----------------check list-------------")
        self.status_display.appendPlainText("kappa tau >> 1")
        self.status_display.appendPlainText(str(self.trajcompute.kappa * self.trajcompute.tau))
        self.status_display.appendPlainText("kappa /g / sqrt(N) >> 1")
        self.status_display.appendPlainText(
            str(self.trajcompute.kappa / self.trajcompute.g / np.sqrt(self.trajcompute.cs * self.trajcompute.clusnum)))
        self.status_display.appendPlainText("kappa /deltaD >> 1")
        self.status_display.appendPlainText(str(self.trajcompute.kappa / self.trajcompute.delD))
        self.status_display.appendPlainText("---------------------------------------")

        self.status_display.appendPlainText("NtauGammac")
        self.status_display.appendPlainText(
            str(self.trajcompute.cs * self.trajcompute.clusnum * self.trajcompute.tau * self.trajcompute.Gammac))
        self.status_display.appendPlainText("unittime")
        self.status_display.appendPlainText(str(self.trajcompute.dt))
        self.compute_process = Atomicbeamclock(*self.trajcompute.params())
        self.compute_process.start()
        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)
        self.compute_process.result_signal.connect(self.save_data)
        self.compute_process.finished.connect(self.reset_compute)

    def timeout(self):
        status = 0
        if self.trajcompute.dependent_var == "evolve":
            data = open("./process/process.txt").read()
            status = len(data)
            # logging.info(f'{status, len(self.trajcompute.vlist)}')
            # logging.info(f'{float(status / len(self.trajcompute.vlist) * 100)}')
            self.pbar_display.setValue(int(status / len(self.trajcompute.vlist) / self.trajcompute.t_length * 100))
        else:
            for ii in range(len(self.trajcompute.vlist)):
                data = open("./process/process" + str(ii) + ".txt").read()
                status += len(data)
            self.pbar_display.setValue(int(status * 1000 / len(self.trajcompute.vlist) / self.trajcompute.t_length * 100))

    def stop_clicked(self, e):
        self.statusBar().showMessage('Stopped')
        self.stop_button.setEnabled(False)
        reply = QMessageBox.question(self, 'Process stopped', 'Are you sure you want to stop the process?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.compute_process.stop()
            self.compute_process.finished.connect(self.reset_compute)
            self.pbar_display.setValue(0)

        if reply == QMessageBox.No:
            pass

    @pyqtSlot(int)
    def reset_compute(self, i):
        self.timer.stop()
        if i == 0:
            if self.trajcompute.dependent_var == "g1calc":
                QMessageBox.about(self, "message", "Simulation completed. g1calc start.")
                folder_reset("./process")
                self.statusBar().showMessage('Computing g1 function')
                self.g1_compute()
            else:
                QMessageBox.about(self, "message", "Simulation completed. Results have been saved.")
                logging.info("reset")
                self.compute_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.statusBar().showMessage('Ready')

        if i == 1:
            QMessageBox.about(self, "message", "Processes have been killed.")
            self.statusBar().showMessage('Ready')
            self.compute_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.statusBar().showMessage('Ready')

    @pyqtSlot(list, np.ndarray)
    def save_data(self, result, vlist):
        self.result = result
        self.trajcompute.analysis(result, vlist, self.trajcompute.params())
        logging.info("saved")
        self.statusBar().showMessage('Saved')

    def draw_xyplot(self):
        try:
            fname = QFileDialog.getOpenFileName(self)
            data = open(fname[0], encoding='utf8', errors='ignore').read().split()
            data = list(map(float, data))
            data = np.asarray(data, dtype=float).reshape(-1, 2)
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.scatter(data[:, 0], data[:, 1], label='test')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("test")
        except:
            # QMessageBox.about("Drawing failed.")
            pass

        self.canvas.draw()

    def draw_contourplot(self):
        path = QFileDialog.getExistingDirectory(self, "Select g1 directory")
        text, ok = QInputDialog.getText(self, 'Save', 'Plot file name')
        self.status_display.clear()
        self.status_display.setPlainText("Source directory: " + path)
        iname = text
        contour = ContourPlot(path, iname)
        contour.contour_plot()

        # contour.drawing_finished.connect(self.drawing_finish)

    @pyqtSlot(object)
    def drawing_finish(self, cp):
        QMessageBox.about(self, "message", "Drawing finished.")
        self.fig.clear()
        ax = self.fig.add_subplot(111)

    def g1_compute(self):
        self.pbar_display.setValue(0)
        self.compute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.g1_calc = G1calc(self.result, self.trajcompute.vlist, self.trajcompute.params())
        self.g1_calc.start()
        self.g1_timer = QTimer(self)
        self.g1_timer.start(1000)
        self.g1_timer.timeout.connect(self.g1_timeout)
        self.g1_calc.finished.connect(self.g1_reset_compute)

    @pyqtSlot()
    def g1_error(self, e):
        QMessageBox(self, "message", "error occurred")

    def g1_timeout(self):
        status = 0
        # for ii in range(len(self.trajcompute.vlist)):
        data = open("./process/g1_process.txt").read()
        status = len(data)
        # logging.info(int(status))
        self.pbar_display.setValue(int(status / len(self.trajcompute.vlist) * 100))

    def g1_reset_compute(self):
        self.g1_timer.stop()
        # folder_reset("./process")
        QMessageBox.about(self, "message", "g1calc finished.")
        self.compute_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pbar_display.setValue(0)
        self.statusBar().showMessage('Ready')

# class Atomsheet(QDialog, MyWindow):
#     def __init__(self):
#         super().__init__()
#         self.setupUI()
#         self.kappa_copy = None
#         self.g_copy = None
#         self.tau_copy = None
#
#     def setupUI(self):
#         self.kappa_label = QLabel("kappa / 2pi (MHz)")
#         self.kappa_display = QLineEdit()
#         self.kappa_display.setText(str(MyWindow.kappa / 2 / np.pi))
#         self.g_label = QLabel("g / 2pi (MHz)")
#         self.g_display = QLineEdit()
#         self.g_display.setText(str(self.g / 2 / np.pi))
#         self.tau_label = QLabel("tau / 2pi (MHz)")
#         self.tau_display = QLineEdit()
#         self.tau_display.setText(str(self.tau / 2 / np.pi))
#         self.atom_apply_button = QPushButton("Apply")
#         self.atom_apply_button.clicked.connect(self.atom_sheet_close)
#
#         self.setWindowTitle("Atom sheet")
#         self.setWindowModality(2)
#         self.resize(200, 300)
#         self.show()
#
#         layout = QVBoxLayout()
#         layout.addWidget(self.kappa_label)
#         layout.addWidget(self.kappa_display)
#         layout.addWidget(self.atom_apply_button)
#
#     def atom_sheet_close(self):
#         self.kappa_copy = self.kappa_display.text()
#         self.g_copy = self.g_display.text()
#         self.tau_copy = self.tau_display.text()
#         self.close()

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    # trajcompute = Atomicbeamclock()
    # trajcompute.start()
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
