# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spectral_radiance.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(905, 559)
        mainWindow.setMaximumSize(QtCore.QSize(16777215, 900))
        self.verticalLayout = QtWidgets.QVBoxLayout(mainWindow)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.sensorInfo = QtWidgets.QGroupBox(mainWindow)
        self.sensorInfo.setObjectName("sensorInfo")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.sensorInfo)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.visualisationWindow = PlotWidget(self.sensorInfo)
        self.visualisationWindow.setMinimumSize(QtCore.QSize(200, 25))
        self.visualisationWindow.setObjectName("visualisationWindow")
        self.horizontalLayout_5.addWidget(self.visualisationWindow)
        self.line = QtWidgets.QFrame(self.sensorInfo)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_5.addWidget(self.line)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pitchLabel = QtWidgets.QLabel(self.sensorInfo)
        self.pitchLabel.setObjectName("pitchLabel")
        self.gridLayout_2.addWidget(self.pitchLabel, 3, 0, 1, 1)
        self.yawLabel = QtWidgets.QLabel(self.sensorInfo)
        self.yawLabel.setObjectName("yawLabel")
        self.gridLayout_2.addWidget(self.yawLabel, 4, 0, 1, 1)
        self.live = QtWidgets.QCheckBox(self.sensorInfo)
        self.live.setChecked(False)
        self.live.setObjectName("live")
        self.gridLayout_2.addWidget(self.live, 0, 0, 1, 1)
        self.rollLabel = QtWidgets.QLabel(self.sensorInfo)
        self.rollLabel.setObjectName("rollLabel")
        self.gridLayout_2.addWidget(self.rollLabel, 2, 0, 1, 1)
        self.boardTempLabel = QtWidgets.QLabel(self.sensorInfo)
        self.boardTempLabel.setObjectName("boardTempLabel")
        self.gridLayout_2.addWidget(self.boardTempLabel, 1, 0, 1, 1)
        self.roll = QtWidgets.QLineEdit(self.sensorInfo)
        self.roll.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.roll.setFont(font)
        self.roll.setAlignment(QtCore.Qt.AlignCenter)
        self.roll.setReadOnly(True)
        self.roll.setObjectName("roll")
        self.gridLayout_2.addWidget(self.roll, 2, 1, 1, 1)
        self.pitch = QtWidgets.QLineEdit(self.sensorInfo)
        self.pitch.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.pitch.setFont(font)
        self.pitch.setAlignment(QtCore.Qt.AlignCenter)
        self.pitch.setReadOnly(True)
        self.pitch.setObjectName("pitch")
        self.gridLayout_2.addWidget(self.pitch, 3, 1, 1, 1)
        self.yaw = QtWidgets.QLineEdit(self.sensorInfo)
        self.yaw.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.yaw.setFont(font)
        self.yaw.setAlignment(QtCore.Qt.AlignCenter)
        self.yaw.setReadOnly(True)
        self.yaw.setObjectName("yaw")
        self.gridLayout_2.addWidget(self.yaw, 4, 1, 1, 1)
        self.boardTemp = QtWidgets.QLineEdit(self.sensorInfo)
        self.boardTemp.setMinimumSize(QtCore.QSize(0, 0))
        self.boardTemp.setMaximumSize(QtCore.QSize(150, 100))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.boardTemp.setFont(font)
        self.boardTemp.setAlignment(QtCore.Qt.AlignCenter)
        self.boardTemp.setReadOnly(True)
        self.boardTemp.setObjectName("boardTemp")
        self.gridLayout_2.addWidget(self.boardTemp, 1, 1, 1, 1)
        self.horizontalLayout_5.addLayout(self.gridLayout_2)
        self.horizontalLayout_7.addWidget(self.sensorInfo)
        self.verticalLayout_8.addLayout(self.horizontalLayout_7)
        self.acParameters = QtWidgets.QGroupBox(mainWindow)
        self.acParameters.setFlat(False)
        self.acParameters.setObjectName("acParameters")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.acParameters)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.gainLabel = QtWidgets.QLabel(self.acParameters)
        self.gainLabel.setObjectName("gainLabel")
        self.gridLayout.addWidget(self.gainLabel, 1, 0, 1, 1)
        self.gainDoubleSpinBox = QtWidgets.QDoubleSpinBox(self.acParameters)
        self.gainDoubleSpinBox.setDecimals(0)
        self.gainDoubleSpinBox.setMinimum(-4.0)
        self.gainDoubleSpinBox.setMaximum(38.0)
        self.gainDoubleSpinBox.setSingleStep(1.0)
        self.gainDoubleSpinBox.setObjectName("gainDoubleSpinBox")
        self.gridLayout.addWidget(self.gainDoubleSpinBox, 1, 1, 1, 1)
        self.exposureSpinBox = QtWidgets.QSpinBox(self.acParameters)
        self.exposureSpinBox.setPrefix("")
        self.exposureSpinBox.setMinimum(200)
        self.exposureSpinBox.setMaximum(10000000)
        self.exposureSpinBox.setProperty("value", 20000)
        self.exposureSpinBox.setObjectName("exposureSpinBox")
        self.gridLayout.addWidget(self.exposureSpinBox, 0, 1, 1, 1)
        self.exposureSlider = QtWidgets.QSlider(self.acParameters)
        self.exposureSlider.setMinimum(2300)
        self.exposureSlider.setMaximum(7000)
        self.exposureSlider.setSingleStep(1)
        self.exposureSlider.setProperty("value", 4300)
        self.exposureSlider.setSliderPosition(4300)
        self.exposureSlider.setOrientation(QtCore.Qt.Horizontal)
        self.exposureSlider.setObjectName("exposureSlider")
        self.gridLayout.addWidget(self.exposureSlider, 0, 2, 1, 1)
        self.exposureLabel = QtWidgets.QLabel(self.acParameters)
        self.exposureLabel.setObjectName("exposureLabel")
        self.gridLayout.addWidget(self.exposureLabel, 0, 0, 1, 1)
        self.depthlabel = QtWidgets.QLabel(self.acParameters)
        self.depthlabel.setObjectName("depthlabel")
        self.gridLayout.addWidget(self.depthlabel, 3, 0, 1, 1)
        self.depth = QtWidgets.QSpinBox(self.acParameters)
        self.depth.setMaximum(400)
        self.depth.setSingleStep(10)
        self.depth.setObjectName("depth")
        self.gridLayout.addWidget(self.depth, 3, 1, 1, 1)
        self.gainSlider = QtWidgets.QSlider(self.acParameters)
        self.gainSlider.setMinimum(-4)
        self.gainSlider.setMaximum(38)
        self.gainSlider.setSingleStep(1)
        self.gainSlider.setOrientation(QtCore.Qt.Horizontal)
        self.gainSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.gainSlider.setTickInterval(2)
        self.gainSlider.setObjectName("gainSlider")
        self.gridLayout.addWidget(self.gainSlider, 1, 2, 1, 1)
        self.binlabel = QtWidgets.QLabel(self.acParameters)
        self.binlabel.setObjectName("binlabel")
        self.gridLayout.addWidget(self.binlabel, 2, 0, 1, 1)
        self.binComboBox = QtWidgets.QComboBox(self.acParameters)
        self.binComboBox.setAcceptDrops(False)
        self.binComboBox.setEditable(False)
        self.binComboBox.setInsertPolicy(QtWidgets.QComboBox.InsertAtBottom)
        self.binComboBox.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.binComboBox.setMinimumContentsLength(2)
        self.binComboBox.setObjectName("binComboBox")
        self.binComboBox.addItem("")
        self.gridLayout.addWidget(self.binComboBox, 2, 1, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_5.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.water = QtWidgets.QRadioButton(self.acParameters)
        self.water.setChecked(True)
        self.water.setObjectName("water")
        self.horizontalLayout_8.addWidget(self.water)
        self.air = QtWidgets.QRadioButton(self.acParameters)
        self.air.setObjectName("air")
        self.horizontalLayout_8.addWidget(self.air)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.errorlog = QtWidgets.QLineEdit(self.acParameters)
        self.errorlog.setObjectName("errorlog")
        self.verticalLayout_5.addWidget(self.errorlog)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.shoot = QtWidgets.QGroupBox(self.acParameters)
        self.shoot.setTitle("")
        self.shoot.setObjectName("shoot")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.shoot)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.darkFrameButton = QtWidgets.QPushButton(self.shoot)
        self.darkFrameButton.setObjectName("darkFrameButton")
        self.verticalLayout_6.addWidget(self.darkFrameButton)
        self.acquisitionButton = QtWidgets.QPushButton(self.shoot)
        self.acquisitionButton.setIconSize(QtCore.QSize(16, 16))
        self.acquisitionButton.setObjectName("acquisitionButton")
        self.verticalLayout_6.addWidget(self.acquisitionButton)
        self.horizontalLayout_3.addWidget(self.shoot)
        self.verticalLayout_8.addWidget(self.acParameters)
        self.verticalLayout_2.addLayout(self.verticalLayout_8)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.saveFolder = QtWidgets.QLineEdit(mainWindow)
        self.saveFolder.setText("")
        self.saveFolder.setReadOnly(True)
        self.saveFolder.setObjectName("saveFolder")
        self.gridLayout_4.addWidget(self.saveFolder, 0, 2, 1, 1)
        self.fname = QtWidgets.QLineEdit(mainWindow)
        self.fname.setAlignment(QtCore.Qt.AlignCenter)
        self.fname.setObjectName("fname")
        self.gridLayout_4.addWidget(self.fname, 1, 2, 1, 1)
        self.labelfname = QtWidgets.QLabel(mainWindow)
        self.labelfname.setObjectName("labelfname")
        self.gridLayout_4.addWidget(self.labelfname, 1, 1, 1, 1)
        self.chooseFolderButton = QtWidgets.QPushButton(mainWindow)
        self.chooseFolderButton.setObjectName("chooseFolderButton")
        self.gridLayout_4.addWidget(self.chooseFolderButton, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_4)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Spectral radiance"))
        self.sensorInfo.setTitle(_translate("mainWindow", "Sensor "))
        self.pitchLabel.setText(_translate("mainWindow", "Pitch - y axis"))
        self.yawLabel.setText(_translate("mainWindow", "Yaw - z axis"))
        self.live.setText(_translate("mainWindow", "Real time"))
        self.rollLabel.setText(_translate("mainWindow", "Roll - x axis"))
        self.boardTempLabel.setText(_translate("mainWindow", "Board temperature "))
        self.acParameters.setTitle(_translate("mainWindow", "Acquisition "))
        self.gainLabel.setText(_translate("mainWindow", "Gain"))
        self.gainDoubleSpinBox.setSuffix(_translate("mainWindow", " dB"))
        self.exposureSpinBox.setSuffix(_translate("mainWindow", " us"))
        self.exposureLabel.setText(_translate("mainWindow", "Exposure"))
        self.depthlabel.setText(_translate("mainWindow", "Current depth"))
        self.depth.setSuffix(_translate("mainWindow", " cm"))
        self.binlabel.setText(_translate("mainWindow", "Binning"))
        self.binComboBox.setCurrentText(_translate("mainWindow", "2x2"))
        self.binComboBox.setItemText(0, _translate("mainWindow", "2x2"))
        self.water.setText(_translate("mainWindow", "Water"))
        self.air.setText(_translate("mainWindow", "Air"))
        self.darkFrameButton.setText(_translate("mainWindow", "Dark frame"))
        self.acquisitionButton.setText(_translate("mainWindow", "Acquisition"))
        self.fname.setText(_translate("mainWindow", "acq_001"))
        self.labelfname.setText(_translate("mainWindow", "Folder name:"))
        self.chooseFolderButton.setText(_translate("mainWindow", "Choose directory"))

from pyqtgraph import PlotWidget
