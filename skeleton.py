# -*- coding: utf-8 -*-
"""
PyQT launcher.
"""

# Importation of modules
import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph
from gui_source_code import Ui_mainWindow
import glob
import datetime
import cameracontrol
from ximea import xiapi
import numpy as np
import threadfile


class MyDialog(QtWidgets.QDialog, cameracontrol.ProcessImage):
    def __init__(self):
        super(MyDialog, self).__init__()
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        # Attributes initialized
        self.savepath = self.ui.saveFolder.text()
        self.campaign = self.ui.CampName.text()
        self.foldername = self.ui.fname.text()  # Name of the folder only

        self.longpath = self.create_path()

        self.bin_choice = {"2x2": "XI_DWN_2x2", "4x4": "XI_DWN_4x4"}
        self.orientation = np.zeros(3)

        # Create ximea Camera instance
        self.status = False
        self.cam = xiapi.Camera()

        self.img = xiapi.Image()  # Create ximea Image instance
        #print("Opening camera")
        #self.cam.open_device_by("XI_OPEN_BY_SN", "16990159")
        #self.status = True

        self.imformat = "XI_RAW16"
        self.bin = self.bin_choice[self.ui.binComboBox.currentText()]
        self.exp = self.ui.exposureSpinBox.value()
        self.gain = self.ui.gainDoubleSpinBox.value()

        if self.ui.water.isChecked():
            self.medium = self.ui.water.text()
        else:
            self.medium = self.ui.air.text()

        # Threads
        #self.euler_thread = threadfile.Euler()  # IMU
        #self.camera_thread = threadfile.CameraThread(self.imformat, self.bin, self.exp, self.gain, self.cam, self.medium)  # CAM

        # Updating pyqtgraph graph appearance
        self.pred = self.pyqtplot(np.array([0]), np.array([0]), "611 nm", "r")
        self.pgreen = self.pyqtplot(np.array([0]), np.array([0]), "530 nm", "g")
        self.pblue = self.pyqtplot(np.array([0]), np.array([0]), "468 nm", "b")

        self.ui.visualisationWindow.setLabel("left", "D.N normalized", size=2, color="red")
        self.ui.visualisationWindow.setLabel("bottom", "Zenith angle [˚]", size=2, color="red")

        # Spinbox with keyboard tracking disable
        self.ui.exposureSpinBox.setKeyboardTracking(False)
        self.ui.gainDoubleSpinBox.setKeyboardTracking(False)

        # Connections and signals ___________________________________________________________________
        self.ui.exposureSlider.valueChanged.connect(self.exposure_slider)
        self.ui.gainSlider.valueChanged.connect(self.gain_slider)

        self.ui.exposureSpinBox.valueChanged.connect(self.exposure_spin)
        self.ui.gainDoubleSpinBox.valueChanged.connect(self.gain_spin)

        self.ui.binComboBox.currentTextChanged.connect(self.bin_combobox)

        self.ui.darkFrameButton.clicked.connect(self.darkframe_button)  # Darkframe button
        self.ui.acquisitionButton.clicked.connect(self.acquisition_button)  # Acquisition button

        self.ui.live.toggled.connect(self.start_realtimedata)  # Toggle of real time data --> starting timer

        #self.ui.fname.editingFinished.connect(self.folder_name_changed)

        self.ui.chooseFolderButton.clicked.connect(self.getdirectory)  # Choose folder button
        self.ui.newProfileButton.clicked.connect(self.new_profile_button)
        self.ui.fname.textChanged.connect(self.folder_name_changed)
        self.ui.CampName.editingFinished.connect(self.camp_name_changed)

        self.ui.air.toggled.connect(self.water_air_radiobutton)
        self.ui.water.toggled.connect(self.water_air_radiobutton)

        # Custom signals from thread
        #self.euler_thread.my_signal.connect(self.display_angle)
        #self.camera_thread.my_signal.connect(self.plot_avg)
        #self.camera_thread.my_signal_temperature.connect(self.ui.boardTemp.setText)
        #self.camera_thread.my_signal_saturation.connect(self.ui.errorlog.setText)

        # Starting IMU Thread (Always running to be able to record and save the angles even when data is not live)
        #self.euler_thread.running = True
        #self.euler_thread.start()

    def start_realtimedata(self, b):
        """
        Function executed when live checkbox is toggled. The function starts the timer (label tim).

        :return:
        """
        if self.ui.live.isChecked():
            # Starting acquisition
            print("Starting data acquisition...")
            self.cam.start_acquisition()

            # Camera thread
            self.camera_thread.running = True
            self.camera_thread.start()

        else:
            self.camera_thread.running = False

            # Stopping acquisition
            print("Stopping acquisition...")
            self.cam.stop_acquisition()

    def water_air_radiobutton(self):
        if self.ui.water.isChecked():
                print(self.ui.water.text())
                self.camera_thread.medium = self.ui.water.text()
                self.medium = self.ui.water.text()
        if self.ui.air.isChecked():
                print(self.ui.air.text())
                self.camera_thread.medium = self.ui.air.text()
                self.medium = self.ui.air.text()

    def exposure_slider_to_spinbox(self, slider_val):
        slider_val /= 1000  # 1x10-4 de resolution --> 1x10-8 pour avoir 1 entre 10 000 000 et 9 999 999
        return round(10**slider_val)

    def exposure_spinbox_to_slider(self, spinbox_val):
        spinbox_val = np.log10(spinbox_val)
        return round(spinbox_val * 1000)

    def exposure_slider(self):
        """
        Change SpinBox value when Slider is moved. Exposure time.
        :return:
        """
        self.ui.exposureSpinBox.setValue(self.exposure_slider_to_spinbox(self.ui.exposureSlider.value()))
        self.exp = self.ui.exposureSpinBox.value()

        self.camera_thread.exp = self.ui.exposureSpinBox.value()  # Update camera thread

    def gain_slider(self):
        """
        Change SpinBox value when Slider is moved. Gain.
        :return:
        """

        self.ui.gainDoubleSpinBox.setValue(self.ui.gainSlider.value())
        self.gain = self.ui.gainDoubleSpinBox.value()

        self.camera_thread.gain = self.ui.gainDoubleSpinBox.value()  # Update camera thread

    def exposure_spin(self):
        """
        Change Slider value when SpinBox value is updated. Exposure time.
        :return:
        """
        self.ui.exposureSlider.setValue(self.exposure_spinbox_to_slider(self.ui.exposureSpinBox.value()))
        self.exp = self.ui.exposureSpinBox.value()

        self.camera_thread.exp = self.ui.exposureSpinBox.value()  # Update camera thread

    def gain_spin(self):
        """
        Change Slider value when SpinBox value is updated. Gain.
        :return:
        """
        self.ui.gainSlider.setValue(self.ui.gainDoubleSpinBox.value())
        self.gain = self.ui.gainDoubleSpinBox.value()

        self.camera_thread.gain = self.ui.gainDoubleSpinBox.value()   # Update camera thread

    def bin_combobox(self):
        """
        Modification of attributes bin when comboBox in changed. Binning
        :return:
        """
        self.bin = self.bin_choice[self.ui.binComboBox.currentText()]
        print(self.bin)

        self.camera_thread.bin = self.bin_choice[self.ui.binComboBox.currentText()]

    def create_path(self):
        today = datetime.datetime.utcnow()
        return "{0}/{1}_{2}/{3}".format(self.savepath, self.campaign, today.strftime("%Y%m%d"), self.foldername)

    def create_directory(self):
        """
        Function to create a directory if it doesn't already exists.

        :return:
        """

        complete_path = self.create_path()

        # Case if the directory already exists
        if os.path.exists(complete_path):

            self.ui.errorlog.setText("Directory already exists")
            print("Directory ", complete_path, " already exists")

        # Case of creation of the directory
        else:
            os.makedirs(complete_path)
            self.ui.errorlog.setText("Directory successfully created")
            print("Directory ", complete_path, " Created ")

        self.ui.saveFolder.setText(complete_path)
        self.longpath = complete_path

    def getdirectory(self):

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open files")

        if folder:
            self.savepath = folder
            self.create_directory()

    def camp_name_changed(self):
        self.campaign = self.ui.CampName.text()
        self.create_directory()

    def folder_name_changed(self):
        self.foldername = self.ui.fname.text()
        self.create_directory()

    def new_profile_button(self):
        """
        Function called when new_profile pushbutton is pressed.

        :return:
        """
        dirn = os.path.dirname(self.longpath)
        filelist = os.listdir(dirn)
        digit = []

        for name in filelist:
            diglist = [int(i) for i in name.split("_") if i.isdigit()]
            digit += diglist

        num = 1
        while num in digit:
            num+=1

        newname = "profile_{0:03}".format(num)

        self.ui.fname.setText(newname)

    def display_angle(self, xAngle, yAngle, zAngle):
        """
        Display roll, yaw and pitch.

        :param xAngle:
        :param yAngle:
        :param zAngle:
        :return:
        """
        self.ui.roll.setText("{0:.3f} ˚".format(xAngle))
        self.ui.pitch.setText("{0:.3f} ˚".format(yAngle))
        self.ui.yaw.setText("{0:.3f} ˚".format(zAngle))

        # Storing orientation
        self.orientation[0], self.orientation[1], self.orientation[2] = xAngle, yAngle, zAngle  # Roll, pitch, yaw
        self.camera_thread.orientation[0], self.camera_thread.orientation[1], self.camera_thread.orientation[2] = xAngle, yAngle, zAngle

    def update_camera(self):

        self.cam.set_imgdataformat(self.imformat)  # Image format
        self.cam.set_downsampling_type("XI_BINNING")
        self.cam.set_downsampling(self.bin)  # Downsampling
        self.cam.set_exposure(self.exp)  # Exposure time
        self.cam.set_gain(self.gain)  # Gain

    def metadata_xiMU(self, structure):
        """"
        Construction of xiMU metadata dictionary.
        """

        if isinstance(structure, xiapi.Image):
            met_dict = {}
            for i in structure._fields_:
                if isinstance(getattr(structure, i[0]), int) or isinstance(getattr(structure, i[0]), float):
                    met_dict[i[0]] = getattr(structure, i[0])

            # Adding roll, yaw and pitch in degrees to metadata
            met_dict["orientation roll"] = self.orientation[0]
            met_dict["orientation pitch"] = self.orientation[1]
            met_dict["orientation yaw"] = self.orientation[2]

            # Adding depth in cm to metadata
            met_dict["depth cm"] = self.ui.depth.value()

            # Adding medium to metadata
            met_dict["medium"] = self.medium.lower()

            return met_dict
        else:
            raise ValueError("Not the type of metadata expected. Should be a xiapi.Ximage instance.")

    def acquisition(self):
        """
        Acquisition of one image from transport buffer.

        :return:
        """
        if self.status:

            self.update_camera()  # Changing parameters of camera
            self.verify_temp()  # Verify if the board temperature is ok

            print("Starting data acquisition...")
            self.cam.start_acquisition()
            self.cam.get_image(self.img)

            data_raw = self.img.get_image_data_numpy()
            data_raw = data_raw[::-1, :]

            # Metadata in dictionary
            met_dict = self.metadata_xiMU(self.img)

            print("Stopping acquisition...")
            self.cam.stop_acquisition()

            return data_raw, met_dict

        else:
            raise ValueError("No device found.")

    def acquisition_button(self):
        """

        :return:
        """
        if self.savepath:
            darkim = glob.glob(self.savepath + self.foldername + "/DARK_*.tif")

            if darkim:
                # Stop real time data
                cond = self.ui.live.isChecked()
                self.ui.live.setChecked(False)

                # Acquisition
                image, metadata = self.acquisition()
                # Save
                today = datetime.datetime.utcnow()
                depth = self.ui.depth.value()
                path = self.longpath + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + str(depth) + "cm" + ".tif"

                self.saveTIFF_xiMU(path, image, metadata)

                if cond:
                    self.ui.live.setChecked(True)
            else:
                self.ui.errorlog.setText("Dark frame does not appear to be acquire.")
        else:
            self.ui.errorlog.setText("No directory selected.")

    def darkframe_button(self):
        """

        :return:
        """
        if self.savepath:
            darkim = glob.glob(self.savepath + self.foldername + "/DARK_*.tif")

            if not darkim:
                cond2 = self.ui.live.isChecked()
                self.ui.live.setChecked(False)

                # Acquisition
                image, metadata = self.acquisition()
                # Save
                today = datetime.datetime.utcnow()
                path = self.longpath + "/DARK_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"  # year/month/day

                self.saveTIFF_xiMU(path, image, metadata)

                if cond2:
                    self.ui.live.setChecked(True)
            else:
                self.ui.errorlog.setText("Dark frame already exists.")
        else:
            self.ui.errorlog.setText("No directory selected.")

    def verify_temp(self):
        temp = self.cam.get_sensor_board_temp()

        self.ui.boardTemp.setText("{0:.3f} ˚C".format(temp))

        if temp >= 65:
            raise ValueError("Board temperature exceeds 65˚C")

    def plot_avg(self, angle, rad_red, rad_green, rad_blue):

        self.pred.setData(angle, rad_red)
        self.pgreen.setData(angle, rad_green)
        self.pblue.setData(angle, rad_blue)

        #self.ui.visualisationWindow.clear()

        #self.pyqtplot(angle, rad_red, "611 nm", "r")
        #self.pyqtplot(angle, rad_green, "530 nm", "g")
        #self.pyqtplot(angle, rad_blue, "468 nm", "b")

        #self.ui.visualisationWindow.setLabel("left", "D.N normalized", size="6pt")
        #self.ui.visualisationWindow.setLabel("bottom", "Zenith angle [˚]", size="6pt")

        #self.ui.visualisationWindow.addLegend(offset=(10, 5))

    def pyqtplot(self, x, y, plotname, color):
        pen = pyqtgraph.mkPen(color=color)
        return self.ui.visualisationWindow.plot(x, y, name=plotname, pen=pen)

    def closeEvent(self, event):  # Should also do a functino for signal KILL code 137.....?
        """
        Over writing existing method.
        :param event:
        :return:
        """
        if self.status:
            print("Closing device")
            self.cam.close_device()

        #self.euler_thread.exit()
        #self.camera_thread.exit()
        event.accept()


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(app.exec_())
