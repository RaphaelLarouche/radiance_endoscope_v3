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
import imu_sensor
from ximea import xiapi
import numpy as np
import radiance
import threadfile


class MyDialog(QtWidgets.QDialog, cameracontrol.ProcessImage):
    def __init__(self):
        super(MyDialog, self).__init__()
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        # Attributes initialized
        self.savepath = False
        self.foldername = "/" + self.ui.fname.text()

        self.bin_choice = {"2x2": "XI_DWN_2x2", "4x4": "XI_DWN_4x4"}

        # Create ximea Camera instance
        self.status = False
        print("Opening camera")
        self.cam.open_device_by("XI_OPEN_BY_SN", "16990159")
        self.status = True

        self.cam = xiapi.Camera()

        # Create ximea Image instance to store image data and metadata
        self.img = xiapi.Image()

        self.imformat = "XI_RAW16"
        self.bin = self.bin_choice[self.ui.binComboBox.currentText()]
        self.exp = self.ui.exposureSpinBox.value()
        self.gain = self.ui.gainDoubleSpinBox.value()

        # IMU object
        self.euler_thread = threadfile.Euler()

        # Updating pyqtgraph appearange
        #self.ui.visualisationWindow.ui.roiBtn.hide()
        #self.ui.visualisationWindow.ui.menuBtn.hide()
        #self.ui.visualisationWindow.ui.histogram.hide()

        # Spinbox with keyboard tracking disable
        self.ui.exposureSpinBox.setKeyboardTracking(False)
        self.ui.gainDoubleSpinBox.setKeyboardTracking(False)

        # Creation of a PyQT timer object for live data visualization
        self.tim_camera = QtCore.QTimer(self)
        self.tim_camera.timeout.connect(self.realtimedata_camera)  # Timer connection with camera

        # Connections ___________________________________________________________________
        self.ui.exposureSlider.valueChanged.connect(self.exposure_slider)
        self.ui.gainSlider.valueChanged.connect(self.gain_slider)

        self.ui.exposureSpinBox.valueChanged.connect(self.exposure_spin)
        self.ui.gainDoubleSpinBox.valueChanged.connect(self.gain_spin)

        self.ui.binComboBox.currentTextChanged.connect(self.bin_combobox)

        self.ui.chooseFolderButton.clicked.connect(self.getdirectory)  # Choose folder button

        self.ui.openSensors.clicked.connect(self.open_sens)  # OPENING SENSORS

        self.ui.darkFrameButton.clicked.connect(self.darkframe_button)  # Darkframe button
        self.ui.acquisitionButton.clicked.connect(self.acquisition_button)  # Acquisition button

        self.ui.live.toggled.connect(self.start_realtimedata)  # Toggle of real time data --> starting timer

        self.ui.fname.editingFinished.connect(self.folder_name_changed)

        self.euler_thread.my_signal.connect(self.display_angle)

    def start_realtimedata(self):
        """
        Function executed when live checkbox is toggled. The function starts the timer (label tim).

        :return:
        """
        if self.ui.live.isChecked():
            # Starting acquisition
            print("Starting data acquisition...")
            self.cam.start_acquisition()

            # Camera thread
            self.tim_camera.start(1000)  # Updating each 1 s to prevent acquisition bug

            # IMU thread
            self.euler_thread.running = True
            self.euler_thread.start()

        else:
            self.tim_camera.stop()
            self.euler_thread.running = False

            # Stopping acquisition
            print("Stopping acquisition...")
            self.cam.stop_acquisition()

    def realtimedata_camera(self):
        """
        Function connected to the timer labeled tim.

        :return:
        """

        if self.status:

            # Camera data
            self.update_camera()  # Changing parameters of camera
            self.verify_temp()  # Verify if the board temperature is ok and updating lcd

            self.cam.get_image(self.img)

            data_raw = self.img.get_image_data_numpy()
            data_raw = data_raw[::-1, :]

            # Metadata in dictionary
            met_dict = self.metadata_xiMU(self.img)  # Metadata in dictionary


            radinstance = radiance.Radiance(data_raw, met_dict, "air", "test")
            radinstance.absolute_radiance()
            radinstance.makeradiancemap([0, 180], [0, 180], angular_res=0.25)

            azi_avg = radinstance.azimuthal_integration()

            # Update image
            #self.ui.visualisationWindow.setImage(data_raw.T)
            #self.ui.visualisationWindow.setImage(radmap.T)

            self.plot_avg(radinstance.zenith_vect * 180/np.pi, azi_avg[:, 0], azi_avg[:, 1], azi_avg[:, 2])

            # # IMU sensor data
            # xAngle, yAngle, zAngle = self.IMU.kalman_filter()  # xAngle - roll, yAngle - pitch, zAngle - Yaw
            # self.display_angle(xAngle, yAngle, zAngle)   # Showing angle

        else:
            raise ValueError("No devices found.")

    def open_sens(self):
        """
        Opening sensor when openSensors button is clicked. Only for the camera.
        :return:
        """

        # Camera
        print("Opening camera")
        self.cam.open_device_by("XI_OPEN_BY_SN", "16990159")
        self.status = True


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

    def gain_slider(self):
        """
        Change SpinBox value when Slider is moved. Gain.
        :return:
        """

        self.ui.gainDoubleSpinBox.setValue(self.ui.gainSlider.value())
        self.gain = self.ui.gainDoubleSpinBox.value()

    def exposure_spin(self):
        """
        Change Slider value when SpinBox value is updated. Exposure time.
        :return:
        """
        self.ui.exposureSlider.setValue(self.exposure_spinbox_to_slider(self.ui.exposureSpinBox.value()))
        self.exp = self.ui.exposureSpinBox.value()

    def gain_spin(self):
        """
        Change Slider value when SpinBox value is updated. Gain.
        :return:
        """
        self.ui.gainSlider.setValue(self.ui.gainDoubleSpinBox.value())
        self.gain = self.ui.gainDoubleSpinBox.value()

    def bin_combobox(self):
        """
        Modification of attributes bin when comboBox in changed. Binning
        :return:
        """
        self.bin = self.bin_choice[self.ui.binComboBox.currentText()]
        print(self.bin)

    def getdirectory(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open files")
        if folder:
            if not [f for f in os.listdir(folder) if not f.startswith('.')]:

                self.ui.saveFolder.setText(folder + self.foldername)
                self.savepath = folder

                self.folder_name_changed()

            else:
                raise FileExistsError("Directory not empty")

    def folder_name_changed(self):

        dirname = self.savepath + "/" + self.ui.fname.text()

        if not os.path.exists(dirname):
            os.mkdir(dirname)

            self.foldername = "/" + self.ui.fname.text()
            self.ui.saveFolder.setText(self.savepath + self.foldername)

            print("Directory ", dirname, " Created ")
        else:
            print("Directory ", dirname, " already exists")

    def display_angle(self, xAngle, yAngle, zAngle):
        """
        Display roll, yaw and pitch in the graphical interface.

        :param xAngle:
        :param yAngle:
        :param zAngle:
        :return:
        """
        # self.ui.roll.display("{0:.2f}".format(xAngle))
        # self.ui.pitch.display("{0:.2f}".format(yAngle))
        # self.ui.yaw.display("{0:.2f}".format(zAngle))

        self.ui.roll.setText("{0:.3f} ˚".format(xAngle))
        self.ui.pitch.setText("{0:.3f} ˚".format(yAngle))
        self.ui.yaw.setText("{0:.3f} ˚".format(zAngle))

    @staticmethod
    def metadata_xiMU(structure):
        """"
        Construction of xiMU metadata dictionary.
        """
        if isinstance(structure, xiapi.Image):
            met_dict = {}
            for i in structure._fields_:
                if isinstance(getattr(structure, i[0]), int) or isinstance(getattr(structure, i[0]), float):
                    met_dict[i[0]] = getattr(structure, i[0])
            return met_dict
        else:
            raise ValueError("Not the type of metadata expected. Should be a xiapi.Ximage instance.")

    def update_camera(self):

        self.cam.set_imgdataformat(self.imformat)  # Image format
        self.cam.set_downsampling_type("XI_BINNING")
        self.cam.set_downsampling(self.bin)  # Downsampling
        self.cam.set_exposure(self.exp)  # Exposure time
        self.cam.set_gain(self.gain)  # Gain

    def acquisition(self):
        """
        Acquisition of one image from transport buffer. No saving.

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
                path = self.savepath + self.foldername + "/IMG_" + today.strftime("%Y%m%d_%H%M%S_UTC_") + str(depth) + ".tif"

                self.saveTIFF_xiMU(path, image, metadata)

                if cond:
                    self.ui.live.setChecked(True)
            else:
                raise FileExistsError("Take darkframe before measurements.")  # Error to be changed
        else:
            raise IsADirectoryError("No directory selected.")

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
                path = self.savepath + self.foldername + "/DARK_" + today.strftime("%Y%m%d_%H%M%S_UTC") + ".tif"  # year/month/day

                self.saveTIFF_xiMU(path, image, metadata)

                if cond2:
                    self.ui.live.setChecked(True)
            else:
                raise FileExistsError("Dark frame already exists.")
        else:
            raise IsADirectoryError("No directory selected.")

    def verify_temp(self):
        temp = self.cam.get_sensor_board_temp()
        self.ui.boardTemp.display(temp)

        if temp >= 65:
            raise ValueError("Board temperature exceeds 65˚C")

    def plot_avg(self, angle, rad_red, rad_green, rad_blue):

        self.ui.visualisationWindow.clear()

        self.pyqtplot(angle, rad_red, "611 nm", "r")
        self.pyqtplot(angle, rad_green, "530 nm", "g")
        self.pyqtplot(angle, rad_blue, "468 nm", "b")

        self.ui.visualisationWindow.setLabel("left", "D.N normalized", size="6pt")
        self.ui.visualisationWindow.setLabel("bottom", "Zenith angle [˚]", size="6pt")

        self.ui.visualisationWindow.addLegend(offset=(10, 5))

    def pyqtplot(self, x, y, plotname, color):
        pen = pyqtgraph.mkPen(color=color)
        self.ui.visualisationWindow.plot(x, y, name=plotname, pen=pen)


    def closeEvent(self, event):  # Should also do a functino for signal KILL code 137.....?
        """
        Over writing existing method.
        :param event:
        :return:
        """

        print("Closing device")
        if self.status:
            self.cam.close_device()

        self.euler_thread.exit()
        event.accept()

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    dialog = MyDialog()
    dialog.showMaximized()
    sys.exit(app.exec_())
