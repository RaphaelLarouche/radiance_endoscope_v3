# -*- coding: utf-8 -*-
"""
File with of threads.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import imu_sensor
import radiance
import time
from ximea import xiapi
import numpy as np


class Euler(QtCore.QThread):
    """

    """
    my_signal = QtCore.pyqtSignal(float, float, float)

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.sensor_imu = imu_sensor.MinIMUv5()
        self.sensor_imu.acc_offsets()
        #self.sensor_imu.magnetometer_calibration()  # To be uncheked
        self.running = False

    def run(self):

        while self.running:
            euler_angles = self.sensor_imu.kalman_filter()
            self.my_signal.emit(euler_angles[0], euler_angles[1], euler_angles[2])
            time.sleep(0.004)  # To be changed?


class CameraThread(QtCore.QThread):
    """

    """

    my_signal = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    my_signal_temperature = QtCore.pyqtSignal(str)
    my_signal_saturation = QtCore.pyqtSignal(str)

    def __init__(self, imformat, bin, exp, gain, cam, medium):
        QtCore.QThread.__init__(self)

        self.cam = cam

        self.img = xiapi.Image()

        self.imformat = imformat
        self.bin = bin
        self.exp = exp
        self.gain = gain

        # Medium
        self.medium = medium

        # Orientation
        self.orientation = np.zeros(3)

        # Condition to run the function run
        self.running = False

    def run(self):

        while self.running:
            self.update_camera()

            self.verify_temp()

            self.cam.get_image(self.img)

            data_raw = self.img.get_image_data_numpy()
            data_raw = data_raw[::-1, :]

            if np.any(data_raw >= (2**12) - 1):
                numpixel = np.sum(data_raw >= (2**12) - 1)
                self.my_signal_saturation.emit("{0} pixels saturated".format(numpixel))
            else:
                self.my_signal_saturation.emit("0 pixel saturated")

            # Metadata in dictionary
            met_dict = self.metadata_xiMU(self.img)

            radclass = radiance.Radiance(data_raw, met_dict, self.medium.lower(), "test")  # Values to be changed
            radclass.absolute_radiance()
            radclass.makeradiancemap([0.5, 179.5], [0.5, 179.5], angular_res=0.5)
            #radclass.makeradiancemap_am(self.medium, self.orientation, angular_res=0.5)

            azi_average = radclass.azimuthal_integration()

            self.my_signal.emit(radclass.zenith_vect * 180/np.pi, azi_average[:, 0], azi_average[:, 1], azi_average[:, 2])

            time.sleep(0.5)  # 0.5 second repetition

    def update_camera(self):
        self.cam.set_imgdataformat(self.imformat)  # Image format
        self.cam.set_downsampling_type("XI_BINNING")
        self.cam.set_downsampling(self.bin)  # Downsampling
        self.cam.set_exposure(self.exp)  # Exposure time
        self.cam.set_gain(self.gain)  # Gain

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

    def verify_temp(self):
        temp = self.cam.get_sensor_board_temp()
        self.my_signal_temperature.emit("{0:.3f} ˚C".format(temp))

        if temp >= 65:
            raise ValueError("Board temperature exceeds 65˚C")
