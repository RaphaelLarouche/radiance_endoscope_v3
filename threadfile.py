# -*- coding: utf-8 -*-
"""
File with of threads.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import imu_sensor
import time


class Euler(QtCore.QThread):
    """

    """
    my_signal = QtCore.pyqtSignal(float, float, float)

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.sensor_imu = imu_sensor.MinIMUv5()
        self.sensor_imu.acc_offsets()
        self.running = False

    def run(self):

        while self.running:
            euler_angles = self.sensor_imu.kalman_filter()
            self.my_signal.emit(euler_angles[0], euler_angles[1], euler_angles[2])
            time.sleep(0.004)


class CameraThread(QtCore.QThread):
    """

    """

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.running = False

    def run(self):
        pass