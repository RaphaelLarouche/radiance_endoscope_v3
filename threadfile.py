# -*- coding: utf-8 -*-
"""
File with the threads.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
import imu_sensor
import time


class euler(QtCore.QThread):

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.sensor_imu = imu_sensor.MinIMUv5()
        self.euler_angles = 0, 0, 0
        self.my_signal = QtCore.pyqtSignal()
        self.running = True

    def run(self):

        while self.running:
            euler_angles = self.sensor_imu.kalman_filter()
            self.my_signal.emit(euler_angles[0], euler_angles[1], euler_angles[2])
            time.sleep(0.004)
