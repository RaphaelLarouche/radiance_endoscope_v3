# -*- coding: utf-8 -*-
"""
Amilioration of the class minimu9v5 found on github.
"""

# Importation of standard modules
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob

import smbus2 as smbus
import time
import math
import _thread as thread

# Other modules
from MinIMU_v5_pi import MinIMU_v5_pi


# Adding filter,
# Adding a function to save the calibration results
# self.b and self.A imported from saved folder

class MinIMUv5(MinIMU_v5_pi):
    """
    Class which upgrade and correct errors in the already existing MinIMU_v5_pi.

    """

    def __init__(self, f=1):

        # Constructor of Class MinIMU_v5_pi
        super().__init__()

        # New attributes of class
        self.f = f
        self.b = np.zeros((3, 1))
        self.A = np.eye(3)

        filemagcal = glob.glob("magnetometer_calib/*.npz")

        if filemagcal:
            magcal = np.load("magnetometer_calib/magnetometer_calib.npz")
            datecalib = magcal["date"]

            print(datecalib)

            self.b = magcal["b"]
            self.A = magcal["A"]

        # For calibration
        #self.sensor = MinIMU_v5_pi()

        # Animation
        self.fig = False
        self.timeanim = False

        # Kalman filter __________________
        self.timekalman = 0

        self.roll_off = 0
        self.pitch_off = 0

        self.P = np.eye(4)
        self.Q = np.eye(4)
        self.R = np.eye(2) * 2

        Qmul = np.array([2, 0.03, 2, 0.03])
        self.Q *= Qmul[:, None]

        self.x_estimate = np.zeros((4, 1))
        self.pitch_hat = 0.0
        self.roll_hat = 0.0
        self.yaw_hat = 0.0

    def reset(self):
        """

        :return:
        """
        self.prevAngle = [[0, 0, 0]]  # x, y, z (roll, pitch, yaw)
        self.prevYaw = [0]
        self.tau = 0.04  # Want this roughly 10x the dt
        self.lastTimeAngle = [0]
        self.lastTimeYaw = [0]

    def timer(self, seconds):
        for i in range(seconds):
            print("Calibration starts in {0:d} s".format(seconds - i))
            time.sleep(1)

    def magnetometer_calibratation(self):

        # Data acquisition
        s = np.array([])
        self.timer(5)

        print("Move the sensor around each axis.")
        print("X Y Z")
        for i in range(3000):
            read_mag = np.array(self.readMagnetometer())
            print("{0:.3f} {1:.3f} {2:.3f}".format(read_mag[0], read_mag[1], read_mag[2]))
            s = np.append(s, read_mag)
            time.sleep(0.009)

        s = np.reshape(s, (-1, 3))
        # Fit
        M, n, d = self.__ellipsoid_fit(s.T)

        self.b = -np.dot(np.linalg.inv(M), n)
        self.A = np.real((self.f / np.sqrt(np.dot(n.T, np.dot(np.linalg.inv(M), n))) - d) * linalg.sqrtm(M))

        # Printing results
        print(self.A)  # Soft iron
        print(self.b)  # Hard iron

        # Save results
        self.save_magcal()

        # Show results
        self.results(s)

        # Reset angle and time data
        self.reset()

    def __ellipsoid_fit(self, s):
        """ Estimate ellipsoid parameters from a set of points.

            Parameters

            ----------
            s : array_like
              The samples (M,N) where M=3 (x,y,z) and N=number of samples.

            Returns
            -------
            M, n, d : array_like, array_like, float
              The ellipsoid parameters M, n, d.

            References
            ----------
            .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
               fitting," in Geometric Modeling and Processing, 2004.
               Proceedings, vol., no., pp.335-340, 2004
        """

        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                      2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                      2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6, :6]
        S_12 = S[:6, 6:]
        S_21 = S[6:, :6]
        S_22 = S[6:, 6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                      [1, -1,  1,  0,  0,  0],
                      [1,  1, -1,  0,  0,  0],
                      [0,  0,  0, -4,  0,  0],
                      [0,  0,  0,  0, -4,  0],
                      [0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(linalg.inv(C), S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array([[v_1[0], v_1[5], v_1[4]],
                      [v_1[5], v_1[1], v_1[3]],
                      [v_1[4], v_1[3], v_1[2]]])  # CORRECTED FROM SOURCE CODE....

        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d

    def results(self, data_before):

        # Compute corrected values
        data_after = np.dot(self.A, data_before.T - self.b).T

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")

        ax1.scatter(data_before[:, 0], data_before[:, 1], data_before[:, 2], color="r", label="Before correction")
        ax1.scatter(data_after[:, 0], data_after[:, 1], data_after[:, 2], color="b", label="Corrected")

        ax1.legend(loc="best")

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        ax2.scatter(data_after[:, 0], data_after[:, 1], label="XY")
        ax2.scatter(data_after[:, 0], data_after[:, 2], label="XZ")
        ax2.scatter(data_after[:, 1], data_after[:, 2], label="YZ")

        ax2.legend(loc="best")

    def save_magcal(self):
        time_tuple = time.gmtime()  # UTC time
        time_string = time.strftime("%d %b %Y, %H:%M:%S", time_tuple)

        np.savez("magnetometer_calib.npz", date=time_string, A=self.A, b=self.b)

    def acc_offsets(self, show=False):
        """
        Measurements of accelerometer angle roll and pitch offsets with 300 points.
        :return:
        """
        roll_offset = 0
        pitch_offset = 0

        rollarr = np.array([])
        pitcharr = np.array([])

        N = 1000
        for n in range(N):
            [Ax, Ay, Az] = self.readAccelerometer()

            pitch = math.degrees(math.atan2(-Ax, math.sqrt(Ay ** 2 + Az ** 2)))
            roll = math.degrees(math.atan2(Ay, Az))

            pitch_offset += pitch
            roll_offset += roll

            rollarr = np.append(rollarr, roll)
            pitcharr = np.append(pitcharr, pitch)

        roll_offset = float(roll_offset)/float(N)
        pitch_offset = float(pitch_offset)/float(N)

        roll_std = np.std(rollarr * np.pi / 180)
        pitch_std = np.std(pitcharr * np.pi / 180)

        #self.Q[0, 0] = roll_std ** 2
        #self.Q[1, 1] = pitch_std ** 2

        print("Accelerometer offsets: " + "roll " + str(roll_offset) + "," + "pitch " +  str(pitch_offset))

        self.roll_off = roll_offset * np.pi/180
        self.pitch_off = pitch_offset * np.pi/180

        if show:
            fig1, ax1 = plt.subplots(1, 2)

            ax1[0].hist(rollarr, bins="auto")
            ax1[0].set_ylabel("Roll angle [˚]")
            ax1[1].hist(pitcharr, bins="auto")
            ax1[1].set_ylabel("Pitch angle [˚]")

            fig2, ax2 = plt.subplots(1, 2)

            ax2[0].plot(np.arange(len(rollarr)), rollarr, 'r-')
            ax2[0].set_ylabel("Roll angle [˚]")
            ax2[0].set_xlabel("Points")
            ax2[1].plot(np.arange(len(pitcharr)), pitcharr, 'r-')
            ax2[1].set_ylabel("Pitch angle [˚]")
            ax2[1].set_xlabel("Points")

        return roll_offset, pitch_offset

    @staticmethod
    def R_roll(roll):
        """
        Rotation matrix roll angle.

        :param roll:
        :return:
        """
        return np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    @staticmethod
    def R_pitch(pitch):
        """
        Rotation matrix pitch angle.

        :param pitch:
        :return:
        """
        return np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    @staticmethod
    def R_yaw(yaw):
        """
        Rotation angle yaw angle.

        :param yaw:
        :return:
        """
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    def kalman_filter(self):
        """
        For roll and pitch angles.

        :return:
        """

        [Ax, Ay, Az] = self.readAccelerometer()
        [Gx, Gy, Gz] = self.readGyro()
        [Mx, My, Mz] = self.magnetometer_correction(self.readMagnetometer())  # Magnetometer readings corrected

        # Transform gyro in rad/s
        Gx *= np.pi/180
        Gy *= np.pi/180
        Gz *= np.pi/180

        # Time update
        if self.timekalman == 0:
            self.timekalman = time.time()

        dt = time.time() - self.timekalman

        # Measurements
        pitch = math.atan2(-Ax, math.sqrt(Ay ** 2 + Az ** 2)) - self.pitch_off
        roll = math.atan2(Ay, Az) - self.roll_off
        yaw = math.atan2(-My, Mx) + np.pi

        # Angular velocity in inertial frame
        roll_dot = Gx + Gy * np.sin(self.roll_hat) * np.tan(self.pitch_hat) + Gz * np.cos(self.roll_hat) * np.tan(self.pitch_hat)
        pitch_dot = Gx * np.cos(self.roll_hat) - Gy * np.sin(self.roll_hat)
        yaw_dot = Gy * np.sin(self.roll_hat)/np.cos(self.pitch_hat) + Gz * np.cos(self.roll_hat)/np.cos(self.pitch_hat)

        # Yaw
        Gza = self.yaw_hat + yaw_dot * dt  # Yaw in intertial frame !!!!
        #Gza = self.yaw_hat + Gz * dt  # In body frame  !!!!

        if Gza - yaw > np.pi:
            Gza -= 2 * np.pi
        if Gza - yaw < - np.pi:
            Gza += 2 * np.pi

        # This combines a LPF on phi, rho, and theta with a HPF on the Gyro values
        alpha = self.tau / (self.tau + dt)
        self.yaw_hat = (alpha * Gza) + ((1 - alpha) * yaw)

        gyro_input = np.array([[roll_dot], [pitch_dot]])

        # Kalman filter matrix
        A = np.array([[1, -dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dt], [0, 0, 0, 1]])
        B = np.array([[dt, 0], [0, 0], [0, dt], [0, 0]])
        C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # STATE ESTIMATE
        self.x_estimate = A.dot(self.x_estimate) + B.dot(gyro_input)
        self.P = A.dot(self.P.dot(np.transpose(A))) + self.Q

        # UPDATE
        y_tilde = np.array([[roll], [pitch]]) - C.dot(self.x_estimate)
        S = C.dot(self.P.dot(np.transpose(C))) + self.R
        K = self.P.dot(np.transpose(C).dot(np.linalg.inv(S)))

        # Final estimate using Kalman Gain
        self.x_estimate = self.x_estimate + K.dot(y_tilde)

        self.roll_hat = self.x_estimate[0, 0]
        self.pitch_hat = self.x_estimate[2, 0]

        rhat = self.roll_hat * 180/np.pi
        phat = self.pitch_hat * 180/np.pi
        yhat = self.yaw_hat * 180/np.pi

        # Posterior error covariance
        self.P = (np.eye(4) - K.dot(C)).dot(self.P)

        # Printing results
        #print("Roll: " + str(rhat) + " Pitch: " + str(phat) + " Yaw: " + str(yhat))

        # Update time for dt calculations
        self.timekalman = time.time()

        return rhat, phat, yhat

    def updateAngleCalib(self):
        """
        Update angle with calibration.
        :return:
        """
        [Ax, Ay, Az] = self.readAccelerometer()
        [Gx_w, Gy_w, Gz_w] = self.readGyro()
        [Mx, My, Mz] = self.magnetometer_correction(self.readMagnetometer())

        if self.lastTimeAngle[0] == 0:  # First time using updatePos
            self.lastTimeAngle[0] = time.time()

        # Find the angle change given by the Gyro
        dt = time.time() - self.lastTimeAngle[0]

        Gx = self.prevAngle[0][0] + Gx_w * dt
        Gy = self.prevAngle[0][1] + Gy_w * dt
        Gz = self.prevAngle[0][2] + Gz_w * dt

        # Using the Accelerometer find pitch and roll
        rho = math.degrees(math.atan2(-Ax, math.sqrt(Ay ** 2 + Az ** 2))) # Pitch
        phi = math.degrees(math.atan2(Ay, Az))  # Roll

        # Yaw
        theta = math.degrees(math.atan2(-My, Mx)) + 180

        if Gz - theta > 180:
            Gz = Gz - 360
        if Gz - theta < -180:
            Gz = Gz + 360

        # This combines a LPF on phi, rho, and theta with a HPF on the Gyro values
        alpha = self.tau / (self.tau + dt)
        xAngle = (alpha * Gx) + ((1 - alpha) * phi)
        yAngle = (alpha * Gy) + ((1 - alpha) * rho)
        zAngle = (alpha * Gz) + ((1 - alpha) * theta)

        # Update previous angle with the current one
        self.prevAngle[0] = [xAngle, yAngle, zAngle]  # Roll, pitch, yaw

        # Update time for dt calculations
        self.lastTimeAngle[0] = time.time()

        return xAngle, yAngle, zAngle   # roll, pitch, yaw

    def magnetometer_correction(self, mag):
        """
        Magnetometer correction.

        :param mag:
        :return:
        """

        mc = np.dot(self.A, np.array(mag).reshape(-1, 1) - self.b).T

        return [mc[:, 0], mc[:, 1], mc[:, 1]]

    def trackAngleCalib(self):
        thread.start_new_thread(self.trackAngleThreadCalib, ())

    def trackAngleThreadCalib(self):
        while True:
            self.updateAngleCalib()
            time.sleep(0.004)

    def trackAngleKalman(self):
        self.acc_offsets()
        thread.start_new_thread(self.track_angle_thread_kalman, ())

    def track_angle_thread_kalman(self):
        while True:
            self.kalman_filter()
            time.sleep(0.004)

    def start_anim(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(8, 5))
        self.acc_offsets()

        self.timeanim = time.time()
        anim = animation.FuncAnimation(self.fig, self.anim_, interval=100)  # Updated every 100 ms
        plt.show()

        self.reset()

    def anim_(self, i):
        time_diff = time.time() - self.timeanim
        roll, pitch, yaw = self.updateAngleCalib()

        roll_k, pitch_k, yaw_k = self.kalman_filter()

        self.ax[0].plot(time_diff, roll, "r.")
        self.ax[1].plot(time_diff, pitch, "r.")
        self.ax[2].plot(time_diff, yaw, "r.")

        self.ax[0].plot(time_diff, roll_k, "b.")
        self.ax[1].plot(time_diff, pitch_k, "b.")
        self.ax[2].plot(time_diff, yaw_k, "b.")


if __name__ == "__main__":
    sens = MinIMUv5()

    #sens.magnetometer_calibration()

    #sens.start_anim()

    sens.trackAngleKalman()
