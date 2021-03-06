# -*- coding: utf-8 -*-
"""
Building on the class minimu9v5 from Github.
"""

# Importation of standard modules
import glob
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import time
import math
import _thread as thread

# Other modules
from MinIMU_v5_pi import MinIMU_v5_pi


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

        # Animation
        self.fig = False
        self.timeanim = False
        self.pl1 = False
        self.pl2 = False
        self.pl3 = False

        #self.magread = np.array([])
        self.magread = []

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

        # Madgwick algorithm _________
        self.timemad = 0
        self.q = np.zeros(4)
        self.q[0] += 1

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

    def magnetometer_calibration(self, save=True):

        # #self.fig, self.ax = plt.subplots(1, 1)
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        # self.pl1, = self.ax.plot([], [])
        # self.pl2, = self.ax.plot([], [])
        # self.pl3, = self.ax.plot([], [])
        #
        # anim = animation.FuncAnimation(self.fig, self.animcalibration_, frames=100, interval=0.004, repeat=False, blit=True)
        # plt.show()

        # Data acquisition
        s = np.array([])
        self.timer(5)

        print("Move the sensor around each axis.")
        for i in range(3000):
            #read_mag = np.array(self.readMagnetometer()[0])
            read_mag = self.readMagnetometer()
            print("{0:.3f} {1:.3f} {2:.3f}".format(read_mag[0], read_mag[1], read_mag[2]))
            s = np.append(s, read_mag)
            self.magread = read_mag

            time.sleep(0.004)

        s = np.reshape(s, (-1, 3))
        # Fit
        M, n, d = self.__ellipsoid_fit(s.T)

        self.b = -np.dot(np.linalg.inv(M), n)
        self.A = np.real((self.f / np.sqrt(np.dot(n.T, np.dot(np.linalg.inv(M), n))) - d) * linalg.sqrtm(M))

        # Printing results
        print(self.A)  # Soft iron
        print(self.b)  # Hard iron

        # Save results
        if save:
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

        E_w, E_v = np.linalg.eig(E)  # Eigenvalues, Eigenvectors

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

        np.savez("magnetometer_calib/magnetometer_calib.npz", date=time_string, A=self.A, b=self.b)

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

        print("Accelerometer offsets: " + "roll " + str(roll_offset) + "," + "pitch " + str(pitch_offset))

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

    def kalman_filter(self, disp=False):
        """
        Kalman filters for roll and pitch angles. The Heading data still use a complementary filter.

        :return:
        """
        # IMU sensors read
        [Ax, Ay, Az] = self.readAccelerometer()
        [Gx, Gy, Gz] = self.readGyro()
        [Mx, My, Mz] = self.magnetometer_correction(self.readMagnetometer())  # Magnetometer readings corrected
        t = self.read_temperature()

        # Time update
        if self.timekalman == 0:
            self.timekalman = time.time()

        dt = time.time() - self.timekalman

        # Transform gyro from dps (degrees per second) to rad/s
        Gx *= np.pi/180
        Gy *= np.pi/180
        Gz *= np.pi/180

        # Measurements roll and pitch
        roll = math.atan2(Ay, Az) - self.roll_off
        pitch = math.atan2(-Ax, math.sqrt(Ay ** 2 + Az ** 2)) - self.pitch_off

        # Filtering
        alpha = self.tau / (self.tau + dt)
        Gxa = self.roll_hat + Gx * dt
        Gya = self.pitch_hat + Gy * dt
        Gza = self.yaw_hat + Gz * dt  # In body frame  !!!!

        roll = (alpha * Gxa) + ((1 - alpha) * roll)
        pitch = (alpha * Gya) + ((1 - alpha) * pitch)

        # Tilt compensation for measurement of magnetometer parallel with magnetic field
        Mxh = Mx * np.cos(pitch) + My * np.sin(pitch) * np.sin(roll) + Mz * np.sin(pitch) * np.cos(roll)
        Myh = -Mz * np.sin(roll) + My * np.cos(roll)
        # yaw = math.atan2(-Myh, Mxh) + np.pi # Not sure if it's this
        yaw = math.atan2(Myh, Mxh)

        if Gza - yaw > np.pi:
            Gza -= 2 * np.pi
        if Gza - yaw < - np.pi:
            Gza += 2 * np.pi

        # This combines a LPF on phi, rho, and theta with a HPF on the Gyro values
        self.yaw_hat = (alpha * Gza) + ((1 - alpha) * yaw)

        # Angular velocity in inertial frame
        roll_dot = Gx + Gy * np.sin(self.roll_hat) * np.tan(self.pitch_hat) + Gz * np.cos(self.roll_hat) * np.tan(self.pitch_hat)
        pitch_dot = Gx * np.cos(self.roll_hat) - Gy * np.sin(self.roll_hat)
        yaw_dot = Gy * np.sin(self.roll_hat)/np.cos(self.pitch_hat) + Gz * np.cos(self.roll_hat)/np.cos(self.pitch_hat)

        # Yaw filtering
        # Gza = self.yaw_hat + yaw_dot * dt  # Yaw in intertial frame !!!!
        # Gza = self.yaw_hat + Gz * dt  # In body frame  !!!!

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

        # Posterior error covariance
        self.P = (np.eye(4) - K.dot(C)).dot(self.P)

        # Transforming angles in rad to degrees
        rhat = self.roll_hat * 180/np.pi
        phat = self.pitch_hat * 180/np.pi
        yhat = self.yaw_hat * 180/np.pi

        # Printing results
        if disp:
            print("roll: {0:+.5f} pitch: {1:+.5f} yaw: {2:+.5f} temperature: {3:.5f}˚C".format(rhat, phat, yhat, t))
            #print("Roll: " + str(rhat) + " Pitch: " + str(phat) + " Yaw: " + str(yhat) + "Temp: " + str(t))

        # Update time for dt calculations
        self.timekalman = time.time()

        return rhat, phat, yhat

    def magdwick_quaternion(self):
        """
        Implementation of the fusion algorithm from Sebastian O.H Madgwick
        (https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/). The algorithm computes the quaternion from the 9DoF
        data of the IMU.
        :return:
        """

        # IMU sensors read
        [ax, ay, az] = self.readAccelerometer()
        [gx, gy, gz] = self.readGyro()
        [mx, my, mz] = self.magnetometer_correction(self.readMagnetometer())  # Magnetometer readings corrected
        #t = self.read_temperature()

        # Time update
        if self.timemad == 0:
            self.timemad = time.time()

        dtime = time.time() - self.timemad

        q1, q2, q3, q4 = self.q[0], self.q[1], self.q[2], self.q[3]

        # Transform gyro from dps (degrees per second) to rad/s
        gx *= np.pi/180
        gy *= np.pi/180
        gz *= np.pi/180

        # Auxiliary variables
        _2q1, _2q2, _2q3, _2q4 = 2.0 * q1, 2.0 * q2, 2.0 * q3, 2.0 * q4
        _2q1q3, _2q3q4 = 2.0 * q1 * q3, 2.0 * q3 * q4
        q1q1, q1q2, q1q3, q1q4 = q1 * q1, q1 * q2, q1 * q3, q1 * q4
        q2q2, q2q3, q2q4 = q2 * q2, q2 * q3, q2 * q4
        q3q3, q3q4 = q3 * q3, q3 * q4
        q4q4 = q4 * q4

        # Normalise accelerometer measurements
        norma = np.sqrt(ax * ax + ay * ay + az * az)
        ax *= 1 / norma
        ay *= 1 / norma
        az *= 1 / norma

        # Normalise magnetometer measurements
        normm = np.sqrt(mx * mx + my * my + mz * mz)
        mx *= 1 / normm
        my *= 1 / normm
        mz *= 1 / normm

        # Earth reference magnetic field direction
        _2q1mx, _2q1my, _2q1mz = 2 * q1 * mx, 2 * q1 * my, 2 * q1 * mz
        _2q2mx = 2 * q2 * mx
        hx = mx * q1q1 - _2q1my * q4 + _2q1mz * q3 + mx * q2q2 + _2q2 * my * q3 + _2q2 * mz * q4 - mx * q3q3 - mx * q4q4
        hy = _2q1mx * q4 + my * q1q1 - _2q1mz * q2 + _2q2mx * q3 - my * q2q2 + my * q3q3 + _2q3 * mz * q4 - my * q4q4

        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = -_2q1mx * q3 + _2q1my * q2 + mz * q1q1 + _2q2mx * q4 - mz * q2q2 + _2q3 * my * q4 - mz * q3q3 + mz * q4q4
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        # Gradient descent algorithm
        s1 = -_2q3 * (2.0 * q2q4 - _2q1q3 - ax) + _2q2 * (2.0 * q1q2 + _2q3q4 - ay) - \
             _2bz * q3 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + \
             (-_2bx * q4 + _2bz * q2) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + \
             _2bx * q3 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)

        s2 = _2q4 * (2.0 * q2q4 - _2q1q3 - ax) + _2q1 * (2.0 * q1q2 + _2q3q4 - ay) - \
             4.0 * q2 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az) + \
             _2bz * q4 * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + \
             (_2bx * q3 + _2bz * q1) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + \
             (_2bx * q4 - _4bz * q2) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)

        s3 = -_2q1 * (2.0 * q2q4 - _2q1q3 - ax) + _2q4 * (2.0 * q1q2 + _2q3q4 - ay) - \
             4.0 * q3 * (1.0 - 2.0 * q2q2 - 2.0 * q3q3 - az) + \
             (-_4bx * q3 - _2bz * q1) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + \
             (_2bx * q2 + _2bz * q4) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + \
             (_2bx * q1 - _4bz * q3) * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)

        s4 = _2q2 * (2.0 * q2q4 - _2q1q3 - ax) + _2q3 * (2.0 * q1q2 + _2q3q4 - ay) + \
             (-_4bx * q4 + _2bz * q2) * (_2bx * (0.5 - q3q3 - q4q4) + _2bz * (q2q4 - q1q3) - mx) + \
             (-_2bx * q1 + _2bz * q3) * (_2bx * (q2q3 - q1q4) + _2bz * (q1q2 + q3q4) - my) + \
             _2bx * q2 * (_2bx * (q1q3 + q2q4) + _2bz * (0.5 - q2q2 - q3q3) - mz)

        norms = np.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
        s1 *= 1 / norms
        s2 *= 1 / norms
        s3 *= 1 / norms
        s4 *= 1 / norms

        # Rate of change of quaternion
        beta = 0.8  # tester pour avoir le meilleur paramètre
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - beta * s4

        # Integration for quaternion
        q1 += qDot1 * dtime
        q2 += qDot2 * dtime
        q3 += qDot3 * dtime
        q4 += qDot4 * dtime

        # Normalise quaternion
        normq = np.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
        self.q[0] = q1 / normq
        self.q[1] = q2 / normq
        self.q[2] = q3 / normq
        self.q[3] = q4 / normq

        rquat, pquat, yquat = self.quaternion_to_euler(self.q)
        print("roll: {0:+.5f} pitch: {1:+.5f} yaw: {2:+.5f}".format(rquat * 180 / np.pi,
                                                                    pquat * 180 / np.pi,
                                                                    yquat * 180 / np.pi))

        self.timemad = time.time()

        return self.q

    @staticmethod
    def quaternion_to_euler(q):
        """

        :return:
        """

        yawq = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
        pitchq = -1.0 * math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
        rollq = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])

        return rollq, pitchq, yawq

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
        Application of magnetometer correction on data.

        :param mag:
        :return:
        """

        mc = np.dot(self.A, np.array(mag).reshape(-1, 1) - self.b).T

        return [mc[:, 0], mc[:, 1], mc[:, 1]]

    def trackAngleCalib(self):
        """
        Starting a new thread using method trackAngleThreadCalib.
        :return:
        """
        thread.start_new_thread(self.trackAngleThreadCalib, ())

    def trackAngleThreadCalib(self):
        """
        Thread using updateAngleCalib which is essentially the same function as the one provided in the super
        class MinIMU_v5_pi.
        :return:
        """
        while True:
            self.updateAngleCalib()
            time.sleep(0.004)

    def trackAngleKalman(self):
        """
        Starting a new thread define in method track_angle_thread_kalman.
        :return:
        """
        thread.start_new_thread(self.track_angle_thread_kalman, ())

    def track_angle_thread_kalman(self):
        """
        Thread using kalman_filter method which is an amelioration of the original IMU code because it uses a Kalman
        Filter.
        :return:
        """
        while True:
            self.kalman_filter(disp=True)
            time.sleep(0.004)

    def trackquaternion(self):

        thread.start_new_thread(self.trackquaternionthread, ())

    def trackquaternionthread(self):

        while True:
            self.magdwick_quaternion()
            time.sleep(0.004)

    def start_anim(self):
        self.fig, self.ax = plt.subplots(1, 3, figsize=(8, 5))

        self.timeanim = time.time()

        ani = animation.FuncAnimation(self.fig, self.anim_, interval=4)  # Updated every 100 ms
        plt.show()
        self.reset()

    def anim_(self, i):
        time_diff = time.time() - self.timeanim
        # roll, pitch, yaw = self.updateAngleCalib()
        # roll_k, pitch_k, yaw_k = self.kalman_filter()

        self.magdwick_quaternion()
        rquat, pquat, yquat = self.quaternion_to_euler(self.q)


        # self.ax[0].plot(time_diff, roll, "r.")
        # self.ax[1].plot(time_diff, pitch, "r.")
        # self.ax[2].plot(time_diff, yaw, "r.")
        #
        # self.ax[0].plot(time_diff, roll_k, "b.")
        # self.ax[1].plot(time_diff, pitch_k, "b.")
        # self.ax[2].plot(time_diff, yaw_k, "b.")

        self.ax[0].plot(time_diff, rquat * 180 / np.pi, "g.")
        self.ax[1].plot(time_diff, pquat * 180 / np.pi, "g.")
        self.ax[2].plot(time_diff, yquat * 180 / np.pi, "g.")

        return self.ax

    def animcalibration_(self, i):
        print("Frame number {}".format(i))

        read_mag = np.array(self.readMagnetometer())
        #self.magread = np.append(self.magread, read_mag)
        self.magread.append(read_mag[0])

        print(self.magread)

        self.pl1.set_data(1, 2)
        self.pl2.set_data(2, 3)

        #self.pl1.set_ydata(read_mag[1])
        # self.pl2.set_data(self.magread[:, 0], self.magread[:, 2])
        # self.pl3.set_data(self.magread[:, 2], self.magread[:, 2])
        # self.ax.scatter(read_mag[0], read_mag[1], c="r", label="XY")
        # self.ax.scatter(read_mag[0], read_mag[2], c="g", label="XZ")
        # self.ax.scatter(read_mag[1], read_mag[2], c="b", label="YZ")

        #return self.pl1, self.pl2, self.pl3
        return [self.pl1, self.pl2, self.pl3]

    def animcalibinit(self):

        self.pl1.set_data([], [])
        self.pl2.set_data([], [])
        self.pl3.set_data([], [])

        # self.pl1 = self.ax.scatter(self.magread, self.magread, c="r")
        # self.pl2 = self.ax.scatter(self.magread, self.magread, c="g")
        # self.pl3 = self.ax.scatter(self.magread, self.magread, c="b")

        return self.pl1, self.pl2, self.pl3,


if __name__ == "__main__":

    sens = MinIMUv5()
    sens.magnetometer_calibration(save=False)  # calibration
    #sens.trackAngleKalman()
    #sens.start_anim()
    sens.trackquaternion()

    plt.show()
