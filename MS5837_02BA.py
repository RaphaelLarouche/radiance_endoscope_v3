#!/usr/bin/python
"""

"""

# Importation of modules
import time
import smbus
import numpy as np
import _thread as thread


class MS5837PressureSensor:
    """
    """

    def __init__(self, oversampling="OSR_8192", sampling_rate=1):

        # Oversampling possibilities ()
        OSRdict = {"OSR_256": 0, "OSR_512": 1, "OSR_1024": 2, "OSR_2048": 3, "OSR_4096": 4, "OSR_8192": 5}
        self.OSR = OSRdict[oversampling]
        self.sampling_rate = sampling_rate  # Hz

        # Hex adresses
        self.MS5837_ADDR = 0x76
        self.MS5837_RESET = 0x1E
        self.MS5837_ADC_READ = 0x00
        self.MS5837_PROM_READ = 0xA0
        self.MS5837_CONVERT_D1 = 0x40
        self.MS5837_CONVERT_D2 = 0x50

        # Bus
        self.bus = smbus.SMBus(1)
        _, self.C = self.initialize()

    def initialize(self):
        """

        :return:
        """

        self.bus.write_byte(self.MS5837_ADDR, self.MS5837_RESET)

        time.sleep(1)

        C = np.zeros(7)
        for i in range(0, 7):
            c = self.bus.read_word_data(self.MS5837_ADDR, self.MS5837_PROM_READ + 2 * i)
            c = ((c & 0xFF) << 8) | (c >> 8)
            C[i] = c
            print(c)

        time.sleep(1)
        return True, C

    def read_d1d2(self):

        # D1
        # self.bus.write_byte(self.MS5837_ADDR, self.MS5837_CONVERT_D1 + 2 * self.OSR)
        # time.sleep(2.5e-6 * 2 ** (8 + self.OSR))
        # d = self.bus.read_i2c_block_data(self.MS5837_ADDR, self.MS5837_ADC_READ, 3)
        # D1 = d[0] << 16 | d[1] << 8 | d[2]

        # D2
        # self.bus.write_byte(self.MS5837_ADDR, self.MS5837_CONVERT_D2 + 2 * self.OSR)
        # time.sleep(2.5e-6 * 2 ** (8 + self.OSR))
        # d = self.bus.read_i2c_block_data(self.MS5837_ADDR, self.MS5837_ADC_READ, 3)
        # D2 = d[0] << 16 | d[1] << 8 | d[2]

        D1 = self.read_data(self.MS5837_CONVERT_D1)
        D2 = self.read_data(self.MS5837_CONVERT_D2)

        return D1, D2

    def read_data(self, address):
        """

        :param address:
        :return:
        """
        self.bus.write_byte(self.MS5837_ADDR, address + 2 * self.OSR)
        time.sleep(2.6e-6 * 2 ** (8 + self.OSR))
        d = self.bus.read_i2c_block_data(self.MS5837_ADDR, self.MS5837_ADC_READ, 3)
        D = d[0] << 16 | d[1] << 8 | d[2]
        return D

    def pressure_temperature(self, D1, D2):
        """

        :param D1:
        :param D2:
        :return:
        """

        # First order
        dT = np.int32(D2 - (self.C[5] * 2**8))
        temp = 2000 + dT * self.C[6] / 2 ** 23

        off = np.int64(self.C[2] * 2 ** 17 + (self.C[4] * dT) / 2 ** 6)
        sens = np.int64(self.C[1] * 2 ** 16 + (self.C[3] * dT) / 2 ** 7)
        press = (D1 * sens / 2 ** 21 - off) / 2 ** 15

        # Second order
        ti = 0
        offi = 0
        sensi = 0
        if (temp / 100) < 20:
            ti = 11 * dT ** 2 / 2 ** 35
            offi = 31 * (temp - 2000) ** 2 / 2 ** 3
            sensi = 63 * (temp - 2000) ** 2 / 2 ** 5

        off2 = np.int64(off - offi)
        sens2 = np.int64(sens - sensi)

        temperature = (temp - ti) / 100  # deg C
        pressure = (((D1 * sens2 / 2 ** 21) - off2) / 2 ** 15) / 100  # mbar

        print("Temperature: {0:.4f}, Pressure: {1:.4f}".format(temperature, pressure))
        return temperature, pressure

    def calculate_depth(self):
        return

    def track_pressure(self):
        thread.start_new_thread(self.update_pressure, ())

    def update_pressure(self):
        while True:
            d1, d2 = self.read_d1d2()
            self.pressure_temperature(d1, d2)
            time.sleep(1 / self.sampling_rate)


if __name__ == "__main__":

    psensor = MS5837PressureSensor(sampling_rate=3)  # 3 Hz in sampling rate
    psensor.track_pressure()
