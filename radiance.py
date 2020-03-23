# -*- coding: utf-8 -*-
"""
File with radiance class to get an azimuthal and zenithal map of the absolute spectral radiance. This class will be use
to view in real-time spectral radiance data.

Optimize version with cleaner function (radiance_endoscope_v2 for original functions).

"""

# Importation of standard modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
from math import pi
import quadpy
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import time

# Importation of other modules
import cameracontrol


# Class radiance
class Radiance(cameracontrol.ProcessImage):
    """

    """

    def __init__(self, image, metadata, medium, savefolder):
        """
        Constructor which import all the calibration files. Entry parameters are the raw UNPROCESSED image, its metadata
        and the medium in which the camera is. The folder where the current data is going to be safe must be specified
        for dark frame subtraction.

        :param image:
        :param metadata:
        :param air_or_water:
        """
        # Importation of calibration data
        genpath = "calibrationfiles/"

        # 1. Absolute radiance calibration coefficients
        cal_coeff = np.load(genpath + "calibration_coefficients_2x2_20191128.npz")
        cal_coeff = cal_coeff["calibration_coefficients"]

        # Air
        if medium == "air":
            # 3. Geometric calibration
            geo = np.load(genpath + "geo_calibration_2x2_air_20200220_1908.npz")

            # 4. Roll=off calibration
            # After Villefranche
            rolloff = np.load(genpath + "rolloff_proto_sph_air_2x2_20200117.npz", allow_pickle=True)
            rolloff = rolloff["rolloff_fitparams"].item()

        # Water
        elif medium == "water":
            # 3. Geometric calibration
            geo = np.load(genpath + "geo_calibration_2x2_water_20200225_2052.npz")

            # 4. Roll-off calibration
            # SAME AS in the AIR ******* TO BE CHANGED
            # After Villefranche
            rolloff = np.load(genpath + "rolloff_proto_sph_air_2x2_20200117.npz", allow_pickle=True)
            rolloff = rolloff["rolloff_fitparams"].item()

        else:
            raise ValueError("Wrong input argument. Expect air or water")

        # Setting instance attributes
        # Define by constructor
        self.image = image
        self.metadata = metadata

        self.medium = medium
        self.savefolder = savefolder

        # New attributes
        self.imradiance = np.array([])

        self.radiance_map = np.array([])
        self.zenith_vect = np.array([])
        self.azimuth_vect = np.array([])

        # Calibration data
        self.calibration_coefficient = cal_coeff
        self.geometric_calibration = geo
        self.rolloff = rolloff

        self.zenith, self.azimuth = self.angularcoordinates_forcedzero(geo["imagesize"], geo["centerpoint"], geo["fitparams"])

    def absolute_radiance(self):
        """
        Method to compute absolute radiance from image and calibration results.

        :return:
        """

        if len(self.image.shape) == 2:

            path = glob.glob(self.savefolder + "/DARK_*")

            if path:
                im_nn = self.darksub(self.image.astype(float), "darkframe")   # 1. Dark frame subtraction (USING METADATA FOR NOW)
            else:
                im_nn = self.darksub(self.image.astype(float), "metadata")

            im_norm = self.exposure_normalization(self.gain_normalization(im_nn))   # 2. Exposure and gain normalization
            im_dws = self.dwnsampling(im_norm, "BGGR")  # 3. Downsampling

            im_roll_off = self.rolloff_correction(im_dws)  # 4. Roll-off correction
            im_abs = im_roll_off  # 5. Absolute radiance

            # 6. Immersion factor
            if self.medium == "water":
                im_abs *= 1.7

            # Setting attributes
            self.imradiance = im_abs

            return im_abs

        else:
            raise ValueError("This function must be done before demosaic.")

    def darksub(self, image, method):
        """
        Removal of dark noise.

        :param image: 2D image bayer mosaic
        :param method: darkframe subtraction (darkframe) or black level metadata (metadata.
        :return:
        """
        if method == "darkframe":
            path = glob.glob(self.savefolder + "/DARK_*")
            dark, metdark = self.readTIFF_xiMU(path[0])  # To be changed because of new method...

            image = image.astype(float)
            image -= dark.astype(float)

        elif method == "metadata":
            image = image.astype(float)
            image -= float(self.metadata["black_level"])

        else:
            raise ValueError("Invalid method argument. Expect dark frame or metadata.")

        return image

    def gain_normalization(self, image):
        """
        Digital numbers normalized by gain.

        :param image:
        :return:
        """

        return image/(self.gain_linear(self.metadata["gain_db"]))

    def exposure_normalization(self, image):
        """

        :param image:
        :return:
        """
        return image/self.exposure_second(self.metadata["exposure_time_us"])

    def rolloff_correction(self, image):
        """
        Application of roll-off correction for the three bands of the radiance camera.

        :param image: Image (in 3D, multi-spectral)
        :return: Image correct for roll-off.
        """

        if len(image.shape) == 3:

            rolloff_keys = ["red channel", "green channel", "blue channel"]  # Image 3rd dimension

            # Pre-allocation
            new_im = np.empty(image.shape)

            # Downsampling zenith
            zen_dws = self.dwnsampling(self.zenith, "BGGR", ave=True)

            for i in range(image.shape[2]):

                # Selection rolloff
                popt = self.rolloff[rolloff_keys[i]]

                # Apply correction to zenith below 110
                cond = zen_dws[:, :, i] <= 110

                # Rolloff correction
                new_im[cond, i] = image[cond, i]/self.rolloff_polynomial(zen_dws[cond, i], *popt)

        else:
            raise ValueError("The roll-off is applied to each band separately. Demosaic must already be done.")

        return new_im

    def radiancemap_rotation(self, medium, orientation, angular_res=1.0):
        """


        :param medium:
        :param orientation:
        :param angular_res:
        :return:
        """

        # Finding coordinates in X, Y and Z of each pixel of the camera
        imsize = self.geometric_calibration["imagesize"]
        distcenter = self.geometric_calibration["centerpoint"]
        fitcoefficients = self.geometric_calibration["fitparams"]

        camtheta, camphi = self.angularcoordinates_forcedzero(imsize, distcenter, fitcoefficients)

        mask = camtheta > 90
        camtheta[mask] = np.nan
        camphi[mask] = np.nan
        camtheta *= pi/180
        camphi *= pi/180

        py, pz, px = self.points_3d(camtheta, camphi)  # x is in direction of optical axis

        # Application of the rotation according to IMU data

        # According to our coordinate system with z pointing upward, the X axis and Y axis of the IMU are interchanged.
        # For this reason, the roll which is now about the Y axis has to change in sign.

        orientation = orientation.astype(float)
        orientation *= pi/180
        roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
        roll = -roll

       # matrix_rotation = np.dot(np.dot(self.rz(yaw), self.ry(roll)), self.rx(pitch))  # Inversion of pitch and roll
        matrix_rotation = np.dot(self.ry(roll), self.rx(pitch))
        npx, npy, npz = self.rotation(px, py, pz, matrix_rotation)

        # Find maximum and minimum in zenith and azimuth angles (according to Z pointing upward).
        azimuth = np.arctan2(npy, npx)
        maskneg = azimuth < 0.0
        azimuth[maskneg] = azimuth[maskneg] + 2*pi
        zenith = np.arccos(npz)

        limits_zen = (np.nanmin(zenith), np.nanmax(zenith))
        limits_az = (0.5 * pi/180, 179.5 * pi/180)

        # 1. Building zenith and azimuth array and meshgrid
        nb_zen = np.round(abs(limits_zen[1] - limits_zen[0]) / (angular_res * pi / 180)) + 1
        nb_azi = np.round(abs(limits_az[1] - limits_az[0]) / (angular_res * pi / 180)) + 1

        zenith_vector = np.linspace(limits_zen[0], limits_zen[1], nb_zen.astype(int))  # rads
        azimuth_vector = np.linspace(limits_az[0], limits_az[1], nb_azi.astype(int))  # rads

        azi, zeni = np.meshgrid(azimuth_vector, zenith_vector)  # Meshgrid

        pxreg, pyreg, pzreg = self.points_3d(zeni, azi)


        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111, projection="3d")
        #
        # ax1.scatter(npx[::10, ::10], npy[::10, ::10], npz[::10, ::10])
        # ax1.scatter(pxreg, pyreg, pzreg)
        # ax1.set_xlabel("Xaxis")
        # ax1.set_ylabel("Yaxis")
        # ax1.set_zlabel("Zaxis")
        #
        #
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(121)
        # ax3 = fig2.add_subplot(122)
        #
        # ax3.imshow(azimuth)
        # ax2.imshow(zenith)
        #
        # plt.show()

    def radiancemap_interpolation(self, medium, orientation, angular_res=1.0):

        if len(self.imradiance.shape) == 3:

            # Finding coordinates in X, Y and Z of each pixel of the camera
            imsize = self.geometric_calibration["imagesize"]
            distcenter = self.geometric_calibration["centerpoint"]
            fitcoefficients = self.geometric_calibration["fitparams"]

            camtheta, camphi = self.angularcoordinates_forcedzero(imsize, distcenter, fitcoefficients)

            mask = camtheta > 90
            camtheta[mask] = np.nan
            camphi[mask] = np.nan
            camtheta *= pi / 180
            camphi *= pi / 180

            py, pz, px = self.points_3d(camtheta, camphi)  # x is in direction of optical axis

            # Application of the rotation according to IMU data

            # According to our coordinate system with z pointing upward, the X axis and Y axis of the IMU are interchanged.
            # For this reason, the roll which is now about the Y axis has to change in sign.

            orientation = orientation.astype(float)
            orientation *= pi / 180
            roll, pitch, yaw = orientation[0], orientation[1], orientation[2]
            roll = -roll

            # matrix_rotation = np.dot(np.dot(self.rz(yaw), self.ry(roll)), self.rx(pitch))  # Inversion of pitch and roll
            matrix_rotation = np.dot(self.ry(roll), self.rx(pitch))
            npx, npy, npz = self.rotation(px, py, pz, matrix_rotation)

            # Regular grid 4 pi steradian
            limits_zen = np.array([0.5, 179.5]) * pi / 180
            limits_az = np.array([0.5, 359.5]) * pi / 180

            nb_zen = np.round(abs(limits_zen[1] - limits_zen[0]) / (angular_res * pi / 180)) + 1
            nb_azi = np.round(abs(limits_az[1] - limits_az[0]) / (angular_res * pi / 180)) + 1

            zenith_vector = np.linspace(limits_zen[0], limits_zen[1], nb_zen.astype(int))  # rads
            azimuth_vector = np.linspace(limits_az[0], limits_az[1], nb_azi.astype(int))  # rads

            azi, zeni = np.meshgrid(azimuth_vector, zenith_vector)  # Meshgrid

            pxreg, pyreg, pzreg = self.points_3d(zeni, azi)  # Wanted points

            # Reference points of the camera (demosaic eventually)
            cond = ~np.isnan(npx)
            points = np.c_[npx[cond], npy[cond], npz[cond]]

            grid_points = np.c_[pxreg.ravel(), pyreg.ravel(), pzreg.ravel()]

            tim1 = time.time()
            tree = cKDTree(points[::10])
            distances, _ = tree.query(grid_points)
            distances = distances.reshape(pxreg.shape)
            tim2 = time.time()

            print(tim2 - tim1)

            #interp = griddata(points, val[cond], (pxreg, pyreg, pzreg), method="nearest")
            #interp = interp.reshape(pxreg.shape).astype(float)

            toofar = distances > 0.08
            #interp[toofar] *= np.nan
            pxreg[toofar] *= np.nan
            pyreg[toofar] *= np.nan
            pzreg[toofar] *= np.nan

            # Dewarping instead of interpolation
            # 3. Application of rotation matrix
            rotation_mat = np.dot(self.ry(-roll), self.rx(-pitch))  # 180 around z and 90 around x
            nnpx, nnpy, nnpz = self.rotation(pxreg, pyreg, pzreg, rotation_mat)

            theta = np.arccos(nnpx)
            phi = np.arctan2(nnpz, nnpy)

            # 4. Dewarping
            dewarped = np.zeros((theta.shape[0], theta.shape[1], 3))

            for i in range(len(dewarped.shape)):
                dewarped[:, :, i] = self.dewarp(self.imradiance[:, :, i], theta, phi)

            # 5. Storing new radiance map image
            self.radiance_map = dewarped
            self.zenith_vect = zenith_vector
            self.azimuth_vect = azimuth_vector

            # Viz
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111, projection="3d")

            #ax1.scatter(npx[::10, ::10], npy[::10, ::10], npz[::10, ::10])
            ax1.scatter(pxreg[::10, ::10], pyreg[::10, ::10], pzreg[::10, ::10])
            ax1.scatter(nnpx[::10, ::10], nnpy[::10, ::10], nnpz[::10, ::10])
            ax1.set_xlabel("Xaxis")
            ax1.set_ylabel("Yaxis")
            ax1.set_zlabel("Zaxis")
            #
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(121)
            ax3 = fig2.add_subplot(122)

            ax2.imshow(distances)
            ax3.imshow(dewarped[:, :, 0])

            plt.show()

            return dewarped

        else:
            raise ValueError("Demosaic should be done before building radiance mapping.")


    def makeradiancemap_am(self, medium, orientation, angular_res=1.0):
        """
        New version of radiance mapping. It now uses information about the medium to set the camera limits in zenith
        and azimuth as well as the roll, pitch and yaw angles.

        :param medium:
        :param orientation:
        :param angular_res:
        :return:
        """

        if len(self.imradiance.shape) == 3:

            degradconst = pi/180

            roll = orientation[0] * degradconst
            pitch = orientation[1] * degradconst
            yaw = orientation[2] * degradconst

            if medium.lower() == "air":
                cam_zen_limits = np.array([0.5, 179.5]) * degradconst  # Preve?nt division by 0 ?
                #cam_az_limits = np.array([-89.5, 89.5]) * degradconst
                cam_az_limits = np.array([0.5, 179.5]) * degradconst

            elif medium.lower() == "water":
                cam_zen_limits = np.array([10, 170]) * degradconst
                cam_az_limits = np.array([10, 170]) * degradconst
            else:
                raise ValueError("Invalid argument medium.")

            # Building meshgrid
            nb_zen = np.round(abs(cam_zen_limits[1] - cam_zen_limits[0]) / (angular_res * degradconst)) + 1
            nb_az = np.round(abs(cam_az_limits[1] - cam_az_limits[0]) / (angular_res * degradconst)) + 1

            zen_vector = np.linspace(cam_zen_limits[0], cam_zen_limits[1], nb_zen.astype(int))  # rads
            az_vector = np.linspace(cam_az_limits[0], cam_az_limits[1], nb_az.astype(int))  # rads

            az, zen = np.meshgrid(az_vector, zen_vector)  # Meshgrid

            # Converting into world coordinates with Z pointing upward
            px, py, pz = self.points_3d(zen, az)

            theta = np.arccos(py)
            phi = np.arctan2(pz, px)

            # Dewarping
            dewarped = np.zeros((theta.shape[0], theta.shape[1], 3))

            for i in range(len(dewarped.shape)):
                dewarped[:, :, i] = self.dewarp(self.imradiance[:, :, i], theta, phi)

            # Rotation of coordinates according to IMU data
            matrix_rotation = np.dot(np.dot(self.rz(yaw), self.ry(pitch)), self.rx(roll))
            npx, npy, npz = self.rotation(px, py, pz, matrix_rotation)

            azimuth_array = np.arctan2(npy, npx)
            zenith_array = np.arccos(npz)

            print(np.min(np.min(zenith_array)) * 180/pi)
            print(np.max(np.max(zenith_array)) * 180/pi)

            print(np.min(np.min(azimuth_array)) * 180 / pi)
            print(np.max(np.max(azimuth_array)) * 180 / pi)


            plt.figure()
            plt.imshow(azimuth_array*180 / pi)

            plt.figure()
            plt.imshow(zenith_array*180 / pi)

            fig20 = plt.figure()
            ax20 = fig20.add_subplot(111, projection="3d")

            ax20.scatter(npx, npy, npz)
            ax20.set_xlabel("Xaxis")
            ax20.set_ylabel("Yaxis")
            ax20.set_zlabel("Zaxis")

            # 5. Storing new radiance map image
            self.radiance_map = dewarped
            self.zenith_vect = zenith_array[:, 0]
            self.azimuth_vect = azimuth_array[0, :]

            return dewarped

        else:
            raise ValueError("Demosaic should be done before building radiance mapping.")

    def makeradiancemap(self, zeni_lims, azi_lims, orientation, angular_res=1.0):
        """

        :param im: 
        :param zeni_lims: 
        :param azi_lims: 
        :param angular_res: 
        :return: 
        """""
        if len(self.imradiance.shape) == 3:

            # 1. Building zenith and azimuth array and meshgrid
            nb_zen = np.round(abs(zeni_lims[1] * pi/180 - zeni_lims[0] * pi/180)/(angular_res * pi/180)) + 1
            nb_azi = np.round(abs(azi_lims[1] * pi/180 - azi_lims[0] * pi/180)/(angular_res * pi/180)) + 1

            zenith_vector = np.linspace(zeni_lims[0] * pi/180, zeni_lims[1] * pi/180, nb_zen.astype(int))  # rads
            azimuth_vector = np.linspace(azi_lims[0] * pi/180, azi_lims[1] * pi/180, nb_azi.astype(int))   # rads

            azi, zen = np.meshgrid(azimuth_vector, zenith_vector)  # Meshgrid

            # 2. Calcul of world coordinates (Z pointing toward zenith)
            px, py, pz = self.points_3d(zen, azi)

            # 3. Application of rotation matrix
            rotation_mat = np.dot(self.rz(pi), self.rx(pi/2))  # 180 around z and 90 around x
            npx, npy, npz = self.rotation(px, py, pz, rotation_mat)

            theta = np.arccos(npz)
            phi = np.arctan2(npy, npx)

            # 4. Dewarping
            dewarped = np.zeros((theta.shape[0], theta.shape[1], 3))

            for i in range(len(dewarped.shape)):
                dewarped[:, :, i] = self.dewarp(self.imradiance[:, :, i], theta, phi)

            # 5. Storing new radiance map image
            self.radiance_map = dewarped
            self.zenith_vect = zenith_vector
            self.azimuth_vect = azimuth_vector

            return dewarped

        else:
            raise ValueError("Demosaic should be done before building the radiance 2d map.")

    def dewarp(self, image, theta, phi):
        """

        :param image:
        :param theta:
        :param phi:
        :return:
        """
        # Center coordinate of image
        cdx, cdy = self.geometric_calibration["centerpoint"][0]/2, self.geometric_calibration["centerpoint"][1]/2
        cdx, cdy = cdx.astype(int) - 1, cdy.astype(int) - 1

        # Inverse mapping to get radial position in function of theta
        rho = self.polynomial_fit_forcedzero(theta * 180/pi, *self.inverse_mapping()) * 0.5  # Division by two (dws)

        # x and y pos
        xcam, ycam = rho * np.cos(phi), rho * np.sin(phi)
        xcam, ycam = xcam.astype(int) + cdx, -ycam.astype(int) + cdy

        return image[ycam, xcam]

    def inverse_mapping(self, rsq=False):
        """

        :param rsq:
        :return:
        """
        radial = np.linspace(0, 350, 500)  # Different for 2x2 and 4x4 binning?
        angles = self.polynomial_fit_forcedzero(radial, *self.geometric_calibration["fitparams"])

        # Curve fit
        popt, pcov = self.geometric_curvefit_forcedzero(angles, radial)

        if rsq:

            # Coefficient of determination
            residuals = radial - self.polynomial_fit_forcedzero(angles, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((radial - np.mean(radial)) ** 2)

            rsquared = 1 - (ss_res / ss_tot)

            # Printing results
            print(popt)
            print(rsquared)

        return popt

    def azimuthal_integration(self):
        """
        Azimuthal average of spectral radiance 2D map.

        :return:
        """

        if self.radiance_map.any():
            return np.trapz(self.radiance_map, x=self.azimuth_vect, axis=1)
        else:
            print("2d map of spectral radiance has not been calculated.")

    def zenithal_integration(self):
        """
        Zenithal average of spectral radiance 2D map.

        :return:
        """

        if self.radiance_map.any():
            return np.trapz(self.radiance_map, x=self.zenith_vect, axis=0)

        else:
            print("2d map of spectral radiance haven't been calculated.")

    def lebedev_integration(self, xyz, rad, theta, phi, order, viz=False):

        th = np.sort(np.array(theta))
        phi = np.sort(np.array(phi))

        # Creation of the lebedev sphere scheme
        scheme = self.lebedev_sphere_creation(order)

        # Selection of the interested data according to theta and phi
        sph_coord = quadpy.sphere._helpers.cartesian_to_spherical(scheme.points)

        # Conversion of phi between -pi, pi to 0, 2pi
        sph_coord[:, 0] = (sph_coord[:, 0] < 0) * (2 * pi + sph_coord[:, 0]) + (sph_coord[:, 0] > 0) * sph_coord[:, 0]

        # Mask
        mask_theta = ((th[0] * pi / 180 <= sph_coord[:, 1]) & (sph_coord[:, 1] <= th[1] * pi / 180))
        mask_phi = ((phi[0] * pi / 180 <= sph_coord[:, 0]) & (sph_coord[:, 0] <= phi[1] * pi / 180))
        masktot = mask_theta & mask_phi

        # Normalization
        area = self.area(theta, phi)
        norm = area / np.sum(scheme.weights[masktot])

        points_inter = self.data_interpolator(xyz[:, 0], xyz[:, 1], xyz[:, 2], rad, scheme.points[masktot, :])
        int = np.dot(points_inter, norm * scheme.weights[masktot])

        # Visualization
        if viz:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(scheme.points[masktot, 0], scheme.points[masktot, 1], scheme.points[masktot, 2],
                       c=points_inter, cmap="BrBG", s=2)

            ax.set_title("3D Lebedev nodes used for integration")
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")

        return int

    @staticmethod
    def lebedev_sphere_creation(order):
        """
        Creation of a lebedev sphere scheme.

        :param order:
        :return:
        """

        if (order in range(5, 33, 2)) or (order in range(35, 137, 6)):

            return quadpy.sphere._lebedev._read("{0:0>3}".format(order))

        else:
            raise ValueError("Invalid quadrature.")

    @staticmethod
    def area(theta, phi):
        """
        Integration of sin(theta) dtheta dphi between specified range of theta and phi.

        :param theta: Must be between 0 and 180
        :param phi: Must be between 0 and 360
        :return:
        """

        the = np.sort(np.array(theta))
        phi = np.sort(np.array(phi))

        return (phi[1] * pi / 180 - phi[0] * pi / 180) * (np.cos(the[0] * pi / 180) - np.cos(the[1] * pi / 180))

    @staticmethod
    def data_interpolator(X, Y, Z, val, newcoord):
        """

        :param acquisition_coord:
        :param val:
        :return:
        """
        A = np.sum(np.sum(~np.isnan(X) == ~np.isnan(Y)))
        B = np.sum(np.sum(~np.isnan(X) == ~np.isnan(Z)))

        if (A == X.shape[0]) and (B == X.shape[0]):
            cond = ~np.isnan(X)

            points = np.array([X[cond], Y[cond], Z[cond]]).T
            interp = griddata(points, val[cond], (newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]), method="nearest")

        else:
            raise ValueError("Arrays don't seams to have the same size.")

        return interp

    @staticmethod
    def combine_sphere_coordinates(arr1, arr2):
        """
        Static method to combine X, Y, or Z coordinates of the two images to generate a complete sphere.

        :param arr1: First camera 3D array (X, Y or Z)
        :param arr2: Second camera 3D array (X, Y or Z)
        :return:
        """

        if arr2.shape[2] == 3 and arr2.shape[2] == 3:

            # Merge of X, Y and Z for the complete sphere
            cmb = np.empty((arr1.shape[0] * arr1.shape[1] * 2, 3))

            for n in range(arr1.shape[2]):
                cmb[:, n] = np.append(arr1[:, :, n].flatten(), arr2[:, :, n].flatten())
            return cmb
        else:
            raise ValueError("Both array should be 3 dimensionals.")

    @staticmethod
    def rotation(PX, PY, PZ, rmat):
        """
        Function that applies a specified 3D rotation matrix on camera coordinates.

        :param PX:
        :param PY:
        :param PZ:
        :param rotation_matrix:
        :return:
        """
        # Application of rotation matrix
        rotation = rmat.dot(np.array([PX.flatten(), PY.flatten(), PZ.flatten()]))

        return rotation[0, :].reshape(PX.shape), rotation[1, :].reshape(PY.shape), rotation[2, :].reshape(PZ.shape)

    @staticmethod
    def points_3d(zeni, azi):
        """

        :param zeni: Zenith 2D array in radian.
        :param azi: Azimuth 2D array in radian.
        :return:
        """

        return np.sin(zeni) * np.cos(azi), np.sin(zeni) * np.sin(azi), np.cos(zeni)

    @staticmethod
    def rx(roll):
        """
        Rotation matrix around x axis.

        :param roll:
        :return:
        """
        return np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    @staticmethod
    def ry(pitch):
        """
        Rotation matrix around y axis.

        :param pitch:
        :return:
        """
        return np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    @staticmethod
    def rz(yaw):
        """
        Rotation matrix around z axis.

        :param yaw:
        :return:
        """
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])


class ProcessRadianceData:
    """

    """

    def down_irradiance(self, type):
        return

    def up_irradiance(self, type):
        return

    def irradiance_cosinus(self):
        return

    def irradiance_scalar(self):
        return

    def interpolator(self):
        return


if __name__ == "__main__":

    # Object
    PI = cameracontrol.ProcessImage()

    #path_im_test = "/Users/raphaellarouche/Desktop/IMG_20200117_194959_UTC.tif"
    #path_im_test = "/Users/raphaellarouche/Desktop/IMG_20200218_225010_UTC_testdewarp2.tif"
    path_im_test = "/home/pi/Desktop/test18mars/ParcBic_20200319/profile_001/IMG_20200319_173420_UTC_0cm.tif"
    impathstr = glob.glob(path_im_test)

    image, metadata = PI.readTiff_xiMU_multiple_exposure(impathstr[0])
    print(image.shape)
    print(metadata.keys())

    # Instance of RadianceClass
    #a = Radiance(image, metadata, "air", "test")
    #image_dwnsampled = PI.dwnsampling(a.image, "BGGR")
    # Showing dewarped image
    #dw = a.makeradiancemap(image_dwnsampled)

    #a.camera_zenith_azimuth_viz()

    #print(a.lebedev_integration_test([90, 180], [0, 360], 125, viz=True))

    # Example normal processing
    # t0 = time.time()
    # RAD = Radiance(image, metadata, "air", "ok")
    # imRAD = RAD.absolute_radiance()
    # imRADmap = RAD.makeradiancemap([0, 180], [0, 180], angular_res=0.25)
    #
    # imRADmap2 = RAD.makeradiancemap_am("air", np.array([45, 0, 0]), angular_res=5)
    #
    # azi_avg = RAD.azimuthal_integration()
    # tf = time.time()
    # dt = tf - t0
    #print(dt)

    RAD = Radiance(image[0, :, :], metadata["Image001"], "air", "ok")
    #RAD.radiancemap_rotation("air", np.array([0, 0, 0]), angular_res=10)
    #RAD.radiancemap_rotation("air", np.array([0, 0, 180]), angular_res=10)

    RAD.absolute_radiance()
    timestart = time.time()
    RAD.radiancemap_interpolation("air", np.array([45, 0, 0]), angular_res=1)
    #RAD.makeradiancemap([0.5, 179.5], [0.5, 179.5], np.array([45, 0, 0]), angular_res=0.5)
    timeend = time.time()

    print(timeend - timestart)

    #plt.show()

    # Test limits of field of view
    # theta_pi1 = np.array([0, 0])
    # theta_phi2 = np.array([pi/2, 2*pi])
    #
    # for j in [theta_pi1, theta_phi2]:
    #     px, py, pz = RAD.points_3d(j[0], j[1])
    #     npx, npy, npz = RAD.rotation(px, py, pz, RAD.rx(pi/2))
    #
    #     theta = np.arccos(npz)
    #     phi = np.arctan2(npy, npx)
    #
    #     print(theta)
    #     print(phi)

    # plt.figure()
    # plt.imshow(imRADmap[:, :, 0])
    #
    # plt.figure()
    # plt.plot(RAD.zenith_vect * 180/pi, azi_avg[:, 0], "r")
    # plt.plot(RAD.zenith_vect * 180/pi, azi_avg[:, 1], "g")
    # plt.plot(RAD.zenith_vect * 180/pi, azi_avg[:, 2], "b")

    # plt.show()
