# -*- coding: utf-8 -*-
"""
New python module to be able to control the miniature camera xiMU and to process data.
"""

# Importation of modules
import numpy as np
import matplotlib.pyplot as plt
import pandas
import exifread
from ximea import xiapi
import matplotlib.animation as animation
import rawpy
from skimage.measure import label, regionprops
import tifffile
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
import os
from scipy import stats
import json


# Others modules


# Class used to process image data
class ProcessImage:
    """
     Class to perform some processing function on raw imagery taken by Ximea xiMU camera.
    """

    def readTIFF_xiMU(self, path):
        """
        Function to read tiff files written using tifffile library.

        :param path: Absolute path to file
        :return: Tuple (raw image, metadata)
        """
        tif = tifffile.TiffFile(path)
        data = tif.asarray()
        metadata = tif.pages[0].tags["ImageDescription"].value.strip()
        tif.close()
        return data, self.extract_dictionary_from_tiff(metadata)

    @staticmethod
    def extract_dictionary_from_tiff(stri):
        stri = stri[1:-1:1]
        newstri = stri.split(",")
        dc = {}
        for i in newstri[0:-2:1]:
            key, val = i.split(":")
            if str(key.strip()[1:-1:1]) == "medium":
                dc[str(key.strip()[1:-1:1])] = str(val.strip()[1:-1:1])
            else:
                dc[str(key.strip()[1:-1:1])] = float(val.strip())
        return dc

    @staticmethod
    def readTiff_xiMU_multiple_exposure(path):

        with tifffile.TiffFile(path) as tif:
            data = tif.asarray()
            metadata = tif.pages[0].tags["ImageDescription"].value

        metadata = json.loads(metadata)  # Transformation of the string description to dictionnary!

        return data, metadata

    @staticmethod
    def readDNG(path):
        """

        :param path:
        :return:
        """
        f = open(path, 'rb')
        metadata = exifread.process_file(f)
        f.close()
        with rawpy.imread(path) as raw:
            image = raw.raw_image
        return image.astype(int), metadata  # int --> int64

    @staticmethod
    def readDNG_rgb(path):
        """
        Read DNG files and directly processing it using postprocess.
        :param path:
        :return:
        """
        with rawpy.imread(path) as raw:
            image = raw.postprocess(use_camera_wb=True)
        return image

    def readDNG_insta360(self, path_name, which_image):
        """
        Function to read and separate image of insta360 ONE.

        :param path_name:
        :param which_image:
        :return:
        """

        image, metadata = self.readDNG(path_name)
        height = int(metadata["Image ImageLength"].values[0])
        half_height = int(height/2)

        if which_image == "close":
            im_c = image[half_height:height:1, :]
        elif which_image == "far":
            im_c = image[0:half_height:1, :]
        else:
            raise ValueError("Argument which image must be either close of far.")
        return im_c, metadata

    @staticmethod
    def dwnsampling(image_mosaic, pattern, ave=True):
        """
         Downsampling with average of green pixels.

        :param image_mosaic:
        :param pattern:
        :param ave:
        :return:
        """
        if len(image_mosaic.shape) == 2:
            if pattern == "RGGB":
                r = image_mosaic[0::2, 0::2]
                if ave:
                    g = image_mosaic[0::2, 1::2]/2 + image_mosaic[1::2, 0::2]/2
                else:
                    g = np.zeros((int(image_mosaic.shape[0]), int(image_mosaic.shape[1]/2)))
                    g[0::2, :] = image_mosaic[0::2, 1::2]
                    g[1::2, :] = image_mosaic[1::2, 0::2]
                b = image_mosaic[1::2, 1::2]
            elif pattern == "GRBG":
                r = image_mosaic[0::2, 1::2]
                if ave:
                    g = image_mosaic[0::2, 0::2]/2 + image_mosaic[1::2, 1::2]/2
                else:
                    g = np.zeros((int(image_mosaic.shape[0]), int(image_mosaic.shape[1]/2)))
                    g[0::2, :] = image_mosaic[0::2, 0::2]
                    g[1::2, :] = image_mosaic[1::2, 1::2]
                b = image_mosaic[1::2, 0::2]
            elif pattern == "GBRG":
                r = image_mosaic[1::2, 0::2]
                if ave:
                    g = image_mosaic[0::2, 0::2]/2 + image_mosaic[1::2, 1::2]/2
                else:
                    g = np.zeros((int(image_mosaic.shape[0]), int(image_mosaic.shape[1] / 2)))
                    g[0::2, :] = image_mosaic[0::2, 0::2]
                    g[1::2, :] = image_mosaic[1::2, 1::2]
                b = image_mosaic[0::2, 1::2]
            elif pattern == "BGGR":
                r = image_mosaic[1::2, 1::2]
                if ave:
                    g = image_mosaic[0::2, 1::2]/2 + image_mosaic[1::2, 0::2]/2
                else:
                    g = np.zeros((int(image_mosaic.shape[0]), int(image_mosaic.shape[1] / 2)))
                    g[0::2, :] = image_mosaic[0::2, 1::2]
                    g[1::2, :] = image_mosaic[1::2, 0::2]
                b = image_mosaic[0::2, 0::2]
            else:
                raise Exception("Not a valid Bayer pattern.")
        else:
            raise Exception("RGB mosaic raw data has to be used.")

        if ave:
            return np.dstack([r, g, b])
        else:
            return r, g, b

    @staticmethod
    def avg2row(array):
        """
        Fonction to average data every two row of the green pixels.
        :param array: initial numpy array
        :return: new average array
        """
        return 0.5 * (array[0::2] + array[1::2])

    @staticmethod
    def interpolation(wl, data):
        """
        Simple 1D linear interpolation.

        :param wl: Wavelength for interpolation
        :param data: Tuple of measured wavelength and quantum efficiency  (wavelength, QE)
        :return:
        """
        wl_m, qe_m = data
        return np.interp(wl, wl_m, qe_m)

    @staticmethod
    def regionproperties(image, minimum, *maximum):
        binary = image > minimum
        if maximum:
            binary = (image > minimum) & (image < maximum)

        # Region properties of binary image
        labels = label(binary.astype(int))
        region_properties = regionprops(labels)

        # Sorting
        sorted_regionproperties = sorted(region_properties, key=lambda region: region.area)

        return binary, sorted_regionproperties[::-1]

    @staticmethod
    def specific_regionproperties(regionproperties, cond="maxarea"):
        if isinstance(cond, str):
            if cond == "maxarea":
                return regionproperties[0]
            elif cond == "minarea":
                inv = regionproperties.reverse
                return inv[0]
            else:
                raise ValueError("Argument must be maxarea or minarea.")
        elif isinstance(cond, int):
            return regionproperties[cond]
        else:
            raise TypeError("Argument can be either int or str.")

    @staticmethod
    def data_around_centroid(image, centroid, radius):
        """
        Taking the image, a centroid from a connected region in the threshold image (region properties), the function
        return the data around the centroid according to a given radius.

        Can be done for image with stack RGB data? To be tested

        :param image: Image
        :param centroid: Centroid from region properties with scikit-image
        :param radius: Radius around centroid
        :return:
        """
        # Rounding centroid
        centroid_y, centroid_x = round(centroid[0]), round(centroid[1])
        imshape = image.shape
        # Pixel coordinates
        grid_x, grid_y = np.meshgrid(np.arange(0, imshape[1], 1), np.arange(0, imshape[0], 1))
        # Subtraction of centroid to pixel coordinates
        ngrid_x, ngrid_y = grid_x - centroid_x, grid_y - centroid_y

        # Norm calculation
        norm = np.sqrt(ngrid_x ** 2 + ngrid_y ** 2)
        # Binary image of norm below or equal to radius
        bin = norm <= radius

        return bin, image[bin]


    @staticmethod
    def dark_removal(image, darkframe):
        """
        Function to remove dark noise from raw image shots. It must be performed before demosaic. A constant found in
        metadata or a per-pixel black frame taken before measurements can be used for this noise reduction.

        :param image_meta:
        :param darkframe_meta:
        :return:
        """

        if len(image.shape) == 2:
            image -= darkframe
        else:
            raise Exception("Dark noise correction should be done before demosaic.")
        return image

    @staticmethod
    def white_balance(image_mosaic, pattern, wb_mul):
        if len(image_mosaic.shape) == 2:
            image_mosaic = image_mosaic.astype(float)
            wb_mul /= wb_mul[1]
            if pattern == "RGGB":
                image_mosaic[0::2, 0::2] *= wb_mul[0]
                image_mosaic[1::2, 1::2] *= wb_mul[2]
            elif pattern == "GRBG":
                image_mosaic[0::2, 1::2] *= wb_mul[0]  # Red
                image_mosaic[1::2, 0::2] *= wb_mul[2]  # Blue
            elif pattern == "GBRG":
                image_mosaic[1::2, 0::2] *= wb_mul[0]  # Red
                image_mosaic[0::2, 1::2] *= wb_mul[2]  # Blue
            elif pattern == "BGGR":
                image_mosaic[1::2, 1::2] *= wb_mul[0]  # Red
                image_mosaic[0::2, 0::2] *= wb_mul[2]  # Blue
            else:
                raise Exception("Not a valid Bayer pattern.")
        else:
            raise Exception("White balance correction should be done before demosaic.")

        return image_mosaic

    @staticmethod
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def cam2srbg_multiplication(image, matrix):
        if len(image.shape) == 3:
            r = matrix[0, 0] * image[:, :, 0] + matrix[0, 1] * image[:, :, 1] + matrix[0, 2] * image[:, :, 2]
            g = matrix[1, 0] * image[:, :, 0] + matrix[1, 1] * image[:, :, 1] + matrix[1, 2] * image[:, :, 2]
            b = matrix[2, 0] * image[:, :, 0] + matrix[2, 1] * image[:, :, 1] + matrix[2, 2] * image[:, :, 2]
        else:
            raise ValueError("Process only done for RGB image.")
        return np.dstack([r, g, b])

    def raw2rgb(self, image_mosaic, metadata_dict, demosaic="dw"):
        """
        Based on matlab ...

        :param image_mosaic:
        :param metadata_dict:
        :return:
        """

        # 1. Dark substraction and clipping
        image_mosaic -= metadata_dict["black_level"]
        image_mosaic = image_mosaic/(metadata_dict["saturation_level"] - metadata_dict["black_level"])
        print(np.max(image_mosaic))
        image_mosaic = np.clip(image_mosaic, 0, 1)
        print(np.max(image_mosaic))

        # 2. White balance
        wb_mul = metadata_dict["AsShotNeutral"]**-1
        image_mosaic_wb = self.white_balance(image_mosaic, metadata_dict["CFA"], wb_mul)
        image_mosaic_wb = np.clip(image_mosaic_wb, 0, 1)

        # 3. Demosaic
        if demosaic == "dw":
            img = self.dwnsampling(image_mosaic_wb, metadata_dict["CFA"], ave=True)
        else:
            img = self.demosaic_algorithm(image_mosaic_wb, metadata_dict["CFA"], algorithm=demosaic)
        img = np.clip(img, 0, 1)

        # 4. Colour space conversion
        xyz2cam = metadata_dict["XYZtocam"]  # Found in DNG metadata
        srgb2xyz = np.array([[0.4163290, 0.3931464, 0.1547446],
                             [0.2216999, 0.7032549, 0.0750452],
                             [0.0136576, 0.0913604, 0.7201920]])

        cam2xyz = np.linalg.inv(xyz2cam)   # Camera space color to XYZ
        xyz2srgb = np.linalg.inv(srgb2xyz)  # XYZ to sRGB

        #srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])  # D65
        #srgb2xyz = np.array([[0.4163290, 0.3931464, 0.1547446], [0.2216999, 0.7032549, 0.0750452], [0.0136576, 0.0913604, 0.7201920]])  # D50
        #cam2srgb = np.array([[1.3982832e+00, - 3.9828327e-01, - 4.7435487e-09], [-1.7877797e-07, 1.0000001e+00, 1.8976083e-08], [-9.9375832e-09, - 4.2938303e-02, 1.0429384e+00]])

        cam2srgb = cam2xyz * xyz2srgb  # Camera space color to sRGB
        norm = np.sum(cam2srgb, axis=1)
        cam2srgb = cam2srgb / norm[:, None]

        im_c = self.cam2srbg_multiplication(img, cam2srgb)

        #4. Brightness
        grayim = self.rgb2gray(im_c)
        grayscale = 0.28/grayim.mean()
        im_c *= grayscale
        im_c = np.clip(im_c, 0, 1)

        # Gamma
        nl_srgb = (1.055*im_c**(1/2.4)) - 0.055

        # Image finale
        imfinal = np.clip(nl_srgb, 0, 1) * 255

        return imfinal.astype(np.uint8)

    def raw2rgb_ximu(self, image_mosaic, metadata, demosaic="dw"):
        """
        Transforming raw image in RGB standard for the prototype (xiMU CMOS sensor).

        :param image_mosaic:
        :param metadata:
        :param demosaic:
        :return:
        """

        # Extraction of useful metadata from xiMU metadata structure
        black_level = metadata["black_level"]
        #saturation_level = metadata.data_saturation  # PAS CA !
        saturation_level = 4096
        AsShotNeutral = np.array([0.83, 1, 0.65])  # D50T from Tan et al. 2013
        xyz2cam = np.ones((3, 3))
        cfapattern = "BGGR"

        dict_metadata = {"black_level": black_level,
                         "saturation_level": saturation_level,
                         "AsShotNeutral": AsShotNeutral,
                         "XYZtocam": xyz2cam.reshape((3, 3)),
                         "CFA": cfapattern}

        return self.raw2rgb(image_mosaic, dict_metadata, demosaic)

    def raw2rgb_insta360(self, image_mosaic, metadata, demosaic="dw"):
        """
        Transforming raw image in RGB standard for the consumer-graded camera (Insta360 ONE).

        :param image_mosaic:
        :param metadata:
        :param demosaic:
        :return:
        """

        # Extraction of useful metadata
        black_level = int(str(metadata["Image Tag 0xC61A"].values[0]))
        saturation_level = metadata["Image Tag 0xC61D"].values[0]
        AsShotNeutral = np.array(self.ratio2float(metadata["Image Tag 0xC628"].values))
        xyz2cam = np.array(self.ratio2float(metadata["Image Tag 0xC621"].values))
        cfapattern = metadata["Image CFAPattern"].values

        dict_metadata = {"black_level": black_level,
                         "saturation_level": saturation_level,
                         "AsShotNeutral": AsShotNeutral,
                         "XYZtocam": xyz2cam.reshape((3, 3)),
                         "CFA": self.CFApattern_insta360(cfapattern)}

        return self.raw2rgb(image_mosaic, dict_metadata, demosaic)

    @staticmethod
    def ratio2float(array):
        if isinstance(array[0], exifread.utils.Ratio):
            spl = [str(j).split(sep="/") for j in array]
            return [float(i[0])/float(i[1]) if len(i) == 2 else float(i[0]) for i in spl]
        else:
            raise TypeError("Wrong data type inside array.")

    @staticmethod
    def show_metadata_insta360(met):
        """
        Function to print metadata of insta360 ONE camera.

        :param met: metadata in dict format
        :return: nothing
        """
        for i, j in zip(met.keys(), met.values()):
            print("{0} : {1}".format(i, j))

    @staticmethod
    def CFApattern_insta360(array):
        """
        Returning CFA pattern array of camera Insta360 ONE from metadata.

        :param array: Array corresponding to exif tage of CFA pattern.
        :return: Matrix of bayer CFA pattern.
        """
        cfastr = ""
        exifvalues = {0: "R", 1: "G", 2: "B"}
        for i in array:
            cfastr += exifvalues[i]
        return cfastr

    def fit_imagecenter(self, center, centroids, angles, popt):
        """
        Function to fit image center that minimized residuals on theoretical angle and the fitted angles.

        :param center: Center coordinates (x, y) which is image column and row (must be numpy array).
        :param centroids: Coordinates of centroids (numpy array of tuple with x and y coordinates).
        :param angles: Theoretical angles of the centroids
        :param popt: Constants of the polynomial fit.

        :return: Average of the residuals
        """

        # Radial coordinates
        xmean, ymean = center[0], center[1]
        dx, dy = centroids["x"] - xmean, centroids["y"] - ymean
        radial = np.sqrt(dx**2 + dy**2)

        # Fitted angles
        fit_angles = self.polynomial_fit(radial, *popt)

        # Normaly a1 and a2 should have the same size
        a1 = abs(angles.reshape(-1))
        a2 = abs(fit_angles.reshape(-1))

        linfit = stats.linregress(a1[np.isfinite(a1)], a2[np.isfinite(a2)])
        residuals = abs(a2 - (a1 * linfit[0] + linfit[1]))

        return np.nanmean(residuals)

    def angularcoordinates(self, imagesize, image_center, fit_params):
        """
        Function that return the zenith and azimuth coordinates of each pixels.

        :param imagesize: WARNING!!! (row, column) so (y, x) as the convention for image in python
        :param image_center: Image center coordinate numpy array [x, y]
        :param fit_params: Polynomial coefficients
        :return: zenith and azimuth coordinates in tuple format (zenith, azimuth)
        """
        xcenter, ycenter = image_center[0], image_center[1]
        ximsize, yimsize = imagesize[1], imagesize[0]

        xcoord, ycoord = np.meshgrid(np.arange(0, int(ximsize)), np.arange(0, int(yimsize)))

        new_xcoord, new_ycoord = xcoord - xcenter, ycoord - ycenter
        radi = np.sqrt(new_xcoord ** 2 + new_ycoord ** 2)

        zenith = self.polynomial_fit(radi, *fit_params)

        azimuth = np.degrees(np.arctan2(new_ycoord, new_xcoord))
        azimuth[azimuth < 0] = azimuth[azimuth < 0] + 360

        return zenith, azimuth

    def angularcoordinates_forcedzero(self, imagesize, image_center, fit_params):
        """
        Function that return the zenith and azimuth coordinates of each pixels using polynomial fit FORCED to zero.

        :param imagesize: WARNING!!! (row, column) so (y, x) as the convention for image in python
        :param image_center: Image center coordinate numpy array [x, y]
        :param fit_params: Polynomial coefficients
        :return: zenith and azimuth coordinates in tuple format (zenith, azimuth)
        """
        xcenter, ycenter = image_center[0], image_center[1]
        ximsize, yimsize = imagesize[1], imagesize[0]

        xcoord, ycoord = np.meshgrid(np.arange(0, int(ximsize)), np.arange(0, int(yimsize)))

        new_xcoord, new_ycoord = xcoord - xcenter, ycoord - ycenter
        radi = np.sqrt(new_xcoord ** 2 + new_ycoord ** 2)

        zenith = self.polynomial_fit_forcedzero(radi, *fit_params)

        azimuth = np.degrees(np.arctan2(new_ycoord, new_xcoord))
        azimuth[azimuth < 0] = azimuth[azimuth < 0] + 360

        return zenith, azimuth

    @staticmethod
    def gain_linear(gain_db):
        """
        Function to return the gain in linear value.

        :param gain_db: gain [db]
        :return: linear gain
        """
        return 10**(gain_db/20)

    @staticmethod
    def exposure_second(exposure_us):
        """
        Function transforming exposure time in us to exposure time in s.

        :param exposure_us: exposure time [us]
        :return: exposure time [s]
        """
        return exposure_us*1E-6

    # Fit functions
    @staticmethod
    def polynomial_fit(x, a0, a1, a2, a3, a4):
        """
        Polynomial fit of degree 4. This is mostly used for geometric calibration.

        :param x:
        :param a0:
        :param a1:
        :param a2:
        :param a3:
        :param a4:
        :return:
        """
        return a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4

    @staticmethod
    def polynomial_fit_forcedzero(x, a1, a2, a3, a4):
        """
        Polynomial fit with a0 forced to zero. This is mostly used for geometric calibration.

        :param x:
        :param a1:
        :param a2:
        :param a3:
        :param a4:
        :return:
        """
        return a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4

    @staticmethod
    def rolloff_polynomial(x, a0, a2, a4, a6, a8):
        """
        Polynomial fit with even coefficients for roll-off fittting.

        :param x:
        :param a0:
        :param a2:
        :param a4:
        :param a6:
        :param a8:
        :return:
        """
        return a0 + a2*x**2 + a4*x**4 + a6*x**6 + a8*x**8

    def geometric_curvefit(self, radial_distance, angles):
        """

        :param radial_distance:
        :param angles:
        :return:
        """

        return curve_fit(self.polynomial_fit, radial_distance, angles)

    def geometric_curvefit_forcedzero(self, radial, angles):
        """

        :param radial:
        :param angles:
        :return:
        """
        return curve_fit(self.polynomial_fit_forcedzero, radial, angles)

    def rolloff_curvefit(self, angles, rolloff):
        """
        Curve fit of roll-off.

        :param angles:
        :param rolloff:
        :return:
        """

        return curve_fit(self.rolloff_polynomial, angles, rolloff)

    @staticmethod
    def rsquare(func, popt, covmat, x, y):

        # Std of coefficient parameters
        perr = np.sqrt(np.diag(covmat))

        # Rsquare
        residuals = y - func(x, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        rsquared = 1 - (ss_res / ss_tot)

        return rsquared, perr

    @staticmethod
    def saveTIFF_xiMU(path, rawimage, metadata):
        """
        Saving tiff image to folder.

        :param path:
        :param rawimage:
        :param metadata:
        :return:
        """
        if len(rawimage.shape) == 2:
            tifffile.imwrite(path, rawimage.astype(int), metadata=metadata)
        else:
            raise ValueError("Only raw bayer mosaic image can be saved.")

    def saveTIFF_xiMU_multiple_exposure(self, path, list_raw_images, list_metadata):
        """
        Save multiple image taken with different exposures. The input must be a list of the raw images with a list of the
        corresponding metadata.

        :return:
        """

        # Building dictionnary of dictionnary
        met_tot = {}
        for n, metadata in enumerate(list_metadata):
            imname = "Image{0:03}".format(n+1)
            met_tot[imname] = metadata

        images_stack = np.stack(list_raw_images)
        tifffile.imwrite(path, images_stack.astype(int), metadata=met_tot)


# First class TakeImage
class TakeImage(ProcessImage):
    """
    Module to control camera and grad frame.
    """

    def __init__(self, imageformat="raw"):

        # Opening quantum efficiency data of xiMU MT9P031
        fn = os.path.join(os.path.dirname(__file__), "MT9P031_RSR/MT9P031.csv")
        self.df_sr = pandas.read_csv(fn, sep=";")

        # Dictionaries for image format and on-chip binning
        # On-chip binning
        # 1x1 binning = 4.6  fps = 0.217  s (217 ms)  between each image
        # 2x2 binning = 14.3 fps = 0.0699 s (69.9 ms) between each image
        # 4x4 binning = 31.6 fps = 0.0316 s (31.6 ms) between each image
        self.frmt_dict = {"raw": "XI_RAW16", "mono": "XI_MONO16", "color24bits": "XI_RGB24", "color32bits": "XI_RGB32"}
        self.dwsam_dict = {"1x1": "XI_DWN_1x1", "2x2": "XI_DWN_2x2", "4x4": "XI_DWN_4x4"}

        # Create ximea Camera instance
        self.cam = xiapi.Camera()
        # Create ximea Image instance to store image data and metadata
        self.img = xiapi.Image()

        # Opening camera
        print("Opening device")
        self.cam.open_device_by("XI_OPEN_BY_SN", "16990159")

        # Principal parameters that can be controlled
        self.imformat = self.frmt_dict[imageformat]
        self.dwsam = "XI_DWN_1x1"
        self.exp = 10000
        self.gain = 0

        self.update_cam_param()  # Updating camera parameters (only time image format is set)

        # Figure visualization
        #self.fig = plt.figure(figsize=(11, 7))
        self.fig = plt.figure()
        if self.imformat == "XI_RAW16":
            self.ax = []
            self.text = []
            for i in range(6):
                a = self.fig.add_subplot(3, 2, i+1)
                self.ax.append(a)
            for i in range(1, 7, 2):
                a = self.ax[i]
                self.text.append(a.text(1.2, 0.5, "", transform=a.transAxes))
        else:
            self.ax = self.fig.add_subplot(1, 1, 1)

    def __str__(self):
        """
        Printing updated CMOS parameters.
        :return:
        """
        info1 = "Image format: {0:s}".format(self.cam.get_imgdataformat())
        info2 = "Binning mode: {0:s}".format(self.cam.get_downsampling())
        info3 = "Exposure set to {0:f} us (min: {1:f}, max: {2:f})".format(self.cam.get_exposure(),
                                                                         self.cam.get_exposure_minimum(),
                                                                         self.cam.get_exposure_maximum())
        info4 = "Gain set to {0:f} dB or {1:.2f}X (min: {2:f}, max: {3:f})".format(self.cam.get_gain(),
                                                                                 self.gain_db_to_linear(),
                                                                                 self.cam.get_gain_minimum(),
                                                                                 self.cam.get_gain_maximum())
        prin = "CAMERA PARAMETERS SET\n"
        for i in [info1, info2, info3, info4]:
            prin += i + "\n"
        return prin

    def metadata_xiMU(self, structure):
        """"
        """
        if isinstance(structure, xiapi.Image):
            met_dict = {}
            for i in structure._fields_:
                if isinstance(getattr(structure, i[0]), int) or isinstance(getattr(structure, i[0]), float):
                    met_dict[i[0]] = getattr(structure, i[0])
            if len(met_dict) == 30:
                return met_dict
            else:
                raise ValueError("Expect 32 values in the dictionary, got {0}.".format(len(met_dict)))
        else:
            raise ValueError("Not the type of metadata expected. Should be a xiapi.Ximage instance.")


    def check_temperature(self):
        """
        Checking board temperature. Only one temperature sensor even though there is multiple parameters in the xiapi
        documentation.

        :return: Temperature of the board.
        """
        temp = self.cam.get_sensor_board_temp()
        print("Board temperature {0:.4f} ˚C".format(temp))
        return temp

    def gain_db_to_linear(self):
        return 10**(self.gain/20)

    def update_cam_param(self):

        # Image format
        self.cam.set_imgdataformat(self.imformat)

        # Binning by averaging same color pixels of the imaging matrix (instead of skipping)
        self.cam.set_downsampling_type("XI_BINNING")
        self.cam.set_downsampling(self.dwsam)

        self.cam.set_exposure(self.exp)
        self.cam.set_gain(self.gain)

    def acquisition(self, binning="1x1", exposuretime=10000, gain=0, video=False):
        """
        Acquisition of one image from transport buffer. Different camera parameters can be specified and adjusted before
        frame acquisition.

        :param imageformat: Format of image (raw, mono, color24bits, color32bits with alpha)
        :param binning: On-board averaging of neighbors pixels of same spectral band (1x1, 2x2, 4x4)
        :param exposuretime: Integration time in microseconds
        :param gain: Gain in dB
        :param video: Live stream image

        :return: Tuple with image and metadata.
        """

        # Update camera parameters
        self.dwsam = self.dwsam_dict[binning]
        self.exp = exposuretime
        self.gain = gain

        # Update camera
        self.update_cam_param()

        # Printing new camera parameters
        print(self.__str__())

        if self.check_temperature() < 65:

            # Starting data acquisition
            print("Starting data acquisition...")
            self.cam.start_acquisition()

            if video:
                anim = animation.FuncAnimation(self.fig, self._anim, interval=250)
                plt.show()

            self.cam.get_image(self.img)
            data_raw = self.img.get_image_data_numpy()
            data_raw = data_raw[::-1, :]  # versing imageIn

            # Metadata in dictionary
            met_dict = self.metadata_xiMU(self.img)

            # Stopping acquisition
            print("Stopping acquisition...")
            self.cam.stop_acquisition()

        else:
            self.end()
            print("Board temperature exceeds 65˚C")
            data_raw, met_dict = False, False

        return data_raw, met_dict

    def _anim(self, i):

        if self.check_temperature() < 65:
            self.cam.get_image(self.img)
            data_raw = self.img.get_image_data_numpy()
            data_raw = data_raw[::-1, :]

            if self.imformat == "XI_RAW16":
                rgb = self.dwnsampling(data_raw, "BGGR", ave=True)
                self.imshowrgb(rgb)
            else:
                self.imshow_monocolor(data_raw)

    def imshowrgb(self, rgbdata):
        """

        :param rgb: 3 dimensionnal numpy array
        :return:
        """
        titles = ["Red channel", "Green channel", "Blue channel"]
        for n, p in enumerate(zip(range(1, 7, 2), range(0, 6, 2))):

            p1, p2 = p[0], p[1]

            # Image
            i = rgbdata[..., n]

            imbin, regsprops = self.regionproperties(i, self.img.black_level+50)
            reprops = self.specific_regionproperties(regsprops, cond="maxarea")

            minr, minc, maxr, maxc = reprops.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=0.5)
            histogram, bin = np.histogram(i[minr:maxr, minc:maxc], 4096, [0, 4096])

            # Histogram plot
            self.ax[p2].clear()
            # self.ax[p2].hist(i.ravel(), bins=75)
            self.ax[p2].plot(histogram)

            self.ax[p2].set_title(titles[n])
            self.ax[p2].set_xlabel("pixel value")
            self.ax[p2].set_ylabel("pixel")

            # Image
            self.ax[p1].clear()
            pixel_info = "$\mu$ = {0:.3f}\nMax: {1:.3f}\nMin: {2:.3f}".format(i.mean(), i.max(), i.min())
            self.ax[p1].add_patch(rect)
            self.ax[p1].imshow(i, cmap="nipy_spectral")
            self.ax[p1].set_title(titles[n])
            self.text[n].set_text(pixel_info)

        self.fig.tight_layout()

    def imshow_monocolor(self, data):
        """

        :param data:
        :return:
        """
        self.ax.imshow(data)

    def show_rsr(self):
        wl = np.arange(400, 751, 1)

        fig_spectral = plt.figure()
        ax_spectral = fig_spectral.add_subplot(1, 1, 1)

        color = {"Red": 'r', "Green": "g", "Blue": "b"}

        for i in range(0, len(self.df_sr.columns), 2):
            inter = self.interpolation(wl, (self.df_sr[self.df_sr.columns[i]], self.df_sr[self.df_sr.columns[i+1]]))
            rsr = inter/inter.max()

            # Spectral average and extent
            spectral_extent = np.trapz(rsr, wl)
            avg_wl = np.trapz(rsr*wl, wl)/spectral_extent
            semi_spectral_extent = spectral_extent/2

            # Peak wavelength
            ind_peak = np.argmax(rsr)
            wl_peak = wl[ind_peak]

            # Text
            txt = "({0:.0f} $\pm$ {1:.0f}) nm".format(avg_wl, semi_spectral_extent)

            if wl_peak < 500:
                cl, y_peak = color["Blue"], 1
            elif (wl_peak < 580) and (wl_peak >= 500):
                cl, y_peak = color["Green"], 0.9
            else:
                cl, y_peak = color["Red"], 0.8

            # Plotting data
            ax_spectral.plot(wl, rsr, color=cl)
            ax_spectral.errorbar(avg_wl, y_peak, yerr=None, xerr=semi_spectral_extent,  marker="o", color=cl, label=txt)

        ax_spectral.set_xlabel("Wavelength [nm]")
        ax_spectral.set_ylabel("Relative spectral response")
        ax_spectral.legend(loc="best")

    def end(self):
        # Closing device
        print("Closing device")
        self.cam.close_device()


if __name__ == "__main__":

    import glob

    files = glob.glob("/home/pi/Desktop/test18mars/ParcBic_20200319/profile_001" + "/IMG*.tif")

    a = ProcessImage()

    for f in files[5:6]:
        data, met = a.readTiff_xiMU_multiple_exposure(f)

        for i, m in enumerate(met.keys()):
            if m != "shape":
                plt.figure()
                plt.imshow(data[i, :, :])
                plt.title(m + str(met[m]["exposure_time_us"]))
                plt.show()

    #Script when the files cameracontrol.py is executed. Can be used to test modules function.
    # exp = [200, 2000, 200000]
    # raw_ima = []
    # met = []
    # ac = TakeImage(imageformat="raw")
    # for i in exp:
    #     acim = ac.acquisition(exposuretime=i, gain=0, binning="4x4", video=False)
    #     raw_ima.append(acim[0])
    #     met.append(acim[1])

    # plt.figure()
    # plt.imshow(acim[0])
    # plt.show()

    #ac.saveTIFF_xiMU("test.tif", acim[0], acim[1])

    #im_open, meta_open = ac.readTIFF_xiMU("test.tif")
    #print(len(meta_open))

    # ac.show_rsr()
    #

    # # Saving data
    # dict_meta = {}
    # for i in acim[1]._fields_:
    #     if isinstance(getattr(acim[1], i[0]), int) or isinstance(getattr(acim[1], i[0]), float):
    #         print("{0:s}: {1}".format(i[0], getattr(acim[1], i[0])))
    #         dict_meta[i[0]] = getattr(acim[1], i[0])
    # print(dict_meta["size"])
    #
    # ac.saveTIFF_xiMU("test.tif", acim[0], dict_meta)

    # Opening file

    # ac = ProcessImage()
    #
    # im_open, meta_open = ac.readTIFF_xiMU("test.tif")

    # print(meta_open.bs)

    # print(meta_open)

    #print(acim[0] == im_open)

    # plt.figure()
    # plt.imshow(im_open)

    # ***** TEST ProcessImage with insta360 DNG *****

    # data = np.arange(256).reshape((16, 16)).astype(int)
    # print(data)
    #
    #
    # ob = ProcessImage()
    # image_insta, metadata_insta = ob.readDNG_insta360("IMG_20180831_180648_086.dng", "far")
    # im_dw = ob.dwnsampling(image_insta, pattern="RGGB", ave=True)
    #
    # im_c = ob.raw2rgb_insta360(image_insta, metadata_insta, demosaic="dw")
    #
    # plt.figure()
    # plt.imshow(im_c)
    #
    # im_comp = ob.readDNG_rgb("IMG_20180831_180648_086.dng")
    #
    # plt.figure()
    # plt.imshow(im_comp)


    # ***** TEST ProcessImage with insta360 DNG *****

    # Using libraw
    # with rawpy.imread("IMG_20180831_180648_086.dng") as raw:
    #     cm1 = raw.color_matrix
    #     print(cm1)
    #     print(cm1[:, 0:3])
    #     print(raw.rgb_xyz_matrix)
    #     a = np.linalg.inv(cm1[:, 0:3])
    #     print(a)

    # ob = ProcessImage()
    #
    #
    # image_insta, metadata_insta = ob.readDNG("IMG_20180831_180648_086.dng")
    #
    #
    # image = ob.insta360_raw2rgb(image_insta, metadata_insta, demosaic="dw")
    # plt.figure()
    # plt.imshow(image)
    #
    # plt.figure()
    # plt.imshow(ob.rgb2gray(image))
    #
    # image2 = ob.readDNG_rgb("IMG_20180831_180648_086.dng")
    # plt.figure()
    # plt.imshow(image2)

    # image_dark, metadata_dark = ob.readDNG("IMG_20191018_143522_001_dark.dng")

    # allo = ob.insta360_raw2rgb(image_insta, metadata_insta)
    # allo2 = ob.dwnsampling(ob.insta360DNG((image_insta, metadata_insta), "far"), "RGGB")
    # plt.figure()
    # plt.imshow(allo)


    #allo2 = np.array(image_insta==allo, dtype=int)
    #print(plt.imshow(allo2))
    # print(metadata_insta)


    # for attr, value in vars(b).items():
    #     print(attr, value)

    # # Figure three bands
    # sep = ProcessImage().band_separation(acim, pattern="BGGR", allsamesize=True)
    #
    # bandsfig, bandsax = plt.subplots(nrows=1, ncols=3)
    #
    # bandsax[0].imshow(sep[0])
    # bandsax[0].set_title("$\mu$ = {0:.3f}".format(sep[0].mean()))
    #
    # bandsax[1].imshow(sep[1])
    # bandsax[1].set_title("$\mu$ = {0:.3f}".format(sep[1].mean()))
    #
    # bandsax[2].imshow(sep[2])
    # bandsax[2].set_title("$\mu$ = {0:.3f}".format(sep[2].mean()))

    #plt.show()
