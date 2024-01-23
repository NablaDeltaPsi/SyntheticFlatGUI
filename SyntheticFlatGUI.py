import tkinter as tk
import tkinter.font
import tkinter.messagebox # for pyinstaller
from tkinter.filedialog import askopenfilename
from configparser import ConfigParser
import glob, os, sys, copy, numpy as np, math
import cv2, pickle, bz2
from rawpy import imread
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import threading
from functools import lru_cache
import datetime as dt
from astropy.io import fits

GUINAME = "SyntheticFlatGUI"
VERSION = '1.4'

# STATIC SETTINGS =============================================================
IGNORE_EDGE = 10          # ignore pixels close to the image edges
IGNORE_RADII = 5          # ignore pixels extreme radii (close to 0 and maximum)
RADIAL_RESOLUTION = 100  # should be larger than 16 bit (65536)
DERIVE_RAD = True
DEBUG_MODE = True

RAWTYPES = ['arw', 'crw', 'cr2', 'cr3', 'nef', 'raf', 'rw2']
TIFTYPES = ['tif', 'tiff']
FITSTYPES = ['fit', 'fits']
OTHERTYPES = ['jpeg', 'jpg', 'png']
FILETYPES = RAWTYPES + TIFTYPES + FITSTYPES + OTHERTYPES

# STATIC MAJOR FUNCTIONS =====================================================
def load_image(file, picklepath):
    print("")
    print(file)
    if not os.path.isfile(file):
        raise ValueError("File does not exist!\n\n")
    try:
        print("Search and read pickle files ... ", end='')
        im_deb = pickle.load(bz2.BZ2File(picklepath + os.sep + os.path.basename(file).replace('.', '_') + ".pkl", 'rb'))
        origshape = pickle.load(bz2.BZ2File(picklepath + os.sep + os.path.basename(file).replace('.', '_') + "_origshape.pkl", 'rb'))
        header = pickle.load(bz2.BZ2File(picklepath + os.sep + os.path.basename(file).replace('.', '_') + "_header.pkl", 'rb'))
        print("success.")
        print("shape: ", im_deb.shape)
    except:
        # no pickle file found
        print("no success.")

        # load image depending on type
        imagetype = os.path.basename(file).split(".")[1].lower()
        header = ""
        if imagetype in RAWTYPES:
            print("read raw image (" + imagetype.upper() + ") ...")
            im_load = imread(file).raw_image_visible
        elif imagetype in TIFTYPES:
            print("read tif type image (" + imagetype.upper() + ") ...")
            im_load = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_ANYDEPTH, cv2.IMREAD_UNCHANGED
        elif imagetype in FITSTYPES:
            print("read fits type image (" + imagetype.upper() + ") ...")
            im_load, header = fits.getdata(file, ext=0, header=True)
            print(header)
        elif imagetype in OTHERTYPES:
            print("read other type image (" + imagetype.upper() + ") ...")
            im_load = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_ANYDEPTH, cv2.IMREAD_UNCHANGED
        else:
            raise ValueError("image type not supported!!")

        # remember original shape
        origshape = im_load.shape
        print("original shape: ", origshape)

        # order axes
        im_deb = order_axes(im_load, type='HWD')
        print("ordered shape: ", im_deb.shape)

        # debayer, if necessary
        if len(im_deb.shape) == 3:
            print("no need to debayer ...")
        elif len(im_deb.shape) == 2:
            print("debayer ...")
            im_deb = debayer(im_load)
        else:
            raise ValueError('Bad input shape')
        print("debayered shape: ", im_deb.shape)

        for c in range(im_deb.shape[2]):
            cpx_x = int(im_deb.shape[0] / 2)
            cpx_y = int(im_deb.shape[1] / 2)
            print(im_deb[cpx_x-1:cpx_x+1,cpx_y-1:cpx_y+1,c])

    return im_deb, origshape, header


def corr_gradient(image, resolution_factor=4):
    # measure slopes
    height, width, colors = image.shape
    slopes_x = []
    slopes_y = []
    for c in range(colors):
        rowmeans = []
        for row in range(height):
            if row < IGNORE_EDGE or row > height - IGNORE_EDGE:
                continue
            if row % resolution_factor == 0:
                rowmeans.append(np.mean(sigmaclip(image[row,IGNORE_EDGE:-IGNORE_EDGE,c], low=2, high=2)[0]))
        slope_y = np.mean(np.diff(rowmeans)) * height / resolution_factor
        slopes_y.append(slope_y)

        colmeans = []
        for col in range(width):
            if col < IGNORE_EDGE or col > width - IGNORE_EDGE:
                continue
            if col % resolution_factor == 0:
                colmeans.append(np.mean(sigmaclip(image[IGNORE_EDGE:-IGNORE_EDGE,col,c], low=2, high=2)[0]))
        slope_x = np.mean(np.diff(colmeans)) * width / resolution_factor
        slopes_x.append(slope_x)

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        gradient = X * slope_x + Y * slope_y - (slope_x + slope_y) / 2
        image[:, :, c] = image[:, :, c] - gradient

    # print
    print("gradient slopes x: ", slopes_x)
    print("gradient slopes y: ", slopes_y)

    return image


def calc_histograms(image, circular=False):
    image_rgb = merge_green(image)
    height = image_rgb.shape[0]
    width = image_rgb.shape[1]
    for c in range(3):
        if circular:
            for i in range(image_rgb.shape[0]):
                for j in range(image_rgb.shape[1]):
                    thisdist = dist_from_center(i, j, height, width)
                    if thisdist > image_rgb.shape[0] / 2 or thisdist > image_rgb.shape[1] / 2:
                        image_rgb[i, j, c] = np.nan

        if c == 1:
            vals = np.append(image_rgb[:, :, 1], image_rgb[:, :, 2])
        else:
            vals = image_rgb[:, :, c]
        vals = vals.flatten()

        # caLculate histogram
        counts, bins = np.histogram(vals, np.linspace(0, 2 ** 12, 2 ** 8))
        bins = bins[1:]
        if c == 0:
            data = bins
        data = np.column_stack((data, counts))
    return data


def calc_rad_profile(image, statistics=2, extrapolate_max=True, resolution_factor=4):

    # measure time
    start = dt.datetime.now()

    # acquire pixel values
    image_height = image.shape[0]
    image_width = image.shape[1]
    radii = []
    rad_counts = {}
    for i in range(image_height):
        for j in range(image_width):
            rad = int(dist_from_center(i, j, image_height, image_width))
            if not (image[i, j, 0] > 0 and image[i, j, 1] > 0 and image[i, j, 2] > 0 and image[i, j, 3] > 0):
                continue
            if i < IGNORE_EDGE or j < IGNORE_EDGE or i > image_height-IGNORE_EDGE or j > image_width-IGNORE_EDGE:
                continue
            if not (i % resolution_factor == 0 and j % resolution_factor == 0):
                continue
            if not rad in radii:
                radii.append(rad)
                rad_counts[rad] = [[], [], []]
            rad_counts[rad][0].append(image[i, j, 0])
            rad_counts[rad][1].append(image[i, j, 1])
            rad_counts[rad][1].append(image[i, j, 2])
            rad_counts[rad][2].append(image[i, j, 3])
    maxrad = dist_from_center(0, 0, image_height, image_width)
    rad_profile = np.zeros((len(radii), 4))
    rad_profile_raw_mean = np.zeros((len(radii), 4))
    index = 0
    for rad in sorted(radii):
        rad_profile[index, 0] = rad / maxrad
        rad_profile[index, 1] = apply_statistics(rad_counts[rad][0], statistics)
        rad_profile[index, 2] = apply_statistics(rad_counts[rad][1], statistics)
        rad_profile[index, 3] = apply_statistics(rad_counts[rad][2], statistics)
        rad_profile_raw_mean[index, 0] = rad / maxrad
        rad_profile_raw_mean[index, 1] = np.mean(rad_counts[rad][0])
        rad_profile_raw_mean[index, 2] = np.mean(rad_counts[rad][1])
        rad_profile_raw_mean[index, 3] = np.mean(rad_counts[rad][2])
        index += 1
    rad_profile = rad_profile[~np.isnan(rad_profile).any(axis=1), :]
    print("rad_profile shape: ", rad_profile.shape)

    # safety
    if not rad_profile.shape[0] > IGNORE_RADII * 2:
        print("\nSomething ist wrong!!\nInterrupt!\n\n")

    # cut edges
    if IGNORE_RADII:
        mask = rad_profile[:, 0] > IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile[mask, :]
        mask = rad_profile_cut[:, 0] < 1 - IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile_cut[mask, :]
    else:
        rad_profile_cut = copy.deepcopy(rad_profile)

    # cut data inside maximum
    if extrapolate_max:
        maxind = []
        for c in range(3):
            this_filtered = rad_profile_cut[:, c + 1]
            this_filtered = savgol_filter(this_filtered, window_length=odd_int(rad_profile_cut.shape[0] / 10),
                                                       polyorder=2, mode='interp')
            this_filtered = savgol_filter(this_filtered, window_length=odd_int(rad_profile_cut.shape[0] / 5),
                                                       polyorder=2, mode='interp')
            mi = np.argmax(this_filtered)
            maxind.append(mi)
        print("maximum index: ", maxind)
        if np.max(maxind) > int(rad_profile_cut.shape[0]/2):
            print("maximum index too large -> skip maximum cut")
        else:
            rad_profile_cut = rad_profile_cut[max(maxind):, :]
        # for c in range(3):
        #     rad_profile_cut[:min_maxind,c+1] = max(rad_profile_cut[:,c+1])

    # smooth data and interpolate
    radii = np.linspace(0, 1, RADIAL_RESOLUTION)
    rad_profile_smoothed = radii
    slopes_inner = []
    slopes_outer = []
    for c in range(3):
        y_new = savgol_filter(rad_profile_cut[:, c + 1], window_length=odd_int(rad_profile_cut.shape[0] / 10), polyorder=2, mode='interp')
        y_new = savgol_filter(y_new, window_length=odd_int(rad_profile_cut.shape[0] / 5), polyorder=2, mode='interp')

        # extrapolate inner
        x_new = list(rad_profile_cut[:, 0].flatten())
        y_new = list(y_new.flatten())
        xdist = x_new[1] - x_new[0]
        ylast = y_new[0]
        xlast = x_new[0]
        slope = (y_new[1] - y_new[0]) / xdist
        if slope > 0:
            slope = 0
        slopes_inner.append(int(slope))
        xadd = np.arange(0, xlast - xdist, xdist)
        # quadratic function with given value and slope at xlast and zero derivative at x=0
        yadd = ylast + slope / 2 * (xadd ** 2 / xlast - xlast)
        x_new = list(xadd) + x_new
        y_new = list(yadd) + y_new

        # extrapolate outer
        x_new = x_new[:-4]  # whatever reason, otherwise step in profile
        y_new = y_new[:-4]  # whatever reason, otherwise step in profile
        xdist = x_new[-1] - x_new[-2]
        ylast = y_new[-1]
        xlast = x_new[-1]
        slope = (y_new[-1] - y_new[-2]) / xdist
        slopes_outer.append(int(slope))
        xadd = np.arange(xlast + xdist, 1, xdist)
        # linear function
        yadd = ylast + slope * (xadd - xlast)
        x_new = x_new + list(xadd)
        y_new = y_new + list(yadd)

        # interpolate onto continuous values
        f = interp1d(x_new, y_new, kind='quadratic', fill_value='extrapolate')
        y_new = f(radii)
        rad_profile_smoothed = np.column_stack((rad_profile_smoothed, y_new))

    # print extrapolation info
    print("extrapolated with inner slopes: ", slopes_inner)
    print("extrapolated with outer slopes: ", slopes_outer)

    # normalize by smoothed curve
    for c in range(3):
        rad_profile_raw_mean[:, c + 1] = rad_profile_raw_mean[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile_cut[:, c + 1] = rad_profile_cut[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile_smoothed[:, c + 1] = rad_profile_smoothed[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])

    # print minimum value
    print("minimum:", np.min(rad_profile_smoothed[:, 1:]))

    # print execution time
    stop = dt.datetime.now()
    print("execution time:", int((stop-start).total_seconds() + 0.5), "seconds")

    return rad_profile_raw_mean, rad_profile, rad_profile_cut, rad_profile_smoothed


def calc_synthetic_flat(rad_profile, grey_flat=False, tif_size=(4024, 6024)):

    # measure time
    start = dt.datetime.now()

    # initialize image
    height, width = sorted(tif_size[:2])
    halfheight = int(height / 2)
    halfwidth = int(width / 2)
    im_syn = np.zeros((int(height), int(width)))

    # normalize (again)
    for c in range(3):
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile[:, c + 1])

    # match profile to output size
    maxrad_this = dist_from_center(0, 0, height, width)

    # set kmax for reusing radii
    if DERIVE_RAD:
        kmax = int(height / 2)
    else:
        kmax = 2

    # iterate one quadrant and write pixels in all four
    # height = 10 -> iterate di from 0 to 4
    for di in range(halfheight):
        for dj in range(halfwidth):

            # absolute pixel indices
            i = halfheight + di
            j = halfwidth + dj

            # check already written
            if not im_syn[i, j] == 0:
                continue

            # radial position of pixel
            r = dist_from_center(i, j, height, width) / maxrad_this

            # search match within small range in radial profile
            r_index = int((RADIAL_RESOLUTION - 1) * r)

            # write pixel brightness in all quadrants
            # since the radii of the multiples are already known, use them here (r_index * k)
            di_, dj_ = to_centersystem(i, j, height, width)
            for k in range(1, kmax):
                k_di_ = di_ * k
                k_dj_ = dj_ * k
                k_r_index = r_index * k

                # doesn't work for integer positions in centersystem
                if k_di_.is_integer() or k_dj_.is_integer():
                    continue

                if k_di_ < halfheight and k_dj_ < halfwidth:
                    for q in range(4):
                        k_i, k_j = to_cornersystem(k_di_, k_dj_, height, width, quadrant=q)
                        if not im_syn[k_i, k_j] == 0:
                            print("Already written", di, dj, di_, dj_, k_di_, k_dj_, q, k_i, k_j, sep=' ')
                            sys.exit()
                        else:
                            print("Write pixel    ", di, dj, di_, dj_, k_di_, k_dj_, q, k_i, k_j, sep=' ')
                        write_flat_pixel(im_syn, rad_profile, grey_flat, k_r_index, k_i, k_j)
                else:
                    break

    # convert to 16 bit
    im_syn = im_syn / np.max(im_syn)
    print("zeros (%):", "{:.10f}".format(100.0 - 100.0 * np.count_nonzero(im_syn) / np.prod(im_syn.shape)))
    print("nan (%):", "{:.10f}".format(100.0 * np.count_nonzero(np.isnan(im_syn)) / np.prod(im_syn.shape)))
    print("minimum:  ", np.min(im_syn[im_syn > 0]))
    im_syn = (2 ** 16 - 1) * im_syn
    im_syn = im_syn.astype(np.uint16)

    # print execution time
    stop = dt.datetime.now()
    print("execution time:", int((stop-start).total_seconds() + 0.5), "seconds")

    return im_syn



# STATIC MINOR FUNCTIONS =====================================================

def order_axes(image, type='HWD'):
    image_shape = image.shape
    axes_sort = np.argsort(image_shape)[::-1]
    image = np.transpose(image, axes=axes_sort)
    if type == 'HWD':
        image = np.swapaxes(image, 0, 1)
    if len(image.shape) == 3 and type == 'DHW':
        image = np.swapaxes(image, 0, 2)
    return image

def separate_axes(image):
    # try to find a rgb dimension
    color_axis = None
    for d in range(len(image.shape)):
        if image.shape[d] < 5:
            color_axis = d
            break

    # calc tuple of image_axes
    image_axes = []
    for d in range(len(image.shape)):
        if not d == color_axis:
            image_axes.append(d)
    image_axes = tuple(image_axes)

    return image_axes, color_axis

def write_flat_pixel(im_syn, rad_profile, grey_flat, r_index, i, j):
    i = int(i)
    j = int(j)
    # calculate brightness and set new image pixel
    if grey_flat:
        im_syn[i, j] = rad_profile[r_index, 2]
    else:
        if i % 2 == 0 and j % 2 == 0:  # links oben
            im_syn[i, j] = rad_profile[r_index, 1]  # R
        if i % 2 == 0 and j % 2 == 1:  # rechts oben
            im_syn[i, j] = rad_profile[r_index, 2]  # G
        if i % 2 == 1 and j % 2 == 0:  # links unten
            im_syn[i, j] = rad_profile[r_index, 2]  # G
        if i % 2 == 1 and j % 2 == 1:  # rechts unten
            im_syn[i, j] = rad_profile[r_index, 3]  # B


def debayer(image, separate_green=True):
    rows = image.shape[0]
    cols = image.shape[1]
    db_image = np.zeros((int(rows / 2), int(cols / 2), 4))
    for n in range(rows):
        for m in range(cols):
            if n % 2 == 0 and m % 2 == 0:  # links oben
                db_image[int((n + 0) / 2), int((m + 0) / 2), 0] = image[n, m]  # R
            if n % 2 == 0 and m % 2 == 1:  # rechts oben
                db_image[int((n + 0) / 2), int((m - 1) / 2), 1] = image[n, m]  # G
            if n % 2 == 1 and m % 2 == 0:  # links unten
                db_image[int((n - 1) / 2), int((m + 0) / 2), 2] = image[n, m]  # G
            if n % 2 == 1 and m % 2 == 1:  # rechts unten
                db_image[int((n - 1) / 2), int((m - 1) / 2), 3] = image[n, m]  # B
    if not separate_green:
        db_image = merge_green(db_image)
        print("shape", db_image.shape)
    return db_image


def merge_green(image):
    image_r = image[:, :, 0]
    image_g = (image[:, :, 1] + image[:, :, 2]) / 2
    image_b = image[:, :, 3]
    image = np.stack((image_r, image_g, image_b), axis=2)
    return image


def bayer(image):
    if len(image.shape) < 3:
        return image
    rows = image.shape[0]
    cols = image.shape[1]
    colors = image.shape[2]
    b_image = np.zeros((rows * 2, cols * 2))
    for i in range(rows):
        for j in range(cols):
            for c in range(colors):
                if c == 0:  # R
                    b_image[2 * i + 0, 2 * j + 0] = image[i, j, c]  # links oben
                elif c == 1:  # G
                    b_image[2 * i + 1, 2 * j + 0] = image[i, j, c]  # rechts oben
                    if colors == 3:
                        b_image[2 * i + 0, 2 * j + 1] = image[i, j, c]  # links unten
                elif c == 3:  # G
                    b_image[2 * i + 0, 2 * j + 1] = image[i, j, c]  # links unten
                else:  # B
                    b_image[2 * i + 1, 2 * j + 1] = image[i, j, c]  # rechts unten
    return b_image


def to_centersystem(x, y, height, width):
    # index at half height is actually not center but 0.5 off
    # pixel 4 and 5 both are at radius 0.5 for an array with length 10
    x_ = x - height / 2 + 0.5
    y_ = y - width  / 2 + 0.5
    return x_, y_


def to_cornersystem(x_, y_, height, width, quadrant=0):
    x = height / 2 - 0.5
    y = width  / 2 - 0.5
    if quadrant == 0: # bottom right with positive x_ and y_
        x += x_
        y += y_
    elif quadrant == 1:
        x -= x_
        y += y_
    elif quadrant == 3:
        x -= x_
        y -= y_
    else:
        x += x_
        y -= y_
    return int(x), int(y)

def dist_from_center(i, j, height, width, centersystem=False):
    if centersystem:
        dx = i
        dy = j
    else:
        dx, dy = to_centersystem(i, j, height, width)
    dx = abs(dx)
    dy = abs(dy)
    dx, dy = sorted([dx, dy])  # sort helps caching of symmetric function
    return cached_dist(dx, dy)


@lru_cache(maxsize=None)
def cached_dist(dx, dy):
    return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


def create_folder(folder_path):
    try:
        os.mkdir(folder_path)
    except:
        pass
    return folder_path


def odd_int(number):
    if int(number) % 2 == 1:
        return int(number)
    else:
        return int(number) + 1


def contains(string, substrings):
    if not isinstance(substrings, list):
        substrings = [substrings]
    for sub in substrings:
        if string.lower().find(sub.lower()) >= 0:
            return True
    return False


def rgbtohex(r, g, b):
    return f'#{r:02x}{g:02x}{b:02x}'


def apply_statistics(array, statistics):
    if statistics[:10] == 'sigma clip':
        number = statistics[10:].strip()
        sigma_clip = float(number)
        result = sigma_clip_mean(array, sigma_clip=sigma_clip)
    elif statistics == 'median':
        result = np.median(array)
    elif statistics == 'min':
        result = np.min(array)
    elif statistics == 'max':
        result = np.max(array)
    else:
        result = np.mean(array)
    return result


def sigma_clip_mean(array, sigma_clip=2.0):
    reduced = sigmaclip(array, low=sigma_clip, high=sigma_clip)[0]
    result = np.mean(reduced)
    return result


def write_csv(data, savepath, original_file, suffix):
    fmt = '%.10f', '%.10f', '%.10f', '%.10f'
    np.savetxt(savepath + os.sep + os.path.basename(original_file).split('.')[0] + suffix + ".csv",
               data, delimiter=",", fmt=fmt)


# IMAGE CLASS =====================================================
class Image():
    def __init__(self, file, debug=False):

        # init attributes
        self.file = file
        self.debug = debug
        self.image = None
        self.header = ''
        self.origpath = os.path.dirname(file)
        self.origtype = os.path.basename(self.file).split(".")[1].lower()
        self.origshape = None
        self.origdebayer = None
        self.outpath = ''
        self.outpath_pickle = ''
        self.outpath_csv = ''
        self.outtype = ''
        self.outdebayer = None
        self.radprof = None
        self.image_flat = None

        self.set_outpaths()
        self.set_debug()

    def set_outpaths(self):
        self.outpath = create_folder(self.origpath + os.sep + GUINAME)
        self.outpath_pickle = create_folder(self.outpath + os.sep + "pickle")
        self.outpath_csv = create_folder(self.outpath + os.sep + "csv")

    def set_outtype(self):
        if self.origtype in RAWTYPES:
            self.outtype = 'tif'
        elif self.origtype in TIFTYPES:
            self.outtype = 'tif'
        else:
            self.outtype = 'fit'

    def set_origdebayer(self):
        if len(self.origshape) >= 3:
            self.origdebayer = True
        else:
            self.origdebayer = False

    def set_debug(self):
        if not self.debug:
            return
        self.image = np.ones((200, 300, 4))
        self.origshape = (400, 600)
        self.set_origdebayer()
        self.outtype = 'tif'
        radprof_x = np.linspace(0, 1, RADIAL_RESOLUTION)
        radprof_y = np.linspace(1, 0.7, RADIAL_RESOLUTION)
        self.radprof = np.column_stack((radprof_x, radprof_y, radprof_y, radprof_y))
        self.image_flat = np.ones((200, 300, 4))

    def load(self):
        if self.debug:
            return
        self.image, self.origshape, self.header = load_image(self.file, self.outpath_pickle)
        self.set_outtype()
        self.set_origdebayer()

    def write_pickle(self):
        if self.debug:
            return
        # without 190 MB, 0 min
        # gzip     30 MB, 3 min
        # lzma     20 MB, 3 min
        # bz2      20 MB, 0 min <-- checked: pyinstaller size didnt increase
        pickle_filename = self.outpath_pickle + os.sep + os.path.basename(self.file).replace('.', '_') + ".pkl"
        pickle_filename_origshape = self.outpath_pickle + os.sep + os.path.basename(self.file).replace('.', '_') + "_origshape.pkl"
        pickle_filename_header = self.outpath_pickle + os.sep + os.path.basename(self.file).replace('.', '_') + "_header.pkl"
        if not os.path.isfile(pickle_filename) or not os.path.isfile(pickle_filename_origshape) or not os.path.isfile(pickle_filename_header):
            pickle.dump(self.image, bz2.BZ2File(pickle_filename, 'wb'))
            pickle.dump(self.origshape, bz2.BZ2File(pickle_filename_origshape, 'wb'))
            pickle.dump(self.header, bz2.BZ2File(pickle_filename_header, 'wb'))

    def write_image(self, suffix, flat=False):
        print("write image \"", suffix, "\"")

        # determine case
        if len(self.origshape) >= 3 and not self.origtype in RAWTYPES:
            self.outdebayer = True
        else:
            self.outdebayer = False

        if flat:
            print("input", self.image_flat.shape)
            image_write = self.image_flat
        else:
            print("input", self.image.shape)
            image_write = self.image

        # reformat
        if self.outdebayer:
            # use as is
            image_write = image_write
        else:
            # bayer image, if it is debayered
            image_write = bayer(image_write)
        print("output", image_write.shape)

        # write
        if self.outtype == 'tif':
            image_write = np.float32(image_write / np.max(image_write))
            print(self.outpath + os.sep + os.path.basename(self.file).split('.')[0] + suffix + ".tif")
            cv2.imwrite(self.outpath + os.sep + os.path.basename(self.file).split('.')[0] + suffix + ".tif", image_write)

    def gradcorr(self, resolution_factor):
        if self.debug:
            return
        self.image = corr_gradient(self.image, resolution_factor=resolution_factor)

    def subtract_bias(self, bias_value):
        if self.debug:
            return
        self.image = self.image - bias_value

    def calc_histogram(self, circular=False):
        if self.debug:
            return
        data = calc_histograms(self.image, circular=circular)
        write_csv(data, self.outpath_csv, self.file, "_histogram")

    def calc_rad_profile(self, statistics=2, extrapolate_max=True, resolution_factor=4):
        if self.debug:
            return
        radprof1, radprof2, radprof3, radprof4 = calc_rad_profile(self.image, statistics=statistics, extrapolate_max=extrapolate_max, resolution_factor=resolution_factor)
        write_csv(radprof1, self.outpath_csv, self.file, "_radprof_0_raw_mean")
        write_csv(radprof2, self.outpath_csv, self.file, "_radprof_1_clipped")
        write_csv(radprof3, self.outpath_csv, self.file, "_radprof_2_cut")
        write_csv(radprof4, self.outpath_csv, self.file, "_radprof_3_smooth")
        self.radprof = radprof4

    def calc_synthetic_flat(self, grey_flat=False):
        self.image_flat = calc_synthetic_flat(self.radprof, grey_flat=grey_flat, tif_size=self.origshape)
        self.image = bayer(self.image) / self.image_flat


# TKINTER CLASS =====================================================

class NewGUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUINAME + " v" + VERSION)
        self.lastpath = ""
        self.loaded_files = []
        self.bias_value = 0
        self.running = False
        self.asked_stop = False

        padding = 5

        self.root.protocol("WM_DELETE_WINDOW",  self.on_close)

        # icon and DPI
        try:
            self.root.iconbitmap(GUINAME + ".ico")
            self.root.update() # important: recalculate the window dimensions
        except:
            print("Found no icon.")

        # menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # mainoptions
        options = tk.Menu(menubar, tearoff=0)
        self.opt_gradient  = tk.BooleanVar()
        self.opt_histogram = tk.BooleanVar()
        self.opt_radprof   = tk.BooleanVar()
        self.opt_synthflat = tk.BooleanVar()
        options.add_checkbutton(label="Correct gradient", onvalue=1, offvalue=0, variable=self.opt_gradient)
        options.add_checkbutton(label="Calculate histogram", onvalue=1, offvalue=0, variable=self.opt_histogram)
        options.add_checkbutton(label="Calculate radial profile", onvalue=1, offvalue=0, variable=self.opt_radprof, command=self.toggle_radprof)
        options.add_checkbutton(label="Export synthetic flat", onvalue=1, offvalue=0, variable=self.opt_synthflat, command=self.toggle_synthflat)
        menubar.add_cascade(label="Options", menu=options)

        # settings
        settings = tk.Menu(menubar, tearoff=0)
        self.set_write_pickle    = tk.BooleanVar()
        self.set_export_corr_input = tk.BooleanVar()
        self.set_circular_hist   = tk.BooleanVar()
        self.set_grey_flat       = tk.BooleanVar()
        self.set_extrapolate_max = tk.BooleanVar()
        settings.add_checkbutton(label="Write pickle file", onvalue=1, offvalue=0, variable=self.set_write_pickle)
        settings.add_checkbutton(label="Histogram of largest circle", onvalue=1, offvalue=0, variable=self.set_circular_hist)
        settings.add_checkbutton(label="Extrapolate inside max", onvalue=1, offvalue=0, variable=self.set_extrapolate_max)
        settings.add_checkbutton(label="Export corrected input images", onvalue=1, offvalue=0, variable=self.set_export_corr_input)
        settings.add_checkbutton(label="Grey synthetic flat", onvalue=1, offvalue=0, variable=self.set_grey_flat)
        menubar.add_cascade(label="Settings", menu=settings)

        # statistics
        statistics = tk.Menu(menubar, tearoff=0)
        self.radio_statistics = tk.StringVar(self.root)
        self.sigmas = ["mean", "median", "min", "max", "sigma clip 0.5", "sigma clip 1.0",
                       "sigma clip 2.0", "sigma clip 3.0", "sigma clip 4.0", "sigma clip 8.0"]
        for opt in self.sigmas:
            statistics.add_radiobutton(label=opt, value=opt, variable=self.radio_statistics)
        menubar.add_cascade(label="Statistics", menu=statistics)

        # resolution
        resolution = tk.Menu(menubar, tearoff=0)
        self.radio_resolution = tk.StringVar(self.root)
        self.resolution_factor = ['full', '1/2', '1/4', '1/8', '1/16']
        for opt in self.resolution_factor:
            resolution.add_radiobutton(label=opt, value=opt, variable=self.radio_resolution)
        menubar.add_cascade(label="Resolution", menu=resolution)

        # Reset
        menubar.add_command(label="Reset", command=self.reset_config)

        # buttons
        self.button_load = tk.Button(text="Load files", command=self.load_files)
        self.button_load.grid(row=0, column=0, sticky='NWSE', padx=padding, pady=padding, columnspan=2)
        self.button_load = tk.Button(text="Set bias value", command=self.ask_bias)
        self.button_load.grid(row=1, column=0, sticky='NWSE', padx=padding, pady=padding)
        self.button_load = tk.Button(text="Bias from file", command=self.ask_bias_file)
        self.button_load.grid(row=1, column=1, sticky='NWSE', padx=padding, pady=padding)
        self.button_start = tk.Button(text="Start", command=lambda: threading.Thread(target=self.process).start())
        self.button_start.grid(row=2, column=0, sticky='NWSE', padx=padding, pady=padding)
        self.button_stop = tk.Button(text="Stop", command=self.stop)
        self.button_stop.grid(row=2, column=1, sticky='NWSE', padx=padding, pady=padding)

        # labels
        self.label_files_var = tk.StringVar()
        self.label_files_var.set("0 files")
        self.label_files = tk.Label(textvariable=self.label_files_var, justify='left')
        self.label_files.grid(row=0, column=2, sticky='NWSE', padx=padding, pady=padding)
        self.label_bias_var = tk.StringVar()
        self.label_bias_var.set(0)
        self.label_bias = tk.Label(textvariable=self.label_bias_var, justify='left')
        self.label_bias.grid(row=1, column=2, sticky='NWSE', padx=padding, pady=padding)
        self.label_status_var = tk.StringVar()
        self.label_status_var.set("ready")
        self.label_status = tk.Label(textvariable=self.label_status_var, justify='left')
        self.label_status.grid(row=2, column=2, sticky='NWSE', padx=padding, pady=padding)
        self.update_labels()

        # configure
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=3)
        for i in range(3):
            self.root.grid_rowconfigure(i, weight=1)

        # default configs
        self.load_config_file()

        # mainloop
        self.root.mainloop()

    def stop(self):
        print("asked_stop:", self.asked_stop, end=' ')
        self.asked_stop = True
        print("to", self.asked_stop)
        self.update_labels(status="stopping...")

    def check_stop(self):
        if self.asked_stop:
            self.running = False
            self.update_labels(status="interrupted")
            print("\nInterrupted!\n")
            self.asked_stop = False
            return True
        else:
            return False

    def on_close(self):
        print("... save config file")
        config_object = ConfigParser()

        config_object["BASICS"] = {}
        config_object["BASICS"]["window size"]      = self.root.winfo_geometry()
        config_object["BASICS"]["lastpath"]         = self.lastpath
        config_object["BASICS"]["radio_statistics"] = self.radio_statistics.get()
        config_object["BASICS"]["radio_resolution"] = self.radio_resolution.get()
        config_object["BASICS"]["bias_value"]   = str(self.bias_value)

        config_object["OPTIONS"] = {}
        config_object["OPTIONS"]["opt_gradient"]  = str(self.opt_gradient.get())
        config_object["OPTIONS"]["opt_histogram"] = str(self.opt_histogram.get())
        config_object["OPTIONS"]["opt_radprof"]   = str(self.opt_radprof.get())
        config_object["OPTIONS"]["opt_synthflat"] = str(self.opt_synthflat.get())

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"]     = str(self.set_write_pickle.get())
        config_object["SETTINGS"]["set_export_corr_input"]  = str(self.set_export_corr_input.get())
        config_object["SETTINGS"]["set_circular_hist"]    = str(self.set_circular_hist.get())
        config_object["SETTINGS"]["set_grey_flat"]        = str(self.set_grey_flat.get())
        config_object["SETTINGS"]["set_extrapolate_max"]  = str(self.set_extrapolate_max.get())

        with open(GUINAME + ".conf", 'w') as conf:
            config_object.write(conf)

        self.root.destroy()

    def reset_config(self, reset_window=False):
        config_object = ConfigParser()
        config_object["BASICS"] = {}
        if reset_window:
            config_object["BASICS"]["window size"] = '318x128+313+94'
        else:
            config_object["BASICS"]["window size"] = self.root.winfo_geometry()
        config_object["BASICS"]["lastpath"] = './'
        config_object["BASICS"]["radio_statistics"] = 'sigma clip 2.0'
        config_object["BASICS"]["radio_resolution"] = '1/4'
        config_object["BASICS"]["bias_value"] = '0'

        config_object["OPTIONS"] = {}
        config_object["OPTIONS"]["opt_gradient"] = 'True'
        config_object["OPTIONS"]["opt_histogram"] = 'False'
        config_object["OPTIONS"]["opt_radprof"] = 'True'
        config_object["OPTIONS"]["opt_synthflat"] = 'True'

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"] = 'True'
        config_object["SETTINGS"]["set_export_corr_input"] = 'True'
        config_object["SETTINGS"]["set_circular_hist"] = 'True'
        config_object["SETTINGS"]["set_grey_flat"] = 'True'
        config_object["SETTINGS"]["set_extrapolate_max"] = 'True'

        self.apply_config(config_object)
        self.update_labels(status="ready")

    def apply_config(self, config_object):
        self.root.geometry(config_object["BASICS"]["window size"])
        self.lastpath = config_object["BASICS"]["lastpath"]
        self.radio_statistics.set(config_object["BASICS"]["radio_statistics"])
        self.radio_resolution.set(config_object["BASICS"]["radio_resolution"])
        self.bias_value = int(config_object["BASICS"]["bias_value"])

        self.opt_gradient.set(config_object["OPTIONS"]["opt_gradient"]   == 'True')
        self.opt_histogram.set(config_object["OPTIONS"]["opt_histogram"] == 'True')
        self.opt_radprof.set(config_object["OPTIONS"]["opt_radprof"]     == 'True')
        self.opt_synthflat.set(config_object["OPTIONS"]["opt_synthflat"] == 'True')

        self.set_write_pickle.set(config_object["SETTINGS"]["set_write_pickle"]   == 'True')
        self.set_export_corr_input.set(config_object["SETTINGS"]["set_export_corr_input"]   == 'True')
        self.set_circular_hist.set(config_object["SETTINGS"]["set_circular_hist"] == 'True')
        self.set_grey_flat.set(config_object["SETTINGS"]["set_grey_flat"]         == 'True')
        self.set_extrapolate_max.set(config_object["SETTINGS"]["set_extrapolate_max"] == 'True')

        self.update_labels()

    def toggle_radprof(self):
        if not self.opt_radprof.get():
            self.opt_synthflat.set(0)

    def toggle_synthflat(self):
        if self.opt_synthflat.get():
            self.opt_radprof.set(1)

    def load_config_file(self):
        # read
        if os.path.exists(GUINAME + ".conf"):
            config_object = ConfigParser()
            config_object.read(GUINAME + ".conf")
            self.apply_config(config_object)

        # default
        else:
            self.reset_config(reset_window=True)

    def load_files(self):
        if self.running:
            return
        user_input = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('Images', ";".join(["*." + x for x in FILETYPES])), ('all', '.*')])
        if user_input:
            self.loaded_files = user_input
            self.update_labels(file=str(len(self.loaded_files)) + " files")
            self.update_labels(status="ready")
            self.lastpath = os.path.dirname(self.loaded_files[0])
        return

    def ask_bias(self):
        if self.running:
            return
        user_input = tk.simpledialog.askinteger(title="Bias value", prompt="Which bias value should be subtracted from the image?")
        self.bias_value = int(user_input)
        self.label_bias_var.set(self.bias_value)
        self.update_labels()
        return

    def ask_bias_file(self):
        if self.running:
            return
        self.update_labels(status="calc bias...")
        user_input_file = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('RAW format (supported)', RAW_TYPES), ('Image format (not supported)', IMAGE_TYPES), ('all', '.*')])
        im_raw = imread(user_input_file[0]).raw_image_visible
        self.bias_value = int(sigma_clip_mean(im_raw))
        self.label_bias_var.set(self.bias_value)
        self.update_labels(status="ready")
        return

    def update_labels(self, file="", status=""):
        if file:
            self.label_files_var.set(file)
        if status:
            self.label_status_var.set(status)
            print("\n>> ", status)
        if contains(self.label_status_var.get().lower(), ["ready", "finish"]):
            self.label_status.configure(background=rgbtohex(180, 230, 180))
        elif contains(self.label_status_var.get().lower(), ["error", "interr", "stop", "no file"]):
            self.label_status.configure(background=rgbtohex(250, 180, 180))
        else:
            self.label_status.configure(background=rgbtohex(250, 220, 180))
        if len(self.loaded_files) > 0:
            self.label_files.configure(background=rgbtohex(210, 230, 255))
        else:
            self.label_files.configure(background=rgbtohex(250, 180, 180))
        if self.bias_value > 0:
            self.label_bias.configure(background=rgbtohex(210, 230, 255))
        self.label_bias_var.set(self.bias_value)
        self.root.update()
        return

    def process(self):
        if self.running:
            return
        else:
            self.running = True
            self.update_labels(status="running...")
        self.asked_stop = False
        try:
            counter = 0

            # safety check
            if not self.loaded_files:
                self.running = False
                self.update_labels(status="no file chosen.")
                return

            # resolution factor
            if self.radio_resolution.get()[2:].isnumeric():
                resolution_factor = int(self.radio_resolution.get()[2:])
            else:
                resolution_factor = 1

            for file in self.loaded_files:

                # set and display
                counter += 1
                self.update_labels(file=os.path.basename(file) + " (" + str(counter) + "/" + str(len(self.loaded_files)) + ")")

                # load
                self.update_labels(status="load...")
                if DEBUG_MODE:
                    imobj = Image(file, debug=True)
                else:
                    imobj = Image(file)
                imobj.load()
                if self.check_stop(): return

                # write pickle
                if self.set_write_pickle.get():
                    self.update_labels(status="save pickle file...")
                    imobj.write_pickle()
                    if self.check_stop(): return

                # write original image
                if self.set_export_corr_input.get():
                    self.update_labels(status="write original tif...")
                    imobj.write_image("_0_input")
                    if self.check_stop(): return

                # gradient
                if self.opt_gradient.get():
                    self.update_labels(status="calc gradient...")
                    imobj.gradcorr(resolution_factor)
                    if self.check_stop(): return

                    # write gradient-corrected image
                    if self.set_export_corr_input.get():
                        self.update_labels(status="write gradcorr tif...")
                        imobj.write_image("_1_gradcorr")
                        if self.check_stop(): return

                # subtract bias
                self.update_labels(status="subtract bias...")
                imobj.subtract_bias(self.bias_value)
                if self.check_stop(): return

                # histogram
                if self.opt_histogram.get():
                    self.update_labels(status="calc histogram...")
                    imobj.calc_histogram(circular=self.set_circular_hist.get())
                    if self.check_stop(): return

                # radial profile
                if self.opt_radprof.get():
                    self.update_labels(status="calc radial profile...")
                    imobj.calc_rad_profile(statistics=self.radio_statistics.get(), extrapolate_max=self.set_extrapolate_max.get(), resolution_factor=resolution_factor)
                    if self.check_stop(): return

                # synthetic flat
                if self.opt_synthflat.get():

                    # calculate synthetic flat
                    self.update_labels(status="calc synthetic flat...")
                    imobj.calc_synthetic_flat(grey_flat=self.set_grey_flat.get())
                    if self.check_stop(): return

                    # write synthetic flat tif
                    self.update_labels(status="write synthflat tif...")
                    imobj.write_image("_2_synthflat", flat=True)
                    if self.check_stop(): return

                    # correct input
                    if self.set_export_corr_input.get():
                        self.update_labels(status="export flat-corrected...")
                        imobj.write_image("_3_flatcorr")
                        if self.check_stop(): return

                self.update_labels(status="finished.")
                print("cached_dist:", cached_dist.cache_info())
                cached_dist.cache_clear()
                print("Finished file.")

        except Exception as e:
            print("\nERROR!!")
            print("during status:", self.label_status_var.get())
            print("message:", e)
            self.update_labels(status="unknown error...")
            raise e
            return
        finally:
            self.running = False
            self.update_labels(file=str(len(self.loaded_files)) + " files")
        return


if __name__ == '__main__':
    new = NewGUI()
    # for q in range(4):
    #     print(to_cornersystem(1.0, 1.0, 10, 10, quadrant=q))

