import tkinter as tk
import tkinter.font
import tkinter.messagebox # for pyinstaller
from tkinter.filedialog import askopenfilename
import glob, os, sys, copy, numpy as np#, datetime as dt
from configparser import ConfigParser
import rawpy, cv2
import pickle, bz2
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt

GUINAME = "SyntheticFlatGUI"
VERSION = '1.0'

# STATIC SETTINGS =============================================================
IGNORE_EDGE = 10          # ignore pixels close to the image edges
IGNORE_RADII = 5          # ignore pixels extreme radii (close to 0 and maximum)
RADIAL_RESOLUTION = 100000  # should be larger than 16 bit (65536)
ADD_NOISE_LEVEL = 0

RAW_TYPES = '.arw .crw .cr2 .cr3 .nef .raf .rw2'
IMAGE_TYPES = '.tif .tiff .jpeg .jpg .png'

# STATIC MAJOR FUNCTIONS =====================================================
def load_image(file):
    print("")
    print(file)
    if not os.path.isfile(file):
        print("File does not exist!\n\n")
        sys.exit()
    try:
        im_deb = pickle.load(bz2.BZ2File(os.path.dirname(file) + os.sep + "Pickle_files" + os.sep +
                                         os.path.basename(file).split('.')[0] + ".pkl", 'rb'))
        rawshape = pickle.load(bz2.BZ2File(os.path.dirname(file) + os.sep + "Pickle_files" + os.sep +
                                         os.path.basename(file).split('.')[0] + "_rawshape.pkl", 'rb'))
        print("Use image from pickle file.")
        print("shape: ", im_deb.shape)
    except:
        print("read file ...")
        im_raw = rawpy.imread(file).raw_image_visible
        rawshape = im_raw.shape
        print("shape: ", im_raw.shape)
        print("debayer ...")
        im_deb = debayer(im_raw)
        print("shape: ", im_deb.shape)
    return im_deb, rawshape

def write_pickle(im_deb, rawshape, file):
    # without 190 MB, 0 min
    # gzip     30 MB, 3 min
    # lzma     20 MB, 3 min
    # bz2      20 MB, 0 min <-- checked: pyinstaller size didnt increase
    savepath = create_folder(file, "Pickle_files")
    pickle_filename = savepath + os.sep + os.path.basename(file).split('.')[0] + ".pkl"
    pickle_filename_rawshape = savepath + os.sep + os.path.basename(file).split('.')[0] + "_rawshape.pkl"
    if not os.path.isfile(pickle_filename) or not os.path.isfile(pickle_filename_rawshape):
        print("write pickle file ...")
        pickle.dump(im_deb, bz2.BZ2File(pickle_filename, 'wb'))
        pickle.dump(rawshape, bz2.BZ2File(pickle_filename_rawshape, 'wb'))
        print("maxrad original: ", int(dist_from_center(0, 0, im_deb)))

def edit_image(im_raw, bias_value=0, shrink_factor='4'):
    if bias_value > 0:
        print("subtract bias (" + str(bias_value) + ") ...")
        im_raw = im_raw - bias_value

    if shrink_factor.isnumeric():
        print("resize ...")
        im_raw = resize(im_raw, int(shrink_factor))
        print("shape: ", im_raw.shape)

    if IGNORE_EDGE:
        print("ignore edge ...")
        edge_size = IGNORE_EDGE
        im_raw = im_raw[edge_size:-edge_size, edge_size:-edge_size, :]
        print("shape: ", im_raw.shape)

    return im_raw


def corr_gradient(image, file):
    # save original image
    image_write = bayer(image[:, :, 1:])
    savepath = create_folder(file, "Corr_gradient")
    image_write = np.float32(image_write / np.max(image_write))
    cv2.imwrite(savepath + os.sep + os.path.basename(file).split('.')[0] + ".tif", image_write)

    # measure slopes
    height, width, colors = image.shape
    slopes_x = []
    slopes_y = []
    for c in range(colors):
        rowmeans = []
        for row in range(height):
            rowmeans.append(np.mean(sigmaclip(image[row,:,c], low=2, high=2)[0]))
        slope_y = np.mean(np.diff(rowmeans)) * height
        slopes_y.append(slope_y)

        colmeans = []
        for col in range(width):
            colmeans.append(np.mean(sigmaclip(image[:,col,c], low=2, high=2)[0]))
        slope_x = np.mean(np.diff(colmeans)) * width
        slopes_x.append(slope_x)

        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        gradient = X * slope_x + Y * slope_y - (slope_x + slope_y) / 2
        image[:, :, c] = image[:, :, c] - gradient

    # print
    print("gradient slopes x: ", slopes_x)
    print("gradient slopes y: ", slopes_y)

    # save corrected image
    image_write = bayer(image[:, :, 1:])
    savepath = create_folder(file, "Corr_gradient")
    image_write = np.float32(image_write / np.max(image_write))
    cv2.imwrite(savepath + os.sep + os.path.basename(file).split('.')[0] + "_corr.tif", image_write)
    return image


def calc_histograms(image, file, circular=False):
    print("calculate histograms ...")
    for c in range(3):

        if circular:
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    thisdist = dist_from_center(i, j, image)
                    if thisdist > image.shape[0] / 2 or thisdist > image.shape[1] / 2:
                        image[i, j, c] = np.nan

        if c == 1:
            vals = np.append(image[:, :, 1], image[:, :, 2])
        else:
            vals = image[:, :, c]
        vals = vals.flatten()

        # caLculate histogram
        counts, bins = np.histogram(vals, np.linspace(0, 2 ** 12, 2 ** 8))
        bins = bins[1:]
        if c == 0:
            data = bins
        data = np.column_stack((data, counts))

    # save
    savepath = create_folder(file, "Histograms")
    fmt = '%.5f', '%d', '%d', '%d'
    np.savetxt(savepath + os.sep + os.path.basename(file).split('.')[0] + "_histograms.csv", data,
               delimiter=",",
               fmt=fmt)

def nearest_neighbor_pixelmap(im_deb, file):
    rows, cols, colors = im_deb.shape
    flat_pixel_values = []
    flat_maxneighbors = []
    for c in range(colors):
        maxneighbors = np.zeros((rows, cols))
        for n in range(rows):
            if n % 500 == 0: print("color: ", c, ", row: ", n)
            for m in range(cols):
                neighbors = []
                for nd in [-1, 0, 1]:
                    for md in [-1, 0, 1]:
                        if not (nd == 0 and md == 0):
                            try:
                                neighbors.append(im_deb[n + nd, m + md, c])
                            except:
                                pass
                maxneighbors[n, m] = max(neighbors)
        flat_pixel_values += list(im_deb[:,:,c].flatten())
        flat_maxneighbors += list(maxneighbors.flatten())

    fig = plt.figure(figsize=(7, 5))
    plt.rcParams['font.size'] = 14
    plt.hist2d(flat_pixel_values, flat_maxneighbors, bins=300, range=[[0, 2000], [0, 2000]], norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
    # plt.xticks(range(100,20000,200))
    # plt.yticks(range(100,20000,200))
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 2000])
    plt.ylim([0, 2000])
    plt.xlabel('Pixel value')
    plt.ylabel('Max of 8 neighbors')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    fig.tight_layout()
    savepath = create_folder(file, "Nearest_neighbor_pixelmap")
    plt.savefig(savepath + os.sep + os.path.basename(file).split(".")[0] + '_pixelmap.png', dpi=300)
    plt.show()


def calc_rad_profile(image, file, statistics=2, extrapolate_max=True):
    print("calculate radial profiles ...")
    image_width = image.shape[1]
    image_height = image.shape[0]
    maxrad = int(dist_from_center(0, 0, image))
    radii = []
    rad_counts = {}
    rad_pixels = {}
    for i in range(image_height):
        for j in range(image_width):
            rad = int(dist_from_center(i, j, image))
            if not (image[i, j, 0] > 0 and image[i, j, 1] > 0 and image[i, j, 2] > 0 and image[i, j, 3] > 0):
                continue
            if not rad in radii:
                radii.append(rad)
                rad_counts[rad] = [[], [], []]
                rad_pixels[rad] = [[], [], []]
            rad_counts[rad][0].append(image[i, j, 0])
            rad_counts[rad][1].append(image[i, j, 1])
            rad_counts[rad][1].append(image[i, j, 2])
            rad_counts[rad][2].append(image[i, j, 3])
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
        print("cut edges ...")
        mask = rad_profile[:, 0] > IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile[mask, :]
        mask = rad_profile_cut[:, 0] < 1 - IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile_cut[mask, :]
    else:
        rad_profile_cut = copy.deepcopy(rad_profile)

    # cut data inside maximum
    if extrapolate_max:
        print("cut inside max ...")
        maxind = []
        slopes = []
        for c in range(3):
            this_filtered = rad_profile_cut[:, c + 1]
            this_filtered = savgol_filter(this_filtered, window_length=odd_int(rad_profile_cut.shape[0] / 10),
                                                       polyorder=2, mode='interp')
            this_filtered = savgol_filter(this_filtered, window_length=odd_int(rad_profile_cut.shape[0] / 5),
                                                       polyorder=2, mode='interp')
            mi = np.argmax(this_filtered)
            maxind.append(mi)
        print("maximum index: ", maxind)
        rad_profile_cut = rad_profile_cut[max(maxind):, :]
        # for c in range(3):
        #     rad_profile_cut[:min_maxind,c+1] = max(rad_profile_cut[:,c+1])

    # smooth data and interpolate
    print("smooth data...")
    radii = np.linspace(0, 1, RADIAL_RESOLUTION)
    rad_profile_smoothed = radii
    for c in range(3):
        y_new = savgol_filter(rad_profile_cut[:, c + 1], window_length=odd_int(rad_profile_cut.shape[0] / 10),
                                           polyorder=2, mode='interp')
        y_new = savgol_filter(y_new, window_length=odd_int(rad_profile_cut.shape[0] / 5), polyorder=2,
                                           mode='interp')

        if extrapolate_max:
            # add central value for interpolation
            x_new = list(rad_profile_cut[:, 0].flatten())
            y_new = list(y_new.flatten())
            xdist = x_new[1] - x_new[0]
            ylast = y_new[0]
            xlast = x_new[0]
            slope = (y_new[1] - y_new[0]) / xdist
            # slope = np.mean(np.diff(y_new[:int(len(y_new)/10)])) / xdist
            slopes.append(int(slope))
            xadd = np.arange(0, xlast * 0.999, xdist)
            # quadratic function with given value and slope at xlast and zero derivative at x=0
            yadd = ylast + slope / 2 * (xadd ** 2 / xlast - xlast)
            # print(np.diff(yadd) / xdist) # derivative for troubleshooting
            x_new = list(xadd) + x_new
            y_new = list(yadd) + y_new
        else:
            x_new = rad_profile_cut[:, 0]

        # y_new = savgol_filter(y_new, window_length=odd_int(rad_profile_cut.shape[0]/5), polyorder=2, mode='interp')

        f = interp1d(x_new, y_new, kind='quadratic', fill_value='extrapolate')
        y_new = f(radii)
        rad_profile_smoothed = np.column_stack((rad_profile_smoothed, y_new))

    if extrapolate_max:
        print("extrapolate with slopes: ", slopes)

    # normalize by smoothed curve
    for c in range(3):
        rad_profile_raw_mean[:, c + 1] = rad_profile_raw_mean[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile_cut[:, c + 1] = rad_profile_cut[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])
        rad_profile_smoothed[:, c + 1] = rad_profile_smoothed[:, c + 1] / np.max(rad_profile_smoothed[:, c + 1])

    # save
    print("save csv ...")
    savepath = create_folder(file, "Radial_profiles")
    fmt = '%.5f', '%.5f', '%.5f', '%.5f'
    np.savetxt(savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile_0_raw_mean.csv", rad_profile_raw_mean, delimiter=",", fmt=fmt)
    np.savetxt(savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile_1_clipped.csv", rad_profile, delimiter=",", fmt=fmt)
    np.savetxt(savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile_2_cut.csv", rad_profile_cut, delimiter=",", fmt=fmt)
    np.savetxt(savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile_3_smooth.csv", rad_profile_smoothed[::int(RADIAL_RESOLUTION / 1000)], delimiter=",", fmt=fmt)
    return rad_profile_smoothed


def export_tif(rad_profile, file, grey_flat=False, tif_size=(4024, 6024), max_value=1):
    # export tif
    print("export tif...")
    print("grey flat: ", grey_flat)
    print("tif size:  ", tif_size)
    print("max value: ", max_value)
    if len(tif_size) == 3:
        # like (4024, 6024, 3)
        # save as debayered image
        save_debayered = True
        im_syn = np.zeros((int(tif_size[0]), int(tif_size[1]), 3))
    else:
        # (4024, 6024)
        # save as bayer
        save_debayered = False
        im_syn = np.zeros((int(tif_size[0]), int(tif_size[1])))

    # normalize (again)
    for c in range(3):
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile[:, c + 1])

    # match profile to output size
    maxrad_this = dist_from_center(0, 0, im_syn)
    print("maxrad synthetic: ", int(maxrad_this))

    # iterate image and write pixels
    for i in range(im_syn.shape[0]):
        if not i == 0 and i % int(im_syn.shape[0] / 10) == 0:
            print(i, end=" >> ")
        if not i == 0 and i % int(im_syn.shape[0] / 2) == 0:
            print("")
        for j in range(im_syn.shape[1]):

            # radial position of pixel
            r = dist_from_center(i, j, im_syn) / maxrad_this

            # search match within small range in radial profile
            # r_index = get_closest(r, rad_profile[:,0])
            r_index = int((RADIAL_RESOLUTION - 1) * r)

            # check deviation
            if abs(rad_profile[r_index, 0] - r) > 0.001:
                print("Radial deviation is larger than 0.1%!")
                print("rad: ", r, ", found: ", rad_profile[r_index, 0])
                print("Interrupt!!")
                sys.exit()

            # calculate brightness and set new image pixel
            if save_debayered:
                for c in range(3):
                    im_syn[i, j, c] = rad_profile[r_index, c + 1]
            else:
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

    # add noise for dithering
    if ADD_NOISE_LEVEL > 0:
        if save_debayered:
            im_syn = im_syn + ADD_NOISE_LEVEL * (np.random.rand(im_syn.shape[0], im_syn.shape[1], 3) - 0.5)
        else:
            im_syn = im_syn + ADD_NOISE_LEVEL * (np.random.rand(im_syn.shape[0], im_syn.shape[1]) - 0.5)

    # final row print
    print(i)

    # convert to 16 bit and save image
    savepath = create_folder(file, "Radial_profiles_tifs")
    im_syn = max_value * (2 ** 16 - 1) * im_syn / np.max(im_syn)
    im_syn = im_syn.astype(np.uint16)
    print("16-bit maximum: ", np.max(im_syn))
    cv2.imwrite(savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile.tif", im_syn)


# STATIC MINOR FUNCTIONS =====================================================

def channel(index):
    if index == 0:
        return 'B'
    if index == 1:
        return 'G'
    if index == 2:
        return 'R'


def mindist(array_):
    array = list(set(array_[:1000000]))[:100]
    array.sort()
    mindist = max(array)
    for i in range(1, len(array)):
        thisdist = array[i] - array[i - 1]
        if thisdist > 0 and thisdist < mindist:
            mindist = thisdist
    return mindist


def debayer(array):
    rows = array.shape[0]
    cols = array.shape[1]
    db_array = np.zeros((int(rows / 2) + 1, int(cols / 2) + 1, 4))
    for n in range(rows):
        for m in range(cols):
            if n % 2 == 0 and m % 2 == 0:  # links oben
                db_array[int((n + 0) / 2), int((m + 0) / 2), 0] = int(array[n, m])  # R
            if n % 2 == 0 and m % 2 == 1:  # rechts oben
                db_array[int((n + 0) / 2), int((m - 1) / 2), 1] = int(array[n, m])  # G
            if n % 2 == 1 and m % 2 == 0:  # links unten
                db_array[int((n - 1) / 2), int((m + 0) / 2), 2] = int(array[n, m])  # G
            if n % 2 == 1 and m % 2 == 1:  # rechts unten
                db_array[int((n - 1) / 2), int((m - 1) / 2), 3] = int(array[n, m])  # B
    return db_array


def bayer(array):
    rows = array.shape[0]
    cols = array.shape[1]
    colors = array.shape[2]
    b_array = np.zeros((rows * 2, cols * 2))
    for i in range(rows):
        for j in range(cols):
            for c in range(colors):
                if c == 0:  # R
                    b_array[2 * i + 0, 2 * j + 0] = array[i, j, c]  # links oben
                elif c == 1:  # G
                    b_array[2 * i + 1, 2 * j + 0] = array[i, j, c]  # rechts oben
                    b_array[2 * i + 0, 2 * j + 1] = array[i, j, c]  # links unten
                else:  # B
                    b_array[2 * i + 1, 2 * j + 1] = array[i, j, c]  # rechts unten
    return b_array


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


def resize(array, factor):
    image_axes, color_axis = separate_axes(array)
    if color_axis:
        new_array = np.zeros((
            int(array.shape[image_axes[0]] / factor) + 1,
            int(array.shape[image_axes[1]] / factor) + 1,
            array.shape[color_axis]
        ))
    else:
        new_array = np.zeros((
            int(array.shape[image_axes[0]] / factor) + 1,
            int(array.shape[image_axes[1]] / factor) + 1
        ))
    for n in range(array.shape[image_axes[0]]):
        if n % factor == 0:
            for m in range(array.shape[image_axes[1]]):
                if m % factor == 0:
                    if color_axis:
                        for c in range(array.shape[color_axis]):
                            new_array[int(n / factor), int(m / factor), c] = array[n, m, c]
                    else:
                        new_array[int(n / factor), int(m / factor)] = array[n, m]
    return new_array


def dist_from_center(i, j, mat):
    n = len(mat[:, 0])
    m = len(mat[0, :])
    rad = np.sqrt((i - n / 2) ** 2 + (j - m / 2) ** 2)
    return rad


def create_folder(file, folder_name):
    savepath = os.path.dirname(file) + os.sep + folder_name
    try:
        os.mkdir(savepath)
    except:
        pass
    return savepath


def calc_quadrant_med(image, color_index=0, relative=False):
    image_axes, color_axis = separate_axes(image)
    if color_axis:
        image = image.take(indices=color_index, axis=color_axis)
    height_half = int(image.shape[0] / 2)
    width_half = int(image.shape[1] / 2)
    quadrant_1 = image[:height_half, :width_half]
    quadrant_2 = image[:height_half, width_half:]
    quadrant_3 = image[height_half:, :width_half]
    quadrant_4 = image[height_half:, width_half:]
    if relative:
        norm = np.median(image) / 100
    else:
        norm = 1
    quadrant_med = []
    quadrant_med.append(np.median(quadrant_1) / norm)
    quadrant_med.append(np.median(quadrant_2) / norm)
    quadrant_med.append(np.median(quadrant_3) / norm)
    quadrant_med.append(np.median(quadrant_4) / norm)
    return quadrant_med


def clip(number):
    if number < 0:
        return 0
    elif number > 1:
        return 1
    else:
        return number

def odd_int(number):
    if int(number) % 2 == 1:
        return int(number)
    else:
        return int(number) + 1

def get_closest(val, list):
    # list must be sorted!
    closest = np.inf
    closest_ind = 0
    start = int(clip(val - 0.05) * len(list))
    stop = int(clip(val + 0.05) * len(list))
    for i in range(start, stop):
        if abs(list[i] - val) < closest:
            closest = abs(list[i] - val)
            closest_ind = i
        else:
            # larger for the first time
            break
    return closest_ind

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

class NewGUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUINAME + " v" + VERSION)
        self.lastpath = ""
        self.loaded_files = []
        self.current_file = "0 files"
        self.current_status = "ready"
        self.bias_value = 0
        self.running = False
        
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
        self.opt_pixelmap  = tk.BooleanVar()
        self.opt_histogram = tk.BooleanVar()
        self.opt_radprof   = tk.BooleanVar()
        self.opt_synthflat = tk.BooleanVar()
        options.add_checkbutton(label="Correct gradient", onvalue=1, offvalue=0, variable=self.opt_gradient, command=self.verify_modes)
        options.add_checkbutton(label="Nearest neighbor pixelmap", onvalue=1, offvalue=0, variable=self.opt_pixelmap, command=self.verify_modes)
        options.add_checkbutton(label="Calculate histogram", onvalue=1, offvalue=0, variable=self.opt_histogram, command=self.verify_modes)
        options.add_checkbutton(label="Calculate radial profile", onvalue=1, offvalue=0, variable=self.opt_radprof, command=self.verify_modes)
        options.add_checkbutton(label="Export synthetic flat", onvalue=1, offvalue=0, variable=self.opt_synthflat, command=self.verify_modes)
        menubar.add_cascade(label="Options", menu=options)

        # settings
        settings = tk.Menu(menubar, tearoff=0)
        self.set_write_pickle    = tk.BooleanVar()
        self.set_circular_hist   = tk.BooleanVar()
        self.set_grey_flat       = tk.BooleanVar()
        self.set_debayered_flat  = tk.BooleanVar()
        self.set_extrapolate_max = tk.BooleanVar()
        self.set_scale_flat      = tk.BooleanVar()
        settings.add_checkbutton(label="Write pickle file", onvalue=1, offvalue=0, variable=self.set_write_pickle)
        settings.add_checkbutton(label="Histogram of largest circle", onvalue=1, offvalue=0, variable=self.set_circular_hist)
        settings.add_checkbutton(label="Extrapolate inside max", onvalue=1, offvalue=0, variable=self.set_extrapolate_max)
        settings.add_checkbutton(label="Export synthetic flat as grey", onvalue=1, offvalue=0, variable=self.set_grey_flat)
        settings.add_checkbutton(label="Export synthetic flat debayered", onvalue=1, offvalue=0, variable=self.set_debayered_flat)
        settings.add_checkbutton(label="Scale synthetic flat like original", onvalue=1, offvalue=0, variable=self.set_scale_flat)
        menubar.add_cascade(label="Settings", menu=settings)

        # statistics
        statistics = tk.Menu(menubar, tearoff=0)
        self.radio_statistics = tk.StringVar(self.root)
        self.sigmas = ["mean", "median", "min", "max", "sigma clip 0.5", "sigma clip 1.0",
                       "sigma clip 2.0", "sigma clip 3.0", "sigma clip 4.0", "sigma clip 8.0"]
        for opt in self.sigmas:
            statistics.add_radiobutton(label=opt, value=opt, variable=self.radio_statistics)
        menubar.add_cascade(label="Statistics", menu=statistics)

        # shrink
        shrink = tk.Menu(menubar, tearoff=0)
        self.radio_shrink = tk.StringVar(self.root)
        self.shrink_factor = ['off', '2', '4', '8', '16']
        for opt in self.shrink_factor:
            shrink.add_radiobutton(label=opt, value=opt, variable=self.radio_shrink)
        menubar.add_cascade(label="Shrink", menu=shrink)

        # buttons
        self.button_load = tk.Button(text="Load files", command=self.load_files)
        self.button_load.grid(row=0, column=0, sticky='NWSE', padx=padding, pady=padding, columnspan=2)
        self.button_load = tk.Button(text="Set bias value", command=self.ask_bias)
        self.button_load.grid(row=1, column=0, sticky='NWSE', padx=padding, pady=padding)
        self.button_load = tk.Button(text="Bias from file", command=self.ask_bias_file)
        self.button_load.grid(row=1, column=1, sticky='NWSE', padx=padding, pady=padding)
        self.button_start = tk.Button(text="Start", command=self.process)
        self.button_start.grid(row=2, column=0, sticky='NWSE', padx=padding, pady=padding, columnspan=2)

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

    def on_close(self):
        print("... save config file")
        config_object = ConfigParser()

        config_object["BASICS"] = {}
        config_object["BASICS"]["window size"]      = self.root.winfo_geometry()
        config_object["BASICS"]["lastpath"]         = self.lastpath
        config_object["BASICS"]["radio_statistics"] = self.radio_statistics.get()
        config_object["BASICS"]["radio_shrink"]     = self.radio_shrink.get()
        config_object["BASICS"]["bias_value"]   = str(self.bias_value)

        config_object["OPTIONS"] = {}
        config_object["OPTIONS"]["opt_gradient"]  = str(self.opt_gradient.get())
        config_object["OPTIONS"]["opt_pixelmap"]  = str(self.opt_pixelmap.get())
        config_object["OPTIONS"]["opt_histogram"] = str(self.opt_histogram.get())
        config_object["OPTIONS"]["opt_radprof"]   = str(self.opt_radprof.get())
        config_object["OPTIONS"]["opt_synthflat"] = str(self.opt_synthflat.get())

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"]     = str(self.set_write_pickle.get())
        config_object["SETTINGS"]["set_circular_hist"]    = str(self.set_circular_hist.get())
        config_object["SETTINGS"]["set_grey_flat"]        = str(self.set_grey_flat.get())
        config_object["SETTINGS"]["set_debayered_flat"]   = str(self.set_debayered_flat.get())
        config_object["SETTINGS"]["set_extrapolate_max"]  = str(self.set_extrapolate_max.get())
        config_object["SETTINGS"]["set_scale_flat"]       = str(self.set_scale_flat.get())

        with open(GUINAME + ".conf", 'w') as conf:
            config_object.write(conf)

        self.root.destroy()

    def verify_modes(self):
        if self.opt_synthflat.get():
            self.opt_radprof.set(1)

    def load_config_file(self):
        config_object = ConfigParser()

        # read
        if os.path.exists(GUINAME + ".conf"):
            config_object.read(GUINAME + ".conf")

        # default
        else:
            config_object["BASICS"] = {}
            config_object["BASICS"]["window size"] = '318x128+313+94'
            config_object["BASICS"]["lastpath"]    = './'
            config_object["BASICS"]["radio_statistics"]  = 'sigma clip 2.0'
            config_object["BASICS"]["radio_shrink"]  = '4'
            config_object["BASICS"]["bias_value"]  = '0'

            config_object["OPTIONS"] = {}
            config_object["OPTIONS"]["opt_gradient"]  = 'True'
            config_object["OPTIONS"]["opt_pixelmap"]  = 'False'
            config_object["OPTIONS"]["opt_histogram"] = 'False'
            config_object["OPTIONS"]["opt_radprof"]   = 'True'
            config_object["OPTIONS"]["opt_synthflat"] = 'True'

            config_object["SETTINGS"] = {}
            config_object["SETTINGS"]["set_write_pickle"]    = 'True'
            config_object["SETTINGS"]["set_circular_hist"]   = 'True'
            config_object["SETTINGS"]["set_grey_flat"]       = 'False'
            config_object["SETTINGS"]["set_debayered_flat"]  = 'False'
            config_object["SETTINGS"]["set_extrapolate_max"] = 'True'
            config_object["SETTINGS"]["set_scale_flat"]      = 'False'

        # apply
        self.root.geometry(config_object["BASICS"]["window size"])
        self.lastpath = config_object["BASICS"]["lastpath"]
        self.radio_statistics.set(config_object["BASICS"]["radio_statistics"])
        self.radio_shrink.set(config_object["BASICS"]["radio_shrink"])
        self.bias_value = int(config_object["BASICS"]["bias_value"])

        self.opt_gradient.set(config_object["OPTIONS"]["opt_gradient"]   == 'True')
        self.opt_pixelmap.set(config_object["OPTIONS"]["opt_pixelmap"]   == 'True')
        self.opt_histogram.set(config_object["OPTIONS"]["opt_histogram"] == 'True')
        self.opt_radprof.set(config_object["OPTIONS"]["opt_radprof"]     == 'True')
        self.opt_synthflat.set(config_object["OPTIONS"]["opt_synthflat"] == 'True')

        self.set_write_pickle.set(config_object["SETTINGS"]["set_write_pickle"]   == 'True')
        self.set_circular_hist.set(config_object["SETTINGS"]["set_circular_hist"] == 'True')
        self.set_grey_flat.set(config_object["SETTINGS"]["set_grey_flat"]         == 'True')
        self.set_grey_flat.set(config_object["SETTINGS"]["set_debayered_flat"]        == 'True')
        self.set_extrapolate_max.set(config_object["SETTINGS"]["set_extrapolate_max"] == 'True')
        self.set_scale_flat.set(config_object["SETTINGS"]["set_scale_flat"]       == 'True')

        self.update_labels()

    def load_files(self):
        if self.running:
            return
        user_input = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('RAW format (supported)', RAW_TYPES), ('Image format (not supported)', IMAGE_TYPES), ('all', '.*')])
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
        self.root.update()

    def ask_bias_file(self):
        if self.running:
            return
        self.update_labels(status="calc bias...")
        user_input_file = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('all', '.*'), ('raw', '.arw')])
        im_raw = rawpy.imread(user_input_file[0]).raw_image_visible
        self.bias_value = int(sigma_clip_mean(im_raw))
        self.label_bias_var.set(self.bias_value)
        self.update_labels(status="ready")
        return

    def update_labels(self, file="", status=""):
        if file:
            self.current_file = file
        if status:
            self.current_status = status
        self.label_files_var.set(os.path.basename(self.current_file))
        self.label_status_var.set(self.current_status)
        self.label_bias_var.set(self.bias_value)
        self.root.update()
        return

    def process(self):
        if self.running:
            # print("exit")
            # self.root.quit()
            # self.running = False
            # self.root.mainloop()
            # self.update_labels(status="ready")
            return
        else:
            self.running = True
            self.update_labels(status="running...")
        try:
            if not self.loaded_files:
                self.running = False
                self.update_labels(status="no file chosen.")
                return
            for file in self.loaded_files:

                # set and display
                self.update_labels(file=file)

                # load
                #print(dt.datetime.now())
                self.update_labels(status="load and debayer...")
                image, rawshape = load_image(file)

                # write pickle
                if self.set_write_pickle.get():
                    self.update_labels(status="save pickle file...")
                    write_pickle(image, rawshape, self.current_file)

                # edit
                self.update_labels(status="edit image...")
                image_edit = edit_image(image,
                                        bias_value=self.bias_value,
                                        shrink_factor=self.radio_shrink.get()
                                        )

                # gradient
                if self.opt_gradient.get():
                    self.update_labels(status="correct gradient...")
                    image_edit = corr_gradient(image_edit, self.current_file)

                # pixelmap
                if self.opt_pixelmap.get():
                    self.update_labels(status="calculate pixelmap...")
                    nearest_neighbor_pixelmap(image_edit, file)

                # histogram
                if self.opt_histogram.get():
                    self.update_labels(status="calculate histogram...")
                    calc_histograms(image_edit, self.current_file, self.set_circular_hist.get())

                # radial profile
                if self.opt_radprof.get():
                    self.update_labels(status="calculate radial profile...")
                    rad_profile_smoothed = calc_rad_profile(image_edit, self.current_file,
                                                            statistics=self.radio_statistics.get(),
                                                            extrapolate_max=self.set_extrapolate_max.get())

                # synthetic flat
                if self.opt_synthflat.get():
                    self.update_labels(status="export synthetic flat...")
                    if self.set_debayered_flat.get():
                        tif_size = (rawshape[0], rawshape[1], 3)
                    else:
                        tif_size = (rawshape[0], rawshape[1])
                    if self.set_scale_flat.get():
                        max_value = sigma_clip_mean(image) / 16384
                    else:
                        max_value = 1
                    export_tif(rad_profile_smoothed, self.current_file,
                               grey_flat=self.set_grey_flat.get(),
                               tif_size=tif_size,
                               max_value=max_value)

                self.update_labels(status="finished.")
                print("Finished file.")


        except Exception as e:
            print("\nERROR!!\n\n")
            self.update_labels(status="unknown error...")
            print(e)
            return

        finally:
            self.running = False
            self.update_labels(file=str(len(self.loaded_files)) + " files")

        return


if __name__ == '__main__':
    new = NewGUI()

