import tkinter as tk
import tkinter.font
import tkinter.messagebox # for pyinstaller
from tkinter.filedialog import askopenfilename
import glob, os, sys, copy
import numpy as np
from configparser import ConfigParser
import rawpy, cv2, scipy
import pickle, gzip

GUINAME = "SyntheticFlatGUI"
VERSION = '1.0'

# STATIC SETTINGS =============================================================
REDUCE_SIZE_FACTOR = 4    # reduce size for faster calculations and fits
IGNORE_EDGE = 10          # ignore pixels close to the image edges
IGNORE_RADII = 5          # ignore pixels extreme radii (close to 0 and maximum)
RADIAL_RESOLUTION = 100000  # should be larger than 16 bit (65536)
CORR_GRADIENT_MEAS_SIZE = 50  # size of squares at quarter-positions for brightness measurements
ADD_NOISE_LEVEL = 0

# STATIC MAJOR FUNCTIONS =====================================================
def load_image(file):
    print("")
    print(file)
    if not os.path.isfile(file):
        print("File does not exist!\n\n")
        sys.exit()
    try:
        im_raw = pickle.load(gzip.open(
            os.path.dirname(file) + os.sep + "Pickle_files" + os.sep + os.path.basename(file).split('.')[
                0] + ".pkl", 'rb'))
        print("Use image from pickle file.")
        print("shape: ", im_raw.shape)
    except:
        print("read file ...")
        im_raw = rawpy.imread(file).raw_image
        print("shape: ", im_raw.shape)
        print("debayer ...")
        im_raw = debayer(im_raw)
        print("shape: ", im_raw.shape)

    return im_raw

def write_pickle(im_raw, file):
    savepath = create_folder(file, "Pickle_files")
    pickle_filename = savepath + os.sep + os.path.basename(file).split('.')[0] + ".pkl"
    if not os.path.isfile(pickle_filename):
        print("write pickle file ...")
        pickle.dump(im_raw, gzip.open(pickle_filename, 'wb'))
        print("maxrad original: ", int(dist_from_center(0, 0, im_raw)))

def edit_image(im_raw, bias_value=0):
    if bias_value > 0:
        print("subtract bias (" + str(bias_value) + ") ...")
        im_raw = im_raw - bias_value

    if REDUCE_SIZE_FACTOR:
        print("resize ...")
        im_raw = resize(im_raw, REDUCE_SIZE_FACTOR)
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

    # calculate
    height, width, colors = image.shape
    mpos_x = [int(width / 4), int(width * 3 / 4)]
    mpos_y = [int(height / 4), int(height * 3 / 4)]
    mpos_d = CORR_GRADIENT_MEAS_SIZE
    for c in range(colors):
        mint_0 = np.median(image[mpos_y[0] - mpos_d:mpos_y[0] + mpos_d, mpos_x[0] - mpos_d:mpos_x[0] + mpos_d, c])
        mint_x = np.median(image[mpos_y[0] - mpos_d:mpos_y[0] + mpos_d, mpos_x[1] - mpos_d:mpos_x[1] + mpos_d, c])
        mint_y = np.median(image[mpos_y[1] - mpos_d:mpos_y[1] + mpos_d, mpos_x[0] - mpos_d:mpos_x[0] + mpos_d, c])
        mint_1 = np.median(image[mpos_y[1] - mpos_d:mpos_y[1] + mpos_d, mpos_x[1] - mpos_d:mpos_x[1] + mpos_d, c])
        # print("quadrants: ", calc_quadrant_med(image, color_index=c, relative=True))
        # print("ints: ", int(mint_0), int(mint_x), int(mint_y), int(mint_1))
        slope_x = (
                mint_x + mint_1 - mint_0 - mint_y)  # aus irgendeinem Grund besser nicht *2 (slope Nenner sollte 0.5 sein)
        slope_y = (mint_y + mint_1 - mint_0 - mint_x)
        # print("slopes: ", int(slope_x), int(slope_y), int(slope_x + slope_y))
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        gradient = X * slope_x + Y * slope_y - (slope_x + slope_y) / 2
        image[:, :, c] = image[:, :, c] - gradient
        # print("quadrants: ", calc_quadrant_med(image, color_index=c, relative=True))
        # print("")

        # check integration areas
        # image[mpos_y[0]-mpos_d:mpos_y[0]+mpos_d, mpos_x[0]-mpos_d:mpos_x[0]+mpos_d, c].fill(0)
        # image[mpos_y[0]-mpos_d:mpos_y[0]+mpos_d, mpos_x[1]-mpos_d:mpos_x[1]+mpos_d, c].fill(0)
        # image[mpos_y[1]-mpos_d:mpos_y[1]+mpos_d, mpos_x[0]-mpos_d:mpos_x[0]+mpos_d, c].fill(0)
        # image[mpos_y[1]-mpos_d:mpos_y[1]+mpos_d, mpos_x[1]-mpos_d:mpos_x[1]+mpos_d, c].fill(0)

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


def calc_rad_profile(image, file, sigma_clip=2, extrapolate_max=True):
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
        rad_profile[index, 1] = safe_sigmaclip_mean(rad_counts[rad][0], sigma_clip)
        rad_profile[index, 2] = safe_sigmaclip_mean(rad_counts[rad][1], sigma_clip)
        rad_profile[index, 3] = safe_sigmaclip_mean(rad_counts[rad][2], sigma_clip)
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
            this_filtered = scipy.signal.savgol_filter(this_filtered, window_length=int(rad_profile_cut.shape[0] / 10),
                                                       polyorder=2, mode='interp')
            this_filtered = scipy.signal.savgol_filter(this_filtered, window_length=int(rad_profile_cut.shape[0] / 5),
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
        y_new = scipy.signal.savgol_filter(rad_profile_cut[:, c + 1], window_length=int(rad_profile_cut.shape[0] / 10),
                                           polyorder=2, mode='interp')
        y_new = scipy.signal.savgol_filter(y_new, window_length=int(rad_profile_cut.shape[0] / 5), polyorder=2,
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

        # y_new = scipy.signal.savgol_filter(y_new, window_length=int(rad_profile_cut.shape[0]/5), polyorder=2, mode='interp')

        f = scipy.interpolate.interp1d(x_new, y_new, kind='quadratic', fill_value='extrapolate')
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


def export_tif(rad_profile, file, grey_flat=False, tif_size=(4024, 6024)):
    # export tif
    print("export tif")
    if len(tif_size) == 3:
        # like (4024, 6024, 3)
        # save as debayered image
        save_debayered = True
        print("save debayered tif ...")
        im_syn = np.zeros((int(tif_size[0]), int(tif_size[1]), 3))
    else:
        # (4024, 6024)
        # save as bayer
        save_debayered = False
        print("save bayer tif ...")
        im_syn = np.zeros((int(tif_size[0]), int(tif_size[1])))

    # normalize (again)
    for c in range(3):
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile[:, c + 1])

    # match profile to output size
    maxrad_this = dist_from_center(0, 0, im_syn)
    print("maxrad synthetic: ", int(maxrad_this))

    # iterate image and write pixels
    for i in range(im_syn.shape[0]):
        if i % int(im_syn.shape[0] / 10) == 0:
            print(i, end=" >> ")
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
    im_syn = im_syn / np.max(im_syn)
    im_syn = ((2 ** 16 - 1) * im_syn).astype(np.uint16)
    print(im_syn.shape)
    cv2.imwrite(
        savepath + os.sep + os.path.basename(file).split('.')[0] + "_radial_profile.tif",
        im_syn)


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

def safe_sigmaclip_mean(array, sigma_clip):
    reduced = scipy.stats.sigmaclip(array, low=sigma_clip, high=sigma_clip)[0]
    meanval = np.mean(reduced)
    if np.isnan(meanval):
        meanval = np.median(array)
        print("check")
    return meanval

class NewGUI():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(GUINAME + " v" + VERSION)
        self.lastpath = ""
        self.loaded_files = []
        self.current_file = "0 files"
        self.current_status = "ready"
        self.bias_value = 0
        
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
        options.add_checkbutton(label="Calculate radial profile", onvalue=1, offvalue=0, variable=self.opt_radprof)
        options.add_checkbutton(label="Export synthetic flat", onvalue=1, offvalue=0, variable=self.opt_synthflat)
        menubar.add_cascade(label="Options", menu=options)

        # settings
        settings = tk.Menu(menubar, tearoff=0)
        self.set_write_pickle  = tk.BooleanVar()
        self.set_circular_hist = tk.BooleanVar()
        self.set_grey_flat   = tk.BooleanVar()
        self.set_extrapolate_max = tk.BooleanVar()
        settings.add_checkbutton(label="Write pickle file", onvalue=1, offvalue=0, variable=self.set_write_pickle)
        settings.add_checkbutton(label="Histogram of largest circle", onvalue=1, offvalue=0, variable=self.set_circular_hist)
        settings.add_checkbutton(label="Export synthetic flat as grey", onvalue=1, offvalue=0, variable=self.set_grey_flat)
        settings.add_checkbutton(label="Extrapolate inside max", onvalue=1, offvalue=0, variable=self.set_extrapolate_max)
        menubar.add_cascade(label="Settings", menu=settings)

        # sigma clipping
        mode = tk.Menu(menubar, tearoff=0)
        self.radio_clip = tk.StringVar(self.root)
        self.sigmas = ["0.5", "1.0", "2.0", "3.0", "4.0", "8.0", "off"]
        for opt in self.sigmas:
            mode.add_radiobutton(label=opt, value=opt, variable=self.radio_clip)
        menubar.add_cascade(label="Sigma clipping", menu=mode)

        # buttons
        self.button_load = tk.Button(text="Load files", command=self.load_files)
        self.button_load.grid(row=0, column=0, sticky='NWSE', padx=padding, pady=padding)
        self.button_load = tk.Button(text="Set bias value", command=self.ask_bias)
        self.button_load.grid(row=1, column=0, sticky='NWSE', padx=padding, pady=padding)
        self.button_start = tk.Button(text="Start", command=self.process)
        self.button_start.grid(row=2, column=0, sticky='NWSE', padx=padding, pady=padding)

        # labels
        self.label_files_var = tk.StringVar()
        self.label_files_var.set("0 files")
        self.label_files = tk.Label(textvariable=self.label_files_var, justify='left')
        self.label_files.grid(row=0, column=1, sticky='NWSE', padx=padding, pady=padding)
        self.label_bias_var = tk.StringVar()
        self.label_bias_var.set("bias = 0")
        self.label_bias = tk.Label(textvariable=self.label_bias_var, justify='left')
        self.label_bias.grid(row=1, column=1, sticky='NWSE', padx=padding, pady=padding)
        self.label_status_var = tk.StringVar()
        self.label_status_var.set("ready")
        self.label_status = tk.Label(textvariable=self.label_status_var, justify='left')
        self.label_status.grid(row=2, column=1, sticky='NWSE', padx=padding, pady=padding)

        # configure
        for i in range(2):
            self.root.grid_columnconfigure(i, weight=1)
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
        config_object["BASICS"]["window size"] = self.root.winfo_geometry()
        config_object["BASICS"]["lastpath"] = self.lastpath
        config_object["BASICS"]["radio_clip"] = self.radio_clip.get()
        config_object["BASICS"]["bias_value"] = str(self.bias_value)

        config_object["OPTIONS"] = {}
        config_object["OPTIONS"]["opt_gradient"]  = str(self.opt_gradient.get())
        config_object["OPTIONS"]["opt_histogram"] = str(self.opt_histogram.get())
        config_object["OPTIONS"]["opt_radprof"]   = str(self.opt_radprof.get())
        config_object["OPTIONS"]["opt_synthflat"] = str(self.opt_synthflat.get())

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"]    = str(self.set_write_pickle.get())
        config_object["SETTINGS"]["set_circular_hist"]   = str(self.set_circular_hist.get())
        config_object["SETTINGS"]["set_grey_flat"]        = str(self.set_grey_flat.get())
        config_object["SETTINGS"]["set_extrapolate_max"] = str(self.set_extrapolate_max.get())

        with open(GUINAME + ".conf", 'w') as conf:
            config_object.write(conf)

        self.root.destroy()

    def load_config_file(self):
        config_object = ConfigParser()

        # read
        if os.path.exists(GUINAME + ".conf"):
            config_object.read(GUINAME + ".conf")

        # default
        else:
            config_object["BASICS"] = {}
            config_object["BASICS"]["window size"] = '379x213+421+21'
            config_object["BASICS"]["lastpath"]    = './'
            config_object["BASICS"]["radio_clip"]  = '2.0'
            config_object["BASICS"]["bias_value"]  = '0'

            config_object["OPTIONS"] = {}
            config_object["OPTIONS"]["opt_gradient"]  = 'True'
            config_object["OPTIONS"]["opt_histogram"] = 'True'
            config_object["OPTIONS"]["opt_radprof"]   = 'True'
            config_object["OPTIONS"]["opt_synthflat"] = 'True'

            config_object["SETTINGS"] = {}
            config_object["SETTINGS"]["set_write_pickle"]  = 'True'
            config_object["SETTINGS"]["set_circular_hist"] = 'True'
            config_object["SETTINGS"]["set_grey_flat"]     = 'False'
            config_object["SETTINGS"]["set_extrapolate_max"] = 'True'

        # apply
        self.root.geometry(config_object["BASICS"]["window size"])
        self.lastpath = config_object["BASICS"]["lastpath"]
        self.radio_clip.set(config_object["BASICS"]["radio_clip"])
        self.bias_value = int(config_object["BASICS"]["bias_value"])

        self.opt_gradient.set(config_object["OPTIONS"]["opt_gradient"] == 'True')
        self.opt_histogram.set(config_object["OPTIONS"]["opt_histogram"] == 'True')
        self.opt_radprof.set(config_object["OPTIONS"]["opt_radprof"] == 'True')
        self.opt_synthflat.set(config_object["OPTIONS"]["opt_synthflat"] == 'True')

        self.set_write_pickle.set(config_object["SETTINGS"]["set_write_pickle"] == 'True')
        self.set_circular_hist.set(config_object["SETTINGS"]["set_circular_hist"] == 'True')
        self.set_grey_flat.set(config_object["SETTINGS"]["set_grey_flat"] == 'True')
        self.set_extrapolate_max.set(config_object["SETTINGS"]["set_extrapolate_max"] == 'True')

        self.update_labels()

    def load_files(self):
        user_input = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('all', '.*'), ('.arw', '.arw')])
        if user_input:
            self.loaded_files = user_input
            self.update_labels(file=str(len(self.loaded_files)) + " files")
            self.lastpath = os.path.dirname(self.loaded_files[0])
        return

    def ask_bias(self):
        user_input = tk.simpledialog.askinteger(title="Bias value", prompt="Which bias value should be subtracted from the image?")
        if user_input:
            self.bias_value = int(user_input)
        self.label_bias_var.set("bias = " + str(self.bias_value))
        self.root.update()

    def update_labels(self, file="", status=""):
        if file:
            self.current_file = file
        if status:
            self.current_status = status
        self.label_files_var.set(os.path.basename(self.current_file))
        self.label_status_var.set(self.current_status)
        self.label_bias_var.set(self.bias_value)
        self.root.update()

    def process(self):
        self.update_labels(status="running...")
        for file in self.loaded_files:

            # set and display
            self.update_labels(file=file)

            # process
            self.update_labels(status="load and debayer...")
            image = load_image(file)
            if self.set_write_pickle.get():
                self.update_labels(status="save pickle file...\n(takes some time but faster next time)")
                write_pickle(image, self.current_file)
            self.update_labels(status="edit image...")
            image_edit = edit_image(image, bias_value=self.bias_value)
            if self.opt_gradient.get():
                self.update_labels(status="correct gradient...")
                image_edit = corr_gradient(image_edit, self.current_file)
            if self.opt_histogram.get():
                self.update_labels(status="calculate histogram...")
                calc_histograms(image_edit, self.current_file, self.set_circular_hist.get())
            if self.opt_radprof.get():
                self.update_labels(status="calculate radial profile...")
                try:
                    sigma_clip = float(self.radio_clip.get())
                except:
                    sigma_clip = np.inf
                rad_profile_smoothed = calc_rad_profile(image_edit, self.current_file,
                                                        sigma_clip=sigma_clip,
                                                        extrapolate_max=self.set_extrapolate_max.get())
            if self.opt_synthflat.get():
                self.update_labels(status="export synthetic flat...")
                export_tif(rad_profile_smoothed, self.current_file, grey_flat=self.set_grey_flat.get(), tif_size=(image.shape[0], image.shape[1]))

            print("Finished file.")

        print("\nFinished.")
        self.update_labels(file=str(len(self.loaded_files)) + " files")
        self.update_labels(status="finished.")
        return


if __name__ == '__main__':
    new = NewGUI()

