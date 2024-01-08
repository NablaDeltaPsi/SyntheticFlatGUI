import tkinter as tk
import tkinter.font
import tkinter.messagebox # for pyinstaller
from tkinter.filedialog import askopenfilename
import glob, os, sys, copy, numpy as np, math
from configparser import ConfigParser
import rawpy, cv2
import pickle, bz2
from scipy.stats import sigmaclip
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
from functools import lru_cache
import datetime as dt

GUINAME = "SyntheticFlatGUI"
VERSION = '1.2'

# STATIC SETTINGS =============================================================
IGNORE_EDGE = 10          # ignore pixels close to the image edges
IGNORE_RADII = 5          # ignore pixels extreme radii (close to 0 and maximum)
RADIAL_RESOLUTION = 100000  # should be larger than 16 bit (65536)

RAW_TYPES = '.arw .crw .cr2 .cr3 .nef .raf .rw2'
IMAGE_TYPES = '.tif .tiff .jpeg .jpg .png'

# STATIC MAJOR FUNCTIONS =====================================================
def load_image(file):
    print("")
    print(file)
    print("Trying to load pickle file...")
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
        pickle.dump(im_deb, bz2.BZ2File(pickle_filename, 'wb'))
        pickle.dump(rawshape, bz2.BZ2File(pickle_filename_rawshape, 'wb'))


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


def nearest_neighbor_pixelmap(im_deb, file, resolution_factor=4):
    rows, cols, colors = im_deb.shape
    flat_pixel_values = []
    flat_maxneighbors = []
    for c in range(colors):
        maxneighbors = np.zeros((rows, cols))
        for n in range(rows):
            if n % 500 == 0:
                print("color: ", c, ", row: ", n)
            if n % resolution_factor == 0:
                continue
            for m in range(cols):
                if m % resolution_factor == 0:
                    continue
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


def calc_rad_profile(image, statistics=2, extrapolate_max=True, resolution_factor=4):

    # measure time
    start = dt.datetime.now()

    # acquire pixel values
    image_height = image.shape[0]
    image_width = image.shape[1]
    maxrad = int(dist_from_center(0, 0, image_height, image_width))
    radii = []
    rad_counts = {}
    rad_pixels = {}
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
        mask = rad_profile[:, 0] > IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile[mask, :]
        mask = rad_profile_cut[:, 0] < 1 - IGNORE_RADII / maxrad
        rad_profile_cut = rad_profile_cut[mask, :]
    else:
        rad_profile_cut = copy.deepcopy(rad_profile)

    # cut data inside maximum
    if extrapolate_max:
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
        if np.max(maxind) > int(rad_profile_cut.shape[0]/2):
            print("maximum index too large -> skip maximum cut")
        else:
            rad_profile_cut = rad_profile_cut[max(maxind):, :]
        # for c in range(3):
        #     rad_profile_cut[:min_maxind,c+1] = max(rad_profile_cut[:,c+1])

    # smooth data and interpolate
    radii = np.linspace(0, 1, RADIAL_RESOLUTION)
    rad_profile_smoothed = radii
    for c in range(3):
        y_new = savgol_filter(rad_profile_cut[:, c + 1], window_length=odd_int(rad_profile_cut.shape[0] / 10),
                                           polyorder=2, mode='interp')
        y_new = savgol_filter(y_new, window_length=odd_int(rad_profile_cut.shape[0] / 5), polyorder=2,
                                           mode='interp')

        # extrapolate inside cut max value
        if extrapolate_max:
            # add central value for interpolation
            x_new = list(rad_profile_cut[:, 0].flatten())
            y_new = list(y_new.flatten())
            xdist = x_new[1] - x_new[0]
            ylast = y_new[0]
            xlast = x_new[0]
            slope = (y_new[1] - y_new[0]) / xdist
            # slope = np.mean(np.diff(y_new[:int(len(y_new)/10)])) / xdist
            if slope > 0:
                slope = 0
            slopes.append(int(slope))
            xadd = np.arange(0, xlast * 0.999, xdist)
            # quadratic function with given value and slope at xlast and zero derivative at x=0
            yadd = ylast + slope / 2 * (xadd ** 2 / xlast - xlast)
            # print(np.diff(yadd) / xdist) # derivative for troubleshooting
            x_new = list(xadd) + x_new
            y_new = list(yadd) + y_new
        else:
            x_new = rad_profile_cut[:, 0]

        # interpolate onto continuous values
        f = interp1d(x_new, y_new, kind='quadratic', fill_value='extrapolate')
        y_new = f(radii)
        rad_profile_smoothed = np.column_stack((rad_profile_smoothed, y_new))

    # print extrapolation info
    if extrapolate_max:
        print("extrapolated with slopes: ", slopes)

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


def calc_synthetic_flat(rad_profile, grey_flat=False, tif_size=(4024, 6024), max_value=1):

    # measure time
    start = dt.datetime.now()

    # initialize image
    height, width = sorted(tif_size)
    halfheight = int(height / 2)
    halfwidth = int(width / 2)
    im_syn = np.zeros((int(height), int(width)))

    # normalize (again)
    for c in range(3):
        rad_profile[:, c + 1] = rad_profile[:, c + 1] / np.max(rad_profile[:, c + 1])

    # match profile to output size
    maxrad_this = dist_from_center(0, 0, height, width)

    # iterate image and write pixels
    for di in range(halfheight):
        for dj in range(halfwidth):

            # absolute pixel indices
            i = halfheight + di
            j = halfwidth + dj

            # halfheight and halfwidth are pixel references
            # but the true center is in between pixels
            halfheight_ = halfheight - 0.5
            halfwidth_  = halfwidth  - 0.5

            # check already written
            if not im_syn[i, j] == 0:
                continue

            # radial position of pixel
            r = dist_from_center(i, j, height, width) / maxrad_this

            # search match within small range in radial profile
            r_index = int((RADIAL_RESOLUTION - 1) * r)

            # write pixel brightness in all quadrants
            # since the radii of the multiples are already known, use them here (r_index * k)
            # write_flat_pixel(im_syn, rad_profile, grey_flat, r_index, i, j)
            for k in range(1, halfheight):
                if di * k < halfheight_ and dj * k < halfwidth_:
                    di_ = di * k + 0.5
                    dj_ = dj * k + 0.5
                    write_flat_pixel(im_syn, rad_profile, grey_flat, r_index * k, halfheight_ + di_, halfwidth_ + dj_)
                    write_flat_pixel(im_syn, rad_profile, grey_flat, r_index * k, halfheight_ + di_, halfwidth_ - dj_)
                    write_flat_pixel(im_syn, rad_profile, grey_flat, r_index * k, halfheight_ - di_, halfwidth_ + dj_)
                    write_flat_pixel(im_syn, rad_profile, grey_flat, r_index * k, halfheight_ - di_, halfwidth_ - dj_)
                else:
                    break

    # convert to 16 bit
    im_syn = im_syn / np.max(im_syn)
    print("zeros (%):", "{:.10f}".format(100.0 - 100.0 * np.count_nonzero(im_syn) / np.prod(im_syn.shape)))
    print("minimum:  ", np.min(im_syn[im_syn > 0]))
    im_syn = max_value * (2 ** 16 - 1) * im_syn
    im_syn = im_syn.astype(np.uint16)

    # print execution time
    stop = dt.datetime.now()
    print("execution time:", int((stop-start).total_seconds() + 0.5), "seconds")

    return im_syn



# STATIC MINOR FUNCTIONS =====================================================

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
                db_image[int((n + 0) / 2), int((m + 0) / 2), 0] = int(image[n, m])  # R
            if n % 2 == 0 and m % 2 == 1:  # rechts oben
                db_image[int((n + 0) / 2), int((m - 1) / 2), 1] = int(image[n, m])  # G
            if n % 2 == 1 and m % 2 == 0:  # links unten
                db_image[int((n - 1) / 2), int((m + 0) / 2), 2] = int(image[n, m])  # G
            if n % 2 == 1 and m % 2 == 1:  # rechts unten
                db_image[int((n - 1) / 2), int((m - 1) / 2), 3] = int(image[n, m])  # B
    if not separate_green:
        db_image = merge_green(db_image)
        print("shape", db_image.shape)
    return db_image


def merge_green(image):
    image_r = image[:, :, 0]
    image_g = ((image[:, :, 1] + image[:, :, 2]) / 2).astype(int)
    image_b = image[:, :, 3]
    image = np.stack((image_r, image_g, image_b), axis=2)
    return image


def bayer(image):
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


def dist_from_center(i, j, height, width):
    dx = int(abs(i - height / 2))
    dy = int(abs(j - width / 2))
    dx, dy = sorted([dx, dy])  # sort helps caching of symmetric function
    return cached_dist(dx, dy)


@lru_cache(maxsize=None)
def cached_dist(dx, dy):
    return math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))


def create_folder(file, folder_name):
    savepath = os.path.dirname(file) + os.sep + folder_name
    try:
        os.mkdir(savepath)
    except:
        pass
    return savepath


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


def write_tif_image(image, original_file, folder, suffix):
    if len(image.shape) >= 3:
        # input is debayered, needs to be bayered
        image_write = bayer(image)
    else:
        # input is already bayered
        image_write = image
    print("write image", suffix, "input shape", image.shape)
    print("write image", suffix, "output shape", image_write.shape)
    savepath = create_folder(original_file, folder)
    image_write = np.float32(image_write / np.max(image_write))
    cv2.imwrite(savepath + os.sep + os.path.basename(original_file).split('.')[0] + suffix + ".tif", image_write)


def write_csv(data, original_file, folder, suffix):
    savepath = create_folder(original_file, folder)
    fmt = '%.5f', '%.5f', '%.5f', '%.5f'
    np.savetxt(savepath + os.sep + os.path.basename(original_file).split('.')[0] + suffix + ".csv",
               data, delimiter=",", fmt=fmt)



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
        self.opt_pixelmap  = tk.BooleanVar()
        self.opt_histogram = tk.BooleanVar()
        self.opt_radprof   = tk.BooleanVar()
        self.opt_synthflat = tk.BooleanVar()
        options.add_checkbutton(label="Correct gradient", onvalue=1, offvalue=0, variable=self.opt_gradient)
        options.add_checkbutton(label="Nearest neighbor pixelmap", onvalue=1, offvalue=0, variable=self.opt_pixelmap)
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
        self.set_scale_flat      = tk.BooleanVar()
        settings.add_checkbutton(label="Write pickle file", onvalue=1, offvalue=0, variable=self.set_write_pickle)
        settings.add_checkbutton(label="Histogram of largest circle", onvalue=1, offvalue=0, variable=self.set_circular_hist)
        settings.add_checkbutton(label="Extrapolate inside max", onvalue=1, offvalue=0, variable=self.set_extrapolate_max)
        settings.add_checkbutton(label="Export corrected input images", onvalue=1, offvalue=0, variable=self.set_export_corr_input)
        settings.add_checkbutton(label="Export synthetic flat as grey", onvalue=1, offvalue=0, variable=self.set_grey_flat)
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
        config_object["OPTIONS"]["opt_pixelmap"]  = str(self.opt_pixelmap.get())
        config_object["OPTIONS"]["opt_histogram"] = str(self.opt_histogram.get())
        config_object["OPTIONS"]["opt_radprof"]   = str(self.opt_radprof.get())
        config_object["OPTIONS"]["opt_synthflat"] = str(self.opt_synthflat.get())

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"]     = str(self.set_write_pickle.get())
        config_object["SETTINGS"]["set_export_corr_input"]  = str(self.set_export_corr_input.get())
        config_object["SETTINGS"]["set_circular_hist"]    = str(self.set_circular_hist.get())
        config_object["SETTINGS"]["set_grey_flat"]        = str(self.set_grey_flat.get())
        config_object["SETTINGS"]["set_extrapolate_max"]  = str(self.set_extrapolate_max.get())
        config_object["SETTINGS"]["set_scale_flat"]       = str(self.set_scale_flat.get())

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
        config_object["OPTIONS"]["opt_pixelmap"] = 'False'
        config_object["OPTIONS"]["opt_histogram"] = 'False'
        config_object["OPTIONS"]["opt_radprof"] = 'True'
        config_object["OPTIONS"]["opt_synthflat"] = 'True'

        config_object["SETTINGS"] = {}
        config_object["SETTINGS"]["set_write_pickle"] = 'True'
        config_object["SETTINGS"]["set_export_corr_input"] = 'True'
        config_object["SETTINGS"]["set_circular_hist"] = 'True'
        config_object["SETTINGS"]["set_grey_flat"] = 'False'
        config_object["SETTINGS"]["set_extrapolate_max"] = 'True'
        config_object["SETTINGS"]["set_scale_flat"] = 'False'

        self.apply_config(config_object)
        self.update_labels(status="ready")

    def apply_config(self, config_object):
        self.root.geometry(config_object["BASICS"]["window size"])
        self.lastpath = config_object["BASICS"]["lastpath"]
        self.radio_statistics.set(config_object["BASICS"]["radio_statistics"])
        self.radio_resolution.set(config_object["BASICS"]["radio_resolution"])
        self.bias_value = int(config_object["BASICS"]["bias_value"])

        self.opt_gradient.set(config_object["OPTIONS"]["opt_gradient"]   == 'True')
        self.opt_pixelmap.set(config_object["OPTIONS"]["opt_pixelmap"]   == 'True')
        self.opt_histogram.set(config_object["OPTIONS"]["opt_histogram"] == 'True')
        self.opt_radprof.set(config_object["OPTIONS"]["opt_radprof"]     == 'True')
        self.opt_synthflat.set(config_object["OPTIONS"]["opt_synthflat"] == 'True')

        self.set_write_pickle.set(config_object["SETTINGS"]["set_write_pickle"]   == 'True')
        self.set_export_corr_input.set(config_object["SETTINGS"]["set_export_corr_input"]   == 'True')
        self.set_circular_hist.set(config_object["SETTINGS"]["set_circular_hist"] == 'True')
        self.set_grey_flat.set(config_object["SETTINGS"]["set_grey_flat"]         == 'True')
        self.set_extrapolate_max.set(config_object["SETTINGS"]["set_extrapolate_max"] == 'True')
        self.set_scale_flat.set(config_object["SETTINGS"]["set_scale_flat"]       == 'True')

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
        self.update_labels()
        return

    def ask_bias_file(self):
        if self.running:
            return
        self.update_labels(status="calc bias...")
        user_input_file = askopenfilename(initialdir=self.lastpath, multiple=True,
              filetypes=[('RAW format (supported)', RAW_TYPES), ('Image format (not supported)', IMAGE_TYPES), ('all', '.*')])
        im_raw = rawpy.imread(user_input_file[0]).raw_image_visible
        self.bias_value = int(sigma_clip_mean(im_raw))
        self.label_bias_var.set(self.bias_value)
        self.update_labels(status="ready")
        return

    def update_labels(self, file="", status=""):
        if file:
            self.label_files_var.set(file)
        if status:
            self.label_status_var.set(status)
            print("status", status)
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
                #print(dt.datetime.now())
                self.update_labels(status="load...")
                image, rawshape = load_image(file)
                if self.check_stop(): return

                # write pickle
                if self.set_write_pickle.get():
                    self.update_labels(status="save pickle file...")
                    write_pickle(image, rawshape, file)
                    if self.check_stop(): return

                # write original image
                if self.set_export_corr_input.get():
                    self.update_labels(status="write original tif...")
                    write_tif_image(image, file, "TIF_images", "_0_input")
                    if self.check_stop(): return

                # gradient
                if self.opt_gradient.get():
                    self.update_labels(status="calc gradient...")
                    image = corr_gradient(image, resolution_factor=resolution_factor)
                    if self.check_stop(): return

                    # write gradient-corrected image
                    if self.set_export_corr_input.get():
                        self.update_labels(status="write gradcorr tif...")
                        write_tif_image(image, file, "TIF_images", "_1_gradcorr")
                        if self.check_stop(): return

                # subtract bias
                self.update_labels(status="subtract bias...")
                image = image - self.bias_value
                if self.check_stop(): return

                # pixelmap
                if self.opt_pixelmap.get():
                    self.update_labels(status="calc pixelmap...")
                    nearest_neighbor_pixelmap(image, file, resolution_factor=resolution_factor)
                    if self.check_stop(): return

                # histogram
                if self.opt_histogram.get():
                    self.update_labels(status="calc histogram...")
                    data = calc_histograms(image, self.set_circular_hist.get())
                    write_csv(data, file, "CSV_files", "_histogram")
                    if self.check_stop(): return

                # radial profile
                if self.opt_radprof.get():
                    self.update_labels(status="calc radial profile...")
                    radprof1, radprof2, radprof3, radprof4 = calc_rad_profile(image,
                                                            statistics=self.radio_statistics.get(),
                                                            extrapolate_max=self.set_extrapolate_max.get(),
                                                            resolution_factor=resolution_factor)
                    write_csv(radprof1, file, "CSV_files", "_radprof_0_raw_mean")
                    write_csv(radprof2, file, "CSV_files", "_radprof_1_clipped")
                    write_csv(radprof3, file, "CSV_files", "_radprof_2_cut")
                    write_csv(radprof4, file, "CSV_files", "_radprof_3_smooth")
                    if self.check_stop(): return

                # synthetic flat
                if self.opt_synthflat.get():

                    # output scaling
                    if self.set_scale_flat.get():
                        max_value = sigma_clip_mean(image) / 16384
                    else:
                        max_value = 1

                    # calculate synthetic flat
                    self.update_labels(status="calc synthetic flat...")
                    image_flat = calc_synthetic_flat(radprof4,
                               grey_flat=self.set_grey_flat.get(),
                               tif_size=(rawshape[0], rawshape[1]),
                               max_value=max_value)
                    if self.check_stop(): return

                    # write synthetic flat tif
                    self.update_labels(status="write synthflat tif...")
                    write_tif_image(image_flat, file, "TIF_images", "_2_synthflat")
                    if self.check_stop(): return

                    # correct input
                    if self.set_export_corr_input.get():
                        self.update_labels(status="export flat-corrected...")
                        image = bayer(image)
                        write_tif_image(image / image_flat, file, "TIF_images", "_3_flatcorr")

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
    #print(own_gcd(40, 60))


