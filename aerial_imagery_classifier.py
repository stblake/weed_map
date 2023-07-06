
# Aerial Imagery Classification Model

# Sam Blake, started 23 December 2022

import time
import os
import glob
import math
import numpy as np
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.stats import halfnorm

from tqdm import tqdm

import numba
from numba import jit, prange, set_num_threads

plt.rcParams["figure.figsize"] = (10, 6)


# Some quick plotting routines. 

def show_greyscale_image(image, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for greyscale images using matplotlib. \
    Options are title = None, axis = 'off'."""
    fig, ax = plt.subplots() 
    ax.imshow(image, interpolation='nearest', cmap=plt.get_cmap('Greys'))
    ax.axis(axis)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return



def show_bgr_image(image, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for BGR images using matplotlib. \
    Options are title = None, axis = 'off'."""
    fig, ax = plt.subplots() 
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis(axis)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return



def show_hls_image(image, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for HLS images using matplotlib. \
    Options are title = None, axis = 'off'."""
    fig, ax = plt.subplots() 
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_HLS2RGB))
    ax.axis(axis)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return



def show_lab_image(image, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for Lab images using matplotlib. \
    Options are title = None, axis = 'off'."""
    fig, ax = plt.subplots() 
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_Lab2RGB))
    ax.axis(axis)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return



def show_bgra_image(image, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for BGRA (blue, green, red, alpha) images \
    using matplotlib. Options are title = None, axis = 'off'."""
    fig, ax = plt.subplots() 
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    ax.axis(axis)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return



def heatmap(image, cmap = 'hot_r', colorbar = False, title = None, axis = 'off', 
    save_figure = False, file_name = 'temp', file_extension = 'PNG'):
    """Quick plotting routine for heat maps using matplotlib. \
    Options are colorbar = False, title = None, axis = 'off'."""
    fig, ax = plt.subplots()
    im = ax.imshow(image, interpolation='bilinear', cmap=cmap, origin='upper')
    ax.axis(axis)
    if colorbar:
        plt.colorbar(im, fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
    if title is not None:
        ax.set_title(title)
    if save_figure:
        plt.savefig(f'{file_name}.{file_extension}')
    plt.ioff()
    plt.show()
    return 



def show_bgr_image_histogram(image, title = None):
    """Histogram of blue, green and red components of an image. Currently under development."""
    blue = image[:,:,0]
    green = image[:,:,1]
    red = image[:,:,2]

    fig, ax = plt.subplots()
    plt.hist(blue.ravel(),  256, [0,256], color = 'blue', histtype = 'step')
    plt.hist(green.ravel(), 256, [0,256], color = 'green', histtype = 'step')
    plt.hist(red.ravel(),   256, [0,256], color = 'red', histtype = 'step')
    plt.show()
    return 



def histogram_equalise(img):
    """Adaptive histogram equalisation for BGR images."""
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    luma = img_hls[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    luma_equalized = clahe.apply(luma)
    img_hls[:,:,1] = luma_equalized
    img_equalised = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    return img_equalised



def plot_spectra_table_bgr(spectra, ncols = 25, npx_per_row = 32, title = None):
    """Plot table of spectra in BGR colour space."""
    nrows = int(np.ceil(len(spectra)/ncols))
    image = np.zeros((npx_per_row*nrows, npx_per_row*ncols, 3), dtype = np.uint8)
    image[:,:,:] = 255 # white background
    
    # Sort spectra by brightness. (I don't especially like this sorting function.)
    # Ref: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color 
    brightness = [0.2126*R + 0.7152*G + 0.0722*B for B,G,R in spectra]

    sorted_index = sorted(range(len(brightness)), key=lambda k: brightness[k])
    spectra_sorted = spectra[sorted_index]
    
    for k,spec in enumerate(spectra_sorted):
        i,j = k//ncols, k%ncols
        image[npx_per_row*i:npx_per_row*(i + 1), npx_per_row*j:npx_per_row*(j + 1), :] = spec

    show_bgr_image(image, title = title)
    
    return 



def plot_spectra_table_hls(spectra, ncols = 25, npx_per_row = 32, title = None):
    """Plot table of spectra in HLS colour space."""

    nrows = int(np.ceil(len(spectra)/ncols))
    image = np.zeros((npx_per_row*nrows, npx_per_row*ncols, 3), dtype = np.uint8)
    image[:,:,:] = 255 # white background
    
    brightness = [L for H,L,S in spectra]

    sorted_index = sorted(range(len(brightness)), key=lambda k: brightness[k])
    spectra_sorted = spectra[sorted_index]
    
    for k,spec in enumerate(spectra_sorted):
        i,j = k//ncols, k%ncols
        image[npx_per_row*i:npx_per_row*(i + 1), npx_per_row*j:npx_per_row*(j + 1), :] = spec

    show_hls_image(image, title = title)
    
    return 



def plot_spectra_table_lab(spectra, ncols = 25, npx_per_row = 32, title = None):
    """Plot table of spectra in Lab colour space."""
    
    nrows = int(np.ceil(len(spectra)/ncols))
    image = np.zeros((npx_per_row*nrows, npx_per_row*ncols, 3), dtype = np.uint8)
    # White background in Lab-space -- cv2.cvtColor(np.array([[[255,255,255]]], dtype=np.uint8), cv2.COLOR_BGR2Lab)
    image[:,:,0] = 255 
    image[:,:,1] = 128
    image[:,:,2] = 128

    brightness = [L for L,a,b in spectra]

    sorted_index = sorted(range(len(brightness)), key=lambda k: brightness[k])
    spectra_sorted = spectra[sorted_index]
    
    for k,spec in enumerate(spectra_sorted):
        i,j = k//ncols, k%ncols
        image[npx_per_row*i:npx_per_row*(i + 1), npx_per_row*j:npx_per_row*(j + 1), :] = spec

    show_lab_image(image, title = title)
    
    return 



def normalise(a):
    """Rescale an array linearly such that np.amin(a) == 0 and np.amax(a) == 1."""
    return (a - np.amin(a))/(np.amax(a) - np.amin(a))



def remove_near_duplicate_spectra(array, tol = 1):
    """Near-duplicate spectra will skew the results, so we remove them."""
    rounded = tol*np.rint(array/tol)
    _,indices,counts = np.unique(rounded, axis = 0, return_index = True, return_counts = True)
    return array[indices], counts/np.sum(counts)



def remove_common_spectra(spectra_a, spectra_b, tol = 1):
    """Spectra common to both samples will only skew results, so we remove them prior \
    before computing their spectral contribution to the detection."""
    rounded_a = tol*np.rint(spectra_a/tol)
    rounded_b = tol*np.rint(spectra_b/tol)
    hash_a = [hash(arr.data.tobytes()) for arr in rounded_a]
    hash_b = [hash(arr.data.tobytes()) for arr in rounded_b]
    indx_a = [i for i,h in enumerate(hash_a) if h not in hash_b]
    indx_b = [i for i,h in enumerate(hash_b) if h not in hash_a]
    return spectra_a[indx_a], spectra_b[indx_b]



@jit(nopython=True,fastmath=True)
def image_spectra_diff(image, spectra, classification):
    """Computes the RMSE of each pixel in the image with the sample spectra datasets."""

    nrows,ncols = classification.shape

    for i in range(nrows):
        for j in range(ncols):

            spectra_min = 1.

            for s in spectra:
                # root-mean-square error
                diff = (image[i,j,0] - s[0])**2 + \
                       (image[i,j,1] - s[1])**2 + \
                       (image[i,j,2] - s[2])**2
                diff /= 3.
                diff = np.sqrt(diff)
                # min error -- our analogue for the closest spectral match
                if diff < spectra_min:
                    spectra_min = diff

            classification[i,j] = spectra_min

    return



@jit(nopython=True,fastmath=True)
def image_spectra_diff_weighted(image, spectra, weights, classification):
    """Computes the RMSE of each pixel in the image with the sample spectra datasets."""

    nrows,ncols = classification.shape

    for i in range(nrows):
        for j in range(ncols):

            spectra_min = 1e10

            for s,w in zip(spectra, weights):
                # root-mean-square error
                diff = (image[i,j,0] - s[0])**2 + \
                       (image[i,j,1] - s[1])**2 + \
                       (image[i,j,2] - s[2])**2
                diff /= 3.
                diff = np.sqrt(diff)/w
                # min error -- our analogue for the closest spectral match
                if diff < spectra_min:
                    spectra_min = diff

            classification[i,j] = spectra_min

    return



def denoise(image, classification_confidence, morph_close = 20, morph_open = 20):
    """denoise requires a 4-channel BGRA image."""
    img_bw = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    _,img_binarized = cv2.threshold(img_bw, 254, 255, cv2.THRESH_BINARY)

    img_binarized = 255*(img_binarized > 0).astype(np.uint8)

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))

    mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    mask = mask < 255

    denoised = image.copy()
    denoised[:,:,3] *= mask

    classification_confidence = np.where(mask, classification_confidence, 0)

    return denoised, classification_confidence



# Note: This code assumes the input is a BGR, 8-bit, 3-channel colour space. 

# Reference for Lab colour space: https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab

def imagery_statistical_binary_classification(
                     input_image_bgr, 
                     samples_A_bgr, 
                     samples_B_bgr, 
                     image_blur_kernel_size = (3,3),
                     image_resize_ratio = 10,
                     template_blur_kernel_size = (3,3), 
                     template_resize_size = 32, 
                     n_sigma_thresholds = [1,2,3], 
                     duplicate_tolerance = 1, 
                     common_tolerance = 1, 
                     morph_close = 20, 
                     morph_open = 20,
                     equalise = True, 
                     denoise_classification_map = True, 
                     dilation_size = 0, 
                     colour_space = 'Lab',
                     weighted_differencing = True,
                     z_score = False,
                     limit_outliers = True, 
                     half_normal_dist = False,
                     extents = None,
                     imagery_id = None,
                     export_histogram = False, 
                     export_heatmap = False, 
                     preprocess_image = True, 
                     plotting = False,
                     verbose = False):

    # Image ID string for plot labels. 
    if imagery_id is not None:
        id_str = os.path.basename(imagery_id).replace('.JPG','').replace('.tif','')
    else:
        id_str = 'UNKNOWN'

    if plotting:
        show_bgr_image(input_image_bgr, title = f'{id_str} - original image')

    # Histogram equalisation. 
    if equalise:
        image_bgr = histogram_equalise(input_image_bgr)
        if plotting:
            show_bgr_image(image_bgr, title = f'{id_str} - histogram equalised')
    else:
        image_bgr = input_image_bgr.copy()

    # Convert from BGR to HSL. 
    if colour_space == 'Lab':
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        samples_A = [cv2.cvtColor(sample, cv2.COLOR_BGR2Lab) for sample in samples_A_bgr]
        samples_B = [cv2.cvtColor(sample, cv2.COLOR_BGR2Lab) for sample in samples_B_bgr]
    elif colour_space == 'HLS':
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        samples_A = [cv2.cvtColor(sample, cv2.COLOR_BGR2HLS) for sample in samples_A_bgr]
        samples_B = [cv2.cvtColor(sample, cv2.COLOR_BGR2HLS) for sample in samples_B_bgr]
    else:
        print(f'ERROR: unknown colour space \'{colour_space}\'. Only \'Lab\' and \'HLS\' are supported.')
        return

    # Check shape of input image. (Should be a BGR, 3-channel image.)
    if len(image.shape) != 3 or image.shape[-1] != 3:
        print('ERROR: the image should be a two-dimensional, 3-channel, BGR image.')
        return 

    # Check the shape of each of the spectral samples of A and B. 
    for sample in samples_A:
        if len(sample.shape) != 3 or sample.shape[-1] != 3:
            print('ERROR: the spectral sample in samples_A should be a two-dimensional, 3-channel, BGR image.')
            return

    for sample in samples_B:
        if len(sample.shape) != 3 or sample.shape[-1] != 3:
            print('ERROR: the spectral sample in samples_B should be a two-dimensional, 3-channel, BGR image.')
            return

    # Prepare weed and crop images. 
    samples_A_preprocessed = []
    for sample in samples_A:
        size = (template_resize_size, int(float(sample.shape[1])/sample.shape[0]*template_resize_size))
        pre = cv2.blur(sample, template_blur_kernel_size)
        pre = cv2.resize(pre, size, interpolation = cv2.INTER_AREA)
        samples_A_preprocessed.append(pre.reshape(-1,3))
    samples_A_preprocessed = np.concatenate(samples_A_preprocessed, axis = 0)

    samples_B_preprocessed = []
    for sample in samples_B:
        size = (template_resize_size, int(float(sample.shape[1])/sample.shape[0]*template_resize_size))
        pre = cv2.blur(sample, template_blur_kernel_size)
        pre = cv2.resize(pre, size, interpolation = cv2.INTER_AREA)
        samples_B_preprocessed.append(pre.reshape(-1,3))
    samples_B_preprocessed = np.concatenate(samples_B_preprocessed, axis = 0)
    
    samples_A_preprocessed = samples_A_preprocessed.astype(np.float32)
    samples_B_preprocessed = samples_B_preprocessed.astype(np.float32)

    # Remove similar spectra common to both weeds and crop. 
    samples_A_preprocessed, samples_B_preprocessed = remove_common_spectra(\
          samples_A_preprocessed, samples_B_preprocessed, tol = common_tolerance)
    
    # Remove near duplicates. 
    samples_A_preprocessed, samples_A_weights = remove_near_duplicate_spectra(\
        samples_A_preprocessed, tol = duplicate_tolerance)
    samples_B_preprocessed, samples_B_weights  = remove_near_duplicate_spectra(\
        samples_B_preprocessed, tol = duplicate_tolerance)
    
    # TODO: If too many spectra are common to both sample sets, then we should issue a warning 
    # and possibly abort. This would prevent misuse. 
    if len(samples_A_preprocessed) == 0 or len(samples_B_preprocessed) == 0:
        print(f'ERROR: sample spectra are not sufficiently distinct for classification. Perhaps \
try decreasing \'duplicate_tolerance\', which is currently {duplicate_tolerance}.')
        return

    if verbose:
        print(f'number of distinct spectra in sample A is {len(samples_A_preprocessed)}')
        print(f'number of distinct spectra in sample B is {len(samples_B_preprocessed)}')

    # Plot color patchwork of the distinct spectra. 
    if plotting:
        if colour_space == 'Lab':
            plot_spectra_table_lab(samples_A_preprocessed, title = f'{id_str} - distinct spectra of sample A')
            plot_spectra_table_lab(samples_B_preprocessed, title = f'{id_str} - distinct spectra of sample B')
        elif colour_space == 'HLS':
            plot_spectra_table_hls(samples_A_preprocessed, title = f'{id_str} - distinct spectra of sample A')
            plot_spectra_table_hls(samples_B_preprocessed, title = f'{id_str} - distinct spectra of sample B')

    # Normalise colour space.
    if colour_space == 'Lab':
        samples_A_preprocessed /= np.sqrt(3)*255. 
        samples_B_preprocessed /= np.sqrt(3)*255.
    elif colour_space == 'HLS':
        samples_A_preprocessed[:,0] /= 255.
        samples_A_preprocessed[:,1] /= 255.
        samples_A_preprocessed[:,2] /= 180.

        samples_B_preprocessed[:,0] /= 255.
        samples_B_preprocessed[:,1] /= 255.
        samples_B_preprocessed[:,2] /= 180.

    # Preprocess image. 
    if preprocess_image:
        image_pre = cv2.blur(image, image_blur_kernel_size)
        image_pre = cv2.resize(image_pre, 
            (image.shape[1]//image_resize_ratio, image.shape[0]//image_resize_ratio), 
            interpolation = cv2.INTER_AREA)
    else:
        image_pre = image.copy()

    if plotting:
        if colour_space == 'Lab':
            show_lab_image(image_pre, title = f'{id_str} - preprocessed image')
        elif colour_space == 'HLS':
            show_hls_image(image_pre, title = f'{id_str} - preprocessed image')

    # Normalise colour space of preprocessed image. 
    image_pre = image_pre.astype(np.float32)
    if colour_space == 'Lab':
        image_pre /= np.sqrt(3)*255. # Normalise for Lab.
    elif colour_space == 'HLS':
        image_pre[:,:,0] /= 255. # Normalise for HLS. 
        image_pre[:,:,1] /= 255.
        image_pre[:,:,2] /= 180.

    # Compute classification based on spectral differences. 
    density_A = np.zeros((image_pre.shape[0], image_pre.shape[1]), dtype = np.float32)
    density_B = np.zeros((image_pre.shape[0], image_pre.shape[1]), dtype = np.float32)

    if weighted_differencing:
        image_spectra_diff_weighted(image_pre, samples_A_preprocessed, samples_A_weights, density_A)
        image_spectra_diff_weighted(image_pre, samples_B_preprocessed, samples_B_weights, density_B)
    else:
        image_spectra_diff(image_pre, samples_A_preprocessed, density_A)
        image_spectra_diff(image_pre, samples_B_preprocessed, density_B)

    # Experimental - smooth spectral match density.
    density_A = cv2.blur(density_A, template_blur_kernel_size)
    density_B = cv2.blur(density_B, template_blur_kernel_size)

    # Experimental - limit range of outliers. 
    if limit_outliers:
        threshold_A = np.mean(density_A) + 2.0*np.std(density_A)
        threshold_B = np.mean(density_B) + 2.0*np.std(density_B)
        threshold = min(threshold_A, threshold_B)
        density_A[density_A > threshold] = np.nan
        density_B[density_B > threshold] = np.nan

    # Experimental
    if z_score:
        density_A = (density_A - np.mean(density_A))/np.std(density_A)
        density_B = (density_B - np.mean(density_B))/np.std(density_B)

    density = density_A - density_B
    
    # Resize to original image size. Using bi-linear interpolation here, and not a 
    # higher order interpolation is important. 
    density_A = cv2.resize(density_A, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
    density_B = cv2.resize(density_B, (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
    density   = cv2.resize(density,   (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)
   
    density_A = cv2.blur(density_A, template_blur_kernel_size)
    density_B = cv2.blur(density_B, template_blur_kernel_size)
    density   = cv2.blur(density,   template_blur_kernel_size)

    # Mask no_data (0,0,0) for geotiff. For some drones (9,9,9) is no-data value.
    image_sum = np.sum(input_image_bgr, axis = 2)
    density_A[image_sum < 10] = np.nan
    density_B[image_sum < 10] = np.nan
    density[image_sum < 10] = np.nan

    # Plot histograms of spectral match difference. 
    if plotting:
        fig, ax = plt.subplots()
        plt.hist(density_A.ravel(), bins=80)
        plt.title(f'{id_str} - histogram of differences for spectra A')
        plt.show()

        fig, ax = plt.subplots()
        plt.hist(density_B.ravel(), bins=80)
        plt.title(f'{id_str} - histogram of differences for spectra B')
        plt.show()

        if not export_histogram:
            fig, ax = plt.subplots()
            plt.hist(density.ravel(), bins=80)
            plt.title(f'{id_str} - histogram of spectral match difference')
            plt.show()

    if export_histogram:
        fig, ax = plt.subplots()
        plt.hist(density.ravel(), bins=80)
        plt.title(f'{id_str} - histogram of spectral match difference')
        plt.savefig(f'{id_str}_MATCH_HISTOGRAM.PNG')

    # Remove spectra which are extreme/outlier non-matches. 
    if not half_normal_dist:
        upper_cutoff = -np.amin(density)
        density[density > upper_cutoff] = np.nan

        # Plot histograms of spectral match difference. 
        if plotting:
            fig, ax = plt.subplots()
            plt.hist(density.ravel(), bins=80, color='salmon')
            plt.title(f'{id_str} - histogram of spectral match difference after removing outliers')
            plt.show()

    # Plot heatmap of density. 
    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(density_A, interpolation='nearest', cmap='jet_r',
                       origin='upper',
                       vmin=np.nanmin(density_A), 
                       vmax=np.nanmax(density_A),
                       extent=extents)
        if extents == None:
            ax.axis('off')
        ax.ticklabel_format(useOffset=False)
        plt.title(f'{id_str} - image RMSE with spectra A')
        plt.colorbar(im,fraction=0.046*density_A.shape[0]/density_A.shape[1], pad=0.04)
        plt.show()

        fig, ax = plt.subplots()
        im = ax.imshow(density_B, interpolation='nearest', cmap='jet_r',
                       origin='upper',
                       vmin=np.nanmin(density_B), 
                       vmax=np.nanmax(density_B),
                       extent=extents)
        if extents == None:
            ax.axis('off')
        ax.ticklabel_format(useOffset=False)
        plt.title(f'{id_str} - image RMSE with spectra B')
        plt.colorbar(im,fraction=0.046*density_B.shape[0]/density_B.shape[1], pad=0.04)
        plt.show()

        if not export_heatmap:
            fig, ax = plt.subplots()
            im = ax.imshow(density, interpolation='bilinear', cmap='RdBu',
                           origin='upper',
                           vmin=np.nanmin(density), 
                           vmax=np.nanmax(density),
                           extent=extents)
            if extents == None:
                ax.axis('off')
            ax.ticklabel_format(useOffset=False)
            plt.title(f'{id_str} - RMS difference of image with spectra')
            plt.colorbar(im,fraction=0.046*density.shape[0]/density.shape[1], pad=0.04)
            plt.show()

    if export_heatmap:
        fig, ax = plt.subplots()
        im = ax.imshow(density, interpolation='bilinear', cmap='RdBu',
                       origin='upper',
                       vmin=np.nanmin(density), 
                       vmax=np.nanmax(density),
                       extent=extents)
        if extents == None:
            ax.axis('off')
        ax.ticklabel_format(useOffset=False)
        plt.title(f'{id_str} - RMS difference of image with spectra')
        plt.colorbar(im,fraction=0.046*density.shape[0]/density.shape[1], pad=0.04)
        plt.savefig(f'{id_str}_HEATMAP.PNG')

    # Compute std (assumes a half-normal distribution.) 
    if half_normal_dist:
        density_half = np.concatenate([density[density < 0.], -density[density < 0.]])
        mean, std = 0., np.nanstd(density_half)
    else:
        mean, std = 0., np.nanstd(density)

    if verbose:
        print(f'mean = {mean:.4f}, std = {std:.4f}')

    # Compute percentage of image in each confidence interval (relative to entire image).
    n_total_pixels = image.shape[0]*image.shape[1]
    if verbose: 
        n_pc_total = 0.
        for i,n_sigma in enumerate(n_sigma_thresholds,1):
            n_px = np.sum(density < mean - n_sigma*std)
            n_pc = 100.*n_px/n_total_pixels
            n_pc_total += n_pc
            print(f'{n_px:8} [px] in {n_sigma:.2f} CI, {n_pc:.4f} [%] of all pixels. {n_pc_total:.4f} [%] cumulative total.')

        print(f'Sample not detected in {100. - n_pc_total:.2f} [%] of all pixels.')

    # Create (RGBA) classification image. 
    cmap = mpl.cm.get_cmap('hot_r')
    colours255 = [np.append(np.round(255*np.array(cmap(r)[:3])[::-1]), 255) for r in np.linspace(0.,0.66,len(n_sigma_thresholds) + 1)]

    classification_image = np.full(image.shape, 255, dtype = np.uint8) 
    classification_image = cv2.cvtColor(classification_image, cv2.COLOR_RGB2RGBA) # alpha channel
    classification_image[:,:,3] = 0

    classification_confidence = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)

    for i,n_sigma in enumerate(n_sigma_thresholds,1):
        classification_image[density < mean - n_sigma*std] = colours255[i]
        classification_confidence[density < mean - n_sigma*std] = int(10*n_sigma)

    if plotting:
        z_scored = density.copy()
        z_scored = (density - mean)/std
        fig, ax = plt.subplots()
        im = ax.imshow(z_scored, interpolation='bilinear', cmap='RdBu',
                       origin='upper',
                       vmin=-4, vmax=4,
                       extent = extents)
        if extents == None:
            ax.axis('off')
        ax.ticklabel_format(useOffset=False)
        plt.title(f'{id_str} - Z-score of Classification Confidence')
        plt.colorbar(im,fraction=0.046*z_scored.shape[0]/z_scored.shape[1], pad=0.04)
        plt.savefig(f'{id_str}_ZSCORE.PNG')
        plt.show()

        show_bgra_image(classification_image, title = f'{id_str} - Classification Map', \
            save_figure = True, file_name = f'{id_str}_CLASSIFICATION_MAP')


    # Denoise classification map. 
    if denoise_classification_map:
        classification_image, classification_confidence = denoise(classification_image, classification_confidence, 
            morph_close = morph_close, morph_open = morph_open)
        if plotting:
            show_bgra_image(classification_image, title = f'{id_str} - Denoised Classification Map', \
                save_figure = True, file_name = f'{id_str}_DENOISED_CLASSIFICATION_MAP')

    # Create binary classification for spray region.  
    spray_region = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
    spray_region[classification_image[:,:,3] == 255] = 255

    # Dilate to generate spray region (an overestimate of the area enclosing the weeds.) The array 
    # spray_region is a binary image.

    if dilation_size > 0:
        # Dilation size should be odd. 
        if dilation_size%2 == 0:
            dilation_size += 1
        dilation_kernel = np.ones((dilation_size, dilation_size), dtype = np.uint8)
        spray_region = cv2.dilate(spray_region, dilation_kernel, iterations = 1)
        spray_region[spray_region != 0] = 255 # Binarise. 

    if verbose:
        pc_sprayed = 100.*np.sum(spray_region == 255)/n_total_pixels
        print(f'Spray region is {pc_sprayed:.2f} [%] of total region.')

    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(spray_region, interpolation='bilinear', cmap='Pastel1_r',
                    origin='upper', vmin=0., vmax=1., extent = extents)
        if extents == None:
            ax.axis('off')
        ax.ticklabel_format(useOffset=False)
        plt.title(f'{id_str} - Spray Region')
        plt.savefig(f'{id_str}_SPRAY_REGION.PNG')
        plt.show() 

    return density, classification_image, classification_confidence, spray_region









