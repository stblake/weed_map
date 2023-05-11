
# NDVI-based Image Classifier

# Sam Blake, started 1 May, 2023



import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib as mpl
from matplotlib import pyplot as plt

import rasterio
from rasterio.plot import show

def green_red_ratio_filter(image, verbose = False):

    image_blur = cv2.blur(image, (5,5))
    image_blur = image_blur.astype(np.float32)

    image_norm = image_blur.copy()
    
    blue_min = np.nanmin(image_blur[:,:,0])
    blue_max = np.nanmax(image_blur[:,:,0])
    if verbose:
        print(f'blue_min, blue_max = {blue_min}, {blue_max}')
    image_norm[:,:,0] = (image_blur[:,:,0] - blue_min)/(blue_max - blue_min)
    
    green_min = np.nanmin(image_blur[:,:,1])
    green_max = np.nanmax(image_blur[:,:,1])
    if verbose:
        print(f'green_min, green_max = {green_min}, {green_max}')
    image_norm[:,:,1] = (image_blur[:,:,1] - green_min)/(green_max - green_min)
    
    red_min = np.nanmin(image_blur[:,:,2])
    red_max = np.nanmax(image_blur[:,:,2])
    if verbose:
        print(f'red_min, red_max = {red_min}, {red_max}')
    image_norm[:,:,2] = (image_blur[:,:,2] - red_min)/(red_max - red_min)
    
    #ratio = (image_norm[:,:,1] - image_norm[:,:,2])/(\
    #    1e-6 + image_norm[:,:,0]*(image_norm[:,:,1] + image_norm[:,:,2]))

    ratio = (image_norm[:,:,1] - image_norm[:,:,2])/(\
        1e-6 + (image_norm[:,:,1] + image_norm[:,:,2]))

    #image_luma = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
    #image_luma = image_luma.astype(np.float32)
    #image_luma[np.isclose(image_luma, 0.0)] = np.nan
    #image_luma = (image_luma - np.nanmin(image_luma))/(np.nanmax(image_luma) - np.nanmin(image_luma))
    #ratio /= (1. + image_luma**2)
    
    ratio = gaussian_filter(ratio, sigma = 12)    
    return 



def green_red_ratio_filter_threshold(image, n_sigma = 3., dilation_size = 101, n_blur = 13, 
                      luma_threshold_pct = 50, plotting = False, verbose = False):
    
    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(image[:,:,0], cmap='Greys')
        plt.title('blue')
        plt.colorbar(im,fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
        plt.show()  

        fig, ax = plt.subplots()
        im = ax.imshow(image[:,:,1], cmap='Greys')
        plt.title('green')
        plt.colorbar(im,fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
        plt.show() 

        fig, ax = plt.subplots()
        im = ax.imshow(image[:,:,2], cmap='Greys')
        plt.title('red')
        plt.colorbar(im,fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
        plt.show() 
    
    ratio = green_red_ratio_filter(image, verbose = verbose)

    image_sum = np.sum(image, axis = 2)
    ratio[image_sum < 10] = np.nan
    
    # Threshold weed map
    mean, std = np.nanmean(ratio), np.nanstd(ratio)
    if verbose:
        print(f'mean, std = {mean}, {std}')

    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(ratio, vmin = mean - 3*std, vmax = mean + 3*std)
        plt.title('ratio')
        plt.colorbar(im,fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
        plt.show()    
        
    threshold = mean + n_sigma*std
    if verbose:
        print(f'threshold = {threshold}')
        
    weed_map = ratio.copy()
    weed_map = weed_map.astype(np.uint8)
    weed_map[ratio < threshold] = 0
    weed_map[ratio >= threshold] = 255

    # Remove high luma regions. 
    image_luma = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:,:,1]
    image_luma = cv2.blur(image_luma, (11,11))
    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(image_luma, cmap='Greys', vmin = 0, vmax = 255)
        plt.title('luma')
        plt.colorbar(im,fraction=0.046*image.shape[0]/image.shape[1], pad=0.04)
        plt.show()
    image_luma = image_luma.astype(np.float32)
    image_luma[image_luma == 0] = np.nan
    luma_threshold = np.nanpercentile(image_luma, luma_threshold_pct)
    if verbose:
        print(f'{luma_threshold_pct} [%] luma percentile = {luma_threshold}')
      
    luma_mask = image_luma > luma_threshold
    
    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(np.where(luma_mask, 255, 0), cmap='Greys', vmin = 0, vmax = 255)
        plt.title('luma mask')
        plt.show()
    
    weed_map[luma_mask] = 0
    
    # Denoise. 
    weed_map = cv2.medianBlur(weed_map, 2*n_blur + 1)

    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(weed_map, cmap='Greys', vmin = 0, vmax = 255)
        plt.title('weed map')
        plt.show()

    # Spray map
    dilation_kernel = np.ones((dilation_size, dilation_size), dtype = np.uint8)
    spray_map = cv2.dilate(weed_map, dilation_kernel, iterations = 1)
    spray_map[spray_map != 0] = 255 # Binarise. 

    if plotting:
        fig, ax = plt.subplots()
        im = ax.imshow(spray_map, cmap='Greys', vmin = 0, vmax = 255)
        plt.title('spray map')
        plt.show()
        
    return weed_map, spray_map



