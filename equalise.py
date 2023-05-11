

import os
import glob
import numpy as np
import cv2
import rasterio
from argparse import ArgumentParser


def histogram_equalise(img):
    """Adaptive histogram equalisation for BGR images."""
    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    luma = img_hls[:,:,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    luma_equalized = clahe.apply(luma)
    img_hls[:,:,1] = luma_equalized
    img_equalised = cv2.cvtColor(img_hls, cv2.COLOR_HLS2BGR)
    return img_equalised

def equalise_all():

    working_dir = os.getcwd()
    filenames = glob.glob(working_dir + '/*.tif')

    for filename in filenames:
        print(filename)
        if 'EQUALISED' in filename:
            continue
        
        output_filename = filename.replace('.tif',f'_EQUALISED.tif')

        try:
            dataset = rasterio.open(filename)
            red,green,blue = dataset.read(1), dataset.read(2), dataset.read(3)
        except:
            continue

        img = np.zeros((blue.shape[0], blue.shape[1], 3), dtype = np.uint8)
        img[:,:,0] = blue
        img[:,:,1] = green
        img[:,:,2] = red
        
        equalised = histogram_equalise(img)

        kwargs = dataset.meta
        kwargs.update(
            dtype=rasterio.uint8,
            count=3,
            compress='lzw')

        with rasterio.open(output_filename, 'w', **kwargs) as dst:
            dst.write_band(1, equalised[:,:,2])
            dst.write_band(2, equalised[:,:,1])
            dst.write_band(3, equalised[:,:,0])

        os.system(f'/bin/rm \'{filename}\'')

    return 1


if __name__ == "__main__":
    equalise_all()
