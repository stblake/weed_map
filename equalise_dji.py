

import os
import glob
import numpy as np
import cv2
import rasterio
from argparse import ArgumentParser

import PIL
from PIL import Image

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
    filenames = glob.glob(working_dir + '/*.JPG')

    for filename in filenames:
        print(filename)
        if 'EQUALISED' in filename:
            continue
        
        output_filename = filename.replace('.JPG',f'_EQUALISED.JPG')

        try:
            img = cv2.imread(filename)
        except:
            continue

        equalised = histogram_equalise(img)

        cv2.imwrite(output_filename, equalised)

        # Copy EXIF data from original image to classification map.
        image_with_exif = Image.open(filename)
        exif = image_with_exif.info['exif']

        image_wo_exif = Image.open(output_filename)
        image_wo_exif.save(output_filename, 'JPEG', exif=exif)

        #os.system(f'/bin/rm \'{filename}\'')

    return 1


if __name__ == "__main__":
    equalise_all()
