
# Preprocessor - JPEG compression

# Sam Blake, 10 July, 2023


# JPEG to JPEG compression whilst preserving EXIF data. For GEOTIFF to GEOTIFF
# compression use gdal_translate: 

# gdal_translate rgb.tif out_ycbcr.tif -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR

# Ref: https://gis.stackexchange.com/questions/76351/gdal-translate-why-jpg-compressed-tif-is-2-times-greater-than-jpg-file

import os
import cv2
import glob
from PIL import Image
from argparse import ArgumentParser



def jpeg_compress_dir():

	# Read command line args. 
	parser = ArgumentParser()
	parser.add_argument("-f", "--format", dest="file_format", default="JPG")
	parser.add_argument("-c", "--compression", dest="compression", default=95)
	args = parser.parse_args()


	# Import all file names matching the given format. 
	working_dir = os.getcwd()
	image_filenames = glob.glob(working_dir + '/*.' + args.file_format)

	for filename in image_filenames:
		if 'COMPRESSED' in filename:
			continue

		print(filename)

		output_filename = filename.replace(f'.{args.file_format}', f'_COMPRESSED.JPG')

		img = cv2.imread(filename)
		cv2.imwrite(output_filename, img, [cv2.IMWRITE_JPEG_QUALITY, int(args.compression)])

		compression_ratio = float(os.path.getsize(output_filename))/float(os.path.getsize(filename))
		print(f'compression rate = {compression_ratio:.4}')

		# Copy EXIF data and re-export.
		image_with_exif = Image.open(filename)
		exif = image_with_exif.info['exif']
		image_wo_exif = Image.open(output_filename)
		image_wo_exif.save(output_filename, args.file_format.replace('JPG','JPEG'), exif=exif)

if __name__ == "__main__":
    jpeg_compress_dir()
