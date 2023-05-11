
# Aerial image preprocessor 

# Sam Blake, started 8 March, 2023.

import os
import glob
import cv2
from PIL import Image
from argparse import ArgumentParser



def preprocess_dir():

	# Read command line args. 
	parser = ArgumentParser()
	parser.add_argument("-f", "--format", dest="file_format", default="JPG")
	parser.add_argument("-b", "--blur", dest="blur", default=(3,3))
	parser.add_argument("-d", "--downsize", dest="downsize", default=5)
	args = parser.parse_args()

	# Import all file names matching the given format. 
	working_dir = os.getcwd()
	image_filenames = glob.glob(working_dir + '/*.' + args.file_format)

	for filename in image_filenames:
		print(filename)
		if 'DOWNSAMPLED' in filename:
			continue
		output_filename = filename.replace(f'.{args.file_format}', f'_DOWNSAMPLED.{args.file_format}')
		if os.path.exists(output_filename):
			continue

		# Import image. 
		image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

		# Run preprocessing routine. 
		preprocessed = preprocess_image(image, args.blur, args.downsize)

		# Export preprocessed image.
		cv2.imwrite(output_filename, preprocessed)

		# Copy EXIF data and re-export.
		image_with_exif = Image.open(filename)
		exif = image_with_exif.info['exif']
		image_wo_exif = Image.open(output_filename)
		image_wo_exif.save(output_filename, args.file_format.replace('JPG','JPEG'), exif=exif)

	return 1



def preprocess_image(image, image_blur_kernel_size, image_resize_ratio):
	image_pre = cv2.blur(image, image_blur_kernel_size)
	image_pre = cv2.resize(image_pre, 
		(image.shape[1]//image_resize_ratio, image.shape[0]//image_resize_ratio), 
		interpolation = cv2.INTER_AREA)
	return image_pre



if __name__ == "__main__":
    preprocess_dir()
