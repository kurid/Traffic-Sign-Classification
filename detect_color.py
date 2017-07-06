# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import os


def threshold_color(image_name):
	# load the image
	# image = cv2.imread(args["image"])
	image = cv2.imread(image_name)
	# (src1, alpha, src2, beta, gamma, dst)
	image = cv2.medianBlur(image,5)
	# define the list of boundaries

		# ([86, 31, 4], [220, 88, 50]),
		# ([25, 146, 190], [62, 174, 250]),
		# ([103, 86, 65], [145, 133, 128])

	boundaries = [
		([30, 15, 90], [110, 80, 255])]

	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")

		# lower2 = np.array([145, 133, 128], dtype = "uint8")
		# upper2 = np.array([250, 250, 250], dtype = "uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		mask1 = cv2.inRange(image, lower, upper)
		mask2 = cv2.inRange(image, lower, upper)
		output = cv2.bitwise_and(image, image, mask = mask1 | mask2)
		output = cv2.GaussianBlur(output, (5, 5), 0)

		# show the images
		cv2.imshow("images", np.hstack([image, output]))
		cv2.waitKey(0)



# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())

path = './test_images/'

for filename in os.listdir(path):
	print filename
	threshold_color(path + filename)

