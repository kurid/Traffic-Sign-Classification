# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import os

def equalizeHist(image_name):
	img = cv2.imread(image_name)

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# cv2.imshow('Color input image', img)
	# cv2.imshow('Histogram equalized', img_output)

	return img_output


def is_inner_box(cnts, cnt):
	for c in cnts:
		minx = c[0][0][0]
		miny = c[0][0][1]
		maxx = c[3][0][0]
		maxy = c[3][0][1]
		
		minx1 = cnt[0][0][0]
		miny1 = cnt[0][0][1]
		maxx1 = cnt[3][0][0]
		maxy1 = cnt[3][0][1]
		if minx1 > minx and miny1 > miny and maxx1 < maxx and maxy1 < maxy:
			print cnt
			print "aaaaaaa"
			return True
	return False



def remove_inner_boxes(cnts, image):
	print 'AAAAA '
	print len(cnts)
	print cnts 
	for cnt in cnts :
		if is_inner_box(cnts, cnt):
			continue
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	
	cv2.imshow("Image", image)
	cv2.waitKey(0)



def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def threshold_color(image_name):
	# load the image

	# image = cv2.imread(image_name)
	image = equalizeHist(image_name)
	image = adjust_gamma(image, 0.8)
	# define the list of boundaries

	# ([86, 31, 4], [220, 88, 50]),
	# ([25, 146, 190], [62, 174, 250]),
	# ([103, 86, 65], [145, 133, 128])

	boundaries = [
		([17, 15, 90], [80, 80, 255])]

	# boundaries = [([17, 15, 90], [50, 56, 255])]

	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")

		lower2 = np.array([0, 0, 50], dtype = "uint8")
		upper2 = np.array([70, 20, 255], dtype = "uint8")

		lower3 = np.array([20, 20, 50], dtype = "uint8")
		upper3 = np.array([50, 50, 100], dtype = "uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		mask1 = cv2.inRange(image, lower, upper)
		mask2 = cv2.inRange(image, lower2, upper2)
		mask3 = cv2.inRange(image, lower3, upper3)
		output = cv2.bitwise_and(image, image, mask = mask1 | mask2 | mask3)
		# output = cv2.GaussianBlur(output, (21, 21), 0)
		output = cv2.medianBlur(output, 9)


		imgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray, 0,200,cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		

		# print len(contours)
		# print np.amin(contours[0], axis = 0)[0][0]
		# print np.amax(contours[0], axis = 0)[0]
		
		cnts = []

		for c in contours :
			minx = np.amin(c, axis = 0)[0][0] - 5
			miny = np.amin(c, axis = 0)[0][1] - 5
			maxx = np.amax(c, axis = 0)[0][0] + 5
			maxy = np.amax(c, axis = 0)[0][1] + 5
			cnt = np.array([ [[minx, miny]], [[minx, maxy]], [[maxx, miny]], [[maxx, maxy]] ])
			cnts.append(cnt)
		

		remove_inner_boxes(cnts, image)


		# show the images
		# cv2.imshow("images", np.hstack([image, output]))
		# cv2.waitKey(0)


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())

path = './test_images/'
for filename in os.listdir(path):
	print filename
	threshold_color(path + filename)



