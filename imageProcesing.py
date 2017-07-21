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

	return img_output


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



def boxes_intersect(a, b):
	a_min_x = a[0][0][0]
	a_min_y = a[0][0][1]
	a_max_x = a[3][0][0]
	a_max_y = a[3][0][1]

	b_min_x= b[0][0][0]
	b_min_y= b[0][0][1]
	b_max_x= b[3][0][0]
	b_max_y= b[3][0][1]


	if a_max_x < b_min_x :
		return False  #a is left of b
	if a_min_x > b_max_x : 
		return False # a is right of b
	if a_max_y < b_min_y :
		return False # a is above b
	if a_min_y > b_max_y :
		return False # a is below b
	return not (is_inner_box(a, b) or is_inner_box(b, a)) # boxes overlap


def merge_intersecting_boxes(cnts,image):
	newCnts = []
	for i in xrange(len(cnts)):
		for j in xrange(len(cnts)):
			if i == j:
				continue
			if boxes_intersect(cnts[i], cnts[j]) :
				minx = min(cnts[i][0][0][0], cnts[j][0][0][0])
				miny = min(cnts[i][0][0][1], cnts[j][0][0][1])
				maxx = min(cnts[i][3][0][0], cnts[j][3][0][0])
				maxy = min(cnts[i][3][0][1], cnts[j][3][0][1])
				newCnts.append(np.array([ [[minx, miny]], [[minx, maxy]], [[maxx, miny]], [[maxx, maxy]] ]))
	return newCnts
				
