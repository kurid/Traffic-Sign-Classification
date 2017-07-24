from pyimagesearch.shapedetector import ShapeDetector
import numpy as np
import argparse
import cv2
import os
import argparse
import imutils



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


def detect_circles(image):
	output = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
	 
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	 
		# show the output image
	cv2.imshow("output", np.hstack([image, output]))
	cv2.waitKey(0)


def detect_shape(image):
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it

	blurred = cv2.GaussianBlur(resized, (5, 5), 0)
	gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
	thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	sd = ShapeDetector()
	for c in cnts:
		# compute the center of the contour
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue

		cX = int((M["m10"] / M["m00"]) * ratio)
		cY = int((M["m01"] / M["m00"]) * ratio)

		# detect the shape of the contour and label the color
		shape = sd.detect(c)
		if shape != 'circle':
			continue
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape and labeled
		# color on the image
		c = c.astype("float")
		c *= ratio
		c = c.astype("int")
		text = "{}".format(shape)
		cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		cv2.putText(image, text, (cX, cY),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

				
