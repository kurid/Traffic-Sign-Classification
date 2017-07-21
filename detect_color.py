# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import imageProcesing as imp



def is_inner_box(a, b):
	minx = a[0][0][0]
	miny = a[0][0][1]
	maxx = a[3][0][0]
	maxy = a[3][0][1]
	
	minx1 = b[0][0][0]
	miny1 = b[0][0][1]
	maxx1 = b[3][0][0]
	maxy1 = b[3][0][1]
	return minx1 > minx and miny1 > miny and maxx1 < maxx and maxy1 < maxy


def inner_box_found(cnts, cnt):
	for c in cnts:
		if is_inner_box(c, cnt) :
			return True
	return False



def remove_inner_boxes(cnts, image):
	new_cnts = []
	for cnt in cnts :
		if inner_box_found(cnts, cnt):
			continue
		new_cnts.append(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	return new_cnts



def threshold_color(image_name):
	# load the image

	image = imp.equalizeHist(image_name)
	image = imp. adjust_gamma(image, 0.8)

	boundaries = [
		([0, 0, 55],[80, 40, 255])]



	# boundaries = [([17, 15, 90], [50, 56, 255])]

	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")

		lower2 = np.array([60, 40, 200], dtype = "uint8")
		upper2 = np.array([120, 100, 255], dtype = "uint8")

		lower3 = np.array([80, 40, 120], dtype = "uint8")
		upper3 = np.array([120, 75, 150], dtype = "uint8")

		lower4 = np.array([10, 0, 30], dtype = "uint8")
		upper4 = np.array([50, 25, 65], dtype = "uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		mask1 = cv2.inRange(image, lower, upper)
		mask2 = cv2.inRange(image, lower2, upper2)
		mask3 = cv2.inRange(image, lower3, upper3)
		mask4 = cv2.inRange(image, lower4, upper4)
		output = cv2.bitwise_and(image, image, mask = mask1 | mask2 | mask3 | mask4)
		# output = cv2.GaussianBlur(output, (21, 21), 0)
		output = cv2.medianBlur(output, 9)


		imgray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
		ret,thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
		
		cnts = []

		for c in contours :
			minx = np.amin(c, axis = 0)[0][0] - 7
			miny = np.amin(c, axis = 0)[0][1] - 7
			maxx = np.amax(c, axis = 0)[0][0] + 7
			maxy = np.amax(c, axis = 0)[0][1] + 7
			cnt = np.array([ [[minx, miny]], [[minx, maxy]], [[maxx, miny]], [[maxx, maxy]] ])
			cnts.append(cnt)


		# cnts.extend(merge_intersecting_boxes(cnts,image))
		cnts = remove_inner_boxes(cnts, image)

		if len(cnts) != 0 :
			x, y, width, height = cv2.boundingRect(cnts[0])
			roi = image[y:y+height, x:x+width]
			if  len(roi) != 0 and len(roi[0]) != 0 :
				cv2.imshow("roi", roi)
		else :
			cv2.imshow("images", np.hstack([image, output]))


		# show the images
		cv2.waitKey(0)


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image")
# args = vars(ap.parse_args())

path = './test_images/'
for filename in os.listdir(path):
	print filename
	threshold_color(path + filename)



