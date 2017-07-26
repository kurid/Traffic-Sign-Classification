# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import imageProcesing as imp
from pyimagesearch.shapedetector import ShapeDetector


def is_inner_box(a, b):
	minx, miny, maxx, maxy     = a[0][0][0], a[0][0][1], a[3][0][0], a[3][0][1]
	minx1, miny1, maxx1, maxy1 = b[0][0][0], b[0][0][1], b[3][0][0], b[3][0][1]
	return minx1 > minx and miny1 > miny and maxx1 < maxx and maxy1 < maxy


def inner_box_found(cnts, cnt):
	for c in cnts:
		if is_inner_box(c, cnt) :
			return True
	return False


def remove_inner_boxes(cnts):
	new_cnts = []
	for cnt in cnts :
		if inner_box_found(cnts, cnt):
			continue
		new_cnts.append(cnt)
		x,y,w,h = cv2.boundingRect(cnt)
		#drow bounding box
		# cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	return new_cnts

def find_contours(image):
	imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_BINARY)
	_ , contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnts = []
	for c in contours :
		minx = np.amin(c, axis = 0)[0][0] - 5
		miny = np.amin(c, axis = 0)[0][1] - 5
		maxx = np.amax(c, axis = 0)[0][0] + 5
		maxy = np.amax(c, axis = 0)[0][1] + 5
		cnt = np.array([ [[minx, miny]], [[minx, maxy]], [[maxx, miny]], [[maxx, maxy]] ])
		cnts.append(cnt)

	return cnts

def	color_threshold(image):
	boundaries = [
		([0, 0, 65],[100, 40, 255]),
		([0, 0, 30],[60, 25, 65]),
		([55, 40, 180],[135, 120, 255]),
		([65, 40, 100],[120, 75, 150])]

	# loop over the boundaries

	red_mask = 0 
	for (lower, upper) in boundaries:
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
		mask = cv2.inRange(image, lower, upper)
		red_mask = red_mask | mask

	output = cv2.bitwise_and(image, image, mask = red_mask)
	output = cv2.medianBlur(output, 9)
	return output


def cut_rois(image_name, show_image):
	# load the image
	original_image = cv2.imread(image_name)
	if original_image == None:
		return []
	image = imp.preprocesing(original_image.copy())
	output = color_threshold(image)
	cnts = find_contours(output)
	# cnts.extend(merge_intersecting_boxes(cnts,image))
	cnts = remove_inner_boxes(cnts)
	if show_image:
		cv2.imshow("images", np.hstack([original_image, image, output]))

	rois = []
	for cnt in cnts:
		x, y, width, height = cv2.boundingRect(cnt)
		roi = original_image[y:y+height, x:x+width]
		if  width * height > 4000 and len(roi) != 0 and len(roi[0]) != 0 :
			# if imp.detect_circles(roi) or imp.detect_shape(roi):
			rois.append(roi)
			if show_image:
				cv2.imshow("roi", roi)
				cv2.waitKey(0)

	return rois

def test():	
	path = './ellipse/'
	for filename in os.listdir(path):
		print filename
		rois = cut_rois(path + filename, True)


test()