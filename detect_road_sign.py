from sklearn import svm
import os
import imageProcesing as imp 
import cv2
from sklearn import svm
import cPickle
import numpy as np



with open('my_dumped_classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

size = 128.0

path = './detect/'
for  image_name in os.listdir(path):
		image_path = path +  image_name
		print image_path
		image = cv2.imread(image_path)
		r = size / image.shape[0]
		dim = (int(image.shape[1] * r), int(size))
 
		resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		if len(resized) < size or len(resized[0]) < size:
			continue
		cropped = resized[0:size, 0:size]
		hog = imp.calculate_hog(cropped)
		print clf.predict(hog.tolist())
		cv2.imshow("cropped", cropped)
		cv2.waitKey(0)
