from sklearn import svm
import os
import imageProcesing as imp 
import cv2
from sklearn import svm
import cPickle
import numpy as np


def fit(X,Y):
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(X, Y)
	clf.decision_function_shape = "ovr"
	return clf

size = 128.0


path = './learning'
X = []
 # X = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]]
Y = []
for folder in os.listdir(path):
	label = folder.split('_')[0]
	for image_name in os.listdir(path + "/" + folder):
		image_path = path + "/" + folder + "/" + image_name
		print image_path
		image = cv2.imread(image_path)
		r = size / image.shape[0]
		dim = (int(image.shape[1] * r), int(size))
 
		resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		if len(resized) < size or len(resized[0]) < size:
			continue
		cropped = resized[0:size, 0:size]
		hog = imp.calculate_hog(cropped)
		# print len(cropped)
		# print len(cropped[0])
		# print len(hog)
		# print "AAAAA"
		X.append(hog.tolist())
		Y.append(label)

clf = fit(X,Y)
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)
		# cv2.imshow("cropped", cropped)
		# cv2.waitKey(0)

		
	# rois = cut_rois(path + filename, True)