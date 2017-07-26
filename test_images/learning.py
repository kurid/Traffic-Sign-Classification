from sklearn import svm



path = './learning/'
for filename in os.listdir(path):
	print filename
	# rois = cut_rois(path + filename, True)