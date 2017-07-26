import os
import find_roi as fr
import cv2

def create_folder(category_id):
    if not os.path.exists(category_id):
        os.makedirs(category_id)


path = './daset_images/images'
cut_images_path = './daset_images/images_cut'
for folder in os.listdir(path):
	print folder
	cut_folder = cut_images_path + "/" + folder +'_cut'
	create_folder(cut_folder)
	for image_name in os.listdir(path + "/" + folder):
		i = 0
		image_path = path + "/" + folder + "/" + image_name
		print image_path
		rois = fr.cut_rois(image_path, False)
		for img in rois:
			a = image_name.split('.')
		 	cv2.imwrite(cut_folder + "/" + a[0] + '_' + str(i) + '.jpg', img)
		 	i += 1 
