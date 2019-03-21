import numpy as np
import pandas as pd 
import pydicom as dicom
import os, json, sys
import scipy.ndimage
from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import h5py
import re

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



# datagen = ImageDataGenerator(
#         rotation_range=45,
#         width_shift_range=0.3,
#         height_shift_range=0.3,
#         shear_range=0.3,
#         zoom_range=0.3,
#         horizontal_flip=True,
#         fill_mode='nearest')



def flip_image(img):

	img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.7)
	plt.imshow(img_adapteq, cmap=plt.cm.gray)
	
	outfile_name = "/tmp/img1.jpeg"
	plt.savefig(outfile_name)
	#print("Saved: {}".format(image_name))

	new_img = np.fliplr(img)

	img_adapteq = exposure.equalize_adapthist(new_img, clip_limit=0.7)
	plt.imshow(img_adapteq, cmap=plt.cm.gray)
	
	outfile_name = "/tmp/img2.jpeg"
	plt.savefig(outfile_name)
	#print("Saved: {}".format(image_name))




def get_img_generator():

	datagen = ImageDataGenerator(
		samplewise_center=False, 
		samplewise_std_normalization=False, 
		horizontal_flip = True, 
		vertical_flip = False, 
		height_shift_range = 0.03, 
		width_shift_range = 0.03, 
		rotation_range = 5, 
		shear_range = 0.005,
		fill_mode = 'nearest',
		zoom_range=0.05,
		data_format = 'channels_last'
	)


	return datagen



def save_augmented_img(x, datagen, data_format = 'channels_last'):

	#img = load_img('iguana.jpg')  # this is a PIL image
	#x = img_to_array(img)  # convert image to numpy array 
	#x = x.reshape((1,) + x.shape)  # reshape image to (1, ..,..,..) to fit keras' standard shape
	#print(x.shape)


	x = x.reshape(x.shape + (1,))  # reshape image to (1, ..,..,..) to fit keras' standard shape
	print(x.shape)

	#Use flow() to apply data augmentation randomly according to the datagenerator
	#and saves the results to the `preview/` directory
	num_image_generated = 0
	for batch in datagen.flow(x, batch_size=1, save_to_dir='/tmp/augment_img/', save_prefix='aug', save_format='jpeg'):
	    num_image_generated += 1
	    if num_image_generated > 5:
	        break # stop the loop after num_image_generated iterations



def train_with_augmented_data(x_train, datagen):

	x_train = x_train.reshape(x_train.shape + (1,)) 
	
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen.fit(x_train)

	# fits the model on batches with real-time data augmentation:
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), steps_per_epoch=len(x_train) / 32, epochs=epochs)

	# # here's a more "manual" example
	# for e in range(epochs):
	# 	print('Epoch', e)
	# 	batches = 0
	# 	for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
	# 		model.fit(x_batch, y_batch)
	# 		batches += 1
	# 		if batches >= len(x_train) / 32:
	# 			# we need to break the loop by hand because
	# 			# the generator loops indefinitely
	# 			break


