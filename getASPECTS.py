#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications import vgg16
from keras.models import Model
import keras
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization, Activation
from keras.models import Sequential
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
from sklearn.metrics import hamming_loss, accuracy_score
import matplotlib.pyplot as plt

#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')


import numpy as np
import os, json, sys
import pandas as pd

import metrics



def predict_labels(data, model_arch, model_weight, out_dir = "/tmp"):

	data_hu, data_text, label, pid, pslices, ref = data

	#model = load_model("{}/{}".format(out_dir, model_file))

	model = None
	with open("{}/{}".format(out_dir, model_arch), "r") as f:
		model = model_from_json(f.read())

	model.load_weights("{}/{}".format(out_dir, model_weight), by_name=True)


	predicted_labels = model.predict(np.stack([data_hu]*3, axis=-1))
	predicted_labels[predicted_labels >= 0.5] = 1
	predicted_labels[predicted_labels < 0.5] = 0
	predicted_labels = predicted_labels.astype('uint8')

	
	return (label, predicted_labels)


def get_RL_labels(labels):

	label_size = labels.shape[1]

	assert(label_size % 2 == 0)

	new_label_size = label_size/2
	rl_labels = []
	
	for idx in xrange(len(labels)):
		
		if 1 in labels[idx, : new_label_size]:
			rl_labels.append([1,0])
		
		elif 1 in labels[idx, new_label_size :]:
			rl_labels.append([0,1])
		else:
			rl_labels.append([0,0])

	rl_labels = np.array(rl_labels)


	return rl_labels


def predict_labels_functional(data, model_arch, model_weight, batch_size = 6, out_dir = "/tmp"):


	model = None
	with open("{}/{}".format(out_dir, model_arch), "r") as f:
		model = model_from_json(f.read())

	model.load_weights("{}/{}".format(out_dir, model_weight), by_name=True)


	data_hu, data_text, label, pid, pslices, ref = data
	rl_labels_train = get_RL_labels(label)

	merged_rl = []
	merged_pred_labels = np.empty((0, label.shape[1]), int)

	print("Given labels shape: {}".format(label.shape))

	for i in range(0, len(label), batch_size):

		#preds = model.evaluate(x = np.stack([dev_data[i:(i+batch_size)]]*3, axis=-1), y = dev_labels[i:(i+batch_size)])
		predicted_RL, predicted_labels = model.predict(np.stack([data_hu[i:(i+batch_size)]]*3, axis=-1))
		
		predicted_labels[predicted_labels >= 0.5] = 1
		predicted_labels[predicted_labels < 0.5] = 0
		predicted_RL = np.argmax(predicted_RL, axis=1)
		predicted_RL = predicted_RL.tolist()

		predicted_labels = predicted_labels.astype('uint8')
		
		merged_pred_labels = np.vstack([merged_pred_labels, predicted_labels])
		merged_rl = merged_rl + predicted_RL


	pd.DataFrame(merged_pred_labels).to_csv("{}/vgg16_m456_predictedTestFunc_labels.csv".format(out_dir))
	pd.DataFrame(label).to_csv("{}/vgg16_m456_actualTestFunc_labels.csv".format(out_dir))

	pd.DataFrame(merged_rl).to_csv("{}/vgg16_m456_predictTestFunc_RL.csv".format(out_dir))

	return (label, merged_pred_labels, merged_rl)


def get_aspects_functional(data, model_dir = "/tmp", data_type = ""):


	model_weight = None
	model_arch = None
	
	for file_name in os.listdir(model_dir):
		
		if file_name.endswith("weights.h5"):
			model_weight = file_name

		elif file_name.endswith("architecture.json"):
			model_arch = file_name


	actual, predicted_label, prediced_rl = predict_labels_functional(data, model_arch, model_weight, out_dir = model_dir)
	
	
	print("Calculate score for functional model")
	
	#score(actual, predicted_label)

	# For now use evaluation like sequential output
	metrics.report(actual, predicted_label, out_dir = model_dir, datatype = "test", modeltype = "sequential")	#functional




def get_aspects(data, region, model_dir = "/tmp", data_type = ""):


	model_weight = None
	model_arch = None
	
	for file_name in os.listdir(model_dir):
		
		if file_name.endswith("weights_{}.h5".format(region)):
			model_weight = file_name

		elif file_name.endswith("architecture_{}.json".format(region)):
			model_arch = file_name


	if model_arch is None or model_weight is None:
		print("Missing or imcomplete model...")
		return

	

	actual, predicted = predict_labels(data, model_arch, model_weight, out_dir = model_dir)
	
	# actual_out = np.concatenate((actual_left, actual_right), axis = -1)
	# predicted_out = np.concatenate((predicted_left, predicted_right), axis = -1)


	pd.DataFrame(predicted).to_csv("{}/predictedTest_{}.csv".format(model_dir, region))
	pd.DataFrame(actual).to_csv("{}/actual_{}.csv".format(model_dir, region))

	print("Calculate score for: {} - {}".format(region, data_type))
	
	metrics.report(actual, predicted, out_dir = model_dir, datatype = "test", modeltype = "sequential")








