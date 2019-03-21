#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 05:27:18 2019


"""
from keras.applications import vgg16
from keras.models import Model
import keras
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization, Activation
from keras.models import Sequential
from keras import optimizers
from keras import backend as K
from sklearn.metrics import hamming_loss, accuracy_score
import matplotlib.pyplot as plt
import talos as ta

#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')


import numpy as np
import os, json, sys
import pandas as pd
import collections

import metrics


##########################

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

##########################


def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


def initialize_VGG_custom(input_shape, output_size, params, weights = "imagenet", include_top=False, activation = 'relu'):


	vgg16_model = keras.applications.vgg16.VGG16(input_shape = input_shape, weights = weights, include_top = include_top)
	x = vgg16_model.output
	x = Flatten()(x)
	x = Dense(params['dense_layer'])(x)
	x = BatchNormalization()(x)
	x = Activation(params['activation'])(x)
	x = Dense(output_size, activation='sigmoid')(x)


	model = Model(inputs=vgg16_model.input, outputs=x)

	for layer in vgg16_model.layers:
		layer.trainable = False
	model.summary()

	return model



def evaluate_model(model, dev_data, dev_labels, batch_size = 6, out_dir = "/tmp"):

	merged_pred_labels = np.empty((0, dev_labels.shape[1]), int)

	for i in range(0, len(dev_labels), batch_size):

		#preds = model.evaluate(x = np.stack([dev_data[i:(i+batch_size)]]*3, axis=-1), y = dev_labels[i:(i+batch_size)])
		predicted_labels = model.predict(np.stack([dev_data[i:(i+batch_size)]]*3, axis=-1))
		predicted_labels[predicted_labels >= 0.5] = 1
		predicted_labels[predicted_labels < 0.5] = 0
		predicted_labels = predicted_labels.astype('uint8')

		merged_pred_labels = np.vstack([merged_pred_labels, predicted_labels])

		# tempPred = np.vstack([tempPred, predicted_labels])
		# predOutput = tempPred[1:,:]    
		# print("\n\n********* Test Performance *********")
		# print("Loss = " + str(preds[0]))
		# print("Dev Accuracy = " + str(preds[1]))


	pd.DataFrame(merged_pred_labels).to_csv("{}/vgg16_m456_predictedDev_labels_dense256_adam.csv".format(out_dir))
	pd.DataFrame(dev_labels).to_csv("{}/vgg16_m456_actualDev_labels_dense256_adam.csv".format(out_dir))

	return merged_pred_labels


def aspects(x_train, y_train, x_val, y_val, params):	

	model = initialize_VGG_custom(input_shape = (x_train.shape[1], x_train.shape[2], 3), output_size = y_train.shape[1], params = params)

	optimizer = optimizers.Adam(lr = params['lr'])
	model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])

	history = model.fit(x = np.stack([x_train]*3, axis=-1), y = y_train, epochs = params['epochs'], batch_size = params['batch_size'], validation_data=(np.stack([x_val]*3, axis=-1), y_val))	#, validation_data=(np.stack([x_val]*3, axis=-1), y_val)

	return (history, model)


def get_model(train_data, dev_data, out_dir = "/tmp", region = "", batch_size = 6, param_tune = True, seed = 0):

	out_dir = "{}/VGG16/".format(out_dir)

	if not(os.path.isdir(out_dir)):
		os.makedirs(out_dir)

	if seed is not None: 
		np.random.seed(seed)


	train_data_hu, train_data_text, train_labels, train_pid, train_pslices, train_ref = train_data
	dev_data_hu, dev_data_text, dev_labels, dev_pid, dev_pslices, dev_ref = dev_data
	#test_data_hu, test_data_text, test_labels, test_pid, test_pslices, test_ref = test_data

	################################
	# Comment out this at the end
	train_data_hu = train_data_hu[:10]
	train_labels = train_labels[:10]

	dev_data_hu = dev_data_hu[:10]
	dev_labels = dev_labels[:10]
	################################

	train_data_hu = train_data_hu.astype(np.float32)
	dev_data_hu = dev_data_hu.astype(np.float32)
	#test_data_hu = test_data_hu.astype(np.float32)
	
	print("Train data shape: {}".format(train_data_hu.shape))
	print("Dev data shape: {}".format(dev_data_hu.shape))
	#print("Test data shape: {}".format(test_data_hu.shape))


	# *** This is for softmax *****
	# train_labels = to_categorical(train_labels)	
	# dev_labels = to_categorical(dev_labels)	

	# Set old strokes to 0 for now
	train_labels[train_labels < 0] = 0
	dev_labels[dev_labels < 0] = 0


	train_labels[train_labels > 1] = 0
	dev_labels[dev_labels > 1] = 0


	
	#######################################

	if param_tune:

		params_to_tune = {
			'lr': [0.0001, 0.001, 0.01, 0.1],
			'batch_size': [6],
			'activation':['relu', 'elu'],
			'optimizer': ['Nadam', 'Adam'],
			'dense_layer': [32, 64, 128, 256, 512],
			'epochs': [10,15,20,25,30]
		}

		


		# Temporary set of params to check if it works. Final model should use params_to_tune
		# params_to_tune_t = {
		# 	'lr': [0.0001, 0.001],
		# 	'activation':['relu'],
		# 	'dense_layer': [256],
		# 	'epochs': [1]

		# }

		# params_to_tune_t = collections.OrderedDict(sorted(params_to_tune_t.items()))



		ta_scanOut = ta.Scan(train_data_hu, train_labels, params = params_to_tune, model = aspects, x_val = dev_data_hu, y_val = dev_labels)


		print("Scan details:\n{}".format(ta_scanOut.details))
		

		r = ta.Reporting(ta_scanOut)

		print("Max score: {}".format(r.high()))
		print("Number of rounds: {}".format(r.rounds()))

		print("Max scored round: {}".format(r.rounds2high()))
		print("Best params:\n{}".format(r.best_params()))

		
		#r.plot_bars('val_acc', 'dense_layer', 'lr')

		#ta_scanOut.saved_models

		#r.plot_corr()

		#ta.Deploy(ta_scanOut, 'aspects_v1')


	else:

		#best_params_file = "{}/best_tuned_params_handpicked.json".format(out_dir)

		best_params_file = os.path.join(__location__, "best_tuned_params_handpicked.json")

		if not(os.path.isfile(best_params_file)):
			print("Invalid best params file! Provide correct file.")
			print("Exiting ...")
			sys.exit()

		best_params_dict = None
		with open(best_params_file, 'r') as p_file:
			best_params_dict = json.load(p_file)

		print(best_params_dict)
		

		history, model = aspects(train_data_hu, train_labels, dev_data_hu, dev_labels, best_params_dict)
	
		# model = initialize_VGG_custom(input_shape = (train_data_hu.shape[1], train_data_hu.shape[2], 3), output_size = train_labels.shape[1])
		
		# #sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
		# #model.compile(optimizer = sgd, loss = 'binary_crossentropy', metrics = ["accuracy"])

		# model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ["accuracy"])	
		# history = model.fit(x = np.stack([train_data_hu]*3, axis=-1), y = train_labels, epochs = 1, batch_size = batch_size, validation_data=(np.stack([dev_data_hu]*3, axis=-1), dev_labels))	#, validation_data=(np.stack([dev_data_hu]*3, axis=-1), dev_labels)

		# #model.save("{}/deepASPECTS_vgg16_adam_dense256_{}.h5".format(out_dir, region))


		model.save_weights("{}/deepASPECTS_vgg16_tuned_weights_{}.h5".format(out_dir, region))
		with open("{}/deepASPECTS_vgg16_tuned_architecture_{}.json".format(out_dir, region), 'w') as outf:
			outf.write(model.to_json())


		# Plot training & validation accuracy values
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig("{}/accuracy_plot.png".format(out_dir))
		plt.close()


		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig("{}/loss_plot.png".format(out_dir))
		plt.close()



		predicted_labels = evaluate_model(model, dev_data_hu, dev_labels, batch_size = batch_size, out_dir = out_dir)

		metrics.report(dev_labels, predicted_labels, out_dir = out_dir, datatype = "dev")



		# preds = model.evaluate(x = np.stack([dev_data_hu]*3, axis=-1), y = dev_labels)

		# print("\n\n********* Test Performance *********")
		# print("Loss = " + str(preds[0]))
		# print("Test Accuracy = " + str(preds[1]))

		# print("\n\n*************************************")

		# predicted_labels = model.predict(np.stack([dev_data_hu]*3, axis=-1))
		# predicted_labels[predicted_labels >= 0.5] = 1
		# predicted_labels[predicted_labels < 0.5] = 0
		# predicted_labels = predicted_labels.astype('uint8')

		# pd.DataFrame(predicted_labels).to_csv("{}/vgg16_m456_predicted_adam_dense256_{}.csv".format(out_dir, region))
		# pd.DataFrame(dev_labels).to_csv("{}/vgg16_m456_actual_adam_dense256_{}.csv".format(out_dir, region))

		# pd.DataFrame(history.history).to_csv("{}/vgg16_m456_history_adam_dense256_{}.csv".format(out_dir, region))

		###################

		
		