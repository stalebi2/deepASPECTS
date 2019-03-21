#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 05:27:18 2019


"""
from keras.applications import vgg16
from keras.models import Model
import keras
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, BatchNormalization, Activation, concatenate
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




def functional_VGG(input_shape, output_size, params, weights = "imagenet", include_top=False, frozen_len = None):
	
    vgg16_model = keras.applications.vgg16.VGG16(weights = weights, input_shape = input_shape, include_top=include_top)
    
    X = vgg16_model.output
    X = Flatten(name = 'flatten')(X)
    X = Dense(params['dense_layer'], name = 'fc1')(X)
    X = BatchNormalization(name = 'bn1')(X)
    X = Activation(params['activation'], name = 'ac1')(X)
    
    RL = Dense(2, activation = 'softmax', name = 'RL')(X)
    merge = concatenate([RL, X], name = 'combined')
    Sig = Dense(output_size, activation = 'sigmoid', name = 'Regions')(merge)


    # create model
    model_functional = Model(inputs = vgg16_model.input, outputs = [RL, Sig])
    model_functional.summary()

    for layer in vgg16_model.layers:
		layer.trainable = False
    

    model_functional.summary()

    return model_functional



def evaluate_model(model, dev_data, dev_labels, dev_labels_rl , batch_size = 6, out_dir = "/tmp"):



	merged_rl = []
	merged_pred_labels = np.empty((0, dev_labels.shape[1]), int)

	for i in range(0, len(dev_labels), batch_size):

		#preds = model.evaluate(x = np.stack([dev_data[i:(i+batch_size)]]*3, axis=-1), y = dev_labels[i:(i+batch_size)])
		predicted_RL, predicted_labels = model.predict(np.stack([dev_data[i:(i+batch_size)]]*3, axis=-1))
		
		predicted_labels[predicted_labels >= 0.5] = 1
		predicted_labels[predicted_labels < 0.5] = 0
		predicted_RL = np.argmax(predicted_RL, axis=1)
		predicted_RL = predicted_RL.tolist()
		

		predicted_labels = predicted_labels.astype('uint8')
		
		merged_pred_labels = np.vstack([merged_pred_labels, predicted_labels])
		merged_rl = merged_rl + predicted_RL	 #np.vstack([merged_rl, predicted_RL])

		# predOutput = tempPred[1:,:]    
		# print("\n\n********* Test Performance *********")
		# print("Loss = " + str(preds[0]))
		# print("Dev Accuracy = " + str(preds[1]))

	# print(merged_pred_labels)
	# print(merged_rl)


	pd.DataFrame(merged_pred_labels).to_csv("{}/vgg16_m456_predictedDevFunc_labels_dense256_adam.csv".format(out_dir))
	pd.DataFrame(dev_labels).to_csv("{}/vgg16_m456_actualDevFunc_labels_dense256_adam.csv".format(out_dir))

	return (merged_pred_labels, merged_rl)


def aspects(x_train, y_train, y_train_rl, x_val, y_val, y_val_rl, params):	

	model = functional_VGG(input_shape = (x_train.shape[1], x_train.shape[2], 3), output_size = y_train.shape[1], params = params)

	optimizer = optimizers.Adam(lr = params['lr'])
	
	losses = {
		"RL": "categorical_crossentropy",
		"Regions": "binary_crossentropy",
	}

	model.compile(optimizer = optimizer, loss = losses, metrics = ["accuracy"])

	train_out_dict = {
		"RL": y_train_rl,
		"Regions": y_train
	}

	dev_out_dict = {
		"RL": y_val_rl,
		"Regions": y_val
	}


	history = model.fit(x = np.stack([x_train]*3, axis=-1), y = train_out_dict, epochs = params['epochs'], batch_size = params['batch_size'], validation_data=(np.stack([x_val]*3, axis=-1), dev_out_dict), verbose=1)	#, validation_data=(np.stack([x_val]*3, axis=-1), y_val)

	return (history, model)


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


def get_model(train_data, dev_data, out_dir = "/tmp", batch_size = 6, param_tune = True, seed = 0):

	print("Param-Tune: {}".format(param_tune))

	out_dir = "{}/VGG16_functional/".format(out_dir)

	if not(os.path.isdir(out_dir)):
		os.makedirs(out_dir)

	if seed is not None: 
		np.random.seed(seed)


	train_data_hu, train_data_text, train_labels, train_pid, train_pslices, train_ref = train_data
	dev_data_hu, dev_data_text, dev_labels, dev_pid, dev_pslices, dev_ref = dev_data


	################################
	# Comment out this at the end
	# train_data_hu = train_data_hu[:10]
	# train_labels = train_labels[:10]

	# dev_data_hu = dev_data_hu[:10]
	# dev_labels = dev_labels[:10]
	################################

	rl_labels_train = get_RL_labels(train_labels)
	rl_labels_dev = get_RL_labels(dev_labels)

	train_data_hu = train_data_hu.astype(np.float32)
	dev_data_hu = dev_data_hu.astype(np.float32)
	
	print("Train data shape: {}".format(train_data_hu.shape))
	print("Dev data shape: {}".format(dev_data_hu.shape))
	

	# Set old strokes to 0 for now
	train_labels[train_labels < 0] = 0
	dev_labels[dev_labels < 0] = 0


	train_labels[train_labels > 1] = 0
	dev_labels[dev_labels > 1] = 0


	
	#######################################

	if param_tune:

		print("Functional param-tuning Not implemented yet ...")


		# params_to_tune = {
		# 	'lr': [0.0001, 0.001, 0.01, 0.1],
		# 	'batch_size': [6],
		# 	'activation':['relu', 'elu'],
		# 	'optimizer': ['Nadam', 'Adam'],
		# 	'dense_layer': [32, 64, 128, 256, 512],
		# 	'epochs': [10,15,20,25,30]
		# }

		


		# Temporary set of params to check if it works. Final model should use params_to_tune
		params_to_tune = {
			'lr': [0.0001, 0.001],
			'activation':['relu'],
			'dense_layer': [256],
			'epochs': [1]

		}

		# params_to_tune_t = collections.OrderedDict(sorted(params_to_tune_t.items()))



		# ta_scanOut = ta.Scan(train_data_hu, train_labels, params = params_to_tune, model = aspects, x_val = dev_data_hu, y_val = dev_labels)


		# print("Scan details:\n{}".format(ta_scanOut.details))
		

		# r = ta.Reporting(ta_scanOut)

		# print("Max score: {}".format(r.high()))
		# print("Number of rounds: {}".format(r.rounds()))

		# print("Max scored round: {}".format(r.rounds2high()))
		# print("Best params:\n{}".format(r.best_params()))

		
		


	else:
		print("In else part ...")

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
		

		history, model = aspects(train_data_hu, train_labels, rl_labels_train, dev_data_hu, dev_labels, rl_labels_dev, best_params_dict)
	
		model.save_weights("{}/deepASPECTS_vgg16functional_tuned_weights.h5".format(out_dir))
		
		with open("{}/deepASPECTS_vgg16functional_tuned_architecture.json".format(out_dir), 'w') as outf:
			outf.write(model.to_json())



		predicted_labels, predicted_rl = evaluate_model(model, dev_data_hu, dev_labels, rl_labels_dev, batch_size = batch_size, out_dir = out_dir)

		metrics.report(dev_labels, predicted_labels, rl_labels_dev, predicted_rl, out_dir = out_dir, datatype = "dev", modeltype = "sequential")	#functional

		pd.DataFrame(history.history).to_csv("{}/vgg16_m456_history_vgg16_functional.csv".format(out_dir))

		# Plot training & validation accuracy values
		plt.plot(history.history['Regions_acc'])
		plt.plot(history.history['RL_acc'])
		plt.plot(history.history['val_Regions_acc'])
		plt.plot(history.history['val_RL_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train Sig', 'Train Softmax', 'Val Sig', 'Val Softmax'], loc='upper left')
		plt.savefig("{}/accuracy_plot_functional.png".format(out_dir))
		plt.close()


		plt.plot(history.history['loss'])
		plt.plot(history.history['RL_loss'])
		plt.plot(history.history['val_Regions_loss'])
		plt.plot(history.history['val_RL_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train Sig', 'Train Softmax', 'Val Sig', 'Val Softmax'], loc='upper left')
		plt.savefig("{}/loss_plot_functional.png".format(out_dir))
		plt.close()


	