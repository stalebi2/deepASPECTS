#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import hamming_loss, accuracy_score
import matplotlib.pyplot as plt

import numpy as np
import os, json, sys
import pandas as pd



def sensitivity_specificity(y_true, y_pred):

	trueNegative = falseNegative = truePositive = falsePositive = wrongPositive = 0

	sensitivity = -100.0
	specificity = -100.0
	
	for i in xrange(y_true.shape[0]):
		
		trueLabel = y_true[i]
		predLabel = y_pred[i]
		
		if np.sum(trueLabel - predLabel) == 0:
			
			if np.sum(trueLabel) == 0:
				trueNegative += 1
			else:
				truePositive += 1
		else:
			if np.sum(trueLabel) == 0:
				falsePositive += 1
			else:
				if np.sum(predLabel) == 0:
					falseNegative += 1
				else: 
					wrongPositive += 1
	
	sensitivity_demon = (truePositive + wrongPositive + falseNegative)
	if sensitivity_demon > 0:
		sensitivity = truePositive*100.0/(truePositive + wrongPositive + falseNegative)

	specificity_demon = (trueNegative + falsePositive)
	if specificity_demon > 0:
		specificity = trueNegative*100.0/(trueNegative + falsePositive)

	print("Sensitivity (%): {}".format(sensitivity))
	print("Specificity (%): {}".format(specificity))

	return (sensitivity, specificity)




def custom_hamming_loss(y_true, y_pred):


	hammingLoss = 0.0
	for idx in xrange(y_true.shape[0]):
		hammingLoss = hammingLoss + hamming_loss(y_true[idx], y_pred[idx])

	hammingLoss = hammingLoss*100.0/len(y_true)
		
	print("Hamming loss (%): {}".format(hammingLoss))

	return hammingLoss


def custom_exact_match_score(y_true, y_pred):

	exactMatchScore = 0.0
	for idx in xrange(y_true.shape[0]):
		exactMatchScore = exactMatchScore + accuracy_score(y_true[idx], y_pred[idx])

	exactMatchScore = exactMatchScore*100.0/len(y_true)
	
	print("Exact-match-score (%): {}".format(exactMatchScore))

	return exactMatchScore


def report(y_true, y_pred, y_true_rl = None, y_pred_rl = None, out_dir = "/tmp", datatype = "default", modeltype = "sequential"):

	if modeltype == "sequential" or y_true_rl is None or y_pred_rl is None:
		
		hammingLoss = custom_hamming_loss(y_true, y_pred)
		exactMatchScore = custom_exact_match_score(y_true, y_pred)

		sensitivity, specificity = sensitivity_specificity(y_true, y_pred)

		report = {}

		report["hamming_loss"] = hammingLoss
		report["exact_match_score"] = exactMatchScore
		report["sensitivity"] = sensitivity
		report["specificity"] = specificity


		outfile = "{}/aspects_metrics_report_{}.json".format(out_dir, datatype)
		with open(outfile, 'w') as outf:	
			json.dump(report, outf, sort_keys=True, indent=4)

		print("Metrics summary saved in: {}".format(outfile))

	else:

		print("Implement this part")