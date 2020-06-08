import argparse
import logging
import numpy as np
import math
import pandas as pd
import pickle
from sklearn import preprocessing as skl
import warnings

from perceptron_multiple import *
from utils import *

warnings.filterwarnings('ignore')

def ej1(filename, model_name):
	logi("Starting Exercise 1")
	logi("Reading Data")

	data = pd.read_csv(filename, header=None)
	data = normalize(data, -1, 1)
	# data = data.sample(int(len(data)*0.8))

	x = data[list(range(1,11))].to_numpy()
	z = np.array(data.result.tolist())

	learning_rate = 10e-3 * 3
	epsilon = 0.001
	max_epoch = 2000

	logi("LEARNING_RATE: {0}; EPSILON: {1}; MAX_EPOCH: {2}".format(learning_rate, epsilon, max_epoch))

	s = [10,10,10,1]
	l = len(s)
	w = [0] + [np.random.normal(0, s[0]**(0.5), (s[i-1]+1, s[i])) for i in range(1, l)]
	
	logi("Architecture: {0}; # Hidden Layers: {1}; # Training data: {2}".format(s,l-2,len(x)))

	w, error = train(x, w, z, l, s, learning_rate, epsilon, max_epoch)

	save_model(w, model_name+".pkl")
	return

def ej2(filename, model_name):
	logi("Starting Exercise 2")
	logi("Reading Data")
	data = pd.read_csv(data_path+files[1], header=None)
	for c in data.columns:
    	data[c] = skl.minmax_scale(data[c], feature_range=(-1, 1))

    x = data[list(range(0,8))].to_numpy()
	z = data[list(range(8,10))].to_numpy()

	learning_rate = 10e-3 * 6
	epsilon = 0.0001
	max_epoch = 2000

	s = [8,8,8,2]
	l = len(s) # Cantidad de capas
	w = [0] + [np.random.normal(0, s[0]**(0.5), (s[i-1]+1, S[i])) for i in range(1, l)]
	w = train(x, W, z, learning_rate, epsilon, max_epoch)

	logi("LEARNING_RATE: {0}; EPSILON: {1}; MAX_EPOCH: {2}".format(learning_rate, epsilon, max_epoch))
	logi("Architecture: {0}; # Hidden Layers: {1}; # Training data: {2}".format(s,l-2,len(x)))

	save_model(w, model_name+".pkl")
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--info')
	parser.add_argument('--debug')
	args = parser.parse_args()
	if args.debug == "DEBUG":
		logging.basicConfig(level=logging.DEBUG)
	elif args.info == "INFO":
		logging.basicConfig(level=logging.INFO)
	
	logi = logging.info
	logd = logging.debug

	files = {'ej1':'tp1_ej1_training.csv','ej2':'tp1_ej2_training.csv'}
	data_path = "../data/"

	# ej1(data_path+files["ej1"], '../models/model_ej1')
	ej2(data_path+files["ej1"], '../models/model_ej1')