import logging
import numpy as np
import math
import pandas as pd
import pickle
from sklearn import preprocessing as skl

logi = logging.info
logd = logging.debug


def tanh(x, w):
	return np.tanh(x.dot(w))

def add_bias(m):
	bias = -np.ones((len(m),1))
	return np.concatenate((m,bias), axis=1)

def sub_bias(m):
	return m[:,:-1]

def normalize(data, range_bot, range_top):
	logi("Normalizing Data between ({}; {})".format(range_bot, range_top))
	result = data[0]
	for c in data.columns[1:]:
		data[c] = skl.minmax_scale(data[c], feature_range=(range_bot, range_top))
	data['result'] = data.apply(lambda x: -1 if x[0] == 'M' else 1, axis=1)
	return data

def save_model(model, filename='model.pkl'):
	logi("Saving model in {}".format(filename))
	with open(filename, 'wb') as file:
		pickle.dump(model, file)
	return

def load_model(filename='model.pkl'):
	logi("Loading model from {}".format(filename))
	with open(filename, 'rb') as file:
		model = pickle.load(file)
	return model
