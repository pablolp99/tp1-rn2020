import logging
import numpy as np
import math
from matplotlib import pyplot as plt, cm
import pandas as pd
import pickle
from sklearn import preprocessing as skl

logi = logging.info
logd = logging.debug

def normalize(data, range_bot, range_top):
	for c in data.columns:
		logd("Normalizing column {}".format(c))
		data[c] = skl.minmax_scale(data[c], feature_range=(range_bot, range_top))
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
