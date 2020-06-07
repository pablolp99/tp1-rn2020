import numpy as np
import math
from matplotlib import pyplot as plt, cm
import pandas as pd
import pickle
from sklearn import preprocessing as skl

from perceptron_multiple import *
from utils import *

def ejercicio1(data_name, model_name):
	# Cargo los datos del problema 1
	data = pd.read_csv(data_name, header=None)
	
	# Normalizo los datos entre (-1, 1)
	data = normalize(data, -1, 1)

	data['result'] = data.apply(lambda x: -1 if x[0] == 'M' else 1, axis=1)
	x = data[list(range(1,11))].to_numpy()
	z = np.array(data.result.tolist())

	learning_rate = 10e-3 * 2
	epsilon = 0.0001
	max_epoch = 1000

	P = len(x)
	S = [10,20,1]
	L = len(S)
	W = [0] + [np.random.normal(0, S[0]**(0.5), (S[i-1]+1, S[i])) for i in range(1, L)]
	W = train(x, W, z, learning_rate, epsilon, max_epoch)
	save_model(W, model_name)

def ejercicio2(data_name, model_name):
	return

if __name__ == '__main__':
	files = ['tp1_ej1_training.csv','tp1_ej1_training.csv']
	data_path = "../data/"
	ejercicio1(data_path+files[0], '../models/model_ej1.pkl')