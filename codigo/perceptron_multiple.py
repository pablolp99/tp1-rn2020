import logging
import numpy as np
import math
from matplotlib import pyplot as plt, cm
import pandas as pd
import pickle
from sklearn import preprocessing as skl

logi = logging.info
logd = logging.debug

def tanh(xp, w):
	return np.tanh(xp.dot(w))

def add_bias(matrix):
	bias = -np.ones((len(matrix),1))
	return np.concatenate((matrix,bias), axis=1)

def sub_bias(matrix):
	return matrix[:,:-1]

def activation(xh, s, L, W):
	y = [np.zeros((1, s[i]+1))[0] for i in range(L-1)] + [np.zeros((1, s[-1]))[0]]
	y_temp = xh
	for k in range(1, L):
		y[k-1] = add_bias(y_temp)
		y_temp = tanh(y[k-1],W[k])
	y[-1] = y_temp
	return y

def correction(y, zh, S, L, W, lr):
	dw = [0] + [np.zeros((S[i-1]+1, S[i])) for i in range(L-1)]
	e = zh - y[-1]
	dy = 1 - y[-1] ** 2
	# Mismo concepto que el Y pero para atras
	# Mismas dimensiones que Y
	d = [np.zeros((1, S[i]+1))[0] for i in range(L-1)] + [np.zeros((1, S[-1]))[0]]
	d[-1] = e * dy
	for k in range(L-1,0,-1):
		dw[k] = lr * (y[k-1].T.dot(d[k]))
		e = d[k].dot(W[k].T)
		dy = 1 - y[k-1] ** 2
		d[k-1] = sub_bias(e * dy)
	return dw

def adaptation(w, dw, L):
	for k in range(1, L):
		w[k] += dw[k]
	return w

def estimation(zh, yh):
	return np.linalg.norm(zh-yh[-1])**2

def train(xp, w, z, l, s, lr, epsilon, max_epoch):
	logi("Starting Train")
	error = 1
	epoch = 1
	while error > epsilon and epoch < max_epoch:
		logi("EPOCH {}".format(epoch+1))
		error = 0
		for h in range(0, len(xp)):
			yh = activation(xp[h:h+1], s, l, w)
			eh = estimation(z[h:h+1], yh)
			dw = correction(yh, z[h:h+1], s, l, w, lr)
			w = adaptation(w, dw, l)
			error += np.linalg.norm(eh)
		logd(error)
		epoch += 1
	return w