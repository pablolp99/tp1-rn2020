import numpy as np
import math
import pandas as pd
from sklearn import preprocessing as skl
from tqdm import tqdm

from utils import *

logi = logging.info
logd = logging.debug

def activation(xh, w, l, s):
	y = [np.zeros((1, s[i]+1))[0] for i in range(l-1)] + [np.zeros((1, s[-1]))[0]]
	y_temp = xh
	for k in range(1, l):
		y[k-1] = add_bias(y_temp)
		y_temp = tanh(y[k-1],w[k])
	y[-1] = y_temp
	return y

def correction(y, zh, lr, w, l, s):
	dw = [0] + [np.zeros((s[i-1]+1, s[i])) for i in range(l-1)]
	e = zh - y[-1]
	dy = 1 - y[-1] ** 2
	d = [np.zeros((1, s[i]+1))[0] for i in range(l-1)] + [np.zeros((1, s[-1]))[0]]
	d[-1] = e * dy
	for k in range(l-1,0,-1):
		dw[k] = lr * (y[k-1].T.dot(d[k]))
		e = d[k].dot(w[k].T)
		dy = 1 - y[k-1] ** 2
		d[k-1] = sub_bias(e * dy)
	return dw

def adaptation(w, dw, l):
	for k in range(1, l):
		w[k] += dw[k]
	return w

def estimation(zh, yh):
	return np.linalg.norm(zh-yh[-1])**2

def train(xp, w, z, l, s, lr, epsilon, max_epoch):
	logi("Training")
	error = 1
	epoch = 1
	with tqdm(total=max_epoch) as pbar:
		pbar.update(1)
		while error > epsilon and epoch < max_epoch:
			error = 0
			for h in range(0, len(xp)):
				yh = activation(xp[h:h+1], w, l, s)
				eh = estimation(z[h:h+1], yh)
				dw = correction(yh, z[h:h+1], lr, w, l, s)
				w = adaptation(w, dw, l)
				error += np.linalg.norm(eh)
			epoch += 1
			pbar.update(1)
			pbar.set_description("Error: {}".format(round(error,3)))
	return w