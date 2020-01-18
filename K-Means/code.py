import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt 
import math



def initialize_mean(a, K, X):
	M = []
	for i in range(0, K):
		Y = []
		Y.append(X[i][0])
		Y.append(X[i][1])
		M.append(Y)
	return M

def new_mean(a, K,Class):
	M1 = []
	for i in range(0, K):
		Y1 = []
		x = 0
		y = 0
		for j in range(0, len(Class[i])):
			x += Class[i][j][0]
			y += Class[i][j][1]
		x = float(x/a)
		y = float(y/a)
		Y1.append(x)
		Y1.append(y)
		M1.append(Y1)
	return M1

def Euclid_distance(a,b):
	S = 0
	for i in range(0, 2):
		S += pow(a[i]-b[i],2)
	return S

def classify(a, K, X, M):
	Class = []
	for i in range(0, K):
		Y = []
		Class.append(Y)
	for i in range(0, a):
		dis = []
		for j in range(0, K):
			dis.append(Euclid_distance(X[i], M[j]))
		t = dis.index(min(dis))
		Class[t].append(X[i])
	return Class



df = pd.read_csv('group04.txt',  delim_whitespace=True, skiprows=1, nrows=4893,names=('X1', 'X2'),
                    dtype={'X1': np.float, 'X2': np.float})

X = np.array(list(zip(df.X1, df.X2)))
a = len(df.X1)
K = 3
M = initialize_mean(a, K, X)
Class = classify(a,K,X,M)
M = new_mean(a, K, Class)
while True:
	NewClass = classify(a,K,X,M)
	N = new_mean(a, K, NewClass)
	if M == N:
		break
	M = N
	# print(11)

colors = ['r','g','b','y','c']


for j in range(0, K):
	for i in range(0, len(NewClass[j])):
		plt.scatter(NewClass[j][i][0],NewClass[j][i][1], c=colors[j],s=7)
plt.show()
