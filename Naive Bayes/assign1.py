import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import random
import math

def covar(Y):
	covar_mat = []
	for i in range(0, 3):
		X = np.cov(Y[i][0].transpose())
		covar_mat.append(X)
	return covar_mat

def mean_vec(Y):
	mean_vec = []
	for i in range(0, 3):
		t1 = Y[i][0].transpose()
		me = []
		X = np.mean(t1[0])
		T = np.mean(t1[1])
		me.append(X)
		me.append(T)
		mean_vec.append(me)
	return mean_vec

def probability(Y, M, C):
	X1 = (Y-M).transpose()
	Y1 = np.linalg.inv(C)
	Z1 = np.dot(Y1,X1)
	Z1 = np.dot(Y-M , Z1)
	X1 = np.linalg.det(C)
	Z1 = Z1 + math.log(X1)
	Z1 = -Z1
	return Z1 


def classify(covar_mat, mean_vec, Y, train_no):
	prob = []
	classify_Z = []
	for i in range(0, 3):
		prob.append(0)
		X = []
		classify_Z.append(X)
	for i in range(0,3):
		for j in range(0, len(Y[i][1])):
			for k in range(0, 3):
				prob[k] = probability(Y[i][1][j], mean_vec[k], covar_mat[k])
			# print(prob)
			t = prob.index(max(prob))
			# print("\n\n")
			classify_Z[t].append(Y[i][1][j])
	return classify_Z




txt1 = pd.read_csv('Class1.txt', delim_whitespace=True, names=('X1', 'X2'), dtype={'X1':np.float, 'X2':np.float})
txt2 = pd.read_csv('Class2.txt', delim_whitespace=True, names=('X1', 'X2'), dtype={'X1':np.float, 'X2':np.float})
txt3 = pd.read_csv('Class3.txt', delim_whitespace=True, names=('X1', 'X2'), dtype={'X1':np.float, 'X2':np.float})

X1 = np.array(list(zip(txt1.X1, txt1.X2)))
X2 = np.array(list(zip(txt2.X1, txt2.X2)))
X3 = np.array(list(zip(txt3.X1, txt3.X2)))

train_no = 0.5

l1 = len(X1)
l2 = len(X2)
l3 = len(X3)

Y1 = np.split(X1, [int(l1*train_no)])
Y2 = np.split(X2, [int(l2*train_no)])
Y3 = np.split(X3, [int(l3*train_no)])

Y = []
Y.append(Y1)
Y.append(Y2)
Y.append(Y3)
# print(Y[0][0][0][0])

# print(len(Y1[0]))


X = []
X.append(X1)
X.append(X2)
X.append(X3)

# print(X1[0])
# print(X2[0])
# print(X3[0])



colors = ['r', 'g', 'b']

# plt.subplot(2,1,1)
for i in range(0, 3):
	for j in range(int(l1*train_no), l1):	
		plt.scatter(X[i][j][0],X[i][j][1], c=colors[i],s=7)

plt.show()

covar_mat = covar(Y)
mean_vec = mean_vec(Y)
# for i in range(0, 3):
# 	print(mean_vec[i])
# 	print("\n\n")

# for i in range(0, 3):
# 	print(covar_mat[i])
# 	print("\n\n")
classify_Z = classify(covar_mat, mean_vec, Y, train_no)
for i in range(0, 3):
	print(len(classify_Z[i]))
# plt.subplot(2,2,2)
for i in range(0, 3):
	for j in range(0, len(classify_Z[i])):
		plt.scatter(classify_Z[i][j][0], classify_Z[i][j][1], c=colors[i],s=7)

plt.show()



