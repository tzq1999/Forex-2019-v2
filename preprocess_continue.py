# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:27:46 2019

@author: Chenghai Li
"""

import csv
from args import args
import numpy as np
from matplotlib import pyplot as plt

arg = args()

step = arg.step
input_length = arg.input_length
predict_length = arg.predict_length
split_ratio = arg.split_ratio

file = open(r'data/EURUSD.csv')
file_csv = csv.reader(file)
data = []

for row in file_csv:
    row[2] = float(row[2])
    row[3] = float(row[3])
    row[4] = float(row[4])
    row[5] = float(row[5])
    data.append(row[2 : 6])

train = np.array(data[ : int(len(data) * split_ratio)], dtype = np.float32)
test = np.array(data[int(len(data) * split_ratio) : ], dtype = np.float32)
print('All data:', len(data), '  Train:', len(train), '  test:', len(test))


x_train = []
y_train = []

for i in range (0, len(train) - input_length - predict_length, step):
    x_train.append(train[i : i + input_length])
    y_train.append(train[i + input_length : i + input_length + predict_length, -1])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train_uni = np.zeros_like(x_train)
y_train_uni = np.zeros_like(y_train)
y_train_check = np.zeros_like(y_train)
for i in range(len(x_train)):
    
    s = x_train[i][0][0]
    x_train_uni[i] = x_train[i] / s * 100
    y_train_uni[i] = y_train[i] / s * 100
    d = x_train_uni[i].mean()
    x_train_uni[i] -= d
    y_train_uni[i] -= (d + x_train_uni[i][-1][-1])
    y_train_check[i] = y_train_uni[i]
    for j in range (predict_length):
        if y_train_uni[i][j] > 0.3:
            y_train_uni[i][j] = 0.3
        if y_train_uni[i][j] < -0.3:
            y_train_uni[i][j] = -0.3
            
    #for j in range(predict_length - 1):
    #    y_train_uni[i][-j-1] -= y_train_uni[i][-j-2]
    
#plt.hist(y_train_uni.flatten(), bins=200, color='steelblue', normed=True )

y_train_uni *= 1
y_train_check *= 1

x_test = []
y_test = []


for i in range (0, len(test) - input_length - predict_length, step):
    x_test.append(test[i : i + input_length])
    y_test.append(test[i + input_length : i + input_length + predict_length, -1])

x_test = np.array(x_test, dtype = np.float32)
y_test = np.array(y_test, dtype = np.float32)

x_test_uni = np.zeros_like(x_test)
y_test_uni = np.zeros_like(y_test)
y_test_pip = np.zeros_like(y_test)

for i in range(len(x_test)):
    
    s = x_test[i][0][0]
    x_test_uni[i] = x_test[i] / s * 100
    y_test_uni[i] = y_test[i] / s * 100
    d = x_test_uni[i].mean()
    x_test_uni[i] -= d
    y_test_uni[i] -= (d + x_test_uni[i][-1][-1])
    y_test_pip[i] = y_test[i][-1] - x_test[i][-1][-1]
    for j in range (predict_length):
        if y_test_uni[i][j] > 0.3:
            y_test_uni[i][j] = 0.3
        if y_test_uni[i][j] < -0.3:
            y_test_uni[i][j] = -0.3
            
    #for j in range(predict_length - 1):
    #    y_test_uni[i][-j-1] -= y_test_uni[i][-j-2]
    
y_test_uni *= 1

np.save( r'data/x_train_uni.npy', x_train_uni)
np.save( r'data/y_train_uni.npy', y_train_uni)
np.save( r'data/x_test_uni.npy', x_test_uni)
np.save( r'data/y_test_uni.npy', y_test_uni)
np.save( r'data/y_test.npy', y_test)
np.save( r'data/y_test_pip.npy', y_test_pip)

print('Processed data:', len(x_train_uni) + len(x_test_uni), '  Train:', len(x_train_uni), '  Test:', len(x_test_uni))
print('Saved!')