# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:06:35 2019

@author: Chenghai Li
"""

import numpy as np
from talib import RSI

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
    
data = np.array(data)
rsi = RSI(data[:, -1], timeperiod=14)