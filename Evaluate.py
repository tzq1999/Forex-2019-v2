# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:10:36 2019

@author: Chenghai Li
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')
y_test_pip = np.load( r'data/y_test_pip.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

from Model import EncoderLayerp as EncoderLayer

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__(name='my_model')

    self.Transformer = EncoderLayer(4, 2, 64)
    self.LSTM1 = tf.keras.layers.LSTM(32, return_sequences=True )
    self.LSTM2 = tf.keras.layers.LSTM(16, activation='tanh')
    self.FC1 = tf.keras.layers.Dense(predict_length, activation='sigmoid')

  def call(self, inputs):
  
    x = self.Transformer(inputs, True, None)
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = self.FC1(x)

    return x

model = MyModel()
model.load_weights('./checkpoints/ckpt')

class MyModelc(tf.keras.Model):

  def __init__(self):
    super(MyModelc, self).__init__(name='my_modelc')

    self.Transformer = EncoderLayer(4, 2, 64)
    self.LSTM1 = tf.keras.layers.LSTM(32, return_sequences=True )
    self.LSTM2 = tf.keras.layers.LSTM(16, activation='tanh')
    self.FC1 = tf.keras.layers.Dense(predict_length, activation='sigmoid')
  

  def call(self, inputs):
  
    x = self.Transformer(inputs, True, None)
  
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = self.FC1(x)
    
    return x * 2 - 1 

modelc = MyModelc()
modelc.load_weights('./checkpoints/ckpt1')

def pplot(i,test):
    
    if test ==0:
        pred = model.predict(x_train_uni[i].reshape(1,-1,4)).reshape(-1)*2
        plt.plot(pred,c = 'r')
        label = y_train_uni[i].reshape(-1)
        plt.plot(label,c = 'b')
    else:
        pred = model.predict(x_test_uni[i].reshape(1,-1,4)).reshape(-1)*8
        plt.plot(pred,c = 'r')
        label = y_test_uni[i].reshape(-1)
        plt.plot(label,c = 'b')
        
def test():
    
    right = 0
    pred = model.predict( x_test_uni )
    label = y_test_uni
    for i in range ( len( pred ) ):
        if pred[i][0] * label[i][0] > 0:
            right += 1
    return right / len(pred)


def btest(threshold):
    
    right = 0
    all_num = 0
    pred = model.predict( x_test_uni )
    label = y_test_uni
    for i in range ( len( pred) ):
        if pred[i][0] > 0.50 + threshold: 
            if label[i][0] > 0:
                right += 1
            all_num += 1
        if pred[i][0] < 0.50 - threshold: 
            if label[i][0] == 0:
                right += 1
            all_num += 1
    return right / all_num

def ptest(threshold):
    
    plist = []
    profit = 0
    pred = model.predict( x_test_uni )
    label = y_test_pip
    for i in range ( len( pred ) ):
        
        check = 0
        if pred[i] > 0.5 + threshold:
            profit += label[i][0]
            check = 1
        if pred[i] < 0.5 - threshold:
            profit-= label[i][0]
            check = 1
        if check == 1:
            profit -= 0.00020
            
        plist.append(profit)
    plt.plot(plist)
    return profit

def ptestcs(threshold):
    
    plist = []
    profit = 0
    pred = modelc.predict( x_test_uni )
    label = y_test_pip
    for i in range ( len( pred ) ):
        
        check = 0
        if pred[i] >  threshold:
            profit += label[i][0]
            check = 1
        if pred[i] <  - threshold:
            profit-= label[i][0]
            check = 1
        if check == 1:
            profit -= 0.00010
            
        plist.append(profit)
    plt.plot(plist)
    return profit

def ptestc(threshold):
    
    plist = []
    profit = 0
    pred = modelc.predict( x_test_uni )
    label = y_test_pip
    for i in range ( len( pred ) ):
        if abs(pred[i][0]) < threshold:
            continue
        
        profit += label[i][0] * pred[i][0] * 10000
        profit -= abs(pred[i][0]) * 0.0001 * 10000
        plist.append(profit)
        
    plt.plot(plist)
    return profit

#plt.hist(y_train_uni.flatten(), bins=200, color='steelblue', normed=True )
ptestcs(0.01)


