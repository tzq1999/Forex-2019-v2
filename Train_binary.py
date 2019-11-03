# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 13:22:05 2019

@author: Chenghai Li
"""
import os
import numpy as np
import tensorflow as tf
#import neural_structured_learning as nsl
from matplotlib import pyplot as plt

x_train_uni = np.load( r'data/x_train_uni.npy')
y_train_uni = np.load( r'data/y_train_uni.npy')
x_test_uni = np.load( r'data/x_test_uni.npy')
y_test_uni = np.load( r'data/y_test_uni.npy')
y_test = np.load( r'data/y_test.npy')

input_length = x_train_uni.shape[1]
predict_length = y_train_uni.shape[1]

BATCH_SIZE = 64
BUFFER_SIZE = x_train_uni.shape[0]

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, y_test_uni))
test_univariate = test_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

from Model import EncoderLayer

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__(name='my_model')

    self.Transformer = EncoderLayer(4, 2, 64)
    self.LSTM1 = tf.keras.layers.LSTM(32, return_sequences=True )
    self.LSTM2 = tf.keras.layers.LSTM(16, activation='tanh')
    self.FC1 = tf.keras.layers.Dense(predict_length, activation='tanh')

  def call(self, inputs):
  
    x = self.Transformer(inputs, True, None)
    x = self.LSTM1(x)
    x = self.LSTM2(x)
    x = self.FC1(x)
    
    return x

model = MyModel()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
    
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    
learning_rate = CustomSchedule(1)

optimizer = tf.keras.optimizers.RMSprop(0.001)

checkpoint_dir = './checkpoints'

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    monitor = 'val_accuracy',
    save_best_only = True,
    save_freq = 'epoch',
    save_weights_only = True
    )

EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta = 0.01,
    patience = 20,
    baseline = 0.52,
    restore_best_weights = True
    )

model.compile(loss = 'mse',
              optimizer = optimizer,
              metrics=['accuracy']
              )

history = model.fit(train_univariate,
                    epochs = 100,
                    steps_per_epoch = int(x_train_uni.shape[0] / BATCH_SIZE),
                    validation_data = test_univariate,
                    shuffle = True,
                    validation_steps = int(x_test_uni.shape[0] / BATCH_SIZE),
                    callbacks=[checkpoint_callback, EarlyStopping]
                    )

#model.save_weights('./checkpoints/ckpt')



