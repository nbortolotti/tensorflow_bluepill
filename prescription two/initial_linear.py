import time
import os
import tensorflow as tf

import matplotlib.pyplot as plt

path = os.path.dirname(__file__)
full_path = path + '/../support/heart.csv'

dataset = tf.data.experimental.CsvDataset(
    full_path,
    [tf.float32,
     tf.float32,
     ],
    select_cols=[1, 2]
)

for element in dataset:
    print(element)

# variables
# w = tf.Variable(0.0)
# b = tf.Variable(0.0)
#
#
# def prediction(x):
#     return x * w + b
#
#
# def squared_loss(y, y_predicted):
#     return (y - y_predicted) ** 2
#
#
# def huber_loss(y, y_predicted, m=1.0):
#     t = y - y_predicted
#     return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)
