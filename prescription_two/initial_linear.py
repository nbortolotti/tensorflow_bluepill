import time
import os
import tensorflow as tf

import matplotlib.pyplot as plt

path = os.path.dirname(__file__)
full_path = path + '/../support/heart.csv'

ds_load = tf.data.experimental.CsvDataset(
    full_path,
    [tf.float32,
     tf.float32,
     ],
    select_cols=[1, 2]
)

for element in ds_load:
    print(element)

# variables
w = tf.Variable(0.0)
b = tf.Variable(0.0)

n_samples = 10

def prediction(x):
    return x * w + b


def squared_loss(y, y_predicted):
    return (y - y_predicted) ** 2


def huber_loss(y, y_predicted, m=1.0):
    t = y - y_predicted
    return t ** 2 if tf.abs(t) <= m else m * (2 * tf.abs(t) - m)

def train(loss_fn):
  print('Training; loss function: ' + loss_fn.__name__)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

  # Define the function through which to differentiate.
  def loss_for_example(x, y):
    return loss_fn(y, prediction(x))


  grad_fn = tf.implicit_value_and_gradients(loss_for_example)

  start = time.time()
  for epoch in range(100):
    total_loss = 0.0
    for x_i, y_i in tf.Iterator(ds_load):
      loss, gradients = grad_fn(x_i, y_i)

      # Take an optimization step and update variables.
      optimizer.apply_gradients(gradients)
      total_loss += loss
    if epoch % 10 == 0:
      print('Epoch {0}: {1}'.format(epoch, total_loss / n_samples))
  print('Took: %f seconds' % (time.time() - start))



train(huber_loss)
#plt.plot(data[:,0], data[:,1], 'bo')
#plt.plot(data[:,0], data[:,0] * w.numpy() + b.numpy(), 'r', label="huber regression")
#plt.legend()
#plt.show()