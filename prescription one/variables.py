import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

scalar_example = tf.Variable(2, name='scalar_example')
matrix_example = tf.Variable([[1, 2], [3, 4]], name='matrix_example')
big_matrix_example = tf.Variable(tf.zeros([551, 100]), name='big_matrix_example')
