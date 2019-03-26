import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def ragged_add_pill(a, b):
    return tf.add(a, b)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow tensor types')
    pa.add_argument('--operation', dest='operation', required=True, help='operation')
    # sparse, dense, ragged

    args = pa.parse_args()

    # s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
    #                     values=[1., 2., 3.],
    #                     dense_shape=[3, 4])
    #
    # print(s)

    numbers_a = tf.ragged.constant([[3, 5], [], [7, 9, 1], [5], []])
    numbers_b = tf.ragged.constant([[1, 1], [], [1, 1, 1], [1], []])

    print(ragged_add_pill(numbers_a, numbers_b))
