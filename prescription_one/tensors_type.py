import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow tensor types')
    pa.add_argument('--operation', dest='operation', required=True, help='operation')
    # sparse, dense, ragged

    args = pa.parse_args()

    s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                        values=[1., 2., 3.],
                        dense_shape=[3, 4])

    print(s)
