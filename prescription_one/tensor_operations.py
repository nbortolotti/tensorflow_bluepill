import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def constant_tensor_pill(a):
    return a + 7


def square_tensor_pill(a):
    return tf.square(a)


def transpose_tensor_pill(a):
    return tf.transpose(a)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow constants and functions')
    pa.add_argument('--operation', dest='operation', required=True, help='operation')
    # constant, square, transpose

    args = pa.parse_args()

    tensor_pill = tf.constant([[2., 3.], [7., 11.]])

    if args.operation == "constant":
        resp = constant_tensor_pill(tensor_pill)
    elif args.operation == "square":
        resp = square_tensor_pill(tensor_pill)
    elif args.operation == "transpose":
        resp = transpose_tensor_pill(tensor_pill)
    print(resp)