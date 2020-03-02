import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class tf_utilities():
    def generate_constant(self):
        tf_pill = tf.constant([[2., 3.], [7., 11.]])
        return tf_pill


class tf_operations():
    """
    A class used to represent an TF operations

    ...

    Attributes
    ----------
    tf_constant  = tf.constant
        TensorFlow constant to operate into examples

    Methods
    ---------
    constant_tensor_pill(a)

    square_tensor_pill(a)

    transpose_tensor_pill(a)

    """

    def __init__(self, tf_constant):
        self.tf_constant = tf_constant

    def constant_tensor_pill(self, a):
        return a + 7

    def square_tensor_pill(self, a):
        return tf.square(a)

    def transpose_tensor_pill(self, a):
        return tf.transpose(a)


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow constants and functions')
    pa.add_argument('--operation', dest='operation', required=False, help='operation')
    # constant, square, transpose

    args = pa.parse_args()

    utils = tf_utilities()
    operations = tf_operations(utils.generate_constant())

    if args.operation == "constant":
        resp = operations.constant_tensor_pill(operations.tf_constant)
    elif args.operation == "square":
        resp = operations.square_tensor_pill(operations.tf_constant)
    elif args.operation == "transpose":
        resp = operations.transpose_tensor_pill(operations.tf_constant)
    print(resp)
