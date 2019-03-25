import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def shape_tensor_pill(a):
    return a.shape


def type_tensor_pill(a):
    return a.dtype


def slice_tensor_pill(a):
    #return a[1:, :]
    #return a[..., 0, tf.newaxis]
    return a[1,..., tf.newaxis]

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow constants and functions')
    pa.add_argument('--operation', dest='operation', required=True, help='operation')
    # shape, type, slice

    args = pa.parse_args()

    tensor_pill = tf.constant([[2., 3., 5.], [7., 11., 13.]])

    if args.operation == "shape":
        resp = shape_tensor_pill(tensor_pill)
    elif args.operation == "type":
        resp = type_tensor_pill(tensor_pill)
    elif args.operation == "slice":
        resp = slice_tensor_pill(tensor_pill)
    print(resp)
