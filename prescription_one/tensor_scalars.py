import os
import argparse
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def add_pill(a, b):
    sum_operation = tf.add(a, b, name='add')
    return sum_operation


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='tensorflow constants and functions')
    pa.add_argument('--a', dest='a', required=True, help='a parameter')
    pa.add_argument('--b', dest='b', required=True, help='b parameter')
    pa.add_argument('--operation', dest='operation', required=True, help='operation')

    args = pa.parse_args()

    if args.operation == "sum":
        result = add_pill(args.a, args.b)
        print(result)
